"""
GlobalLocalTransformer with Soft Labels + EfficientNet Support + ResNet/VGG
"""
import torch
import torch.nn as nn
import math
from src.models import vgg as vnet
from torchvision import models
from src.models.resnet_backbone import ResNetFeatures

# ==============================================================================
# 1. HELPER MODULES (Attention, ConvBlock, FeedForward)
# ==============================================================================

class GlobalAttention(nn.Module):
    def __init__(self, transformer_num_heads=8, hidden_size=512, transformer_dropout_rate=0.0):
        super().__init__()
        self.num_attention_heads = transformer_num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(transformer_dropout_rate)
        self.proj_dropout = nn.Dropout(transformer_dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, locx, glox):
        locx_query_mix = self.query(locx)
        glox_key_mix = self.key(glox)
        glox_value_mix = self.value(glox)

        query_layer = self.transpose_for_scores(locx_query_mix)
        key_layer = self.transpose_for_scores(glox_key_mix)
        value_layer = self.transpose_for_scores(glox_value_mix)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output

class convBlock(nn.Module):
    def __init__(self, inplace, outplace, kernel_size=3, padding=1, norm='bn'):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplace, outplace, kernel_size=kernel_size, padding=padding, bias=False)
        if norm == 'bn':
            self.bn1 = nn.BatchNorm2d(outplace)
        elif norm == 'gn':
            self.bn1 = nn.GroupNorm(num_groups=min(32, outplace), num_channels=outplace, affine=True)
        elif norm == 'in':
            self.bn1 = nn.InstanceNorm2d(outplace, affine=True, track_running_stats=False)
        else:
            raise ValueError("norm must be 'bn'|'gn'|'in'")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class Feedforward(nn.Module):
    def __init__(self, inplace, outplace, norm='bn'):
        super().__init__()
        self.conv1 = convBlock(inplace, outplace, kernel_size=1, padding=0, norm=norm)
        self.conv2 = convBlock(outplace, outplace, kernel_size=1, padding=0, norm=norm)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# ==============================================================================
# 2. EFFICIENTNET WRAPPER (NUEVO)
# ==============================================================================
class EfficientNetFeatures(nn.Module):
    def __init__(self, name, in_ch=5, out_ch=512, pretrained=True):
        super().__init__()
        weights = 'DEFAULT' if pretrained else None
        
        # Seleccion de variante
        if name == 'efficientnet_b0':
            base = models.efficientnet_b0(weights=weights)
            enc_dim = 1280 # Dimension final de B0
        elif name == 'efficientnet_b3':
            base = models.efficientnet_b3(weights=weights)
            enc_dim = 1536 # Dimension final de B3
        else:
            raise ValueError(f"Variante {name} no implementada en el wrapper.")

        # --- CIRUGIA DE LA PRIMERA CAPA ---
        # EfficientNet espera 3 canales RGB. Nosotros tenemos in_ch (ej. 5).
        # Reemplazamos la primera conv manteniendo los pesos originales en los primeros 3 canales
        # y promediando/inicializando los nuevos.
        original_conv = base.features[0][0]
        new_conv = nn.Conv2d(in_ch, original_conv.out_channels, 
                             kernel_size=original_conv.kernel_size, 
                             stride=original_conv.stride, 
                             padding=original_conv.padding, 
                             bias=False)
        
        # Opcional: Copiar pesos de los canales RGB para no empezar de cero
        with torch.no_grad():
            if in_ch >= 3:
                new_conv.weight[:, :3, :, :] = original_conv.weight
                # Para los canales extra > 3, copiamos el promedio (estrategia comun)
                if in_ch > 3:
                    avg_weight = torch.mean(original_conv.weight, dim=1, keepdim=True)
                    for i in range(3, in_ch):
                        new_conv.weight[:, i:i+1, :, :] = avg_weight
            else:
                new_conv.weight[:, :in_ch, :, :] = original_conv.weight[:, :in_ch, :, :]
        
        base.features[0][0] = new_conv
        
        self.features = base.features
        
        # --- PROYECCION FINAL ---
        # El Transformer espera hidden_size (512). EfficientNet devuelve 1280 o 1536.
        # Usamos una conv 1x1 para proyectar.
        self.proj = nn.Conv2d(enc_dim, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.features(x)     # [B, enc_dim, H/32, W/32]
        x = self.proj(x)         # [B, 512, H/32, W/32]
        x = self.relu(self.bn(x))
        return x

# ==============================================================================
# 3. CLASE PRINCIPAL
# ==============================================================================
class GlobalLocalBrainAge(nn.Module):
    def __init__(self, inplace,
                 patch_size=64,
                 step=-1,
                 nblock=6,
                 drop_rate=0.5,
                 backbone='resnet18',
                 backbone_norm='bn',
                 backbone_pretrained=True,
                 backbone_freeze_bn=False,
                 num_classes=1): # <--- Soporte para Soft Labels
        super().__init__()
        self.patch_size = patch_size
        self.step = step if step > 0 else int(patch_size // 2)
        self.nblock = nblock
        self.num_classes = num_classes

        hidden_size = 512

        # --- SELECCION DE BACKBONE ---
        if backbone == 'vgg8':
            self.global_feat = vnet.VGG8(inplace)
            self.local_feat  = vnet.VGG8(inplace)
        elif backbone == 'vgg16':
            self.global_feat = vnet.VGG16(inplace)
            self.local_feat  = vnet.VGG16(inplace)
        elif backbone in ['resnet18', 'resnet34']:
            self.global_feat = ResNetFeatures(in_ch=inplace, name=backbone,
                                              cifar_style=True,
                                              pretrained=backbone_pretrained,
                                              norm=backbone_norm,
                                              freeze_bn=backbone_freeze_bn)
            self.local_feat  = ResNetFeatures(in_ch=inplace, name=backbone,
                                              cifar_style=True,
                                              pretrained=backbone_pretrained,
                                              norm=backbone_norm,
                                              freeze_bn=backbone_freeze_bn)
        elif backbone in ['efficientnet_b0', 'efficientnet_b3']:
            # Instanciamos el wrapper nuevo
            self.global_feat = EfficientNetFeatures(name=backbone, in_ch=inplace, out_ch=hidden_size, pretrained=backbone_pretrained)
            self.local_feat  = EfficientNetFeatures(name=backbone, in_ch=inplace, out_ch=hidden_size, pretrained=backbone_pretrained)
        else:
            raise ValueError(f'{backbone} model does not supported!')

        self.attnlist = nn.ModuleList()
        self.fftlist = nn.ModuleList()

        for _ in range(nblock):
            atten = GlobalAttention(
                transformer_num_heads=8,
                hidden_size=hidden_size,
                transformer_dropout_rate=drop_rate
            )
            self.attnlist.append(atten)

            fft = Feedforward(inplace=hidden_size*2,
                              outplace=hidden_size,
                              norm=backbone_norm)
            self.fftlist.append(fft)

        self.avg = nn.AdaptiveAvgPool2d(1)
        
        # --- CAPAS DE SALIDA DINAMICAS ---
        self.gloout = nn.Linear(hidden_size, num_classes)
        self.locout = nn.Linear(hidden_size, num_classes)

    def forward(self, xinput):
        _, _, H, W = xinput.size()
        outlist = []

        # Global
        xglo = self.global_feat(xinput)
        xgfeat = torch.flatten(self.avg(xglo), 1)
        glo = self.gloout(xgfeat)
        outlist = [glo]

        # Global flattened for attention
        B2, C2, H2, W2 = xglo.size()
        xglot = xglo.view(B2, C2, H2 * W2).permute(0, 2, 1)

        # Local sliding-window
        for y in range(0, H - self.patch_size, self.step):
            for x in range(0, W - self.patch_size, self.step):
                locx = xinput[:, :, y:y+self.patch_size, x:x+self.patch_size]
                xloc = self.local_feat(locx)

                for n in range(self.nblock):
                    B1, C1, H1, W1 = xloc.size()
                    xloct = xloc.view(B1, C1, H1 * W1).permute(0, 2, 1)

                    tmp = self.attnlist[n](xloct, xglot)
                    tmp = tmp.permute(0, 2, 1).view(B1, C1, H1, W1)
                    tmp = torch.cat([tmp, xloc], 1)

                    tmp = self.fftlist[n](tmp)
                    xloc = xloc + tmp

                xloc = torch.flatten(self.avg(xloc), 1)
                out = self.locout(xloc)
                outlist.append(out)

        return outlist

# ==============================================================================
# 4. TESTING BLOCK
# ==============================================================================
if __name__ == '__main__':
    # Generamos un tensor random [Batch, Canales, H, W]
    x1 = torch.rand(1, 5, 130, 170)
    
    print("--- Testeando GlobalLocalBrainAge con EfficientNet B0 y Soft Labels ---")
    try:
        mod = GlobalLocalBrainAge(inplace=5,
                                  patch_size=64,
                                  step=32,
                                  nblock=2,
                                  backbone='efficientnet_b0',
                                  num_classes=100) # Soft Labels
        
        zlist = mod(x1)
        
        print("Modelo instanciado correctamente.")
        for i, z in enumerate(zlist):
            print(f"Salida {i}: {z.shape}") # Deberia ser [1, 100]
        
        print(f"Cantidad total de salidas (Global + Parches): {len(zlist)}")
        print("--- TEST EXITOSO ---")
        
    except Exception as e:
        print(f"--- TEST FALLIDO: {e} ---")

# Alias for universal readability
GlobalLocalTransformer = GlobalLocalBrainAge
