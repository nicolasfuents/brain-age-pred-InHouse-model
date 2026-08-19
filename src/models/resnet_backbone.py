# resnet_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def _replace_bn(module: nn.Module, norm: str = "bn"):
    """
    Reemplaza BatchNorm2d por GroupNorm o InstanceNorm si se solicita.
    - norm='bn'  -> no cambia nada
    - norm='gn'  -> GroupNorm con num_groups= min(32, num_channels)
    - norm='in'  -> InstanceNorm2d
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            num_ch = child.num_features
            if norm == "gn":
                new_norm = nn.GroupNorm(num_groups=min(32, num_ch), num_channels=num_ch, affine=True)
            elif norm == "in":
                new_norm = nn.InstanceNorm2d(num_ch, affine=True, track_running_stats=False)
            else:
                new_norm = child  # 'bn'
            setattr(module, name, new_norm)
        else:
            _replace_bn(child, norm)
    return module

def _freeze_all_bn(module: nn.Module):
    """
    Coloca todas las BN (si existen) en eval y congela sus parámetros.
    Útil con batch chico.
    """
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

class ResNetFeatures(nn.Module):
    """
    Extrae mapas de features (sin avgpool ni fc).
    name: 'resnet18' | 'resnet34' | 'resnet50'
    in_ch: 1 o 5 (tu caso típico).
    cifar_style=True -> conv1 3x3 s=1, sin maxpool inicial (para cortes 2D).
    norm: 'bn' | 'gn' | 'in'
    pretrained: bool (ImageNet)
    freeze_bn: si True, BN en eval y sin grad.
    out_proj: si se especifica, proyecta el canal de salida a ese número (ej. 512).
    """
    def __init__(self,
                 in_ch=1,
                 name='resnet34',
                 cifar_style=True,
                 pretrained=False,
                 norm='bn',
                 freeze_bn=False,
                 out_proj=None):
        super().__init__()
        assert name in ['resnet18', 'resnet34', 'resnet50']
        self.name = name

        if name == 'resnet18':
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            out_ch = 512
        elif name == 'resnet34':
            base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            out_ch = 512
        else:
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            out_ch = 2048

        # Adaptar conv1 a in_ch (1, 5, etc.) preservando pesos preentrenados
        if in_ch != 3:
            w = base.conv1.weight.data.clone()  # [out, 3, k, k] si venía de ImageNet
            base.conv1 = nn.Conv2d(in_ch, base.conv1.out_channels,
                                   kernel_size=base.conv1.kernel_size,
                                   stride=base.conv1.stride,
                                   padding=base.conv1.padding,
                                   bias=False)
            with torch.no_grad():
                if w.shape[1] == 3 and in_ch == 1:
                    base.conv1.weight.copy_(w.mean(dim=1, keepdim=True))
                elif w.shape[1] == 3 and in_ch > 1:
                    base.conv1.weight.copy_(w.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1) / in_ch)

        # Estilo CIFAR para MRI 2D: conservar detalle temprano
        if cifar_style:
            base.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity()

        # Reemplazo de normalización si se pide (gn/in)
        if norm in ("gn", "in"):
            _replace_bn(base, norm=norm)

        # Congelar BN (si quedó BN) para batches chicos
        if freeze_bn and norm == "bn":
            _freeze_all_bn(base)

        # Guardamos hasta layer4 (sin avgpool ni fc)
        self.stem = nn.Sequential(base.conv1, base.bn1 if hasattr(base, 'bn1') else nn.Identity(),
                                  base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # Proyección opcional para homogeneizar a 512
        self.out_ch = out_ch
        self.proj = None
        self.out_bn = None
        if out_proj is not None and out_proj != out_ch:
            self.proj = nn.Conv2d(out_ch, out_proj, kernel_size=1, bias=False)
            # Normalización de salida consistente con 'norm'
            if norm == 'bn':
                self.out_bn = nn.BatchNorm2d(out_proj)
            elif norm == 'gn':
                self.out_bn = nn.GroupNorm(num_groups=min(32, out_proj), num_channels=out_proj, affine=True)
            elif norm == 'in':
                self.out_bn = nn.InstanceNorm2d(out_proj, affine=True, track_running_stats=False)
            self.out_ch = out_proj

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.proj is not None:
            x = self.proj(x)
            if self.out_bn is not None:
                x = self.out_bn(x)
            x = F.relu(x, inplace=True)
        return x
