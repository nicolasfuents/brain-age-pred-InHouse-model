#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grad_attention.py

Implementación de Grad-Attention (Transformer Attention Rollout ponderado por Gradiente).
Intercepta la matriz de auto-atención en la capa Softmax del último bloque Transformer del modelo global
y la multiplica por los gradientes de retropropagación para aislar los circuitos anatómicos causales.
"""

from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from skimage.transform import resize
from src.xai.integrated_gradients import score_scalar

class AttnCatcherHook:
    def __init__(self):
        self.f_list = []
        self.b_list = []
    def forward(self, module, inp, out):
        self.f_list.append(out.detach().clone())
    def backward(self, module, grad_in, grad_out):
        self.b_list.append(grad_out[0].detach().clone())

def compute_grad_attention(
    model: nn.Module, 
    image: torch.Tensor, 
    loss_type: str = "soft"
) -> np.ndarray:
    """
    Calcula el mapa de Grad-Attention para el modelo Global-Local Transformer.
    Retorna un array de forma (5, H, W) normalizado entre [0, 1].
    """
    model.eval()
    if image.ndim == 3:
        img = image.unsqueeze(0).detach().clone().requires_grad_(True)
    else:
        img = image.detach().clone().requires_grad_(True)
        
    device = img.device
    catcher = AttnCatcherHook()
    
    if not hasattr(model, "attnlist") or len(model.attnlist) == 0:
        return np.zeros(img.shape[1:], dtype=np.float32)
        
    last_block = model.attnlist[-1]
    if not hasattr(last_block, "softmax"):
        return np.zeros(img.shape[1:], dtype=np.float32)
        
    h1 = last_block.softmax.register_forward_hook(catcher.forward)
    h2 = last_block.softmax.register_full_backward_hook(catcher.backward)
    
    outs = model(img)
    if isinstance(outs, (list, tuple)) and len(outs) > 1:
        if loss_type == "soft":
            bins = torch.arange(100, device=device).float()
            local_ages = []
            for out_head in outs[1:]:
                probs = torch.softmax(out_head, dim=1)
                local_ages.append((probs * bins).sum(dim=1).view([]))
            score = torch.stack(local_ages).mean()
        else:
            score = torch.stack([o.view([]) for o in outs[1:]]).mean()
    else:
        score = score_scalar(model, img, loss_type=loss_type, device=device)
        
    model.zero_grad(set_to_none=True)
    score.backward()
    
    h1.remove()
    h2.remove()
    
    if len(catcher.f_list) == 0 or len(catcher.b_list) == 0:
        return np.zeros(img.shape[1:], dtype=np.float32)
        
    n_p = len(catcher.f_list)
    key_len = catcher.f_list[0].shape[-1]
    
    total_key_importance = torch.zeros(key_len, device=device)
    for i in range(n_p):
        attn_probs = catcher.f_list[i]
        grad_out = catcher.b_list[n_p - 1 - i]
        
        grad_attn = (attn_probs * grad_out).mean(dim=(0, 1))
        key_importance = grad_attn.sum(dim=0).abs()
        total_key_importance += key_importance
        
    _, C, H, W = img.shape
    if H == 182 and W == 218:
        gh, gw = 23, 28
    elif H == 182 and W == 182:
        gh, gw = 23, 23
    elif H == 218 and W == 182:
        gh, gw = 28, 23
    else:
        gh = int(np.sqrt(key_len))
        gw = key_len // gh
        
    grid_2d = total_key_importance.view(gh, gw).cpu().numpy()
    grid_resized = resize(grid_2d, (H, W), order=1, preserve_range=True, anti_aliasing=True)
    
    attn_map = np.stack([grid_resized] * C, axis=0)
    
    m = np.nanmax(np.abs(attn_map))
    if m > 0:
        attn_map = attn_map / m
        
    return attn_map.astype(np.float32)
