#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrated_gradients.py

Implementación de Integrated Gradients (IG firmado) a nivel de vóxel.
Calcula la atribución de características integrando los gradientes de la edad predicha
a lo largo de una trayectoria lineal desde una línea de base neutra hasta el escaneo real.
"""

from typing import Optional
import torch
import torch.nn as nn
import numpy as np

def score_scalar(model: nn.Module, img: torch.Tensor, loss_type: str = "soft", device: Optional[torch.device] = None) -> torch.Tensor:
    """Calcula el escalar continuo de predicción de edad."""
    device = device or img.device
    outs = model(img)
    logits = outs[0] if isinstance(outs, (list, tuple)) else outs
    if loss_type == "soft":
        probs = torch.softmax(logits, dim=1)
        bins = torch.arange(100, device=device).float()
        return (probs * bins).sum(dim=1).view([])
    else:
        return logits.view([])

def compute_integrated_gradients(
    model: nn.Module, 
    image: torch.Tensor, 
    loss_type: str = "soft", 
    baseline: Optional[torch.Tensor] = None, 
    steps: int = 50
) -> np.ndarray:
    """
    Calcula el mapa de atribución IG firmado de 5 canales.
    Retorna un array de numpy de forma (5, H, W) normalizado entre [-1, 1].
    Valores positivos (+) = aceleran la edad cerebral estimada.
    Valores negativos (-) = disminuyen la edad cerebral estimada.
    """
    model.eval()
    if image.ndim == 3:
        img = image.unsqueeze(0).detach().clone()
    else:
        img = image.detach().clone()
        
    device = img.device
    if baseline is None:
        base = torch.zeros_like(img)
    else:
        base = baseline.unsqueeze(0).to(device) if baseline.ndim == 3 else baseline.to(device)
        
    alphas = torch.linspace(0.0, 1.0, steps, device=device)
    total_grad = torch.zeros_like(img)
    
    for a in alphas:
        x_step = (base + a * (img - base)).clone().detach().requires_grad_(True)
        score = score_scalar(model, x_step, loss_type=loss_type, device=device)
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)
        if x_step.grad is not None:
            total_grad += x_step.grad
            
    avg_grad = total_grad / max(1, len(alphas))
    ig = (img - base) * avg_grad
    ig_5c = ig[0].detach().cpu().numpy()
    
    max_val = np.nanmax(np.abs(ig_5c))
    if max_val > 0:
        ig_5c = ig_5c / max_val
        
    return ig_5c
