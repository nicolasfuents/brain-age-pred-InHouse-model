#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
occlusion_sensitivity.py

Implementación de Occlusion Sensitivity (Perturbación sistemática por parches oclusivos).
Evalúa la causalidad anatómica directa calculando la caída de predicción:
Delta = Pred_Baseline - Pred_Ocluida
"""

import torch
import torch.nn as nn
import numpy as np
from src.xai.integrated_gradients import score_scalar

@torch.no_grad()
def compute_occlusion_sensitivity(
    model: nn.Module, 
    image: torch.Tensor, 
    loss_type: str = "soft", 
    patch_size: int = 32, 
    stride: int = 16,
    batch_size: int = 64
) -> np.ndarray:
    """
    Ocluye secuencialmente parches deslizantes a lo largo del tensor 2.5D.
    Retorna un array de forma (5, H, W) normalizado entre [-1, 1].
    """
    model.eval()
    if image.ndim == 3:
        img = image.unsqueeze(0).detach().clone()
    else:
        img = image.detach().clone()
        
    device = img.device
    base_pred = score_scalar(model, img, loss_type=loss_type, device=device).item()
    _, C, H, W = img.shape
    
    coords = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            coords.append((y, x))
            
    batch_imgs = []
    for y, x in coords:
        img_occ = img.clone()
        img_occ[..., y:y+patch_size, x:x+patch_size] = 0.0
        batch_imgs.append(img_occ[0])
        
    if not batch_imgs:
        return np.zeros((C, H, W), dtype=np.float32)
        
    batch_tensor = torch.stack(batch_imgs).to(device)
    preds = []
    
    for i in range(0, len(batch_tensor), batch_size):
        sub_batch = batch_tensor[i:i+batch_size]
        outs = model(sub_batch)
        logits = outs[0] if isinstance(outs, (list, tuple)) else outs
        if loss_type == "soft":
            probs = torch.softmax(logits, dim=1)
            bins = torch.arange(100, device=device).float()
            pred_vals = (probs * bins).sum(dim=1).cpu().numpy()
        else:
            pred_vals = logits.view(-1).cpu().numpy()
        preds.extend(pred_vals)
        
    occ_map = np.zeros((C, H, W), dtype=np.float32)
    count_map = np.zeros((C, H, W), dtype=np.float32)
    
    for idx, (y, x) in enumerate(coords):
        delta = base_pred - preds[idx]
        occ_map[:, y:y+patch_size, x:x+patch_size] += delta
        count_map[:, y:y+patch_size, x:x+patch_size] += 1.0
        
    count_map = np.clip(count_map, a_min=1.0, a_max=None)
    occ_map = occ_map / count_map
    
    max_val = np.nanmax(np.abs(occ_map))
    if max_val > 0:
        occ_map = occ_map / max_val
        
    return occ_map
