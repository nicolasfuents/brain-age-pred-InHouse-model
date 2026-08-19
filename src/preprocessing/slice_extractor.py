#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slice_extractor.py

Aplica la máscara intracraneal SOLID_v2, realiza la normalización robusta de contraste P1-P99 a [0, 1]
y extrae las pilas 2.5D de 5 rebanadas centrales para los planos axial, coronal y sagital.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import nibabel as nib
import numpy as np
import torch

def load_and_preprocess_volume(
    nii_path: Path, 
    mask_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga el volumen NIfTI en espacio MNI152 y la máscara intracraneal.
    Aplica el recorte y normalización por percentiles P1-P99 a [0, 1].
    """
    nii = nib.load(str(nii_path))
    vol = np.asarray(nii.get_fdata(), dtype=np.float32)
    
    mask_nii = nib.load(str(mask_path))
    mask = (np.asarray(mask_nii.get_fdata()) > 0).astype(np.uint8)
    
    if vol.shape != mask.shape:
        raise ValueError(
            f"Dimensiones incompatibles entre el volumen {vol.shape} y la máscara MNI {mask.shape}."
        )
    
    # 1. Enmascaramiento intracraneal
    masked_vol = vol * mask
    brain_voxels = masked_vol[mask > 0]
    
    if len(brain_voxels) == 0:
        raise ValueError("El volumen enmascarado no contiene vóxeles cerebrales válidos.")
    
    # 2. Normalización robusta P1-P99
    p1 = float(np.percentile(brain_voxels, 1.0))
    p99 = float(np.percentile(brain_voxels, 99.0))
    
    if p99 <= p1:
        p99 = p1 + 1e-6
        
    clipped = np.clip(masked_vol, p1, p99)
    norm_vol = np.zeros_like(clipped, dtype=np.float32)
    norm_vol[mask > 0] = (clipped[mask > 0] - p1) / (p99 - p1)
    
    return norm_vol, mask

def extract_triplanar_tensors(
    norm_vol: np.ndarray
) -> Dict[str, torch.Tensor]:
    """
    Extrae las 5 rebanadas centrales contiguas por cada orientación anatómica:
    - Axial: Z = [89, 90, 91, 92, 93] -> (5, 182, 218)
    - Coronal: Y = [107, 108, 109, 110, 111] -> (5, 182, 182)
    - Sagital: X = [89, 90, 91, 92, 93] -> (5, 218, 182)
    """
    # Axial: rebanadas en eje Z (dim 2) -> (5, 182, 218)
    axial_slices = norm_vol[:, :, 89:94]
    axial_tensor = np.transpose(axial_slices, (2, 0, 1))
    
    # Coronal: rebanadas en eje Y (dim 1) -> (5, 182, 182)
    coronal_slices = norm_vol[:, 107:112, :]
    coronal_tensor = np.transpose(coronal_slices, (1, 0, 2))
    
    # Sagital: rebanadas en eje X (dim 0) -> (5, 218, 182)
    sagittal_slices = norm_vol[89:94, :, :]
    sagittal_tensor = sagittal_slices
    
    return {
        "axial": torch.from_numpy(axial_tensor.copy()).float(),
        "coronal": torch.from_numpy(coronal_tensor.copy()).float(),
        "sagittal": torch.from_numpy(sagittal_tensor.copy()).float()
    }

def process_nifti_to_tensors(
    nii_path: Path, 
    mask_path: Path, 
    output_dir: Optional[Path] = None
) -> Dict[str, torch.Tensor]:
    """Pipeline completo de extracción de tensores a partir de NIfTI MNI."""
    norm_vol, _ = load_and_preprocess_volume(nii_path, mask_path)
    tensors = extract_triplanar_tensors(norm_vol)
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for plane, tensor in tensors.items():
            out_file = output_dir / f"tensor_{plane}.pt"
            torch.save(tensor, out_file)
            
    return tensors
