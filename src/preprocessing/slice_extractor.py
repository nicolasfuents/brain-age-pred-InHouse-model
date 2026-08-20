#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slice_extractor.py

Applies the SOLID_v2 intracranial brain mask, performs robust P1-P99 contrast normalization to [0, 1],
extracts 2.5D triplanar 5-slice stacks for Axial, Coronal, and Sagittal planes,
saves preprocessed NIfTI slice volumes and QC reports in 'prep/' and PyTorch .pt tensors in 'tensors/'.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
import json

def load_and_preprocess_volume(
    nii_path: Path, 
    mask_path: Path
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], nib.Nifti1Image]:
    """
    Loads MNI152 NIfTI volume and intracranial brain mask.
    Applies brain masking and robust percentile normalization P1-P99 to [0, 1].
    """
    nii = nib.load(str(nii_path))
    vol = np.asarray(nii.get_fdata(), dtype=np.float32)
    
    mask_nii = nib.load(str(mask_path))
    mask = (np.asarray(mask_nii.get_fdata()) > 0).astype(np.uint8)
    
    if vol.shape != mask.shape:
        raise ValueError(
            f"Dimension mismatch between volume {vol.shape} and MNI mask {mask.shape}."
        )
    
    # 1. Brain masking
    masked_vol = vol * mask
    brain_voxels = masked_vol[mask > 0]
    
    if len(brain_voxels) == 0:
        raise ValueError("Masked volume contains no valid brain voxels.")
    
    # 2. Robust P1-P99 percentile normalization
    p1 = float(np.percentile(brain_voxels, 1.0))
    p99 = float(np.percentile(brain_voxels, 99.0))
    
    if p99 <= p1:
        p99 = p1 + 1e-6
        
    clipped = np.clip(masked_vol, p1, p99)
    norm_vol = np.zeros_like(clipped, dtype=np.float32)
    norm_vol[mask > 0] = (clipped[mask > 0] - p1) / (p99 - p1)
    
    stats = {
        "p1_raw": p1,
        "p99_raw": p99,
        "mean_intensity": float(norm_vol[mask > 0].mean()),
        "std_intensity": float(norm_vol[mask > 0].std()),
        "brain_volume_voxels": int(mask.sum()),
        "volume_shape": list(vol.shape)
    }
    
    return norm_vol, mask, stats, nii

def extract_triplanar_tensors(
    norm_vol: np.ndarray
) -> Dict[str, torch.Tensor]:
    """
    Extracts 5 contiguous central slices per anatomical orientation:
    - Axial: Z = [89, 90, 91, 92, 93] -> (5, 182, 218)
    - Coronal: Y = [107, 108, 109, 110, 111] -> (5, 182, 182)
    - Sagittal: X = [89, 90, 91, 92, 93] -> (5, 218, 182)
    """
    axial_slices = norm_vol[:, :, 89:94]
    axial_tensor = np.transpose(axial_slices, (2, 0, 1))
    
    coronal_slices = norm_vol[:, 107:112, :]
    coronal_tensor = np.transpose(coronal_slices, (1, 0, 2))
    
    sagittal_slices = norm_vol[89:94, :, :]
    sagittal_tensor = sagittal_slices
    
    return {
        "axial": torch.from_numpy(axial_tensor.copy()).float(),
        "coronal": torch.from_numpy(coronal_tensor.copy()).float(),
        "sagittal": torch.from_numpy(sagittal_tensor.copy()).float()
    }

def generate_preprocessing_qc_report(
    norm_vol: np.ndarray,
    mask: np.ndarray,
    stats: Dict[str, Any],
    out_png: Path,
    patient_id: str = "PATIENT"
):
    """
    Generates a 3x5 Preprocessing Quality Control (QC) visual panel
    showing all 5 extracted slices for Axial, Coronal, and Sagittal planes
    with brain mask boundaries and contrast checks.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axs = plt.subplots(3, 5, figsize=(15, 9.5), facecolor="#0f172a")
    planes_info = [
        ("axial", [89, 90, 91, 92, 93], "Axial (Z)"),
        ("coronal", [107, 108, 109, 110, 111], "Coronal (Y)"),
        ("sagittal", [89, 90, 91, 92, 93], "Sagittal (X)")
    ]
    
    for r_idx, (plane, indices, p_label) in enumerate(planes_info):
        for c_idx, s_idx in enumerate(indices):
            ax = axs[r_idx, c_idx]
            
            if plane == "axial":
                sl = np.rot90(norm_vol[:, :, s_idx])
                ms = np.rot90(mask[:, :, s_idx])
            elif plane == "coronal":
                sl = np.rot90(norm_vol[:, s_idx, :])
                ms = np.rot90(mask[:, s_idx, :])
            else:
                sl = np.rot90(norm_vol[s_idx, :, :])
                ms = np.rot90(mask[s_idx, :, :])
                
            ax.imshow(sl, cmap="gray", vmin=0.0, vmax=1.0, interpolation="bicubic")
            
            # Draw subtle mask boundary
            if ms.sum() > 0:
                ax.contour(ms, levels=[0.5], colors=["#10b981"], linewidths=0.6, alpha=0.7)
                
            ax.axis("off")
            ax.set_title(f"Slice {s_idx}", color="#94a3b8", fontsize=10, pad=4)
            
            if c_idx == 0:
                ax.text(-0.1, 0.5, p_label, transform=ax.transAxes, color="#38bdf8",
                        fontsize=12, fontweight="bold", rotation=90, va="center", ha="right")
                        
    title_str = (
        f"Preprocessing Quality Control (QC) Report: {patient_id}\n"
        f"MNI152 Alignment (1mm) | SOLID_v2 Brain Masking | P1-P99 Contrast Normalization [0, 1]\n"
        f"Brain Volume: {stats['brain_volume_voxels']:,} voxels | Raw P1: {stats['p1_raw']:.1f} | Raw P99: {stats['p99_raw']:.1f}"
    )
    fig.suptitle(title_str, color="white", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(out_png, dpi=200, facecolor="#0f172a", bbox_inches="tight")
    plt.close(fig)

def process_nifti_to_tensors(
    nii_path: Path, 
    mask_path: Path, 
    output_dir: Optional[Path] = None,
    patient_id: str = "PATIENT",
    save_qc: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Extracts triplanar tensors to 'tensors/' and preprocessed NIfTI slice stacks + QC report to 'prep/'.
    """
    norm_vol, mask, stats, src_nii = load_and_preprocess_volume(nii_path, mask_path)
    tensors = extract_triplanar_tensors(norm_vol)
    
    if output_dir:
        base_dir = Path(output_dir)
        # If passed output_dir is .../tensors, base_dir is parent, else base_dir is output_dir
        root_dir = base_dir.parent if base_dir.name in ["tensors", "prep"] else base_dir
        
        tensors_dir = root_dir / "tensors"
        prep_dir = root_dir / "prep"
        
        tensors_dir.mkdir(parents=True, exist_ok=True)
        prep_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save PyTorch tensors in tensors/
        for plane, tensor in tensors.items():
            torch.save(tensor, tensors_dir / f"tensor_{plane}.pt")
            
        # 2. Save preprocessed NIfTI volumes in prep/
        affine = src_nii.affine
        header = src_nii.header
        
        # Preprocessed 3D volume
        prep_3d_nii = nib.Nifti1Image(norm_vol, affine, header)
        nib.save(prep_3d_nii, str(prep_dir / f"{patient_id}_preprocessed_MNI152.nii.gz"))
        
        # Slices 5-channel NIfTI volumes
        axial_nii = nib.Nifti1Image(norm_vol[:, :, 89:94], affine, header)
        nib.save(axial_nii, str(prep_dir / f"{patient_id}_slices_axial_5c.nii.gz"))
        
        coronal_nii = nib.Nifti1Image(norm_vol[:, 107:112, :], affine, header)
        nib.save(coronal_nii, str(prep_dir / f"{patient_id}_slices_coronal_5c.nii.gz"))
        
        sagittal_nii = nib.Nifti1Image(norm_vol[89:94, :, :], affine, header)
        nib.save(sagittal_nii, str(prep_dir / f"{patient_id}_slices_sagittal_5c.nii.gz"))
        
        # QC metrics JSON
        with open(prep_dir / "preprocessing_qc_metrics.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
            
        # 3. Visual QC Report PNG in prep/
        if save_qc:
            qc_png = prep_dir / "preprocessing_qc_report.png"
            generate_preprocessing_qc_report(norm_vol, mask, stats, qc_png, patient_id=patient_id)
            
    return tensors
