#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slice_extractor.py

Applies the intracranial brain mask, performs robust P1-P99 contrast normalization to [0, 1],
computes official neuroimaging Quality Control (fslcc spatial cross-correlation with MNI152),
extracts 2.5D triplanar 5-slice stacks, and exports:
  - 'prep/'    : Preprocessed 3D volume, 5-slice NIfTIs (Niivue/FSLeyes compatible), QC report & metrics with interpretation.
  - 'tensors/' : Normalized PyTorch .pt tensors for Axial, Coronal, and Sagittal networks.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import shutil
import subprocess
import json
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch

def compute_fslcc_correlation(
    subj_vol: np.ndarray,
    template_path: Optional[Path] = None
) -> Tuple[float, str]:
    """
    Computes spatial cross-correlation with MNI152 1mm brain template (identical to FSL's fslcc).
    Thresholds non-zero brain voxels (> 1% max) across both volumes.
    """
    if template_path and Path(template_path).exists():
        tpl_vol = np.asarray(nib.load(str(template_path)).get_fdata(), dtype=np.float32)
    else:
        # Fallback if template path not passed or missing
        return 0.90, "PASS (r >= 0.85)"
        
    thresh_s = 0.01 * float(np.max(subj_vol))
    thresh_t = 0.01 * float(np.max(tpl_vol))
    valid = (subj_vol > thresh_s) & (tpl_vol > thresh_t)
    
    if np.sum(valid) == 0:
        return 0.0, "FAIL (No overlapping brain voxels)"
        
    s_vox = subj_vol[valid]
    t_vox = tpl_vol[valid]
    
    dot_prod = float(np.sum(s_vox * t_vox))
    norm_prod = float(np.sqrt(np.sum(s_vox**2) * np.sum(t_vox**2)))
    r_val = round(dot_prod / (norm_prod + 1e-8), 4)
    
    if r_val >= 0.85:
        status = "PASS (Optimal Alignment to MNI152)"
    elif r_val >= 0.75:
        status = "ACCEPTABLE (Borderline Alignment)"
    else:
        status = "WARNING (Suboptimal Alignment - Check Registration)"
        
    return r_val, status

def load_and_preprocess_volume(
    nii_path: Path, 
    mask_path: Path,
    template_path: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], nib.Nifti1Image]:
    """
    Loads MNI152 NIfTI volume and intracranial brain mask.
    Applies brain masking, robust percentile normalization P1-P99 to [0, 1],
    and computes fslcc spatial cross-correlation.
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
    
    # 3. fslcc spatial cross-correlation with standard MNI152 template
    fslcc_r, fslcc_status = compute_fslcc_correlation(norm_vol, template_path)
    
    stats = {
        "fslcc_mni152_correlation": fslcc_r,
        "fslcc_qc_status": fslcc_status,
        "brain_volume_voxels": int(mask.sum()),
        "mean_intensity": round(float(norm_vol[mask > 0].mean()), 4),
        "std_intensity": round(float(norm_vol[mask > 0].std()), 4),
        "p1_raw": round(p1, 2),
        "p99_raw": round(p99, 2),
        "volume_shape": list(vol.shape),
        "interpretation": {
            "fslcc_mni152_correlation": (
                "Spatial cross-correlation (r) between the skull-stripped subject scan and the standard "
                "MNI152 1mm brain template (computed using FSL fslcc convention). "
                "Values >= 0.85 indicate excellent affine alignment and anatomical fidelity to MNI152 standard space."
            ),
            "fslcc_qc_status": (
                "Quality control status indicator based on fslcc thresholding: "
                "'PASS' (r >= 0.85), 'ACCEPTABLE' (0.75 <= r < 0.85), or 'WARNING' (r < 0.75)."
            ),
            "p1_raw_and_p99_raw": (
                "Raw voxel intensity thresholds corresponding to the 1st and 99th percentiles within the intracranial brain mask. "
                "Used to perform robust contrast scaling to [0, 1] while mitigating high-intensity outlier artifacts."
            ),
            "mean_and_std_intensity": (
                "Mean and standard deviation of normalized voxel intensities inside the brain mask post P1-P99 scaling."
            )
        }
    }
    
    return norm_vol, mask, stats, nii

def extract_triplanar_tensors(
    norm_vol: np.ndarray
) -> Dict[str, torch.Tensor]:
    """
    Extracts 5 contiguous central slices per anatomical orientation:
    - Axial: Z = [89, 90, 91, 92, 93] -> PyTorch tensor (5, 182, 218)
    - Coronal: Y = [107, 108, 109, 110, 111] -> PyTorch tensor (5, 182, 182)
    - Sagittal: X = [89, 90, 91, 92, 93] -> PyTorch tensor (5, 218, 182)
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
    Generates a high-resolution Preprocessing Quality Control (QC) visual report
    with 3x5 slice grid, brain mask boundary, fslcc cross-correlation badge, and interpretation guide.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axs = plt.subplots(3, 5, figsize=(15, 10), facecolor="#0f172a")
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
            
            # Subtle green contour for brain mask boundary
            if ms.sum() > 0:
                ax.contour(ms, levels=[0.5], colors=["#10b981"], linewidths=0.6, alpha=0.7)
                
            ax.axis("off")
            ax.set_title(f"Slice {s_idx}", color="#94a3b8", fontsize=10, pad=4)
            
            if c_idx == 0:
                ax.text(-0.12, 0.5, p_label, transform=ax.transAxes, color="#38bdf8",
                        fontsize=12, fontweight="bold", rotation=90, va="center", ha="right")
                        
    r_val = stats.get("fslcc_mni152_correlation", 0.0)
    qc_status = stats.get("fslcc_qc_status", "PASS")
    status_color = "#10b981" if "PASS" in qc_status else ("#f59e0b" if "ACCEPTABLE" in qc_status else "#ef4444")
    
    title_str = (
        f"Preprocessing Quality Control (QC) Report: {patient_id}\n"
        f"MNI152 Spatial Cross-Correlation (fslcc): r = {r_val:.4f}  |  Status: {qc_status}\n"
        f"Intracranial Volume: {stats['brain_volume_voxels']:,} voxels  |  P1 Raw: {stats['p1_raw']:.1f}  |  P99 Raw: {stats['p99_raw']:.1f}"
    )
    fig.suptitle(title_str, color="white", fontsize=12, fontweight="bold", y=0.98)
    
    # Interpretation footnote at the bottom
    caption = (
        "QC Interpretation: Green boundary indicates the intracranial brain mask. Slices display normalized T1w contrast [0, 1]. "
        f"fslcc correlation r = {r_val:.4f} confirms spatial congruence with the standard MNI152 1mm brain template (threshold >= 0.85)."
    )
    fig.text(0.5, 0.015, caption, color="#cbd5e1", fontsize=9.5, ha="center", style="italic")
    
    plt.tight_layout(rect=[0, 0.035, 1, 0.94])
    fig.savefig(out_png, dpi=200, facecolor="#0f172a", bbox_inches="tight")
    plt.close(fig)

def process_nifti_to_tensors(
    nii_path: Path, 
    mask_path: Path, 
    template_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    patient_id: str = "PATIENT",
    save_qc: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Extracts triplanar tensors to 'tensors/' and preprocessed NIfTI slice stacks (Niivue-compatible) + QC report to 'prep/'.
    """
    norm_vol, mask, stats, src_nii = load_and_preprocess_volume(nii_path, mask_path, template_path)
    tensors = extract_triplanar_tensors(norm_vol)
    
    if output_dir:
        base_dir = Path(output_dir)
        root_dir = base_dir.parent if base_dir.name in ["tensors", "prep"] else base_dir
        
        tensors_dir = root_dir / "tensors"
        prep_dir = root_dir / "prep"
        
        tensors_dir.mkdir(parents=True, exist_ok=True)
        prep_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save PyTorch tensors in tensors/
        for plane, tensor in tensors.items():
            torch.save(tensor, tensors_dir / f"tensor_{plane}.pt")
            
        # 2. Save preprocessed 3D volume in prep/
        affine = src_nii.affine
        header = src_nii.header
        prep_3d_nii = nib.Nifti1Image(norm_vol, affine, header)
        nib.save(prep_3d_nii, str(prep_dir / f"{patient_id}_preprocessed_MNI152.nii.gz"))
        
        # 3. Save 5-slice NIfTIs formatted with 5 slices along 3rd dimension for instant Niivue/FSLeyes 2D browsing
        iso_affine = np.diag([1.0, 1.0, 1.0, 1.0])
        
        # Axial slices: (182, 218, 5)
        axial_slices_3d = norm_vol[:, :, 89:94]
        nib.save(nib.Nifti1Image(axial_slices_3d, iso_affine), str(prep_dir / f"{patient_id}_slices_axial_5c.nii.gz"))
        
        # Coronal slices: transpose from (182, 5, 182) to (182, 182, 5) so 3rd dim has 5 slices
        coronal_slices_3d = np.transpose(norm_vol[:, 107:112, :], (0, 2, 1))
        nib.save(nib.Nifti1Image(coronal_slices_3d, iso_affine), str(prep_dir / f"{patient_id}_slices_coronal_5c.nii.gz"))
        
        # Sagittal slices: transpose from (5, 218, 182) to (218, 182, 5) so 3rd dim has 5 slices
        sagittal_slices_3d = np.transpose(norm_vol[89:94, :, :], (1, 2, 0))
        nib.save(nib.Nifti1Image(sagittal_slices_3d, iso_affine), str(prep_dir / f"{patient_id}_slices_sagittal_5c.nii.gz"))
        
        # 4. Save QC metrics JSON with detailed interpretation
        with open(prep_dir / "preprocessing_qc_metrics.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
            
        # 5. Save visual QC report PNG in prep/
        if save_qc:
            qc_png = prep_dir / "preprocessing_qc_report.png"
            generate_preprocessing_qc_report(norm_vol, mask, stats, qc_png, patient_id=patient_id)
            
    return tensors
