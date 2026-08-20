#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xai_engine.py

Explainable AI (XAI) Orchestrator.
Computes 3 feature attribution methods (Integrated Gradients, Occlusion Sensitivity, and Grad-Attention),
constructs in-memory 3D ensemble-weighted attribution maps (EWAM), maps attribution to anatomical regions (ROIs)
from the subcortical and cerebellar atlas, and renders visual panel figures.
"""

import os
import csv
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
import nibabel as nib
from skimage.transform import resize
from scipy.ndimage import gaussian_filter

from src.inference.predictor import TriplanarPredictor
from src.xai.integrated_gradients import compute_integrated_gradients
from src.xai.occlusion_sensitivity import compute_occlusion_sensitivity
from src.xai.grad_attention import compute_grad_attention
from src.xai.visualizer import (
    plot_xai_overlays_panel,
    plot_top_rois_bar_chart
)

def load_labels_map(labels_csv_path: Path) -> Dict[int, str]:
    """Loads ID -> Anatomical ROI name mapping dictionary."""
    m = {}
    if labels_csv_path.exists():
        with open(labels_csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    m[int(row[0])] = row[1].strip()
    return m

class XAIEngine:
    def __init__(
        self, 
        predictor: TriplanarPredictor,
        atlas_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        beta_weights: Optional[Dict[str, float]] = None
    ):
        self.predictor = predictor
        repo_root = Path(__file__).resolve().parents[2]
        
        self.atlas_path = atlas_path or (repo_root / "data/atlases/combined_subcortical_cerebellum_1mm.nii.gz")
        self.labels_path = labels_path or (repo_root / "data/atlases/combined_labels.csv")
        
        self.beta_weights = beta_weights or {
            "axial": 0.1306717,
            "coronal": 0.50210947,
            "sagittal": 0.37634322
        }
        
        self.slice_indices = {
            "axial": [89, 90, 91, 92, 93],
            "coronal": [107, 108, 109, 110, 111],
            "sagittal": [89, 90, 91, 92, 93]
        }
        
        self.id2label = load_labels_map(self.labels_path)
        
        # Load and resize atlas in memory if available
        if self.atlas_path.exists():
            atlas_vol = nib.load(str(self.atlas_path)).get_fdata().astype(np.int16)
            self.atlas_rs = resize(
                atlas_vol, (182, 218, 182), order=0, preserve_range=True, anti_aliasing=False
            ).astype(np.int16)
        else:
            self.atlas_rs = None

    def generate_explanations(
        self, 
        tensors: Dict[str, torch.Tensor],
        predictions: Dict[str, Any],
        output_dir: Path,
        patient_id: str = "PATIENT"
    ) -> Dict[str, Any]:
        """
        Executes the full XAI attribution pipeline:
        - 3 methods (IG, Occlusion, Grad-Attention) across all 3 anatomical planes.
        - In-memory 3D EWAM attribution volume construction.
        - Anatomical ROI quantification (Harvard-Oxford Subcortical + SUIT Cerebellum).
        - Generation of multi-method 4x3 visual panel PNG and ROI importance plots.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plane_configs = {
            "axial": (self.predictor.model_axial, "soft"),
            "coronal": (self.predictor.model_coronal, "smoothl1"),
            "sagittal": (self.predictor.model_sagittal, "mse")
        }
        
        plane_maps = {"ig": {}, "occ": {}, "attn": {}}
        print("\n[+] Computing Explainable AI (XAI) feature attribution maps...")
        
        for plane, (model, loss_type) in plane_configs.items():
            print(f"  * Processing {plane.capitalize()} plane (Integrated Gradients, Occlusion, Grad-Attention)...")
            tensor = tensors[plane].to(self.predictor.device)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
                
            # 1. Integrated Gradients
            ig_5c = compute_integrated_gradients(model, tensor, loss_type=loss_type, steps=50)
            plane_maps["ig"][plane] = ig_5c
            
            # 2. Occlusion Sensitivity
            occ_5c = compute_occlusion_sensitivity(model, tensor, loss_type=loss_type, patch_size=32, stride=16)
            plane_maps["occ"][plane] = occ_5c
            
            # 3. Grad-Attention
            attn_5c = compute_grad_attention(model, tensor, loss_type=loss_type)
            plane_maps["attn"][plane] = attn_5c
            
        # Central T1 anatomical slices
        t1_slices = {
            "axial": tensors["axial"].cpu().numpy()[0, 2] if tensors["axial"].ndim == 4 else tensors["axial"].cpu().numpy()[2],
            "coronal": tensors["coronal"].cpu().numpy()[0, 2] if tensors["coronal"].ndim == 4 else tensors["coronal"].cpu().numpy()[2],
            "sagittal": tensors["sagittal"].cpu().numpy()[0, 2] if tensors["sagittal"].ndim == 4 else tensors["sagittal"].cpu().numpy()[2]
        }
        
        # 2D collapsed overlay slices
        overlay_slices = {}
        for m_name in ["ig", "occ", "attn"]:
            overlay_slices[m_name] = {}
            for plane in ["axial", "coronal", "sagittal"]:
                arr_2d = plane_maps[m_name][plane].sum(axis=0)
                m = np.nanmax(np.abs(arr_2d))
                if m > 0:
                    arr_2d = arr_2d / m
                overlay_slices[m_name][plane] = arr_2d
                
        # 1. Generate Multi-Method 4x3 Panel (xai_overlays_panel.png)
        panel_path = output_dir / "xai_overlays_panel.png"
        plot_xai_overlays_panel(
            t1_slices=t1_slices,
            overlay_slices=overlay_slices,
            predictions=predictions,
            out_path=panel_path,
            patient_id=patient_id
        )
        print(f"  * [✓] Multi-method attribution panel saved to: {panel_path}")
        
        # 2. Anatomical ROI Mapping (3D EWAM in memory)
        roi_stats = {"ig": {}, "occ": {}}
        if self.atlas_rs is not None:
            print("  * Quantifying spatial attribution across subcortical/cerebellar ROIs...")
            methods_3d = {}
            for m_name in ["ig", "occ", "attn"]:
                vol_3d = np.zeros((182, 218, 182), dtype=np.float32)
                for p in ["axial", "coronal", "sagittal"]:
                    beta = self.beta_weights.get(p, 0.33)
                    map_5c = plane_maps[m_name][p]
                    idxs = self.slice_indices[p]
                    for c in range(5):
                        idx = idxs[c]
                        slice_raw = map_5c[c]
                        if p == "axial":
                            slice_2d = slice_raw if slice_raw.shape == vol_3d[:, :, idx].shape else slice_raw.T
                            vol_3d[:, :, idx] += beta * slice_2d
                        elif p == "coronal":
                            slice_2d = slice_raw if slice_raw.shape == vol_3d[:, idx, :].shape else slice_raw.T
                            vol_3d[:, idx, :] += beta * slice_2d
                        elif p == "sagittal":
                            slice_2d = slice_raw if slice_raw.shape == vol_3d[idx, :, :].shape else slice_raw.T
                            vol_3d[idx, :, :] += beta * slice_2d
                            
                vol_3d = gaussian_filter(vol_3d, sigma=2.55)
                vol_3d = np.where(self.atlas_rs > 0, vol_3d, 0.0)
                m = np.nanmax(np.abs(vol_3d))
                if m > 0:
                    vol_3d = vol_3d / m
                methods_3d[m_name] = vol_3d
                
            # Extract quantitative metrics per ROI
            for m_name in ["ig", "occ"]:
                vol = methods_3d[m_name]
                for rid in np.unique(self.atlas_rs):
                    if rid == 0: continue
                    mask = (self.atlas_rs == rid)
                    vals = vol[mask]
                    if vals.size > 0:
                        roi_stats[m_name][int(rid)] = {
                            "mean_net": float(vals.mean()),
                            "sum_abs": float(np.abs(vals).sum()),
                            "area_vox": int(vals.size)
                        }
                        
                # Export ROI CSV
                csv_path = output_dir / f"roi_importance_{m_name}.csv"
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["roi_id", "roi_name", "mean_net_attribution", "sum_abs_attribution", "voxel_count"])
                    for rid, s in roi_stats[m_name].items():
                        rname = self.id2label.get(rid, f"ROI_{rid}")
                        writer.writerow([rid, rname, f"{s['mean_net']:.6f}", f"{s['sum_abs']:.4f}", s['area_vox']])
                        
                # Generate Top 10 ROI importance bar chart
                chart_path = output_dir / f"roi_importance_{m_name}.png"
                plot_top_rois_bar_chart(
                    roi_stats=roi_stats[m_name],
                    id2label=self.id2label,
                    method_name=m_name.upper(),
                    out_path=chart_path,
                    top_n=10
                )
            print(f"  * [✓] Top ROI importance charts saved to: {output_dir}")
            
        return {
            "plane_maps": plane_maps,
            "overlay_slices": overlay_slices,
            "roi_stats": roi_stats
        }
