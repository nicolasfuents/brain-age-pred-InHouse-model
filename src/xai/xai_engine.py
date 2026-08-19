#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xai_engine.py

Orquestador principal de interpretabilidad médica (Medical XAI).
Ejecuta las 3 técnicas de explicación (Integrated Gradients, Occlusion Sensitivity y Grad-Attention),
construye el volumen 3D EWAM en memoria, mapea la atribución a regiones anatómicas (ROIs)
del atlas subcortical/cerebelo (sin guardar archivos *_3d.nii.gz pesados), y renderiza las
figuras diagnósticas de nivel clínico.
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
    plot_xai_superimposed,
    plot_top_rois_bar_chart
)

def load_labels_map(labels_csv_path: Path) -> Dict[int, str]:
    """Carga el diccionario de mapeo ID -> Nombre anatómico de la ROI."""
    m = {}
    if labels_csv_path.exists():
        with open(labels_csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Saltar encabezado
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
        
        # Cargar atlas y redimensionar en memoria si existe
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
        Calcula el pipeline completo de XAI:
        - 3 métodos (IG, Occlusion, Grad-Attention) en los 3 planos.
        - Construcción EWAM 3D en memoria (sin escribir *.nii.gz).
        - Cuantificación por ROI anatómica (Harvard-Oxford + SUIT).
        - Generación de figuras PNG diagnósticas.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plane_configs = {
            "axial": (self.predictor.model_axial, "soft"),
            "coronal": (self.predictor.model_coronal, "smoothl1"),
            "sagittal": (self.predictor.model_sagittal, "mse")
        }
        
        plane_maps = {"ig": {}, "occ": {}, "attn": {}}
        print("\n[+] Calculando mapas de explicabilidad médica XAI...")
        
        for plane, (model, loss_type) in plane_configs.items():
            print(f"  * Procesando plano {plane.capitalize()} (IG, Occlusion, Grad-Attention)...")
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
            
        # Rebanadas centrales de T1
        t1_slices = {
            "axial": tensors["axial"].cpu().numpy()[0, 2] if tensors["axial"].ndim == 4 else tensors["axial"].cpu().numpy()[2],
            "coronal": tensors["coronal"].cpu().numpy()[0, 2] if tensors["coronal"].ndim == 4 else tensors["coronal"].cpu().numpy()[2],
            "sagittal": tensors["sagittal"].cpu().numpy()[0, 2] if tensors["sagittal"].ndim == 4 else tensors["sagittal"].cpu().numpy()[2]
        }
        
        # Slices 2D colapsadas directamente de plane_maps
        overlay_slices = {}
        for m_name in ["ig", "occ", "attn"]:
            overlay_slices[m_name] = {}
            for plane in ["axial", "coronal", "sagittal"]:
                arr_2d = plane_maps[m_name][plane].sum(axis=0)
                m = np.nanmax(np.abs(arr_2d))
                if m > 0:
                    arr_2d = arr_2d / m
                overlay_slices[m_name][plane] = arr_2d
                
        # 1. Generar Figura 1: Panel Multimétodo 4x3 (xai_overlays_panel.png)
        panel_path = output_dir / "xai_overlays_panel.png"
        plot_xai_overlays_panel(
            t1_slices=t1_slices,
            overlay_slices=overlay_slices,
            predictions=predictions,
            out_path=panel_path,
            patient_id=patient_id
        )
        
        # 2. Generar Figura 2: Panel Superpuesto (xai_overlays_superimposed.png)
        super_path = output_dir / "xai_overlays_superimposed.png"
        plot_xai_superimposed(
            t1_slices=t1_slices,
            overlay_slices=overlay_slices,
            predictions=predictions,
            out_path=super_path,
            patient_id=patient_id
        )
        
        # 3. Mapeo a ROIs del Atlas Anatómico en memoria (EWAM 3D)
        roi_stats = {"ig": {}, "occ": {}}
        if self.atlas_rs is not None:
            print("  * Mapeando atribución espacial a ROIs del atlas subcortical/cerebelo...")
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
                
            # Extraer estadísticas cuantitativas por ROI
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
                        
                # Exportar CSV de ROIs
                csv_path = output_dir / f"roi_attributions_{m_name}.csv"
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["roi_id", "label", "mean_net", "sum_abs", "area_vox"])
                    for rid, stat in sorted(roi_stats[m_name].items(), key=lambda x: abs(x[1]["mean_net"]), reverse=True):
                        label_name = self.id2label.get(rid, f"ROI_{rid}")
                        writer.writerow([rid, label_name, f"{stat['mean_net']:.6f}", f"{stat['sum_abs']:.6f}", stat["area_vox"]])
                        
            # Generar Gráficos Top-10 ROIs
            plot_top_rois_bar_chart(
                roi_stats=roi_stats["ig"],
                id2label=self.id2label,
                method_name="ig",
                method_title="Integrated Gradients",
                out_path=output_dir / "roi_importance_ig.png",
                patient_id=patient_id
            )
            plot_top_rois_bar_chart(
                roi_stats=roi_stats["occ"],
                id2label=self.id2label,
                method_name="occ",
                method_title="Occlusion Sensitivity",
                out_path=output_dir / "roi_importance_occ.png",
                patient_id=patient_id
            )
            
        return {
            "plane_maps": plane_maps,
            "roi_stats": roi_stats
        }
