#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualizer.py

Módulo de visualización diagnóstica clínica para explicabilidad médica (Medical XAI).
Genera figuras con estética de publicación idénticas al script de referencia:
1. Panel de atribución neural multimétodo (4x3: Planos x Métodos + Colorbars en fondo negro).
2. Panel superpuesto multimétodo (1x3: Oclusión + Grad-Attention).
3. Gráficos de barras horizontales de Top-10 ROIs contribuyentes anatómicos (IG y Oclusión).
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.measure import label, regionprops

def _apply_display_orient(arr: np.ndarray, plane: str) -> np.ndarray:
    """Rota 90 grados en sentido antihorario (k=1) para coincidir con la orientación estándar del script de referencia."""
    return np.rot90(np.asarray(arr), k=1)

def _add_orient_labels(ax, plane: str):
    """Agrega etiquetas anatómicas de orientación (R/L, A/P, S/I)."""
    if plane == "axial":
        lbl = {"left": "R", "right": "L", "top": "A", "bottom": "P"}
    elif plane == "coronal":
        lbl = {"left": "R", "right": "L", "top": "S", "bottom": "I"}
    else:  # sagittal
        lbl = {"left": "P", "right": "A", "top": "S", "bottom": "I"}

    ax.text(0.02, 0.50, lbl["left"], transform=ax.transAxes, va="center", ha="left",
            fontsize=10, color="white", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2))
    ax.text(0.98, 0.50, lbl["right"], transform=ax.transAxes, va="center", ha="right",
            fontsize=10, color="white", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2))
    ax.text(0.50, 0.03, lbl["bottom"], transform=ax.transAxes, va="bottom", ha="center",
            fontsize=10, color="white", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2))
    ax.text(0.50, 0.97, lbl["top"], transform=ax.transAxes, va="top", ha="center",
            fontsize=10, color="white", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2))

def voxel_level_cluster_filter(heatmap: np.ndarray, threshold_ratio: float = 0.15, min_size: int = 20) -> np.ndarray:
    """Filtra ruido a nivel de vóxel preservando únicamente componentes conectados contiguos."""
    thresh = threshold_ratio * np.nanmax(np.abs(heatmap))
    clustered = np.zeros_like(heatmap)
    
    # Componentes positivos
    pos_mask = heatmap > thresh
    if pos_mask.any():
        labeled_pos, _ = label(pos_mask, return_num=True)
        for r in regionprops(labeled_pos):
            if r.area >= min_size:
                for coords in r.coords:
                    idx = tuple(coords)
                    clustered[idx] = heatmap[idx]
                    
    # Componentes negativos
    neg_mask = heatmap < -thresh
    if neg_mask.any():
        labeled_neg, _ = label(neg_mask, return_num=True)
        for r in regionprops(labeled_neg):
            if r.area >= min_size:
                for coords in r.coords:
                    idx = tuple(coords)
                    clustered[idx] = heatmap[idx]
                    
    return clustered

def plot_xai_overlays_panel(
    t1_slices: Dict[str, np.ndarray],
    overlay_slices: Dict[str, Dict[str, np.ndarray]],
    predictions: Dict[str, Any],
    out_path: Path,
    patient_id: str = "PATIENT"
):
    """
    Genera el panel diagnóstico consolidado de 4x3 (3 planos x 3 métodos + barras de color).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axs = plt.subplots(4, 3, figsize=(10, 10.5), gridspec_kw={'height_ratios': [1, 1, 1, 0.04]}, facecolor='black')
    methods_display = ["ig", "occ", "attn"]
    methods_labels = ["Integrated Gradients", "Occlusion Sensitivity", "Grad-Attention"]
    planes_list = ["axial", "coronal", "sagittal"]
    
    for r, plane in enumerate(planes_list):
        for c, m_name in enumerate(methods_display):
            ax = axs[r, c]
            t1 = _apply_display_orient(t1_slices[plane], plane)
            ov = _apply_display_orient(overlay_slices[m_name][plane], plane)
            
            brain_mask = t1 > 0.01
            
            if m_name == "occ":
                ov = voxel_level_cluster_filter(ov, threshold_ratio=0.15, min_size=20)
                cmap = "seismic"
                norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
            elif m_name == "ig":
                cmap = "seismic"
                norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
            else:  # attn
                cmap = "hot"
                norm = mcolors.Normalize(vmin=0, vmax=1)
                
            ov = np.where(brain_mask, ov, 0.0)
            if m_name == "attn":
                ov = np.where(brain_mask, np.maximum(ov, 0.02), 0.0)
                
            ax.imshow(t1, cmap="gray", interpolation="bicubic")
            
            mask = np.abs(ov) > 1e-4
            ov_masked = np.where(mask, ov, np.nan)
            
            ax.imshow(ov_masked, cmap=cmap, norm=norm, alpha=0.5, interpolation="bilinear")
            _add_orient_labels(ax, plane)
            ax.axis("off")
            
            if r == 0:
                ax.set_title(methods_labels[c], color="white", fontsize=12, pad=10, fontweight="bold")
            if c == 0:
                ax.text(-0.08, 0.5, plane.upper(), transform=ax.transAxes, color="white",
                        fontsize=12, fontweight="bold", rotation=90, va="center", ha="right")
                
    # Fila 4: Barras de color
    norm_seismic = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    sm_seismic = plt.cm.ScalarMappable(cmap="seismic", norm=norm_seismic)
    sm_seismic.set_array([])
    cbar_ig = fig.colorbar(sm_seismic, cax=axs[3, 0], orientation="horizontal")
    cbar_ig.set_label("IG Contribution", color="white", fontsize=8, labelpad=2)
    cbar_ig.ax.tick_params(labelsize=7, colors="white")
    
    cbar_occ = fig.colorbar(sm_seismic, cax=axs[3, 1], orientation="horizontal")
    cbar_occ.set_label("Occlusion Delta", color="white", fontsize=8, labelpad=2)
    cbar_occ.ax.tick_params(labelsize=7, colors="white")
    
    norm_hot = mcolors.Normalize(vmin=0, vmax=1)
    sm_hot = plt.cm.ScalarMappable(cmap="hot", norm=norm_hot)
    sm_hot.set_array([])
    cbar_attn = fig.colorbar(sm_hot, cax=axs[3, 2], orientation="horizontal")
    cbar_attn.set_label("Attention Weight", color="white", fontsize=8, labelpad=2)
    cbar_attn.ax.tick_params(labelsize=7, colors="white")
    
    # Título superior
    pred_ens = predictions.get("Pred_Ensemble", 0.0)
    real_age = predictions.get("Chronological_Age")
    bag_val = predictions.get("Raw_BAG")
    
    if real_age is not None and bag_val is not None:
        title_str = (f"Subject: {patient_id}\nMulti-Method Neural Attribution Overlays\n"
                     f"Real Age: {real_age:.2f} yrs | Predicted Age: {pred_ens:.2f} yrs (BAG: {bag_val:+.2f} yrs)")
    else:
        title_str = (f"Subject: {patient_id}\nMulti-Method Neural Attribution Overlays\n"
                     f"Predicted Age (Ensemble): {pred_ens:.2f} yrs")
                     
    fig.suptitle(title_str, color="white", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(out_path, dpi=200, facecolor='black')
    plt.close(fig)
    print(f"[✓] Panel diagnóstico multimétodo guardado en: {out_path}")

def plot_xai_superimposed(
    t1_slices: Dict[str, np.ndarray],
    overlay_slices: Dict[str, Dict[str, np.ndarray]],
    predictions: Dict[str, Any],
    out_path: Path,
    patient_id: str = "PATIENT"
):
    """
    Genera la figura de superposición conjunta (T1 + Oclusión + Grad-Attention) en 1 fila.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    planes_list = ["axial", "coronal", "sagittal"]
    fig_super, axs_super = plt.subplots(1, 3, figsize=(12, 4.5), facecolor='black')
    
    for c_idx, plane in enumerate(planes_list):
        ax = axs_super[c_idx]
        t1 = _apply_display_orient(t1_slices[plane], plane)
        brain_mask = t1 > 0.01
        
        # 1. T1
        ax.imshow(t1, cmap="gray", interpolation="bicubic")
        
        # 2. Occlusion
        ov_occ = _apply_display_orient(overlay_slices["occ"][plane], plane)
        ov_occ = np.where(brain_mask, ov_occ, 0.0)
        ov_occ = voxel_level_cluster_filter(ov_occ, threshold_ratio=0.15, min_size=20)
        mask_occ = np.abs(ov_occ) > 1e-4
        ov_occ_masked = np.where(mask_occ, ov_occ, np.nan)
        norm_seismic = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        ax.imshow(ov_occ_masked, cmap="seismic", norm=norm_seismic, alpha=0.45, interpolation="bilinear")
        
        # 3. Grad-Attention
        ov_attn = _apply_display_orient(overlay_slices["attn"][plane], plane)
        ov_attn = np.where(brain_mask, np.maximum(ov_attn, 0.02), 0.0)
        mask_attn = np.abs(ov_attn) > 1e-4
        ov_attn_masked = np.where(mask_attn, ov_attn, np.nan)
        norm_hot = mcolors.Normalize(vmin=0, vmax=1)
        ax.imshow(ov_attn_masked, cmap="hot", norm=norm_hot, alpha=0.55, interpolation="bilinear")
        
        _add_orient_labels(ax, plane)
        ax.set_title(plane.upper(), color="white", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")
        
    pred_ens = predictions.get("Pred_Ensemble", 0.0)
    real_age = predictions.get("Chronological_Age")
    bag_val = predictions.get("Raw_BAG")
    
    if real_age is not None and bag_val is not None:
        title_str = (f"Subject: {patient_id} — Superimposed Occ & Attn\n"
                     f"Real Age: {real_age:.2f} yrs | Predicted Age: {pred_ens:.2f} yrs (BAG: {bag_val:+.2f} yrs)")
    else:
        title_str = f"Subject: {patient_id} — Superimposed Occ & Attn\nPredicted Age: {pred_ens:.2f} yrs"
        
    fig_super.suptitle(title_str, color="white", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig_super.savefig(out_path, dpi=200, facecolor='black')
    plt.close(fig_super)
    print(f"[✓] Panel superpuesto guardado en: {out_path}")

def plot_top_rois_bar_chart(
    roi_stats: Dict[int, Dict[str, Any]],
    id2label: Dict[int, str],
    method_name: str,
    method_title: str,
    out_path: Path,
    patient_id: str = "PATIENT"
):
    """
    Genera el gráfico de barras horizontales con las 10 regiones anatómicas con mayor impacto.
    Rojo = Acelera la edad cerebral predicha.
    Azul = Rejuvenece / Disminuye la edad cerebral predicha.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    top_rois = sorted(roi_stats.items(), key=lambda x: abs(x[1]["mean_net"]), reverse=True)[:10]
    if not top_rois:
        return
        
    top_rois = list(reversed(top_rois))
    y_labels = [id2label.get(rid, f"ROI_{rid}") for rid, _ in top_rois]
    x_vals = [stat["mean_net"] for _, stat in top_rois]
    
    colors = ["#ff5a5f" if val >= 0 else "#3182bd" for val in x_vals]
    
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='white')
    bars = ax.barh(y_labels, x_vals, color=colors, height=0.6, edgecolor='none')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(axis='both', colors='#333333', labelsize=9)
    ax.axvline(0, color='#666666', linestyle='--', linewidth=0.8, alpha=0.5)
    
    max_abs = max(abs(val) for val in x_vals) if x_vals else 1.0
    offset = 0.02 * max_abs
    for bar in bars:
        width = bar.get_width()
        if abs(width) >= 0.18 * max_abs:
            if width >= 0:
                x_pos = width - offset
                ha_align = 'right'
            else:
                x_pos = width + offset
                ha_align = 'left'
            text_color = 'white'
        else:
            if width >= 0:
                x_pos = width + offset
                ha_align = 'left'
            else:
                x_pos = width - offset
                ha_align = 'right'
            text_color = '#333333'
            
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{width:+.2f}",
                va='center', ha=ha_align, fontsize=8, color=text_color, fontweight='bold')
                
    ax.set_title(f"Top 10 Contributor ROIs ({method_title})\nSubject: {patient_id}", 
                 fontsize=11, pad=12, fontweight='bold', color='#111111')
    ax.set_xlabel("Net Attribution (- Decelerates Age, + Accelerates Age)", fontsize=9, color='#333333', labelpad=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] Gráfico Top 10 ROIs ({method_name}) guardado en: {out_path}")
