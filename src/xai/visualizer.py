#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualizer.py

Visualization module for Explainable AI (XAI) feature attribution maps.
Renders high-resolution multi-method 4x3 visual panels (T1 + IG + Occlusion + Grad-Attention)
and horizontal bar charts for anatomical ROI contributions.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import label

def voxel_level_cluster_filter(
    heatmap: np.ndarray, 
    threshold_ratio: float = 0.15, 
    min_size: int = 20
) -> np.ndarray:
    """Removes isolated background noise clusters below minimum voxel size."""
    if heatmap is None:
        return heatmap
    m = np.nanmax(np.abs(heatmap))
    if m == 0:
        return heatmap
    thr = threshold_ratio * m
    sig_mask = np.abs(heatmap) > thr
    labeled, num_features = label(sig_mask)
    cleaned = np.zeros_like(heatmap)
    for i in range(1, num_features + 1):
        c_mask = (labeled == i)
        if c_mask.sum() >= min_size:
            cleaned[c_mask] = heatmap[c_mask]
    return cleaned

def _apply_display_orient(arr: np.ndarray, plane: str) -> np.ndarray:
    """Applies anatomical display rotation for standard radiological view."""
    if arr.ndim != 2:
        return arr
    if plane in ["axial", "coronal", "sagittal"]:
        return np.rot90(arr)
    return arr

def _add_orient_labels(ax: plt.Axes, plane: str):
    """Adds radiological anatomical orientation labels (R/L, A/P, S/I)."""
    t_props = dict(color="#94a3b8", fontsize=8, fontweight="bold", alpha=0.9)
    if plane == "axial":
        ax.text(0.04, 0.5, "R", transform=ax.transAxes, va="center", ha="left", **t_props)
        ax.text(0.96, 0.5, "L", transform=ax.transAxes, va="center", ha="right", **t_props)
        ax.text(0.5, 0.94, "A", transform=ax.transAxes, va="top", ha="center", **t_props)
        ax.text(0.5, 0.04, "P", transform=ax.transAxes, va="bottom", ha="center", **t_props)
    elif plane == "coronal":
        ax.text(0.04, 0.5, "R", transform=ax.transAxes, va="center", ha="left", **t_props)
        ax.text(0.96, 0.5, "L", transform=ax.transAxes, va="center", ha="right", **t_props)
        ax.text(0.5, 0.94, "S", transform=ax.transAxes, va="top", ha="center", **t_props)
        ax.text(0.5, 0.04, "I", transform=ax.transAxes, va="bottom", ha="center", **t_props)
    elif plane == "sagittal":
        ax.text(0.04, 0.5, "A", transform=ax.transAxes, va="center", ha="left", **t_props)
        ax.text(0.96, 0.5, "P", transform=ax.transAxes, va="center", ha="right", **t_props)
        ax.text(0.5, 0.94, "S", transform=ax.transAxes, va="top", ha="center", **t_props)
        ax.text(0.5, 0.04, "I", transform=ax.transAxes, va="bottom", ha="center", **t_props)

def plot_xai_overlays_panel(
    t1_slices: Dict[str, np.ndarray],
    overlay_slices: Dict[str, Dict[str, np.ndarray]],
    predictions: Dict[str, Any],
    out_path: Path,
    patient_id: str = "PATIENT"
):
    """
    Renders 4x3 Multi-Method Visual Attribution Panel.
    Rows: Axial, Coronal, Sagittal, Colorbars
    Columns: Integrated Gradients, Occlusion Sensitivity, Grad-Attention
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    planes = ["axial", "coronal", "sagittal"]
    methods = ["ig", "occ", "attn"]
    methods_labels = ["Integrated Gradients", "Occlusion Sensitivity", "Grad-Attention"]
    
    fig, axs = plt.subplots(4, 3, figsize=(13, 13.5),
                            gridspec_kw={"height_ratios": [1, 1, 1, 0.07]},
                            facecolor="#0f172a")
                            
    for r, plane in enumerate(planes):
        t1 = _apply_display_orient(t1_slices[plane], plane)
        brain_mask = t1 > 0.01
        
        for c, m_name in enumerate(methods):
            ax = axs[r, c]
            ov = _apply_display_orient(overlay_slices[m_name][plane], plane)
            
            if m_name == "ig":
                cmap = "seismic"
                norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
                ov = np.where(brain_mask, ov, 0.0)
            elif m_name == "occ":
                cmap = "seismic"
                norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
                ov = np.where(brain_mask, ov, 0.0)
                ov = voxel_level_cluster_filter(ov, threshold_ratio=0.15, min_size=20)
            elif m_name == "attn":
                cmap = "hot"
                norm = mcolors.Normalize(vmin=0, vmax=1)
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
                ax.text(-0.08, 0.5, plane.upper(), transform=ax.transAxes, color="#38bdf8",
                        fontsize=12, fontweight="bold", rotation=90, va="center", ha="right")
                
    # Row 4: Colorbars
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
    
    pred_ens = predictions.get("pred_ensemble", predictions.get("Pred_Ensemble", 0.0))
    real_age = predictions.get("chronological_age", predictions.get("Chronological_Age"))
    bag_val = predictions.get("raw_bag", predictions.get("Raw_BAG"))
    
    if real_age is not None and bag_val is not None:
        title_str = (f"Subject: {patient_id} | Multi-Method Feature Attribution Overlays\n"
                     f"Chronological Age: {real_age:.2f} yr | Predicted Age: {pred_ens:.2f} yr | Raw BAG: {bag_val:+.2f} yr")
    else:
        title_str = f"Subject: {patient_id} | Multi-Method Feature Attribution Overlays\nPredicted Brain Age: {pred_ens:.2f} yr"
                     
    fig.suptitle(title_str, color="white", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(out_path, dpi=200, facecolor="#0f172a", bbox_inches="tight")
    plt.close(fig)

def plot_top_rois_bar_chart(
    roi_stats: Dict[int, Dict[str, Any]],
    id2label: Dict[int, str],
    method_name: str,
    method_title: str,
    out_path: Path,
    patient_id: str = "PATIENT"
):
    """
    Renders horizontal bar chart of top 10 contributing anatomical regions.
    Red = Accelerates predicted brain age.
    Blue = Decelerates predicted brain age (preservation).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    top_rois = sorted(roi_stats.items(), key=lambda x: abs(x[1]["mean_net"]), reverse=True)[:10]
    if not top_rois:
        return
        
    top_rois = list(reversed(top_rois))
    y_labels = [id2label.get(rid, f"ROI_{rid}") for rid, _ in top_rois]
    x_vals = [stat["mean_net"] for _, stat in top_rois]
    
    colors = ["#ef4444" if val >= 0 else "#3b82f6" for val in x_vals]
    
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor="white")
    bars = ax.barh(y_labels, x_vals, color=colors, height=0.6, edgecolor="none")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="both", colors="#334155", labelsize=9)
    ax.axvline(0, color="#64748b", linestyle="--", linewidth=0.8, alpha=0.6)
    
    max_abs = max(abs(val) for val in x_vals) if x_vals else 1.0
    offset = 0.02 * max_abs
    for bar in bars:
        width = bar.get_width()
        if abs(width) >= 0.18 * max_abs:
            x_pos = width - offset if width >= 0 else width + offset
            ha_align = "right" if width >= 0 else "left"
            text_color = "white"
        else:
            x_pos = width + offset if width >= 0 else width - offset
            ha_align = "left" if width >= 0 else "right"
            text_color = "#1e293b"
            
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{width:+.2f}",
                va="center", ha=ha_align, fontsize=8, color=text_color, fontweight="bold")
                
    ax.set_title(f"Top 10 Contributor ROIs ({method_title})\nSubject: {patient_id}", 
                 fontsize=11, pad=12, fontweight="bold", color="#0f172a")
    ax.set_xlabel("Net Attribution (- Decelerates Age, + Accelerates Age)", fontsize=9, color="#475569", labelpad=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
