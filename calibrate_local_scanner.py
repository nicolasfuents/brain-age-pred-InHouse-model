#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_local_scanner.py

Local Scanner Age-Bias Calibration Tool (bc-BAG).
Fits an Ordinary Least Squares (OLS) regression over a local Healthy Control (CN) cohort
to orthogonalize Brain Age Gap (BAG) against chronological age, removing regression-to-the-mean
and site/scanner-specific systematic offsets.

Recommended Usage:
  - Minimum Healthy Control Sample Size: N >= 30 (ideally N >= 50) distributed evenly across the age range.
  - Re-calibration cadence: Perform whenever major scanner software/hardware updates occur,
    or upon acquiring new batches of >= 50-100 control subjects.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent

def fit_local_calibration(df_controls: pd.DataFrame) -> Dict[str, Any]:
    """
    Fits OLS linear regression on Healthy Controls to estimate local slope (alpha) and intercept (beta).
    Formula: Raw_BAG_CN = alpha * Chronological_Age + beta
    """
    age_col = None
    for c in ["Chronological_Age", "chronological_age", "age", "Age", "AGE"]:
        if c in df_controls.columns:
            age_col = c
            break
            
    pred_col = None
    for c in ["Pred_Ensemble", "predicted_age", "pred_ensemble", "pred_age", "Pred_Age", "Predicted_Age", "pred"]:
        if c in df_controls.columns:
            pred_col = c
            break
            
    if age_col is None or pred_col is None:
        raise ValueError(
            f"Control cohort CSV must contain chronological age and predicted age columns. "
            f"Columns found: {list(df_controls.columns)}"
        )
        
    df = df_controls[[age_col, pred_col]].dropna().copy()
    if len(df) < 10:
        print(f"[!] Warning: Sample size N={len(df)} is small. Recommended N >= 30 (ideally N >= 50) for stable calibration.")
        
    df["Raw_BAG"] = df[pred_col] - df[age_col]
    
    X = df[[age_col]].values
    y = df["Raw_BAG"].values
    
    lr = LinearRegression().fit(X, y)
    alpha = float(lr.coef_[0])
    beta = float(lr.intercept_)
    
    r_raw, p_raw = stats.pearsonr(df[age_col], df["Raw_BAG"])
    bc_bag = df["Raw_BAG"] - (alpha * df[age_col] + beta)
    r_bc, p_bc = stats.pearsonr(df[age_col], bc_bag)
    
    mae_raw = mean_absolute_error(df[age_col], df[pred_col])
    mae_bc = mean_absolute_error(df[age_col], df[pred_col] - (alpha * df[age_col] + beta))
    
    return {
        "alpha": alpha,
        "beta": beta,
        "r_raw": r_raw,
        "p_raw": p_raw,
        "r_bc": r_bc,
        "p_bc": p_bc,
        "mae_raw": mae_raw,
        "mae_bc": mae_bc,
        "df_clean": df,
        "age_col": age_col,
        "pred_col": pred_col,
        "n_controls": len(df)
    }

def plot_calibration_results(calib_dict: Dict[str, Any], output_path: Path):
    """Generates publication-quality dual-panel scatter plot showing Raw BAG vs. Calibrated bc-BAG."""
    df = calib_dict["df_clean"]
    age_col = calib_dict["age_col"]
    alpha = calib_dict["alpha"]
    beta = calib_dict["beta"]
    mae_raw = calib_dict["mae_raw"]
    mae_bc = calib_dict["mae_bc"]
    
    X = df[[age_col]].values
    y_raw = df["Raw_BAG"].values
    df["bc_BAG_local"] = df["Raw_BAG"] - (alpha * df[age_col] + beta)
    y_bc = df["bc_BAG_local"].values
    
    lr_local = LinearRegression().fit(X, y_raw)
    lr_bc = LinearRegression().fit(X, y_bc)
    
    sns.set_theme(style="whitegrid", rc={
        "axes.facecolor": "#f8fafc",
        "figure.facecolor": "white",
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--"
    })
    
    color = "#3b82f6"  # Professional blue
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Panel 1: Raw BAG
    axs[0].scatter(df[age_col], df["Raw_BAG"], color=color, alpha=0.7, s=35, edgecolor='none')
    x_range = np.linspace(df[age_col].min(), df[age_col].max(), 100).reshape(-1, 1)
    y_range_raw = lr_local.predict(x_range)
    axs[0].plot(x_range, y_range_raw, color='#ef4444', linestyle='-', linewidth=2.0,
                label=f'Bias Trend: y = {alpha:.4f} * x + {beta:.4f}\nr = {calib_dict["r_raw"]:.3f}, MAE = {mae_raw:.2f} yr')
    axs[0].set_title('Raw Brain Age Gap (Uncalibrated)', fontsize=13, fontweight='bold', pad=12, color='#1e293b')
    axs[0].set_xlabel('Chronological Age (yr)', fontsize=11, color='#1e293b')
    axs[0].set_ylabel('Brain Age Gap (yr)', fontsize=11, color='#1e293b')
    axs[0].legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', loc='lower left')
    axs[0].axhline(0, color='#94a3b8', linestyle=':', linewidth=1.2)
    
    # Panel 2: Calibrated bc-BAG
    axs[1].scatter(df[age_col], df["bc_BAG_local"], color='#10b981', alpha=0.7, s=35, edgecolor='none')
    y_range_bc = lr_bc.predict(x_range)
    axs[1].plot(x_range, y_range_bc, color='#059669', linestyle='-', linewidth=2.0,
                label=f'Residual Bias: y = {lr_bc.coef_[0]:.4f} * x + {lr_bc.intercept_:.4f}\nr = {calib_dict["r_bc"]:.3f}, MAE = {mae_bc:.2f} yr')
    axs[1].set_title('Locally Calibrated bc-BAG (Orthogonalized)', fontsize=13, fontweight='bold', pad=12, color='#1e293b')
    axs[1].set_xlabel('Chronological Age (yr)', fontsize=11, color='#1e293b')
    axs[1].legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', loc='lower left')
    axs[1].axhline(0, color='#94a3b8', linestyle=':', linewidth=1.2)
    
    y_max = max(25, float(np.nanmax(np.abs(df["Raw_BAG"]))) + 5)
    for ax in axs:
        ax.set_ylim(-y_max, y_max)
        sns.despine(ax=ax, top=True, right=True)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.tick_params(colors='#475569')
        
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def update_config_file(alpha: float, beta: float, config_path: Path = REPO_ROOT / "config.yaml"):
    """Updates config.yaml with newly estimated alpha and beta coefficients."""
    if not config_path.exists():
        return
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "calibration" not in cfg:
        cfg["calibration"] = {}
    cfg["calibration"]["alpha"] = float(alpha)
    cfg["calibration"]["beta"] = float(beta)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"[✓] config.yaml automatically updated with alpha={alpha:.6f} and beta={beta:.6f}.")

def main():
    parser = argparse.ArgumentParser(
        description="Local Scanner Age-Bias Calibration Tool (bc-BAG) for MRI Brain Age Estimation."
    )
    parser.add_argument(
        "--controls_csv", 
        type=Path, 
        required=True, 
        help="Path to Healthy Controls CSV (must contain chronological age and predicted age columns)."
    )
    parser.add_argument(
        "--clinical_csv", 
        type=Path, 
        default=None, 
        help="Optional path to target clinical cohort CSV to apply the estimated calibration parameters."
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=Path("./calibration_output"), 
        help="Output directory for calibration parameters and diagnostic plots (default: ./calibration_output)."
    )
    parser.add_argument(
        "--update_config", 
        action="store_true", 
        help="Automatically write estimated alpha and beta into config.yaml for subsequent pipeline runs."
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(" LOCAL SCANNER AGE-BIAS CALIBRATION (bc-BAG)")
    print("="*80)
    
    print(f"[+] Ingesting Healthy Control cohort from: {args.controls_csv}")
    df_controls = pd.read_csv(args.controls_csv)
    calib = fit_local_calibration(df_controls)
    
    print(f"\n[✓] Estimated Local Calibration Parameters (N = {calib['n_controls']}):")
    print(f"  * Age Bias Slope (alpha):       {calib['alpha']:.6f}")
    print(f"  * Site/Scanner Intercept (beta): {calib['beta']:.6f}")
    print(f"  * Pre-calibration Pearson r:     {calib['r_raw']:.3f} (p = {calib['p_raw']:.4e})")
    print(f"  * Post-calibration Pearson r:    {calib['r_bc']:.3f} (Orthogonalized to age)")
    print(f"  * Uncalibrated MAE:              {calib['mae_raw']:.2f} years")
    print(f"  * Calibrated MAE:                {calib['mae_bc']:.2f} years")
    
    # Save calibration parameters CSV
    df_params = pd.DataFrame([{
        "alpha_slope": calib["alpha"],
        "beta_site_intercept": calib["beta"],
        "raw_mae_years": calib["mae_raw"],
        "calibrated_mae_years": calib["mae_bc"],
        "pearson_r_raw": calib["r_raw"],
        "pearson_r_calibrated": calib["r_bc"],
        "n_controls": calib["n_controls"]
    }])
    params_path = args.output_dir / "local_calibration_parameters.csv"
    df_params.to_csv(params_path, index=False)
    print(f"\n[✓] Calibration parameters saved to: {params_path}")
    
    # Generate diagnostic scatter plot
    fig_path = args.output_dir / "local_calibration_curve.png"
    plot_calibration_results(calib, fig_path)
    print(f"[✓] Diagnostic calibration plot saved to: {fig_path}")
    
    # Update config.yaml if requested
    if args.update_config:
        update_config_file(calib["alpha"], calib["beta"])
        
    # Apply calibration to clinical cohort if provided
    if args.clinical_csv and args.clinical_csv.exists():
        print(f"\n[+] Applying local calibration to target cohort: {args.clinical_csv}")
        df_clinical = pd.read_csv(args.clinical_csv)
        
        age_col = None
        for c in ["Chronological_Age", "chronological_age", "age", "Age", "AGE"]:
            if c in df_clinical.columns: age_col = c; break
        pred_col = None
        for c in ["Pred_Ensemble", "predicted_age", "pred_ensemble", "pred_age", "Pred_Age", "pred"]:
            if c in df_clinical.columns: pred_col = c; break
            
        if age_col and pred_col:
            df_clinical["raw_bag"] = df_clinical[pred_col] - df_clinical[age_col]
            df_clinical["bc_bag"] = df_clinical["raw_bag"] - (calib["alpha"] * df_clinical[age_col] + calib["beta"])
            out_clinical_path = args.output_dir / f"calibrated_{args.clinical_csv.name}"
            df_clinical.to_csv(out_clinical_path, index=False)
            print(f"[✓] Calibrated cohort saved to: {out_clinical_path}")
        else:
            print("[!] Could not apply calibration: 'age' or 'predicted_age' column missing in clinical CSV.")
            
    print("\n" + "="*80)
    print(" Local calibration completed successfully.")
    print("="*80)
    print(" How to incorporate these calibration coefficients into future runs:")
    print(f"  1. Automated: Re-run calibration with '--update_config' to save directly to config.yaml:")
    print(f"       python calibrate_local_scanner.py --controls_csv {args.controls_csv} --update_config")
    print(f"  2. CLI Flag: Pass '--calibration_file' to run_pipeline.py or batch_inference.py:")
    print(f"       python run_pipeline.py --input_t1 <path> --age <age> --calibration_file {params_path}")
    print(f"  3. Direct Parameters: Pass '--alpha {calib['alpha']:.6f} --beta {calib['beta']:.6f}' to run_pipeline.py.")
    print(f"  4. Manual: In config.yaml, set alpha: {calib['alpha']:.6f} and beta: {calib['beta']:.6f}.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
