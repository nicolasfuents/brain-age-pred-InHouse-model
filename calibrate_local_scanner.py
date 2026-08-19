#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_local_scanner.py

Ajusta la regresión lineal local sobre una cohorte de Controles Sanos (CN)
para eliminar el sesgo sistemático (regresión a la media) y corregir la cohorte clínica.

Estética de visualización sincronizada exactamente con la figura de referencia
(fondo claro, sin etiquetas 'A.'/'B.', sin nombres de base de datos entre paréntesis).
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy import stats

def fit_local_calibration(df_controls: pd.DataFrame) -> dict:
    """Ajusta OLS sobre controles sanos para estimar alpha y beta."""
    age_col = None
    for c in ["Chronological_Age", "age", "Age", "AGE"]:
        if c in df_controls.columns:
            age_col = c
            break
            
    pred_col = None
    for c in ["Pred_Ensemble", "pred_age", "Pred_Age", "Predicted_Age", "pred"]:
        if c in df_controls.columns:
            pred_col = c
            break
            
    if age_col is None or pred_col is None:
        raise ValueError(f"El CSV de controles debe contener columnas de edad cronológica y predicha. Encontradas: {list(df_controls.columns)}")
        
    df = df_controls[[age_col, pred_col]].dropna().copy()
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
        "pred_col": pred_col
    }

def plot_calibration_results(calib_dict: dict, output_path: Path):
    """Genera la figura de dispersión con la estética estándar de publicación."""
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
    
    # Configuración de estilo
    sns.set_theme(style="whitegrid", rc={
        "axes.facecolor": "#f8fafc",
        "figure.facecolor": "white",
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--"
    })
    
    color = "#bee0a3"  # Light Green / Pistachio
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Panel 1: Raw BAG (Antes de la calibración local)
    axs[0].scatter(df[age_col], df["Raw_BAG"], color=color, alpha=0.7, s=35, edgecolor='none')
    x_range = np.linspace(df[age_col].min(), df[age_col].max(), 100).reshape(-1, 1)
    y_range_raw = lr_local.predict(x_range)
    axs[0].plot(x_range, y_range_raw, color='#475569', linestyle='-', linewidth=2.0,
                label=f'Local Bias: y = {alpha:.4f} * x + {beta:.4f}\nRaw MAE: {mae_raw:.2f} yr')
    axs[0].set_title('Raw BAG', fontsize=13, fontweight='bold', pad=12, color='#1e293b')
    axs[0].set_xlabel('Chronological Age (yr)', fontsize=11, color='#1e293b')
    axs[0].set_ylabel('Brain Age Gap (yr)', fontsize=11, color='#1e293b')
    axs[0].legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', loc='lower left')
    axs[0].axhline(0, color='#94a3b8', linestyle=':', linewidth=1.2)
    
    # Panel 2: Corrected BAG (Después de la calibración local)
    axs[1].scatter(df[age_col], df["bc_BAG_local"], color=color, alpha=0.7, s=35, edgecolor='none')
    y_range_bc = lr_bc.predict(x_range)
    axs[1].plot(x_range, y_range_bc, color='#10b981', linestyle='-', linewidth=2.0,
                label=f'Residual Bias: y = {lr_bc.coef_[0]:.4f} * x + {lr_bc.intercept_:.4f}\nCalibrated MAE: {mae_bc:.2f} yr')
    axs[1].set_title('Locally Calibrated bc-BAG', fontsize=13, fontweight='bold', pad=12, color='#1e293b')
    axs[1].set_xlabel('Chronological Age (yr)', fontsize=11, color='#1e293b')
    axs[1].legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', loc='lower left')
    axs[1].axhline(0, color='#94a3b8', linestyle=':', linewidth=1.2)
    
    # Cosmética global
    y_max = max(30, np.nanmax(np.abs(df["Raw_BAG"])) + 5)
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

def main():
    parser = argparse.ArgumentParser(
        description="Calibración Local del Sesgo Etario (bc-BAG) para Resonadores Externos."
    )
    parser.add_argument(
        "--controls_csv", 
        type=Path, 
        required=True, 
        help="CSV con controles sanos (debe contener 'Chronological_Age' y 'Pred_Ensemble')."
    )
    parser.add_argument(
        "--clinical_csv", 
        type=Path, 
        default=None, 
        help="CSV opcional con la cohorte clínica a la cual aplicarle la calibración local."
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=Path("./calibration_output"), 
        help="Directorio de salida para los parámetros y curvas de calibración."
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*75)
    print("CALIBRACIÓN LOCAL DEL SESGO DE EDAD PARA RESONADORES EXTERNOS")
    print("="*75)
    
    print(f"[+] Cargando cohorte de controles sanos: {args.controls_csv}")
    df_controls = pd.read_csv(args.controls_csv)
    
    calib = fit_local_calibration(df_controls)
    
    print(f"\n[✓] Parámetros de Calibración Local Estimados:")
    print(f"  * Pendiente de sesgo (alpha) : {calib['alpha']:.6f}")
    print(f"  * Intercepto de sitio (beta) : {calib['beta']:.6f}")
    print(f"  * Correlación pre-calib (r)   : {calib['r_raw']:.3f} (p = {calib['p_raw']:.4e})")
    print(f"  * Correlación post-calib (r)  : {calib['r_bc']:.3f} (Ortogonalizado)")
    
    # Guardar parámetros CSV
    df_params = pd.DataFrame([{
        "alpha_slope": calib["alpha"],
        "beta_site_intercept": calib["beta"],
        "raw_mae_years": calib["mae_raw"],
        "calibrated_mae_years": calib["mae_bc"],
        "pearson_r_raw": calib["r_raw"],
        "pearson_r_calibrated": calib["r_bc"],
        "n_controls": len(calib["df_clean"])
    }])
    params_path = args.output_dir / "local_calibration_parameters.csv"
    df_params.to_csv(params_path, index=False)
    print(f"[✓] Parámetros guardados en: {params_path}")
    
    # Generar gráfico de calibración local
    fig_path = args.output_dir / "local_calibration_curve.png"
    plot_calibration_results(calib, fig_path)
    print(f"[✓] Gráfico de calibración local guardado en: {fig_path}")
    
    # Aplicar a cohorte clínica si fue proporcionada
    if args.clinical_csv and args.clinical_csv.exists():
        print(f"\n[+] Aplicando calibración local a la cohorte clínica: {args.clinical_csv}")
        df_clinical = pd.read_csv(args.clinical_csv)
        
        age_col = None
        for c in ["Chronological_Age", "age", "Age"]:
            if c in df_clinical.columns: age_col = c; break
        pred_col = None
        for c in ["Pred_Ensemble", "pred_age", "Pred_Age", "pred"]:
            if c in df_clinical.columns: pred_col = c; break
            
        if age_col and pred_col:
            df_clinical["Raw_BAG"] = df_clinical[pred_col] - df_clinical[age_col]
            df_clinical["bc_BAG"] = df_clinical["Raw_BAG"] - (calib["alpha"] * df_clinical[age_col] + calib["beta"])
        elif pred_col:
            print("[!] Aviso: 'Chronological_Age' no encontrada en cohorte clínica. Se calculará sólo bc_Pred.")
            
        out_clinical_path = args.output_dir / f"calibrated_{args.clinical_csv.name}"
        df_clinical.to_csv(out_clinical_path, index=False)
        print(f"[✓] Cohorte clínica calibrada guardada en: {out_clinical_path}")
        
    print("="*75)
    print("Calibración completada con éxito.")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
