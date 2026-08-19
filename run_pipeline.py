#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py

Punto de entrada universal (CLI) para el pipeline end-to-end de estimación de edad cerebral (BAG)
y explicabilidad clínica (XAI).
Soporta carpetas/archivos .zip DICOM, volúmenes NIfTI y tensores preprocesados .pt.
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import torch
import numpy as np
import nibabel as nib

# Añadir el directorio raíz al PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.dicom_reader import handle_input_path, extract_patient_info, convert_dicom_to_nifti
from src.preprocessing.slice_extractor import process_nifti_to_tensors, extract_triplanar_tensors
from src.inference.predictor import TriplanarPredictor
from src.inference.bias_correction import AgeBiasCalibrator
from src.xai.xai_engine import XAIEngine

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    cfg_path = config_path or (REPO_ROOT / "config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración en {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def load_precomputed_tensors(target_path: Path) -> Dict[str, torch.Tensor]:
    """Carga tensores .pt directamente desde un archivo o carpeta."""
    if target_path.is_dir():
        ax_p = target_path / "tensor_axial.pt"
        cor_p = target_path / "tensor_coronal.pt"
        sag_p = target_path / "tensor_sagittal.pt"
        if ax_p.exists() and cor_p.exists() and sag_p.exists():
            return {
                "axial": torch.load(ax_p, map_location="cpu"),
                "coronal": torch.load(cor_p, map_location="cpu"),
                "sagittal": torch.load(sag_p, map_location="cpu")
            }
    elif target_path.is_file() and target_path.suffix == ".pt":
        data = torch.load(target_path, map_location="cpu")
        if isinstance(data, dict):
            if all(k in data for k in ["axial", "coronal", "sagittal"]):
                return {k: data[k] for k in ["axial", "coronal", "sagittal"]}
            elif "image" in data:
                # Si es un tensor directo de 5 rebanadas
                img = data["image"]
                return {"axial": img, "coronal": img, "sagittal": img}
        elif isinstance(data, torch.Tensor):
            return {"axial": data, "coronal": data, "sagittal": data}
            
    raise FileNotFoundError(f"No se pudieron cargar tensores .pt válidos desde {target_path}")

def run_single_subject(
    input_dicom: Optional[Path],
    input_t1: Optional[Path],
    manual_age: Optional[float],
    output_dir: Path,
    run_all_xai: bool,
    config: Dict[str, Any],
    skip_prep: bool = False
) -> Dict[str, Any]:
    """Ejecuta el pipeline end-to-end para un único paciente/escaneo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp_processing"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    patient_id = "PATIENT_001"
    chronological_age = manual_age
    tensors = None
    
    # Modo --skip-prep: Inferencia directa si ya existen los tensores o si el input es .pt
    if skip_prep:
        tensors_dir = output_dir / "tensors"
        if input_t1 and (input_t1.suffix == ".pt" or (input_t1.is_dir() and (input_t1 / "tensor_axial.pt").exists())):
            print(f"\n[+] (--skip-prep) Cargando tensores .pt precomputados directamente desde: {input_t1}")
            tensors = load_precomputed_tensors(input_t1)
            patient_id = input_t1.stem
        elif tensors_dir.exists() and (tensors_dir / "tensor_axial.pt").exists():
            print(f"\n[+] (--skip-prep) Reutilizando tensores existentes en: {tensors_dir}")
            tensors = load_precomputed_tensors(tensors_dir)
            if input_t1: patient_id = input_t1.name.split(".")[0]
        elif input_t1 and (input_t1.name.endswith(".nii") or input_t1.name.endswith(".nii.gz")):
            print(f"\n[+] (--skip-prep) Extracción directa de rebanadas desde volumen NIfTI preprocesado: {input_t1}")
            patient_id = input_t1.name.split(".")[0]
            nii = nib.load(str(input_t1))
            vol = np.asarray(nii.get_fdata(), dtype=np.float32)
            # Si el volumen ya está en rango [0, 1] o [-1, 1], se extrae directo
            tensors = extract_triplanar_tensors(vol)
            
    if tensors is None:
        # 1. Ingesta y conversión estándar (DICOM vs NIfTI)
        if input_dicom:
            print(f"\n[+] Ingesta de estudio DICOM: {input_dicom}")
            dicom_dir = handle_input_path(input_dicom, temp_dir)
            d_name, d_age = extract_patient_info(dicom_dir)
            if d_name != "UNKNOWN_PATIENT":
                patient_id = d_name
            if chronological_age is None and d_age is not None:
                chronological_age = d_age
                print(f"  * Edad extraída automáticamente de cabecera DICOM: {chronological_age:.1f} años")
                
            print("  * Convirtiendo serie DICOM a NIfTI (dcm2niix)...")
            nifti_path = convert_dicom_to_nifti(dicom_dir, temp_dir / "nifti_raw")
        elif input_t1:
            print(f"\n[+] Ingesta de volumen NIfTI T1: {input_t1}")
            nifti_path = input_t1
            patient_id = nifti_path.name.split(".")[0]
        else:
            raise ValueError("Debe proporcionar --input_dicom o --input_t1.")

        # 2. Extracción de pilas 2.5D y normalización P1-P99
        mask_path = REPO_ROOT / config["atlases"]["mask"]
        print(f"\n[+] Extrayendo pilas 2.5D y normalizando intensidades P1-P99...")
        tensors = process_nifti_to_tensors(
            nii_path=nifti_path,
            mask_path=mask_path,
            output_dir=output_dir / "tensors"
        )
        print(f"  * Tensores extraídos exitosamente para Axial, Coronal y Sagital.")

    # 3. Inferencia Triplanar y Ensamble Ridge (TTA siempre activo)
    checkpoints_dir = REPO_ROOT / config["models"]["checkpoints_dir"]
    print(f"\n[+] Ejecutando inferencia triplanar con TTA...")
    predictor = TriplanarPredictor(checkpoints_dir=checkpoints_dir, use_tta=True)
    predictions = predictor.predict(tensors)
    
    # 4. Calibración del Sesgo de Edad (bc-BAG)
    calibrator = AgeBiasCalibrator(
        alpha=config["calibration"]["alpha"],
        beta=config["calibration"]["beta"]
    )
    bag_results = calibrator.calculate_bag(
        pred_age=predictions["Pred_Ensemble"],
        chronological_age=chronological_age
    )
    
    # Consolidar resultados
    results = {
        "Patient_ID": patient_id,
        "Input_File": str(input_dicom or input_t1),
        **predictions,
        **bag_results
    }
    
    # 5. Explicabilidad Médica XAI (Activa únicamente con --all)
    if run_all_xai:
        xai_engine = XAIEngine(predictor=predictor)
        xai_engine.generate_explanations(
            tensors=tensors,
            predictions=results,
            output_dir=output_dir / "xai",
            patient_id=patient_id
        )

    # 6. Exportar resultados
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    csv_path = output_dir / "results.csv"
    pd.DataFrame([results]).to_csv(csv_path, index=False)
    
    # Imprimir resumen en consola
    print("\n" + "="*70)
    print(f"RESUMEN DE PREDICCIÓN DE EDAD CEREBRAL | ID: {patient_id}")
    print("="*70)
    print(f"  * Predicción Axial (ResNet-18 Soft)   : {results['Pred_Axial']:.2f} años")
    print(f"  * Predicción Coronal (ResNet-34 SL1)  : {results['Pred_Coronal']:.2f} años")
    print(f"  * Predicción Sagital (ResNet-18 MSE)  : {results['Pred_Sagittal']:.2f} años")
    print(f"  -------------------------------------------------------------")
    print(f"  * Predicción Ensamble (Ridge Stacker) : {results['Pred_Ensemble']:.2f} años")
    if chronological_age is not None:
        print(f"  * Edad Cronológica                    : {results['Chronological_Age']:.2f} años")
        print(f"  * Raw BAG (Pred - Cronológica)        : {results['Raw_BAG']:+.2f} años")
        print(f"  * bc-BAG (Calibrado / Sin sesgo)      : {results['bc_BAG']:+.2f} años")
    print("="*70)
    print(f"[✓] Resultados guardados en: {output_dir}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline End-to-End de Estimación de Edad Cerebral (BAG) y Explicabilidad XAI."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dicom", type=Path, help="Ruta al directorio o archivo .zip que contiene la serie DICOM.")
    input_group.add_argument("--input_t1", type=Path, help="Ruta al volumen NIfTI T1 (.nii/.nii.gz) o tensor preprocesado (.pt).")
    input_group.add_argument("--input_csv", type=Path, help="Ruta a un archivo CSV para inferencia por lotes (Batch Mode).")
    
    parser.add_argument("--age", type=float, default=None, help="Edad cronológica en años (opcional para NIfTI, auto en DICOM).")
    parser.add_argument("--output_dir", type=Path, default=Path("./output"), help="Directorio donde se guardarán los resultados (por defecto: ./output).")
    parser.add_argument("--all", action="store_true", help="Genera el pipeline completo incluyendo los 3 métodos de interpretabilidad XAI (IG, Occlusion, Grad-Attention) y el panel PNG.")
    parser.add_argument("--skip_prep", "--skip-prep", dest="skip_prep", action="store_true", help="Omite el preprocesamiento previo y ejecuta inferencia directa sobre datos/tensores ya normalizados en MNI152.")
    
    args = parser.parse_args()
    config = load_config()
    
    if args.input_csv:
        print(f"\n[+] Iniciando inferencia en lote desde CSV: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        all_results = []
        for idx, row in df.iterrows():
            subj_dir = args.output_dir / f"subject_{idx+1:03d}"
            t1_p = Path(row["input_t1"]) if "input_t1" in row and pd.notna(row["input_t1"]) else None
            dcm_p = Path(row["input_dicom"]) if "input_dicom" in row and pd.notna(row["input_dicom"]) else None
            age_val = float(row["age"]) if "age" in row and pd.notna(row["age"]) else None
            
            res = run_single_subject(
                input_dicom=dcm_p,
                input_t1=t1_p,
                manual_age=age_val,
                output_dir=subj_dir,
                run_all_xai=args.all,
                config=config,
                skip_prep=args.skip_prep
            )
            all_results.append(res)
            
        summary_csv = args.output_dir / "batch_summary.csv"
        pd.DataFrame(all_results).to_csv(summary_csv, index=False)
        print(f"\n[✓] Inferencia en lote finalizada. Resumen guardado en: {summary_csv}")
    else:
        run_single_subject(
            input_dicom=args.input_dicom,
            input_t1=args.input_t1,
            manual_age=args.age,
            output_dir=args.output_dir,
            run_all_xai=args.all,
            config=config,
            skip_prep=args.skip_prep
        )

if __name__ == "__main__":
    main()
