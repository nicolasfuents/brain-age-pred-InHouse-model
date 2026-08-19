#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_inference.py

Script optimizado para inferencia en lote (Batch Mode).
Carga los modelos en memoria UNA SOLA VEZ y procesa la cohorte completa de forma eficiente.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm

# Limitar hilos de CPU para evitar sobrecarga
torch.set_num_threads(4)

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from run_pipeline import load_config, load_precomputed_tensors
from src.preprocessing.dicom_reader import handle_input_path, extract_patient_info, convert_dicom_to_nifti
from src.preprocessing.slice_extractor import process_nifti_to_tensors, extract_triplanar_tensors
from src.inference.predictor import TriplanarPredictor
from src.inference.bias_correction import AgeBiasCalibrator
from src.xai.xai_engine import XAIEngine

def find_scans_in_directory(input_dir: Path) -> List[Dict[str, Any]]:
    """Escanea recursivamente un directorio para detectar NIfTIs, .pt, zips y carpetas DICOM."""
    items = []
    
    # 1. Volúmenes NIfTI
    for p in sorted(list(input_dir.glob("**/*.nii.gz")) + list(input_dir.glob("**/*.nii"))):
        items.append({"type": "nifti", "path": p, "id": p.name.split(".")[0]})
        
    # 2. Tensores PyTorch precomputados .pt
    for p in sorted(list(input_dir.glob("**/*.pt"))):
        if not p.name.startswith("model_") and not p.name.startswith("tensor_"):
            items.append({"type": "pt_tensor", "path": p, "id": p.stem})
            
    # 3. Archivos Zip DICOM
    for p in sorted(list(input_dir.glob("**/*.zip"))):
        items.append({"type": "dicom_zip", "path": p, "id": p.stem})
        
    # 4. Directorios DICOM (si contienen .dcm)
    for p in sorted([d for d in input_dir.iterdir() if d.is_dir()]):
        has_dcm = any(fn.suffix.lower() == ".dcm" for fn in p.glob("**/*"))
        if has_dcm:
            items.append({"type": "dicom_dir", "path": p, "id": p.name})
            
    return items

def main():
    parser = argparse.ArgumentParser(
        description="Inferencia en Lote (Batch Mode) para Edad Cerebral (BAG)."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_dir", 
        type=Path, 
        help="Directorio raíz que contiene múltiples escaneos (NIfTIs, .pt, zips o carpetas DICOM)."
    )
    input_group.add_argument(
        "--input_csv", 
        type=Path, 
        help="Archivo CSV que contiene rutas y edades (columnas: 'input_t1'/'input_dicom' y 'age')."
    )
    
    parser.add_argument(
        "--output_csv", 
        type=Path, 
        default=Path("./batch_predictions.csv"), 
        help="Ruta del archivo CSV de salida listo para calibrate_local_scanner.py."
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=Path("./batch_output"), 
        help="Directorio donde se guardarán los tensores y figuras individuales."
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Genera el pipeline completo incluyendo los 3 métodos de interpretabilidad XAI y el panel PNG."
    )
    parser.add_argument(
        "--skip_prep", "--skip-prep", 
        dest="skip_prep", 
        action="store_true", 
        help="Omite el preprocesamiento previo y ejecuta inferencia directa sobre datos/tensores ya normalizados en MNI152."
    )
    
    args = parser.parse_args()
    config = load_config()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar el predictor y calibrador UNA SOLA VEZ
    checkpoints_dir = REPO_ROOT / config["models"]["checkpoints_dir"]
    print(f"\n[+] Cargando modelos del ensamble triplanar en memoria...")
    predictor = TriplanarPredictor(checkpoints_dir=checkpoints_dir, use_tta=True)
    calibrator = AgeBiasCalibrator(
        alpha=config["calibration"]["alpha"],
        beta=config["calibration"]["beta"]
    )
    xai_engine = XAIEngine(predictor=predictor) if args.all else None
    mask_path = REPO_ROOT / config["atlases"]["mask"]

    records = []
    if args.input_csv:
        print(f"[+] Procesando lote desde CSV: {args.input_csv}")
        df_meta = pd.read_csv(args.input_csv)
        for _, row in df_meta.iterrows():
            t1_p = Path(row["input_t1"]) if "input_t1" in row and pd.notna(row["input_t1"]) else None
            dcm_p = Path(row["input_dicom"]) if "input_dicom" in row and pd.notna(row["input_dicom"]) else None
            age_v = float(row["age"]) if "age" in row and pd.notna(row["age"]) else None
            subj_id = str(row["Patient_ID"]) if "Patient_ID" in row and pd.notna(row["Patient_ID"]) else (t1_p.stem if t1_p else "SUBJ")
            
            records.append({"type": "nifti" if t1_p else "dicom", "t1_p": t1_p, "dcm_p": dcm_p, "age": age_v, "id": subj_id})
    else:
        print(f"[+] Buscando escaneos en directorio: {args.input_dir}")
        scans = find_scans_in_directory(args.input_dir)
        print(f"  * Encontrados {len(scans)} escaneos válidos.")
        for s in scans:
            t1_p = s["path"] if s["type"] in ["nifti", "pt_tensor"] else None
            dcm_p = s["path"] if s["type"] in ["dicom_dir", "dicom_zip"] else None
            records.append({"type": s["type"], "t1_p": t1_p, "dcm_p": dcm_p, "age": None, "id": s["id"]})
            
    if not records:
        sys.exit("[!] No se encontraron escaneos para procesar.")
        
    all_results = []
    print(f"\n[+] Ejecutando inferencia para {len(records)} sujetos (skip_prep={args.skip_prep})...")
    for idx, r in enumerate(tqdm(records, desc="Inferencia en Lote")):
        subj_out = args.output_dir / f"{r['id']}_{idx+1:03d}"
        subj_out.mkdir(parents=True, exist_ok=True)
        temp_dir = subj_out / "temp_processing"
        
        try:
            tensors = None
            chronological_age = r["age"]
            patient_id = r["id"]
            
            # Carga rápida si --skip-prep
            if args.skip_prep:
                if r["t1_p"] and r["t1_p"].suffix == ".pt":
                    tensors = load_precomputed_tensors(r["t1_p"])
                elif r["t1_p"] and (r["t1_p"].name.endswith(".nii") or r["t1_p"].name.endswith(".nii.gz")):
                    nii = nib.load(str(r["t1_p"]))
                    vol = np.asarray(nii.get_fdata(), dtype=np.float32)
                    tensors = extract_triplanar_tensors(vol)
                    
            if tensors is None:
                if r["dcm_p"]:
                    dicom_dir = handle_input_path(r["dcm_p"], temp_dir)
                    d_name, d_age = extract_patient_info(dicom_dir)
                    if d_name != "UNKNOWN_PATIENT": patient_id = d_name
                    if chronological_age is None and d_age is not None: chronological_age = d_age
                    nifti_path = convert_dicom_to_nifti(dicom_dir, temp_dir / "nifti_raw")
                elif r["t1_p"]:
                    nifti_path = r["t1_p"]
                else:
                    continue
                    
                tensors = process_nifti_to_tensors(
                    nii_path=nifti_path,
                    mask_path=mask_path,
                    output_dir=subj_out / "tensors"
                )
                
            # Inferencia con modelo persistente en memoria
            predictions = predictor.predict(tensors)
            bag_results = calibrator.calculate_bag(
                pred_age=predictions["Pred_Ensemble"],
                chronological_age=chronological_age
            )
            
            res = {
                "Patient_ID": patient_id,
                "Input_File": str(r["dcm_p"] or r["t1_p"]),
                **predictions,
                **bag_results
            }
            
            if xai_engine:
                xai_engine.generate_explanations(
                    tensors=tensors,
                    predictions=res,
                    output_dir=subj_out / "xai",
                    patient_id=patient_id
                )
                
            all_results.append(res)
        except Exception as e:
            print(f"\n[!] Error procesando sujeto {r['id']}: {e}", file=sys.stderr)
            
    df_out = pd.DataFrame(all_results)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)
    
    print("\n" + "="*75)
    print(f"INFERENCIA EN LOTE FINALIZADA | N = {len(df_out)} procesados exitosamente")
    print(f"[✓] CSV consolidado listo para calibración guardado en: {args.output_csv}")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
