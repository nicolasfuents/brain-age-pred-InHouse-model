#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_preprocess.py

High-throughput batch preprocessing pipeline for raw neuroimaging cohorts (DICOM series and native NIfTI volumes).
Orchestrates:
  1. DICOM to NIfTI conversion (dcm2niix).
  2. Brain extraction & Skull Stripping (mri_synthstrip).
  3. 12-DOF Affine registration to MNI152 1mm (FSL flirt).
  4. B-spline bias field correction (ANTs N4BiasFieldCorrection).
  5. SOLID_v2 intracranial masking, robust P1-P99 intensity normalization, and 2.5D triplanar tensor extraction.
  6. Generates a consolidated manifest CSV ready for instant batch inference with --skip-prep.
"""

import sys
import os
import shutil
import argparse
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from src.preprocessing.dicom_reader import handle_input_path, extract_patient_info, convert_dicom_to_nifti
from src.preprocessing.slice_extractor import process_nifti_to_tensors

def load_config(config_path: Path = REPO_ROOT / "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def process_single_subject(
    item: Dict[str, Any],
    output_dir: Path,
    mask_path: Path,
    save_nifti: bool = True
) -> Dict[str, Any]:
    """Preprocesa un único sujeto de forma aislada y robusta."""
    subj_id = item["subject_id"]
    subj_out_dir = output_dir / subj_id
    tensors_dir = subj_out_dir / "t1_tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = subj_out_dir / "temp_prep"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    res = {
        "subject_id": subj_id,
        "age": item.get("age", None),
        "tensor_dir": str(tensors_dir),
        "status": "FAILED",
        "error_msg": ""
    }
    
    # Comprobar si ya está procesado
    if (tensors_dir / "tensor_axial.pt").exists() and (tensors_dir / "tensor_coronal.pt").exists() and (tensors_dir / "tensor_sagittal.pt").exists():
        res["status"] = "ALREADY_PROCESSED"
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        return res
        
    try:
        raw_nii = None
        chrono_age = item.get("age", None)
        
        # 1. Ingesta
        if item.get("input_dicom"):
            dcm_p = Path(item["input_dicom"])
            extracted_dcm = handle_input_path(dcm_p, temp_dir)
            d_name, d_age = extract_patient_info(extracted_dcm)
            if chrono_age is None and d_age is not None:
                chrono_age = d_age
                res["age"] = chrono_age
            raw_nii = convert_dicom_to_nifti(extracted_dcm, temp_dir / "nii_raw")
        elif item.get("input_t1"):
            raw_nii = Path(item["input_t1"])
        else:
            raise ValueError("No se especificó input_dicom ni input_t1.")

        # 2. Comprobar dimensiones espaciales (Nativo vs MNI152)
        nii = nib.load(str(raw_nii))
        mni_nii = raw_nii
        
        if nii.shape != (182, 218, 182):
            script_path = REPO_ROOT / "src" / "preprocessing" / "register_and_n4.sh"
            prep_work_dir = temp_dir / "quasiraw"
            prep_work_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = ["bash", str(script_path), str(raw_nii), str(prep_work_dir)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            
            candidates = list((prep_work_dir / "quasiraw").glob("*desc-6apply*.nii.gz"))
            if not candidates:
                raise FileNotFoundError("El registro quasiraw no generó el archivo desc-6apply.")
            mni_nii = candidates[0]
            
        # 3. Guardar NIfTI preprocesado si se solicita
        if save_nifti:
            final_nii_path = subj_out_dir / f"{subj_id}_preprocessed_MNI152.nii.gz"
            shutil.copyfile(mni_nii, final_nii_path)
            res["preprocessed_nii"] = str(final_nii_path)

        # 4. Extracción 2.5D y normalización P1-P99
        process_nifti_to_tensors(
            nii_path=mni_nii,
            mask_path=mask_path,
            output_dir=tensors_dir
        )
        
        res["status"] = "SUCCESS"
        
    except Exception as e:
        res["status"] = "FAILED"
        res["error_msg"] = str(e)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    return res

def discover_cohort(input_dir: Path) -> List[Dict[str, Any]]:
    """Descubre automáticamente resonancias en un directorio raíz."""
    items = []
    
    # 1. Búsqueda de NIfTI
    nii_files = sorted(list(input_dir.rglob("*.nii")) + list(input_dir.rglob("*.nii.gz")))
    # Filtrar tensores o máscaras auxiliares
    nii_files = [f for f in nii_files if "mask" not in f.name.lower() and "tensor" not in f.name.lower()]
    
    if nii_files:
        for f in nii_files:
            subj_id = f.name.replace(".nii.gz", "").replace(".nii", "")
            items.append({
                "subject_id": subj_id,
                "input_t1": str(f),
                "age": None
            })
        return items

    # 2. Búsqueda de DICOM (carpetas o .zip)
    for entry in sorted(input_dir.iterdir()):
        if entry.is_dir() or entry.suffix.lower() == ".zip":
            items.append({
                "subject_id": entry.stem,
                "input_dicom": str(entry),
                "age": None
            })
            
    return items

def main():
    parser = argparse.ArgumentParser(
        description="High-Throughput Batch Preprocessing Pipeline (Brain Age Prediction & Medical XAI Framework)"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", type=Path, help="Ruta a un directorio que contiene escaneos NIfTI o carpetas/archivos DICOM.")
    input_group.add_argument("--input_csv", type=Path, help="Ruta a un archivo CSV con columnas 'subject_id', 'input_t1'/'input_dicom' y 'age'.")
    
    parser.add_argument("--output_dir", type=Path, default=Path("./preprocessed_cohort"), help="Directorio donde se guardarán los tensores y NIfTIs preprocesados (por defecto: ./preprocessed_cohort).")
    parser.add_argument("--n_jobs", type=int, default=4, help="Número de procesos en paralelo (por defecto: 4).")
    parser.add_argument("--save_nifti", action="store_true", default=True, help="Guarda también el volumen NIfTI preprocesado en espacio MNI152 (por defecto: True).")
    
    args = parser.parse_args()
    config = load_config()
    mask_path = REPO_ROOT / config["atlases"]["mask"]
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Descubrimiento de cohorte
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        cohort = df.to_dict(orient="records")
        for i, item in enumerate(cohort):
            if "subject_id" not in item or pd.isna(item["subject_id"]):
                item["subject_id"] = f"SUBJECT_{i+1:04d}"
    else:
        print(f"\n[+] Descubriendo escaneos en: {args.input_dir}...")
        cohort = discover_cohort(args.input_dir)
        
    print(f"  * Total de escaneos descubiertos: {len(cohort)}")
    if not cohort:
        print("[!] No se encontraron escaneos para procesar.")
        return

    # 2. Procesamiento concurrente
    print(f"\n[+] Iniciando preprocesamiento en lote (n_jobs={args.n_jobs})...")
    results = []
    
    if args.n_jobs > 1 and len(cohort) > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            future_to_item = {
                executor.submit(process_single_subject, item, args.output_dir, mask_path, args.save_nifti): item
                for item in cohort
            }
            with tqdm(total=len(cohort), desc="Preprocesando cohorte") as pbar:
                for future in as_completed(future_to_item):
                    res = future.result()
                    results.append(res)
                    pbar.update(1)
    else:
        for item in tqdm(cohort, desc="Preprocesando cohorte"):
            res = process_single_subject(item, args.output_dir, mask_path, args.save_nifti)
            results.append(res)
            
    # 3. Exportar manifiesto CSV
    manifest_df = pd.DataFrame(results)
    manifest_path = args.output_dir / "preprocessed_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    
    success_count = sum(1 for r in results if r["status"] in ["SUCCESS", "ALREADY_PROCESSED"])
    print("\n" + "="*50)
    print(" RESUMEN DE PREPROCESAMIENTO EN LOTE")
    print("="*50)
    print(f"  * Exitosos / Ya procesados: {success_count} / {len(cohort)}")
    print(f"  * Fallidos:                 {len(cohort) - success_count}")
    print(f"  * Manifiesto generado en:   {manifest_path}")
    print("="*50)
    print(f"\n[💡] Para ejecutar inferencia ultrarrápida sobre esta cohorte preprocesada, ejecute:")
    print(f"     python batch_inference.py --input_csv {manifest_path} --output_csv ./batch_predictions.csv --skip-prep\n")

if __name__ == "__main__":
    main()
