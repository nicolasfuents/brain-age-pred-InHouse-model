#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_pipeline.py

Script principal autónomo para inferencia de Edad Cerebral (Brain Age Gap, BAG)
y explicabilidad mediante Explainable AI (XAI: Integrated Gradients, Occlusion Sensitivity, Grad-Attention).
"""

import sys
import os
import shutil
import argparse
import json
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import nibabel as nib
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from src.preprocessing.dicom_reader import handle_input_path, extract_patient_info, convert_dicom_to_nifti
from src.preprocessing.slice_extractor import process_nifti_to_tensors, extract_triplanar_tensors
from src.inference.predictor import TriplanarPredictor
from src.inference.bias_correction import AgeBiasCalibrator
from src.xai.xai_engine import XAIEngine
from src.xai.visualizer import plot_xai_overlays_panel

def load_config(config_path: Path = REPO_ROOT / "config.yaml") -> Dict[str, Any]:
    """Carga los hiperparámetros y configuraciones del archivo config.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración en {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_precomputed_tensors(target_path: Path) -> Dict[str, torch.Tensor]:
    """Carga tensores .pt preexistentes desde un directorio o tensor combinado."""
    if target_path.is_file() and target_path.suffix == ".pt":
        loaded = torch.load(target_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "axial" in loaded:
            return loaded
        elif isinstance(loaded, torch.Tensor):
            return {"axial": loaded, "coronal": loaded, "sagittal": loaded}
            
    if target_path.is_dir():
        ax_p = target_path / "tensor_axial.pt"
        cor_p = target_path / "tensor_coronal.pt"
        sag_p = target_path / "tensor_sagittal.pt"
        if ax_p.exists() and cor_p.exists() and sag_p.exists():
            return {
                "axial": torch.load(ax_p, map_location="cpu", weights_only=False),
                "coronal": torch.load(cor_p, map_location="cpu", weights_only=False),
                "sagittal": torch.load(sag_p, map_location="cpu", weights_only=False)
            }
            
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
    
    # 1. Modo --skip-prep
    if skip_prep:
        tensors_dir = output_dir / "tensors"
        if input_t1 and (input_t1.suffix == ".pt" or (input_t1.is_dir() and (input_t1 / "tensor_axial.pt").exists())):
            print(f"
[+] (--skip-prep) Loading precomputed .pt tensors from: {input_t1}")
            tensors = load_precomputed_tensors(input_t1)
            patient_id = input_t1.stem
        elif tensors_dir.exists() and (tensors_dir / "tensor_axial.pt").exists():
            print(f"
[+] (--skip-prep) Reusing existing tensors in: {tensors_dir}")
            tensors = load_precomputed_tensors(tensors_dir)
            if input_t1: patient_id = input_t1.name.split(".")[0]
        elif input_t1 and (input_t1.name.endswith(".nii") or input_t1.name.endswith(".nii.gz")):
            print(f"
[+] (--skip-prep) Direct slice extraction from preprocessed MNI NIfTI volume: {input_t1}")
            patient_id = input_t1.name.split(".")[0]
            nii = nib.load(str(input_t1))
            if nii.shape != (182, 218, 182):
                raise ValueError(
                    f"Volume shape {nii.shape} does not match expected MNI152 template (182, 218, 182). "
                    "Remove --skip-prep to run automated affine registration and N4 bias correction."
                )
            mask_path = REPO_ROOT / config["atlases"]["mask"]
            tensors = process_nifti_to_tensors(
                nii_path=input_t1,
                mask_path=mask_path,
                output_dir=output_dir / "tensors"
            )

    if tensors is None:
        # 2. Ingesta y conversión inicial (DICOM vs NIfTI)
        if input_dicom:
            print(f"
[+] Ingesting DICOM study from: {input_dicom}")
            dicom_dir = handle_input_path(input_dicom, temp_dir)
            d_name, d_age = extract_patient_info(dicom_dir)
            if d_name != "UNKNOWN_PATIENT":
                patient_id = d_name
            if chronological_age is None and d_age is not None:
                chronological_age = d_age
                print(f"  * Chronological age automatically extracted from DICOM header: {chronological_age:.1f} years")
                
            print("  * Converting DICOM series to NIfTI (dcm2niix)...")
            nifti_path = convert_dicom_to_nifti(dicom_dir, temp_dir / "nifti_raw")
        elif input_t1:
            print(f"
[+] Ingesting NIfTI T1w volume: {input_t1}")
            nifti_path = input_t1
            patient_id = nifti_path.name.split(".")[0]
        else:
            raise ValueError("Must provide either --input_dicom or --input_t1.")

        # 3. Comprobación de dimensiones espaciales (Nativo vs MNI152)
        nii = nib.load(str(nifti_path))
        if nii.shape != (182, 218, 182):
            print(f"
[+] Input volume is in native space {nii.shape}. Running automated quasiraw preprocessing (FLIRT + N4)...")
            missing_tools = []
            for tool in ["mri_synthstrip", "brainprep"]:
                if shutil.which(tool) is None:
                    missing_tools.append(tool)
            if missing_tools:
                raise EnvironmentError(
                    f"Native preprocessing requires external tools: {', '.join(missing_tools)}. "
                    "Please ensure FSL, ANTs, and FreeSurfer/SynthStrip are in your PATH (e.g. 'module load fsl ants freesurfer' on HPC) "
                    "or provide a pre-registered MNI152 volume (182, 218, 182) with --skip-prep."
                )
            
            script_path = REPO_ROOT / "src" / "preprocessing" / "register_and_n4.sh"
            prep_dir = temp_dir / "quasiraw_out"
            prep_dir.mkdir(parents=True, exist_ok=True)
            cmd = ["bash", str(script_path), str(nifti_path), str(prep_dir)]
            subprocess.run(cmd, check=True)
            
            candidates = list((prep_dir / "quasiraw").glob("*desc-6apply*.nii.gz"))
            if not candidates:
                raise FileNotFoundError(f"Preprocessing completed but desc-6apply NIfTI not found in {prep_dir / 'quasiraw'}")
            nifti_path = candidates[0]
            print(f"  * Volume successfully aligned to MNI152 1mm: {nifti_path}")
        else:
            print(f"  * Volume is already in MNI152 space (182, 218, 182).")

        # 4. Extracción de pilas 2.5D y normalización P1-P99
        mask_path = REPO_ROOT / config["atlases"]["mask"]
        print(f"
[+] Extracting 2.5D slices and normalizing intensities (P1-P99)...")
        tensors = process_nifti_to_tensors(
            nii_path=nifti_path,
            mask_path=mask_path,
            output_dir=output_dir / "tensors"
        )
        print(f"  * Slices extracted successfully for Axial, Coronal, and Sagittal planes.")

    # 5. Inferencia Triplanar y Ensamble Ridge (TTA siempre activo)
    checkpoints_dir = REPO_ROOT / config["models"]["checkpoints_dir"]
    print(f"
[+] Running triplanar inference with Test-Time Augmentation (TTA)...")
    predictor = TriplanarPredictor(checkpoints_dir=checkpoints_dir, use_tta=True)
    predictions = predictor.predict(tensors)
    
    # 6. Calibración del Sesgo de Edad (bc-BAG)
    calibrator = AgeBiasCalibrator(
        alpha=config["calibration"]["alpha"],
        beta=config["calibration"]["beta"]
    )
    bag_results = calibrator.calculate_bag(
        pred_ensemble=predictions["pred_ensemble"],
        chronological_age=chronological_age
    )
    
    # Consolidación de Resultados
    final_results = {
        "subject_id": patient_id,
        "chronological_age": chronological_age,
        "pred_axial": round(predictions["pred_axial"], 2),
        "pred_coronal": round(predictions["pred_coronal"], 2),
        "pred_sagittal": round(predictions["pred_sagittal"], 2),
        "pred_ensemble": round(predictions["pred_ensemble"], 2),
        "raw_bag": bag_results["raw_bag"],
        "bc_bag": bag_results["bc_bag"]
    }
    
    print("
" + "="*50)
    print(f" RESULTADOS FINALES DE EDAD CEREBRAL ({patient_id})")
    print("="*50)
    if chronological_age is not None:
        print(f"  * Edad Cronológica:       {chronological_age:.2f} años")
    print(f"  * Predicción Axial:       {final_results['pred_axial']:.2f} años")
    print(f"  * Predicción Coronal:     {final_results['pred_coronal']:.2f} años")
    print(f"  * Predicción Sagital:     {final_results['pred_sagittal']:.2f} años")
    print(f"  * Predicción Ensamble:    {final_results['pred_ensemble']:.2f} años")
    if chronological_age is not None:
        print(f"  * Raw BAG (Crudo):        {final_results['raw_bag']:+.2f} años")
        print(f"  * Calibrated bc-BAG:      {final_results['bc_bag']:+.2f} años")
    print("="*50)

    # Guardar métricas en JSON y CSV
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)
        
    csv_path = output_dir / "results.csv"
    pd.DataFrame([final_results]).to_csv(csv_path, index=False)
    print(f"
[✓] Métricas cuantitativas guardadas en: {json_path} y {csv_path}")

    # 7. Explicabilidad Médica XAI (Opcional con --all)
    if run_all_xai:
        print(f"
[+] Generando mapas de interpretabilidad XAI (--all)...")
        xai_dir = output_dir / "xai"
        xai_dir.mkdir(parents=True, exist_ok=True)
        
        xai_engine = XAIEngine(
            models=predictor.models,
            device=predictor.device
        )
        
        xai_maps = xai_engine.generate_all_maps(
            tensors=tensors,
            output_dir=xai_dir
        )
        
        panel_path = xai_dir / "xai_overlays_panel.png"
        plot_xai_overlays_panel(
            t1_tensors=tensors,
            xai_maps=xai_maps,
            subject_id=patient_id,
            chrono_age=chronological_age,
            pred_age=final_results["pred_ensemble"],
            raw_bag=final_results["raw_bag"],
            output_path=panel_path
        )
        print(f"[✓] Panel de explicabilidad XAI guardado en: {panel_path}")

    # Limpieza de temporales
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        
    return final_results

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
        print(f"
[+] Iniciando inferencia en lote desde CSV: {args.input_csv}")
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
        print(f"
[✓] Inferencia en lote finalizada. Resumen guardado en: {summary_csv}")
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
