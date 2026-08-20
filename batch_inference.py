#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_inference.py

High-Throughput Batch Inference Pipeline for Brain Age Gap (BAG) Estimation.
Performs fast batch inference over directories containing NIfTI files, DICOM studies, or .pt tensors,
reusing persistent models in memory for maximum throughput.
Outputs a consolidated predictions CSV ready for downstream analysis or local scanner calibration.
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from src.preprocessing.dicom_reader import handle_input_path, extract_patient_info, convert_dicom_to_nifti
from src.preprocessing.slice_extractor import process_nifti_to_tensors, extract_triplanar_tensors
from src.inference.predictor import TriplanarPredictor
from src.xai.xai_engine import XAIEngine

def load_config(config_path: Path = REPO_ROOT / "config.yaml") -> Dict[str, Any]:
    """Loads configuration hyperparameters."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_precomputed_tensors(target_path: Path) -> Dict[str, Any]:
    """Loads existing precomputed .pt tensors."""
    import torch
    if target_path.is_file() and target_path.suffix == ".pt":
        loaded = torch.load(target_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "axial" in loaded:
            return loaded
        elif isinstance(loaded, torch.Tensor):
            return {"axial": loaded, "coronal": loaded, "sagittal": loaded}
    raise FileNotFoundError(f"Could not load valid .pt tensors from {target_path}")

def find_scans_in_directory(input_dir: Path) -> List[Dict[str, Any]]:
    """Recursively scans directory tree to detect NIfTI volumes, .pt tensors, zip archives, and DICOM directories."""
    items = []
    
    # 1. NIfTI volumes
    for p in sorted(list(input_dir.glob("**/*.nii.gz")) + list(input_dir.glob("**/*.nii"))):
        if not any(k in p.name.lower() for k in ["mask", "desc-1", "desc-2", "desc-3", "desc-4", "desc-5", "roi"]):
            items.append({"type": "nifti", "path": p, "id": p.name.split(".")[0]})
            
    # 2. PyTorch .pt tensors
    for p in sorted(list(input_dir.glob("**/*.pt"))):
        if not p.name.startswith("model_") and not p.name.startswith("tensor_"):
            items.append({"type": "pt_tensor", "path": p, "id": p.stem})
            
    # 3. DICOM zip archives
    for p in sorted(list(input_dir.glob("**/*.zip"))):
        items.append({"type": "dicom_zip", "path": p, "id": p.stem})
        
    # 4. DICOM directories (folders containing .dcm)
    for p in sorted([d for d in input_dir.iterdir() if d.is_dir()]):
        has_dcm = any(fn.suffix.lower() == ".dcm" for fn in p.glob("**/*"))
        if has_dcm:
            items.append({"type": "dicom_dir", "path": p, "id": p.name})
            
    return items

def resolve_calibration_coefficients(
    calib_file: Optional[Path],
    cli_alpha: Optional[float],
    cli_beta: Optional[float],
    config_calib: Dict[str, Any]
) -> Tuple[Optional[float], Optional[float]]:
    """Resolves alpha and beta calibration coefficients from file, CLI flags, or config.yaml."""
    if calib_file and calib_file.exists():
        df_cal = pd.read_csv(calib_file)
        if "alpha_slope" in df_cal.columns and "beta_site_intercept" in df_cal.columns:
            return float(df_cal["alpha_slope"].iloc[0]), float(df_cal["beta_site_intercept"].iloc[0])
        elif "alpha" in df_cal.columns and "beta" in df_cal.columns:
            return float(df_cal["alpha"].iloc[0]), float(df_cal["beta"].iloc[0])
            
    if cli_alpha is not None and cli_beta is not None:
        return float(cli_alpha), float(cli_beta)
        
    cfg_a = config_calib.get("alpha")
    cfg_b = config_calib.get("beta")
    if cfg_a is not None and cfg_b is not None:
        return float(cfg_a), float(cfg_b)
        
    return None, None

def main():
    parser = argparse.ArgumentParser(
        description="High-Throughput Batch Inference Pipeline for MRI Brain Age Gap (BAG) Estimation."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_dir", 
        type=Path, 
        help="Root directory containing multiple scans (NIfTI files, .pt tensors, zips, or DICOM directories)."
    )
    input_group.add_argument(
        "--input_csv", 
        type=Path, 
        help="Input manifest CSV containing paths and chronological ages (columns: 'input_t1'/'input_dicom' and 'age')."
    )
    
    parser.add_argument(
        "--output_csv", 
        type=Path, 
        default=Path("./batch_predictions.csv"), 
        help="Path for consolidated predictions CSV (default: ./batch_predictions.csv)."
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=Path("./batch_output"), 
        help="Directory to save individual subject tensors and XAI figures."
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Generate full XAI attribution suite (IG, Occlusion, Grad-Attention, multi-method panel)."
    )
    parser.add_argument(
        "--skip_prep", action="store_true", 
        help="Skip registration and run direct inference on pre-aligned MNI152 volumes or tensors."
    )
    
    # Optional calibration parameters
    parser.add_argument("--calibration_file", type=Path, default=None, help="Path to local_calibration_parameters.csv from calibrate_local_scanner.py.")
    parser.add_argument("--alpha", type=float, default=None, help="Custom local calibration slope (alpha).")
    parser.add_argument("--beta", type=float, default=None, help="Custom local calibration site intercept (beta).")
    
    args = parser.parse_args()
    config = load_config()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve calibration
    resolved_alpha, resolved_beta = resolve_calibration_coefficients(
        calib_file=args.calibration_file,
        cli_alpha=args.alpha,
        cli_beta=args.beta,
        config_calib=config.get("calibration", {})
    )
    
    # 1. Load predictor once in memory
    checkpoints_dir = REPO_ROOT / config["models"]["checkpoints_dir"]
    print(f"\n[+] Loading triplanar ensemble models into memory...")
    predictor = TriplanarPredictor(checkpoints_dir=checkpoints_dir, use_tta=True)
    xai_engine = XAIEngine(predictor=predictor) if args.all else None
    mask_path = REPO_ROOT / config["atlases"]["mask"]

    records = []
    if args.input_csv:
        print(f"[+] Loading batch cohort from CSV: {args.input_csv}")
        df_meta = pd.read_csv(args.input_csv)
        for _, row in df_meta.iterrows():
            t1_p = Path(row["input_t1"]) if "input_t1" in row and pd.notna(row["input_t1"]) else None
            dcm_p = Path(row["input_dicom"]) if "input_dicom" in row and pd.notna(row["input_dicom"]) else None
            age_v = float(row["age"]) if "age" in row and pd.notna(row["age"]) else None
            subj_id = str(row["Patient_ID"]) if "Patient_ID" in row and pd.notna(row["Patient_ID"]) else (t1_p.stem if t1_p else "SUBJ")
            
            records.append({"type": "nifti" if t1_p else "dicom", "t1_p": t1_p, "dcm_p": dcm_p, "age": age_v, "id": subj_id})
    else:
        print(f"[+] Scanning directory tree for candidate scans: {args.input_dir}")
        scans = find_scans_in_directory(args.input_dir)
        print(f"  * Discovered {len(scans)} valid candidate scans.")
        for s in scans:
            t1_p = s["path"] if s["type"] in ["nifti", "pt_tensor"] else None
            dcm_p = s["path"] if s["type"] in ["dicom_dir", "dicom_zip"] else None
            records.append({"type": s["type"], "t1_p": t1_p, "dcm_p": dcm_p, "age": None, "id": s["id"]})
            
    if not records:
        sys.exit("[!] No valid scans found to process. Exiting.")
        
    all_results = []
    print(f"\n[+] Starting batch inference for {len(records)} subjects (skip_prep={args.skip_prep})...")
    for idx, r in enumerate(tqdm(records, desc="Batch Inference")):
        subj_out = args.output_dir / f"{r['id']}_{idx+1:03d}"
        subj_out.mkdir(parents=True, exist_ok=True)
        temp_dir = subj_out / "temp_processing"
        
        try:
            tensors = None
            chronological_age = r["age"]
            patient_id = r["id"]
            
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
                
            predictions = predictor.predict(tensors)
            pred_ens = float(predictions.get("pred_ensemble", predictions.get("Pred_Ensemble")))
            raw_bag = round(pred_ens - chronological_age, 2) if chronological_age is not None else None
            
            bc_bag = None
            if raw_bag is not None and resolved_alpha is not None and resolved_beta is not None:
                bc_bag = round(raw_bag - (resolved_alpha * chronological_age + resolved_beta), 2)
                
            res = {
                "Patient_ID": patient_id,
                "Chronological_Age": chronological_age,
                "Predicted_Age": round(pred_ens, 2),
                "Pred_Ensemble": round(pred_ens, 2),
                "Pred_Axial": round(float(predictions.get("pred_axial", predictions.get("Pred_Axial"))), 2),
                "Pred_Coronal": round(float(predictions.get("pred_coronal", predictions.get("Pred_Coronal"))), 2),
                "Pred_Sagittal": round(float(predictions.get("pred_sagittal", predictions.get("Pred_Sagittal"))), 2),
                "Raw_BAG": raw_bag,
                "bc_BAG": bc_bag
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
            print(f"\n[!] Error processing subject {r['id']}: {e}", file=sys.stderr)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
    df_out = pd.DataFrame(all_results)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)
    
    print("\n" + "="*80)
    print(f" BATCH INFERENCE COMPLETED | N = {len(df_out)} subjects processed successfully")
    print(f" [✓] Consolidated predictions CSV saved to: {args.output_csv}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
