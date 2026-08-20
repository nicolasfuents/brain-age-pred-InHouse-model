#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_pipeline.py

Main end-to-end orchestration pipeline for Brain Age Gap (BAG) estimation
and Explainable AI (XAI) feature attribution maps (Integrated Gradients, Occlusion Sensitivity, Grad-Attention).
Implements the exact training preprocessing pipeline:
  - Skull stripping: mri_synthstrip
  - Quasiraw MNI152 registration & N4 bias field correction: brainprep.sh
  - Spatial & contrast harmonization: SOLID_v2 mask + P1-P99 clipping + MinMax[0, 1]
  - 2.5D triplanar slice extraction & Ridge Stacker Ensemble inference
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

def load_config(config_path: Path = REPO_ROOT / "config.yaml") -> Dict[str, Any]:
    """Loads hyperparameters and configurations from config.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_precomputed_tensors(target_path: Path) -> Dict[str, torch.Tensor]:
    """Loads existing .pt tensors from a directory or combined dictionary."""
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
            
    raise FileNotFoundError(f"Could not load valid .pt tensors from {target_path}")

def run_brainprep_quasiraw(input_nii: Path, prep_dir: Path) -> Path:
    """Executes the training cohort preprocessing script (SynthStrip + brainprep quasiraw)."""
    script_path = REPO_ROOT / "src" / "preprocessing" / "register_and_n4.sh"
    prep_dir.mkdir(parents=True, exist_ok=True)
    
    env = os.environ.copy()
    
    # 1. PATH setup
    conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
    env["PATH"] = f"{conda_prefix}/bin:" + env.get("PATH", "")
    
    # 2. FreeSurfer setup
    if "FREESURFER_HOME" not in env and "EBROOTFREESURFER" in env:
        env["FREESURFER_HOME"] = env["EBROOTFREESURFER"]
    if "FREESURFER_HOME" in env:
        env["PATH"] = f"{env['FREESURFER_HOME']}/bin:" + env["PATH"]
    if "FS_LICENSE" not in env:
        fs_lic = Path.home() / ".licenses" / "freesurfer.lic"
        if fs_lic.exists():
            env["FS_LICENSE"] = str(fs_lic)
            
    # 3. FSL setup
    if "FSLDIR" in env:
        env["PATH"] = f"{env['FSLDIR']}/bin:" + env["PATH"]
        env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
        
    # 4. Dummy dpkg for brainprep (compatibility across Linux/RHEL/CentOS/macOS)
    dummy_dpkg = prep_dir / "dpkg"
    with open(dummy_dpkg, "w") as f:
        f.write("#!/bin/sh\necho ''\n")
    dummy_dpkg.chmod(0o755)
    env["PATH"] = f"{prep_dir}:" + env["PATH"]
    
    # 5. Ghost dir workaround for ANTs N4 inside brainprep
    ghost_dir = os.path.join(os.getcwd(), " " + str(prep_dir / "quasiraw"))
    os.makedirs(ghost_dir, exist_ok=True)
    
    # 6. Execute preprocessing bash script
    cmd = ["bash", str(script_path), str(input_nii), str(prep_dir)]
    subprocess.run(cmd, env=env, check=True)
    
    # 7. Locate desc-6apply output
    candidates = list((prep_dir / "quasiraw").glob("*desc-6apply*.nii.gz"))
    if not candidates:
        raise FileNotFoundError(f"BrainPrep pipeline did not produce desc-6apply file in {prep_dir / 'quasiraw'}")
    return candidates[0]

def run_single_subject(
    input_dicom: Optional[Path],
    input_t1: Optional[Path],
    manual_age: Optional[float],
    output_dir: Path,
    run_all_xai: bool,
    config: Dict[str, Any],
    skip_prep: bool = False
) -> Dict[str, Any]:
    """Runs the end-to-end brain age estimation pipeline for a single subject."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp_processing"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    patient_id = "PATIENT_001"
    chronological_age = manual_age
    tensors = None
    
    # 1. Skip-prep mode
    if skip_prep:
        tensors_dir = output_dir / "tensors"
        if input_t1 and (input_t1.suffix == ".pt" or (input_t1.is_dir() and (input_t1 / "tensor_axial.pt").exists())):
            print(f"\n[+] (--skip-prep) Loading precomputed .pt tensors from: {input_t1}")
            tensors = load_precomputed_tensors(input_t1)
            patient_id = input_t1.stem
        elif tensors_dir.exists() and (tensors_dir / "tensor_axial.pt").exists():
            print(f"\n[+] (--skip-prep) Reusing existing tensors in: {tensors_dir}")
            tensors = load_precomputed_tensors(tensors_dir)
            if input_t1: patient_id = input_t1.name.split(".")[0]
        elif input_t1 and (input_t1.name.endswith(".nii") or input_t1.name.endswith(".nii.gz")):
            print(f"\n[+] (--skip-prep) Direct slice extraction from preprocessed MNI NIfTI volume: {input_t1}")
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
        # 2. Input ingestion (DICOM vs NIfTI)
        if input_dicom:
            print(f"\n[+] Ingesting DICOM study from: {input_dicom}")
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
            print(f"\n[+] Ingesting NIfTI T1w volume: {input_t1}")
            nifti_path = input_t1
            patient_id = nifti_path.name.split(".")[0]
        else:
            raise ValueError("Must provide either --input_dicom or --input_t1.")

        # 3. Spatial dimensions verification (Native vs MNI152)
        nii = nib.load(str(nifti_path))
        if nii.shape != (182, 218, 182):
            print(f"\n[+] Input volume is in native space {nii.shape}. Running automated quasiraw preprocessing (brainprep.sh)...")
            missing_tools = []
            for tool in ["mri_synthstrip", "brainprep", "flirt", "N4BiasFieldCorrection"]:
                if shutil.which(tool) is None:
                    missing_tools.append(tool)
            if missing_tools:
                raise EnvironmentError(
                    f"Native preprocessing requires external tools: {', '.join(missing_tools)}.\n"
                    "Please ensure FSL, ANTs, and FreeSurfer (mri_synthstrip) are installed and available in PATH\n"
                    "(on HPC clusters: 'module load fsl ants freesurfer && source $FSLDIR/etc/fslconf/fsl.sh'),\n"
                    "or provide a pre-registered MNI152 volume (182, 218, 182) with --skip-prep."
                )
            
            prep_dir = temp_dir / "quasiraw_out"
            nifti_path = run_brainprep_quasiraw(nifti_path, prep_dir)
            print(f"  * Volume successfully aligned to MNI152 1mm (quasiraw): {nifti_path}")
        else:
            print(f"  * Volume is already in MNI152 space (182, 218, 182).")

        # 4. 2.5D slice extraction and P1-P99 normalization
        mask_path = REPO_ROOT / config["atlases"]["mask"]
        print(f"\n[+] Extracting 2.5D slices and normalizing intensities (P1-P99)...")
        tensors = process_nifti_to_tensors(
            nii_path=nifti_path,
            mask_path=mask_path,
            output_dir=output_dir / "tensors"
        )
        print(f"  * Slices extracted successfully for Axial, Coronal, and Sagittal planes.")

    # 5. Triplanar Inference & Ridge Stacker (TTA active)
    checkpoints_dir = REPO_ROOT / config["models"]["checkpoints_dir"]
    print(f"\n[+] Running triplanar inference with Test-Time Augmentation (TTA)...")
    predictor = TriplanarPredictor(checkpoints_dir=checkpoints_dir, use_tta=True)
    predictions = predictor.predict(tensors)
    pred_ens_val = float(predictions.get("pred_ensemble", predictions.get("Pred_Ensemble")))
    
    # 6. Brain Age Gap (Raw BAG) & Optional Local Bias Correction
    raw_bag_val = round(pred_ens_val - chronological_age, 2) if chronological_age is not None else None
    
    calib_cfg = config.get("calibration", {})
    alpha = calib_cfg.get("alpha", None)
    beta = calib_cfg.get("beta", None)
    
    bc_bag_val = None
    if raw_bag_val is not None and alpha is not None and beta is not None:
        bc_bag_val = round(raw_bag_val - (float(alpha) * chronological_age + float(beta)), 2)

    # Consolidate results
    final_results = {
        "subject_id": patient_id,
        "chronological_age": chronological_age,
        "predicted_age": round(pred_ens_val, 2),
        "pred_ensemble": round(pred_ens_val, 2),
        "pred_axial": round(float(predictions.get("pred_axial", predictions.get("Pred_Axial"))), 2),
        "pred_coronal": round(float(predictions.get("pred_coronal", predictions.get("Pred_Coronal"))), 2),
        "pred_sagittal": round(float(predictions.get("pred_sagittal", predictions.get("Pred_Sagittal"))), 2),
        "raw_bag": raw_bag_val,
        "bc_bag": bc_bag_val
    }
    
    print("\n" + "="*80)
    print(f" BRAIN AGE PREDICTION & ESTIMATION REPORT: {patient_id}")
    print("="*80)
    if chronological_age is not None:
        print(f"  * Chronological Age:          {chronological_age:.2f} years")
    print(f"  * Predicted Brain Age:        {final_results['predicted_age']:.2f} years")
    if chronological_age is not None:
        print(f"  * Brain Age Gap (Raw BAG):    {final_results['raw_bag']:+.2f} years")
        if bc_bag_val is not None:
            print(f"  * Calibrated Gap (bc-BAG):    {final_results['bc_bag']:+.2f} years (local calibration)")
    print("-"*80)
    print(" Interpretation & Guidelines:")
    print("  - Predicted Brain Age: Model-estimated biological brain age from triplanar MRI.")
    if chronological_age is not None:
        print("  - Raw BAG (Predicted - Chronological): Difference between estimated brain age")
        print("    and chronological age. Positive values indicate an older-appearing brain;")
        print("    negative values indicate a younger-appearing brain.")
        if bc_bag_val is None:
            print("  - Scanner Calibration Note: This Raw BAG is uncalibrated for your specific scanner.")
            print("    To adjust for scanner-specific bias and regression-to-the-mean, calibrate using")
            print("    a local Healthy Control cohort (calibrate_local_scanner.py).")
            print("    * Recommended sample size: N >= 30 (ideally N >= 50) healthy controls")
            print("      spanning the age range of interest.")
            print("    * Re-calibration is recommended periodically when adding new control batches")
            print("      (e.g., every 50-100 new scans or following major scanner software/hardware updates).")
        else:
            print("  - Calibrated bc-BAG: Bias-corrected gap using your local scanner calibration parameters.")
    print("="*80)

    # Save quantitative metrics to JSON and CSV
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)
        
    csv_path = output_dir / "results.csv"
    pd.DataFrame([final_results]).to_csv(csv_path, index=False)
    print(f"\n[✓] Quantitative metrics saved to: {json_path} and {csv_path}")

    # 7. Explainable AI (XAI) Suite (Optional with --all)
    if run_all_xai:
        print(f"\n[+] Generating Explainable AI (XAI) feature attribution suite (--all)...")
        xai_dir = output_dir / "xai"
        xai_engine = XAIEngine(predictor=predictor)
        xai_engine.generate_explanations(
            tensors=tensors,
            predictions=final_results,
            output_dir=xai_dir,
            patient_id=patient_id
        )
        print(f"[✓] XAI feature attribution suite generated successfully in: {xai_dir}")

    # Temporary cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        
    return final_results

def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Brain Age Gap (BAG) Estimation and Explainable AI (XAI) Pipeline."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dicom", type=Path, help="Path to DICOM directory or .zip archive.")
    input_group.add_argument("--input_t1", type=Path, help="Path to T1w NIfTI volume (.nii/.nii.gz) or preprocessed tensor (.pt).")
    input_group.add_argument("--input_csv", type=Path, help="Path to manifest CSV for batch processing.")
    
    parser.add_argument("--age", type=float, default=None, help="Chronological age in years (optional for NIfTI, auto-extracted for DICOM).")
    parser.add_argument("--output_dir", type=Path, default=Path("./output"), help="Output directory (default: ./output).")
    parser.add_argument("--all", action="store_true", help="Generate full XAI attribution suite (IG, Occlusion, Grad-Attention, multi-method panel).")
    parser.add_argument("--skip_prep", "--skip-prep", dest="skip_prep", action="store_true", help="Skip registration and run direct inference on pre-aligned MNI152 volumes or tensors.")
    
    args = parser.parse_args()
    config = load_config()
    
    if args.input_csv:
        print(f"\n[+] Starting batch inference from manifest: {args.input_csv}")
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
        print(f"\n[✓] Batch inference completed. Summary saved to: {summary_csv}")
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
