#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_preprocess.py

High-Throughput Parallel Batch Preprocessing Pipeline for MRI Datasets.
Prepares cohorts of raw T1w MRI scans (DICOM studies or native NIfTI volumes)
by running automated skull-stripping (mri_synthstrip), 12-DOF affine FLIRT registration
and N4 bias field correction to MNI152 (1mm), and extracting normalized 2.5D triplanar slice tensors.
Outputs a standardized manifest CSV ready for batch inference.
"""

import sys
import os
import shutil
import argparse
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import nibabel as nib
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from src.preprocessing.dicom_reader import handle_input_path, extract_patient_info, convert_dicom_to_nifti
from src.preprocessing.slice_extractor import process_nifti_to_tensors

def load_config(config_path: Path = REPO_ROOT / "config.yaml") -> Dict[str, Any]:
    """Loads configuration hyperparameters."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def process_single_item(
    item: Dict[str, Any],
    output_dir: Path,
    mask_path: Path,
    save_nifti: bool = False
) -> Dict[str, Any]:
    """Processes a single subject through skull-stripping, registration, and slice extraction."""
    subj_id = item.get("subject_id", item.get("id", f"sub-{Path(item.get('input_t1', item.get('input_dicom'))).stem}"))
    subj_out_dir = output_dir / subj_id
    subj_out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = subj_out_dir / "temp_prep"
    temp_dir.mkdir(parents=True, exist_ok=True)
    tensors_dir = subj_out_dir / "tensors"
    
    res = {
        "subject_id": subj_id,
        "age": item.get("age", None),
        "tensors_dir": str(tensors_dir),
        "preprocessed_nii": "",
        "status": "FAILED",
        "error_msg": ""
    }
    
    # Check if already processed
    if (tensors_dir / "tensor_axial.pt").exists() and (tensors_dir / "tensor_coronal.pt").exists() and (tensors_dir / "tensor_sagittal.pt").exists():
        res["status"] = "ALREADY_PROCESSED"
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        return res
        
    try:
        raw_nii = None
        chrono_age = item.get("age", None)
        
        # 1. Ingestion
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
            raise ValueError("Must specify either input_dicom or input_t1.")

        # 2. Check spatial dimensions (Native vs MNI152)
        nii = nib.load(str(raw_nii))
        mni_nii = raw_nii
        
        if nii.shape != (182, 218, 182):
            script_path = REPO_ROOT / "src" / "preprocessing" / "register_and_n4.sh"
            prep_work_dir = temp_dir / "quasiraw"
            prep_work_dir.mkdir(parents=True, exist_ok=True)
            
            env = os.environ.copy()
            conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
            env["PATH"] = f"{conda_prefix}/bin:" + env.get("PATH", "")
            if "FREESURFER_HOME" not in env and "EBROOTFREESURFER" in env:
                env["FREESURFER_HOME"] = env["EBROOTFREESURFER"]
            if "FREESURFER_HOME" in env:
                env["PATH"] = f"{env['FREESURFER_HOME']}/bin:" + env["PATH"]
            if "FS_LICENSE" not in env:
                fs_lic = Path.home() / ".licenses" / "freesurfer.lic"
                if fs_lic.exists(): env["FS_LICENSE"] = str(fs_lic)
            if "FSLDIR" in env:
                env["PATH"] = f"{env['FSLDIR']}/bin:" + env["PATH"]
                env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
            dummy_dpkg = prep_work_dir / "dpkg"
            with open(dummy_dpkg, "w") as f:
                f.write("#!/bin/sh\necho ''\n")
            dummy_dpkg.chmod(0o755)
            env["PATH"] = f"{prep_work_dir}:" + env["PATH"]
            ghost_dir = os.path.join(os.getcwd(), " " + str(prep_work_dir / "quasiraw"))
            os.makedirs(ghost_dir, exist_ok=True)
            
            cmd = ["bash", str(script_path), str(raw_nii), str(prep_work_dir)]
            subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            
            candidates = list((prep_work_dir / "quasiraw").glob("*desc-6apply*.nii.gz"))
            if not candidates:
                raise FileNotFoundError("Preprocessing did not produce desc-6apply NIfTI.")
            mni_nii = candidates[0]
            
        # 3. Save preprocessed NIfTI if requested
        if save_nifti:
            final_nii_path = subj_out_dir / f"{subj_id}_preprocessed_MNI152.nii.gz"
            shutil.copyfile(mni_nii, final_nii_path)
            res["preprocessed_nii"] = str(final_nii_path)
            
        # 4. Extract 2.5D normalized tensors
        process_nifti_to_tensors(
            nii_path=mni_nii,
            mask_path=mask_path,
            output_dir=tensors_dir
        )
        
        res["status"] = "SUCCESS"
        
    except Exception as e:
        res["status"] = "ERROR"
        res["error_msg"] = str(e)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
    return res

def scan_directory_for_inputs(data_dir: Path) -> List[Dict[str, Any]]:
    """Scans directory hierarchy for T1w NIfTI or DICOM subjects."""
    items = []
    nii_files = sorted(list(data_dir.rglob("*.nii")) + list(data_dir.rglob("*.nii.gz")))
    
    # Filter out masks or intermediate files
    t1_candidates = [
        f for f in nii_files 
        if not any(k in f.name.lower() for k in ["mask", "desc-1", "desc-2", "desc-3", "desc-4", "desc-5", "roi"])
    ]
    
    for f in t1_candidates:
        subj_name = f.stem.replace(".nii", "")
        items.append({
            "subject_id": subj_name,
            "input_t1": str(f),
            "age": None
        })
    return items

def main():
    parser = argparse.ArgumentParser(
        description="High-Throughput Parallel Batch Preprocessing Pipeline for MRI Datasets."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path, help="Input CSV manifest containing input_t1/input_dicom paths.")
    group.add_argument("--data_dir", type=Path, help="Directory containing raw MRI dataset to auto-discover.")
    
    parser.add_argument("--output_dir", type=Path, default=Path("./batch_preprocessed_output"), help="Output directory.")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel worker processes (default: 4).")
    parser.add_argument("--save_nifti", action="store_true", help="Save intermediate 3D MNI152 aligned NIfTI volumes alongside .pt tensors.")
    
    args = parser.parse_args()
    config = load_config()
    mask_path = REPO_ROOT / config["atlases"]["mask"]
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.manifest:
        df = pd.read_csv(args.manifest)
        items = df.to_dict(orient="records")
        print(f"\n[+] Loaded {len(items)} subjects from manifest: {args.manifest}")
    else:
        items = scan_directory_for_inputs(args.data_dir)
        print(f"\n[+] Discovered {len(items)} candidate scans in directory: {args.data_dir}")
        
    if not items:
        print("[!] No subjects found to process. Exiting.")
        return
        
    print(f"[+] Starting batch preprocessing on {len(items)} subjects with {args.n_jobs} parallel workers...")
    
    results = []
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {
            executor.submit(process_single_item, item, args.output_dir, mask_path, args.save_nifti): item
            for item in items
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            res = future.result()
            results.append(res)
            
    summary_df = pd.DataFrame(results)
    manifest_csv = args.output_dir / "preprocessed_manifest.csv"
    summary_df.to_csv(manifest_csv, index=False)
    
    n_succ = (summary_df["status"] == "SUCCESS").sum()
    n_alr = (summary_df["status"] == "ALREADY_PROCESSED").sum()
    n_err = (summary_df["status"] == "ERROR").sum()
    
    print("\n" + "="*80)
    print(" BATCH PREPROCESSING COMPLETED")
    print("="*80)
    print(f"  * Total Subjects:      {len(summary_df)}")
    print(f"  * Successfully Processed: {n_succ}")
    print(f"  * Previously Processed:   {n_alr}")
    print(f"  * Failed / Errors:        {n_err}")
    print(f"  * Manifest Generated:     {manifest_csv}")
    print("="*80)

if __name__ == "__main__":
    main()
