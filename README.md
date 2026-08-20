# Brain Age Prediction (In-House Model & Explainable AI Framework)

<p align="center">
  <img src="assets/banner.png" alt="Brain Age Prediction & Explainable AI Banner" width="100%">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/)

A standardized, high-performance Deep Learning inference pipeline for **Brain Age Gap (BAG)** estimation and **Explainable AI (XAI)** from structural T1-weighted MRI scans (DICOM studies or NIfTI volumes).

The system employs a multi-planar deep learning architecture combining convolutional neural networks (CNNs) for robust feature extraction with global-local self-attention mechanisms to capture multiscale morphological patterns across brain anatomy.

---

## Table of Contents

- [Installation & Requirements](#installation--requirements)
- [Usage Guide](#usage-guide)
  - [1. Single-Subject Fast Inference](#1-single-subject-fast-inference-run_pipelinepy)
  - [2. Full Inference with Explainable AI](#2-full-inference-with-explainable-ai---all)
  - [3. Large-Scale Batch Inference](#3-large-scale-batch-inference-batch_inferencepy)
  - [4. High-Throughput Batch Preprocessing](#4-high-throughput-batch-preprocessing-batch_preprocesspy)
  - [5. Local Scanner Calibration](#5-local-scanner-calibration-calibrate_local_scannerpy)
- [Model Interpretability Methods (XAI)](#model-interpretability-methods-xai)
- [Train/Val & Test](#trainval--test)
- [Disclaimer](#disclaimer)

---

## Installation & Requirements

### 1. Clone the repository
```bash
git clone https://github.com/nicolasfuents/brain-age-pred-InHouse-model.git
cd brain-age-pred-InHouse-model
```

### 2. Create and activate the Conda environment
```bash
conda env create -f environment.yml
conda activate brain_age_env
```

### 3. External Neuroimaging Dependencies (Preprocessing Pipeline)

To ensure maximum anatomical fidelity and reproducibility, raw MRI scans (DICOM or native NIfTI) must be processed through the standardized `src/preprocessing/` pipeline (identical to the training cohort preprocessing). This requires the following standard neuroimaging tools in your system `PATH`:

* **FSL** (tested with `v6.0.7.18`, compatible with `v6.0+`): `flirt`, `fslreorient2std`, `fslmaths` for 12-DOF affine registration and masking to MNI152 (1mm).
* **ANTs** (tested with `v2.6.2`, compatible with `v2.4+`): `N4BiasFieldCorrection` for B-spline non-parametric bias field correction.
* **FreeSurfer / SynthStrip** (tested with `v7.4.1`, compatible with `v7.0+`): `mri_synthstrip` for deep-learning intracranial skull stripping and brain mask generation.
* **dcm2niix** (tested with `v1.0.20230411`): High-performance DICOM-to-NIfTI conversion.

*Fast Inference Optimization (`--skip_prep`):*
If your volumes are already preprocessed (with our preprocessing pipeline), you can pass `--skip_prep` to bypass external neuroimaging tools and run inference directly using only Python and PyTorch.

---

## Usage Guide

### 1. Single-Subject Fast Inference (`run_pipeline.py`)
```bash
# Inference from DICOM (automatically extracts age from header):
python run_pipeline.py --input_dicom /path/to/DICOM_study/

# Inference from raw T1w NIfTI volume (runs automated SynthStrip + FLIRT + N4):
python run_pipeline.py --input_t1 /path/to/T1w_volume.nii.gz --age 68.5

# Direct inference on preprocessed volume (with our preprocessing pipeline):
python run_pipeline.py --input_t1 /path/to/preprocessed_MNI_volume.nii.gz --age 68.5 --skip_prep
```

### 2. Full Inference with Explainable AI (`--all`)
```bash
# Inference + Integrated Gradients, Occlusion Sensitivity, Grad-Attention & Visual Explanation Panel:
python run_pipeline.py --input_t1 /path/to/T1w_volume.nii.gz --age 68.5 --all
```

### 3. Large-Scale Batch Inference (`batch_inference.py`)
```bash
# High-throughput batch processing across a full cohort directory:
python batch_inference.py \
    --input_dir /path/to/cohort_scans_directory/ \
    --output_csv ./batch_predictions.csv \
    --output_dir ./batch_results
```

### 4. High-Throughput Batch Preprocessing (`batch_preprocess.py`)

Used to preprocess large cohorts in parallel prior to inference (runs deep-learning skull-stripping with SynthStrip, affine registration to MNI152 1mm, and N4 bias field correction):

```bash
# Parallel batch preprocessing on raw scans folder (scans auto-discovered in data_dir):
python batch_preprocess.py \
    --data_dir /path/to/raw_scans_directory/ \
    --output_dir ./preprocessed_cohort \
    --n_jobs 8

# Batch preprocessing from a CSV file (useful when raw scans are distributed across multiple paths or disks):
python batch_preprocess.py \
    --manifest /path/to/cohort_manifest.csv \
    --output_dir ./preprocessed_cohort \
    --n_jobs 8
```

### 5. Local Scanner Calibration (`calibrate_local_scanner.py`)

Fits Ordinary Least Squares (OLS) regression over local Cognitively Normal (CN) healthy controls to remove regression-to-the-mean age bias and scanner-specific contrast offsets:

```bash
python calibrate_local_scanner.py \
    --controls_csv ./controls_predictions.csv \
    --clinical_csv ./clinical_predictions.csv \
    --output_dir ./calibration_results
```

For complete step-by-step instructions, minimum sample size requirements (N >= 30, ideally N >= 50), and how to incorporate the fitted parameters into single-subject and batch runs, refer to **[HOWTO_CALIBRATION.md](HOWTO_CALIBRATION.md)**.

---

## Model Interpretability Methods (XAI)

When specifying the `--all` flag, the pipeline automatically generates 3 complementary visual explanation and feature attribution maps:

1. **Integrated Gradients (Signed IG):** Voxel-wise gradient path integration from a baseline to the input image, highlighting anatomical microstructures that positively (+) or negatively (-) contribute to predicted brain age.
2. **Occlusion Sensitivity:** Systematic sliding-patch perturbation mapping regional sensitivity across brain anatomy.
3. **Grad-Attention (Transformer Rollout):** Self-attention rollout gated by backpropagated gradients, isolating long-range anatomical patterns driving model predictions.

---

## Train/Val & Test

The architecture was trained on **N = 5,913 subjects** and internally evaluated on **N = 1,540 subjects** (total derivation cohort: **N = 7,453 subjects** across OpenBHB, ADNI3, Cam-CAN, AOMIC, OpenNeuro, and JUK):

### 1. Internal Validation Benchmark ($N = 1,540$)

* **Internal Validation Performance:** **MAE = 2.57 years** across the held-out stratified validation cohort ($N = 1,540$).

### 2. External Multi-Site Validation (Out-of-Distribution / OOD)

Evaluated across independent external cohorts and MRI field strengths without re-training:

| External Cohort | Scanner / Field Strength | Sample Size | MAE (Years) |
| :--- | :--- | :--- | :--- |
| **RRIB** | Siemens Tim Trio / 3.0T | $N = 158$ | **3.89 ± 0.27 yr** |
| **ADNI** | Multicenter / 1.5T & 3.0T | $N = 79$ | **4.07 ± 0.44 yr** |
| **OASIS-3** | Siemens / 3.0T | $N = 813$ | **4.15 ± 0.13 yr** |

---

## Disclaimer

This software is for **Research Use Only (RUO)**. It is not approved, certified, or intended for primary medical diagnosis, patient screening, or clinical decision-making. Brain age predictions, brain age gap (BAG) estimates, and explainable AI (XAI) feature attribution maps are mathematical approximations of brain morphology and must be interpreted in conjunction with comprehensive clinical assessments by qualified healthcare professionals.
