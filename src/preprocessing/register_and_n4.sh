#!/usr/bin/env bash
set -euo pipefail

INPUT_NII="$1"
PREP_DIR="$2"

MASK_DIR="${PREP_DIR}/masks"
QUASIRAW_DIR="${PREP_DIR}/quasiraw"
mkdir -p "${MASK_DIR}" "${QUASIRAW_DIR}"

base=$(basename "${INPUT_NII}")
base_noext="${base%.nii*}"

subject_mask="${MASK_DIR}/${base_noext}_mask.nii.gz"
subject_quasiraw="${QUASIRAW_DIR}/${base_noext}_desc-6apply_T1w.nii.gz"

if [ -f "${subject_quasiraw}" ]; then
    exit 0
fi

tmp_brain=$(mktemp --suffix=.nii.gz)

# 1) mri_synthstrip
if [ ! -f "${subject_mask}" ]; then
    mri_synthstrip -i "${INPUT_NII}" -o "${tmp_brain}" -m "${subject_mask}"
    rm -f "${tmp_brain}"
fi

# 2) brainprep quasiraw
brainprep quasiraw "${INPUT_NII}" "${subject_mask}" "${QUASIRAW_DIR}" --no-bids

# 3) Limpieza
find "${QUASIRAW_DIR}" -maxdepth 1 -type f -name "${base_noext}*" ! -name "*desc-6apply*" -exec rm -f {} \;
