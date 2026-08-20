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

# Parche de compatibilidad de brainprep para sistemas sin dpkg (RHEL, CentOS, Gentoo, macOS)
DUMMY_BIN=$(mktemp -d)
if ! command -v dpkg &> /dev/null; then
    cat << 'EOF_DPKG' > "${DUMMY_BIN}/dpkg"
#!/bin/sh
echo "dummy dpkg"
EOF_DPKG
    chmod +x "${DUMMY_BIN}/dpkg"
    export PATH="${DUMMY_BIN}:${PATH}"
fi

# Parche para bug de ANTs en brainprep (directorio fantasma con espacio)
mkdir -p " ${QUASIRAW_DIR}" 2>/dev/null || true

tmp_brain=$(mktemp --suffix=.nii.gz)

# 1) mri_synthstrip
if [ ! -f "${subject_mask}" ]; then
    mri_synthstrip -i "${INPUT_NII}" -o "${tmp_brain}" -m "${subject_mask}"
    rm -f "${tmp_brain}"
fi

# 2) brainprep quasiraw
brainprep quasiraw "${INPUT_NII}" "${subject_mask}" "${QUASIRAW_DIR}" --no-bids

# 3) Limpieza
rm -rf "${DUMMY_BIN}" " ${QUASIRAW_DIR}" 2>/dev/null || true
find "${QUASIRAW_DIR}" -maxdepth 1 -type f -name "${base_noext}*" ! -name "*desc-6apply*" -exec rm -f {} \;
