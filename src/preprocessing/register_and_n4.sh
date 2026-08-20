#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# register_and_n4.sh
# Pipeline estándar de registro quasiraw hacia MNI152 (1mm) y corrección N4.
# Herramientas requeridas: mri_synthstrip, fslreorient2std, fslmaths, flirt, N4BiasFieldCorrection
# ==============================================================================

INPUT_NII="$1"
PREP_DIR="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET_TEMPLATE="${REPO_ROOT}/data/atlases/mni152_t1_1mm_brain.nii.gz"

if [ ! -f "${TARGET_TEMPLATE}" ]; then
    # Fallback a FSLDIR si existe
    if [ -n "${FSLDIR:-}" ] && [ -f "${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz" ]; then
        TARGET_TEMPLATE="${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz"
    else
        echo "[!] Error: No se encontró la plantilla MNI152 en ${TARGET_TEMPLATE}" >&2
        exit 1
    fi
fi

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

WORKDIR=$(mktemp -d -t quasiraw_XXXXXX)
trap 'rm -rf "${WORKDIR}"' EXIT

echo "  * [1/5] Extracción cerebral con mri_synthstrip..."
if [ ! -f "${subject_mask}" ]; then
    mri_synthstrip -i "${INPUT_NII}" -o "${WORKDIR}/synth_brain.nii.gz" -m "${subject_mask}"
fi

echo "  * [2/5] Reorientación estándar (fslreorient2std)..."
fslreorient2std "${INPUT_NII}" "${WORKDIR}/std.nii.gz"
fslreorient2std "${subject_mask}" "${WORKDIR}/stdmask.nii.gz"

echo "  * [3/5] Enmascaramiento y corrección de inhomogeneidad N4..."
fslmaths "${WORKDIR}/std.nii.gz" -mas "${WORKDIR}/stdmask.nii.gz" "${WORKDIR}/brain.nii.gz"
N4BiasFieldCorrection -d 3 -i "${WORKDIR}/brain.nii.gz" -o "${WORKDIR}/bfc.nii.gz" -s 4 -c [50x50x50x50,0.0001] -b [200]

echo "  * [4/5] Registro afín a MNI152 1mm (FLIRT 12-DOF)..."
flirt -in "${WORKDIR}/bfc.nii.gz" -ref "${TARGET_TEMPLATE}" -out "${WORKDIR}/reg.nii.gz" -omat "${WORKDIR}/trf.mat" -dof 12 -cost corratio

echo "  * [5/5] Aplicación de transformación afín y recorte de salida..."
flirt -in "${WORKDIR}/stdmask.nii.gz" -ref "${WORKDIR}/reg.nii.gz" -out "${WORKDIR}/regmask.nii.gz" -applyxfm -init "${WORKDIR}/trf.mat" -interp nearestneighbour
fslmaths "${WORKDIR}/reg.nii.gz" -mas "${WORKDIR}/regmask.nii.gz" "${subject_quasiraw}"

echo "  * Preprocesamiento quasiraw finalizado exitosamente: ${subject_quasiraw}"
