#!/usr/bin/env bash
set -euo pipefail

INPUT_NII="$1"
PREP_DIR="$2"

MASK_DIR="${PREP_DIR}/masks"
QUASIRAW_DIR="${PREP_DIR}/quasiraw"
WRAPPER_DIR="${PREP_DIR}/bin_wrappers"
mkdir -p "${MASK_DIR}" "${QUASIRAW_DIR}" "${WRAPPER_DIR}"

base=$(basename "${INPUT_NII}")
base_noext="${base%.nii*}"

subject_mask="${MASK_DIR}/${base_noext}_mask.nii.gz"
subject_quasiraw="${QUASIRAW_DIR}/${base_noext}_desc-6apply_T1w.nii.gz"

if [ -f "${subject_quasiraw}" ]; then
    exit 0
fi

# Intercept N4BiasFieldCorrection to fix upstream brainprep whitespace bug in -o argument
cat << 'WRAPPER_EOF' > "${WRAPPER_DIR}/N4BiasFieldCorrection"
#!/usr/bin/env bash
REAL_N4=$(which -a N4BiasFieldCorrection | grep -v "bin_wrappers" | head -n 1)
args=()
for arg in "$@"; do
    if [[ "$arg" == \[* ]]; then
        # Remove whitespace after commas inside bracketed arguments
        arg="${arg//, /,}"
    fi
    args+=("$arg")
done
exec "$REAL_N4" "${args[@]}"
WRAPPER_EOF
chmod +x "${WRAPPER_DIR}/N4BiasFieldCorrection"
export PATH="${WRAPPER_DIR}:${PATH}"

tmp_brain=$(mktemp --suffix=.nii.gz)

# 1) mri_synthstrip
if [ ! -f "${subject_mask}" ]; then
    mri_synthstrip -i "${INPUT_NII}" -o "${tmp_brain}" -m "${subject_mask}"
    rm -f "${tmp_brain}"
fi

# 2) brainprep quasiraw
brainprep quasiraw "${INPUT_NII}" "${subject_mask}" "${QUASIRAW_DIR}" --no-bids

# 3) Cleanup intermediate artifacts
find "${QUASIRAW_DIR}" -maxdepth 1 -type f -name "${base_noext}*" ! -name "*desc-6apply*" -exec rm -f {} \;
rm -rf "${WRAPPER_DIR}"
