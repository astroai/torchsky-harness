#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHSKY_ROOT=${TORCHSKY_ROOT:-$(dirname "${ROOT_DIR}")/torchsky}
PROFILE=standard
SUBMIT=0
MODES=()

usage() {
  cat <<'EOF'
Usage: launch_canfar_real_data_fleet.sh [--submit] [--full] [--mode MODE ...]

Dry-runs by default. The standard fleet launches real-data-gates and
catalog-scale-real. --full also launches observed-frame, optimization CPU/CUDA,
GPU validation, and the full SMACS depth chain. --mode replaces the profile.

Required inputs are exact clean commits already pushed to each configured
origin. Real datasets remain on CANFAR project mounts; this script never
downloads them to the laptop.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)
      SUBMIT=1
      ;;
    --full)
      PROFILE=full
      ;;
    --mode)
      shift
      [[ $# -gt 0 ]] || { echo "--mode requires a value" >&2; exit 2; }
      MODES+=("$1")
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ${#MODES[@]} -eq 0 ]]; then
  MODES=(real-data-gates catalog-scale-real)
  if [[ "${PROFILE}" == "full" ]]; then
    MODES+=(observed optimization gpu-validate depth)
  fi
fi

HARNESS_COMMIT=${TORCHSKY_HARNESS_COMMIT:-$(git -C "${ROOT_DIR}" rev-parse HEAD)}
SUBJECT_COMMIT=${TORCHSKY_COMMIT:-$(git -C "${TORCHSKY_ROOT}" rev-parse HEAD)}
HARNESS_REPO=astroai/torchsky-harness
if [[ -z "${TORCHSKY_SUBJECT_REPO:-}" ]]; then
  subject_origin=$(git -C "${TORCHSKY_ROOT}" remote get-url origin)
  TORCHSKY_SUBJECT_REPO=$(printf '%s' "${subject_origin}" \
    | sed -E 's#^(git@github.com:|https://github.com/)##; s#\.git$##')
fi

for repository in "${ROOT_DIR}" "${TORCHSKY_ROOT}"; do
  [[ -d "${repository}/.git" ]] || { echo "not a git repository: ${repository}" >&2; exit 2; }
  [[ -z "$(git -C "${repository}" status --porcelain)" ]] \
    || { echo "dirty repository: ${repository}" >&2; exit 2; }
done

remote_has_revision() {
  local repository=$1
  local revision=$2
  git ls-remote "https://github.com/${repository}.git" \
    | cut -f1 | grep -Fxq "${revision}"
}

remote_has_revision "${HARNESS_REPO}" "${HARNESS_COMMIT}" \
  || { echo "harness revision is not a pushed tip in ${HARNESS_REPO}" >&2; exit 2; }
remote_has_revision "${TORCHSKY_SUBJECT_REPO}" "${SUBJECT_COMMIT}" \
  || { echo "torchsky revision is not a pushed tip in ${TORCHSKY_SUBJECT_REPO}" >&2; exit 2; }

STAMP=$(date -u +%Y%m%d-%H%M%S)
MANIFEST_DIR="${ROOT_DIR}/benchmarks_results/canfar_real_data_fleet_${STAMP}"
mkdir -p "${MANIFEST_DIR}"
printf '%s\n' \
  "created_utc=$(date -u +%FT%TZ)" \
  "profile=${PROFILE}" \
  "submit=${SUBMIT}" \
  "harness_commit=${HARNESS_COMMIT}" \
  "torchsky_commit=${SUBJECT_COMMIT}" \
  "modes=${MODES[*]}" > "${MANIFEST_DIR}/fleet_manifest.txt"

export TORCHSKY_HARNESS_COMMIT="${HARNESS_COMMIT}"
export TORCHSKY_COMMIT="${SUBJECT_COMMIT}"
export TORCHSKY_SUBJECT_REPO
export TORCHSKY_CANFAR_DRY_RUN=$((1 - SUBMIT))

for mode in "${MODES[@]}"; do
  echo "fleet_launch mode=${mode} submit=${SUBMIT}"
  bash "${ROOT_DIR}/scripts/launch_canfar_wcs_sphere.sh" "${mode}" \
    | tee "${MANIFEST_DIR}/${mode}.log"
done

echo "fleet_manifest=${MANIFEST_DIR}/fleet_manifest.txt"
