#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?usage: run_canfar.sh RUN_ID}
harness_commit=${TORCHSKY_HARNESS_COMMIT:?set TORCHSKY_HARNESS_COMMIT to an exact SHA}
subject_commit=${TORCHSKY_COMMIT:?set TORCHSKY_COMMIT to an exact SHA}
persistent_dest=${CFHT_PERSISTENT_DEST:?set CFHT_PERSISTENT_DEST under /arc/projects}
harness_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
work_root=/scratch/${run_id}
artifact_root=${work_root}/artifacts
data_root=${work_root}/data
subject_root=${work_root}/torchsky-subject

case "${persistent_dest}" in
  /arc/projects/*) ;;
  *) echo "CFHT_PERSISTENT_DEST must be under /arc/projects" >&2; exit 2 ;;
esac
test "$(git -C "${harness_root}" rev-parse HEAD)" = "${harness_commit}"
test -z "$(git -C "${harness_root}" status --porcelain --untracked-files=no)"
test ! -e "${work_root}"
test ! -e "${persistent_dest}"
mkdir -p "${artifact_root}/logs" "${artifact_root}/results" "${artifact_root}/manifests" "${artifact_root}/products/reference"
exec > >(tee "${artifact_root}/logs/session.log") 2>&1

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export PIXI_HOME=/scratch/pixi
export PIXI_CACHE_DIR=/scratch/pixi-cache
export PIP_CACHE_DIR=/scratch/pip-cache
export TMPDIR=/scratch/tmp
mkdir -p "${PIXI_HOME}" "${PIXI_CACHE_DIR}" "${PIP_CACHE_DIR}" "${TMPDIR}"
for directory in /opt/astroai/venv/cadc/bin /opt/astroai/bin /opt/conda/bin /usr/local/bin; do
  if [[ -x "${directory}/vcp" || -x "${directory}/gh" ]]; then
    export PATH="${directory}:${PATH}"
  fi
done

persist() {
  local rc=$?
  set +e
  printf '{"exit_code":%d,"finished_utc":"%s"}\n' "${rc}" "$(date -u +%FT%TZ)" > "${artifact_root}/results/exit.json"
  mkdir -p "${persistent_dest}"
  cp -a "${artifact_root}/." "${persistent_dest}/"
  copy_rc=$?
  if [[ ${copy_rc} -ne 0 ]]; then
    echo "persistent artifact copy failed: ${persistent_dest}" >&2
    exit 70
  fi
  exit "${rc}"
}
trap persist EXIT

for command in gh git pixi vls vcp funpack; do
  command -v "${command}" >/dev/null || { echo "missing required command: ${command}" >&2; exit 2; }
done

export TORCHSKY_SUBJECT_ROOT=${subject_root}
export TORCHSKY_SUBJECT_REPO=${TORCHSKY_SUBJECT_REPO:-astroai/torchsky}
source "${harness_root}/scripts/subject_install.sh"
torchsky_install_subject map-compat "${harness_root}"
pixi run --manifest-path "${harness_root}/pixi.toml" -e map-compat -- \
  python -m torchsky_harness.cli subject > "${artifact_root}/manifests/subject.json"
printf '{"harness_commit":"%s","subject_commit":"%s","run_id":"%s"}\n' \
  "${harness_commit}" "${subject_commit}" "${run_id}" > "${artifact_root}/manifests/run.json"

pixi run --manifest-path "${harness_root}/pixi.toml" -e map-compat -- \
  python "${harness_root}/campaigns/cfht_union_pitcairn/discover.py" \
  --scratch-dir "${data_root}" \
  --json-out "${artifact_root}/manifests/discovery.json" \
  --fetch

for reference in "${data_root}"/reference/*.fits; do
  ln "${reference}" "${artifact_root}/products/reference/$(basename "${reference}")"
done

for band in u r; do
  exposure_count=6
  pixi run --manifest-path "${harness_root}/pixi.toml" -e map-compat -- \
    python "${harness_root}/scripts/bench_cfht_megacam.py" \
    --band "${band}" \
    --data-dir "${data_root}/inputs/${band}" \
    --exposures "${exposure_count}" \
    --ccds 36 \
    --device cpu \
    --require-swarp \
    --reference "${data_root}/reference/CFIS.000.229.${band}.fits" \
    --output-dir "${artifact_root}/products" \
    --json-out "${artifact_root}/results/CFIS.000.229.${band}.smoke.json" \
    2>&1 | tee "${artifact_root}/logs/CFIS.000.229.${band}.log"
done
