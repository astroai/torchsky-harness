#!/usr/bin/env bash
set -euo pipefail
script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_root}/image.sh"

usage() {
  printf '%s\n' "Usage: run_canfar.sh OUTPUT_ROOT HARNESS_SHA BASELINE_KIND BASELINE_VALUE CANDIDATE_KIND CANDIDATE_VALUE PINNED_IMAGE PINNED_DIGEST"
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi

[[ $# -eq 8 ]] || { usage >&2; exit 2; }
persistent_dest=$1
harness_sha=$2
baseline_kind=$3
baseline_value=$4
candidate_kind=$5
candidate_value=$6
container_image=$7
container_digest=$8

[[ ${persistent_dest} == /arc/projects/* ]] || { printf '%s\n' "OUTPUT_ROOT must be under /arc/projects" >&2; exit 2; }
[[ ! -e ${persistent_dest} ]] || { printf '%s\n' "refusing existing destination: ${persistent_dest}" >&2; exit 2; }
[[ ${harness_sha} =~ ^[0-9a-f]{40}$ ]] || { printf '%s\n' "HARNESS_SHA must be exact" >&2; exit 2; }
[[ ${container_image} == "${TORCHSKY_RELEASE_PERFORMANCE_IMAGE}" ]] || { printf '%s\n' "PINNED_IMAGE does not match campaign policy" >&2; exit 2; }
[[ ${container_image} == images.canfar.net/astroai/base:26.10 ]] || { printf '%s\n' "PINNED_IMAGE must use the immutable campaign tag" >&2; exit 2; }
[[ ${container_digest} == "${TORCHSKY_RELEASE_PERFORMANCE_IMAGE_DIGEST}" ]] || { printf '%s\n' "PINNED_DIGEST does not match campaign policy" >&2; exit 2; }
[[ ${container_digest} =~ ^sha256:[0-9a-f]{64}$ ]] || { printf '%s\n' "PINNED_IMAGE must contain an immutable digest" >&2; exit 2; }
[[ ${baseline_kind} == sha || ${baseline_kind} == wheel ]] || { printf '%s\n' "invalid baseline kind" >&2; exit 2; }
[[ ${candidate_kind} == sha || ${candidate_kind} == wheel ]] || { printf '%s\n' "invalid candidate kind" >&2; exit 2; }

export PYTHONNOUSERSITE=1
unset PYTHONPATH
resolved_digest=$(python3 -c 'import json,urllib.request; print(json.load(urllib.request.urlopen("https://images.canfar.net/api/v2.0/projects/astroai/repositories/base/artifacts/26.10"))["digest"])')
[[ ${resolved_digest} == "${container_digest}" ]] || { printf '%s\n' "container tag digest drift: ${resolved_digest}" >&2; exit 2; }
export PIXI_HOME=/scratch/env/pixi
export PIXI_CACHE_DIR=/scratch/env/pixi-cache
export PIP_CACHE_DIR=/scratch/data/pip-cache
export TMPDIR=/scratch/data/tmp
mkdir -p /scratch/code /scratch/env /scratch/data "${PIP_CACHE_DIR}" "${TMPDIR}"
artifact_root=/scratch/data/release-performance-output
log_path=/scratch/data/release-performance.log
job_metadata=/scratch/data/canfar_job.json
printf '{"harness_sha":"%s","baseline_kind":"%s","baseline_value":"%s","candidate_kind":"%s","candidate_value":"%s","container_image":"%s","container_digest":"%s","cpu":8,"memory_gb":32,"gpu":0,"concurrency":1,"max_attempts":1}\n' \
  "${harness_sha}" "${baseline_kind}" "${baseline_value}" "${candidate_kind}" "${candidate_value}" "${container_image}" "${container_digest}" > "${job_metadata}"
exec > "${log_path}" 2>&1

on_exit() {
  original_rc=$?
  trap - EXIT
  set +e
  persistence_failed=0
  mkdir -p "${artifact_root}" || persistence_failed=1
  cp "${job_metadata}" "${artifact_root}/canfar_job.json" || persistence_failed=1
  cp "${log_path}" "${artifact_root}/release-performance.log" || persistence_failed=1
  if [[ ${original_rc} -eq 0 ]]; then
    for required in manifest.json release_performance_claims.json release_performance_readiness.json canfar_job.json release-performance.log; do
      [[ -s ${artifact_root}/${required} ]] || persistence_failed=1
    done
  fi
  printf '{"exit_code":%d,"persistence_failed":%s,"finished_utc":"%s"}\n' \
    "${original_rc}" "$([[ ${persistence_failed} -eq 0 ]] && printf false || printf true)" "$(date -u +%FT%TZ)" > "${artifact_root}/EXIT.json" || persistence_failed=1
  mkdir -p "${persistent_dest}" || persistence_failed=1
  cp -a "${artifact_root}/." "${persistent_dest}/" || persistence_failed=1
  if [[ -d ${persistent_dest} ]]; then
    source_count=$(find "${artifact_root}" -type f | wc -l | tr -d ' ')
    destination_count=$(find "${persistent_dest}" -type f | wc -l | tr -d ' ')
    [[ ${source_count} == "${destination_count}" ]] || persistence_failed=1
    while IFS= read -r -d '' source_file; do
      relative=${source_file#"${artifact_root}/"}
      [[ -f ${persistent_dest}/${relative} ]] || { persistence_failed=1; continue; }
      cmp -s "${source_file}" "${persistent_dest}/${relative}" || persistence_failed=1
    done < <(find "${artifact_root}" -type f -print0)
  fi
  if [[ ${persistence_failed} -ne 0 ]]; then
    failure='{"exit_code":70,"reason":"durable artifact persistence verification failed"}'
    printf '%s\n' "${failure}" > "${persistent_dest}/PERSISTENCE_FAILED.json" 2>/dev/null || true
    printf '%s\n' "${failure}" > "${persistent_dest}.persistence-failed.json" 2>/dev/null || true
    printf '{"exit_code":70,"persistence_failed":true,"original_exit_code":%d,"finished_utc":"%s"}\n' \
      "${original_rc}" "$(date -u +%FT%TZ)" > "${persistent_dest}/EXIT.json" 2>/dev/null || true
    exit 70
  fi
  exit "${original_rc}"
}
trap on_exit EXIT

command -v pixi >/dev/null 2>&1 || { printf '%s\n' "pixi is required in the container" >&2; exit 1; }
harness_root=/scratch/code/harness
[[ ! -e ${harness_root} ]] || { printf '%s\n' "refusing existing scratch checkout" >&2; exit 2; }
git clone --filter=blob:none --no-checkout https://github.com/astroai/torchsky-harness.git "${harness_root}"
git -C "${harness_root}" checkout --detach "${harness_sha}"
[[ $(git -C "${harness_root}" rev-parse HEAD) == "${harness_sha}" ]]
[[ -z $(git -C "${harness_root}" status --porcelain) ]]
pixi run --manifest-path "${harness_root}/pixi.toml" -e test -- python -m pip install --no-deps --no-build-isolation -e "${harness_root}"

args=(
  --harness-sha "${harness_sha}"
  --output-root "${artifact_root}"
  --work-root /scratch/code/paired-performance
  --pairs 5
)
if [[ ${baseline_kind} == sha ]]; then
  args+=(--baseline-sha "${baseline_value}")
else
  [[ -f ${baseline_value} ]]
  args+=(--baseline-wheel "${baseline_value}")
fi
if [[ ${candidate_kind} == sha ]]; then
  args+=(--candidate-sha "${candidate_value}")
else
  [[ -f ${candidate_value} ]]
  args+=(--candidate-wheel "${candidate_value}")
fi

pixi run --manifest-path "${harness_root}/pixi.toml" -e test -- python -m torchsky_harness.paired_performance "${args[@]}"
