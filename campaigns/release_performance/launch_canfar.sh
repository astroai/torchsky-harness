#!/usr/bin/env bash
set -euo pipefail
script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${script_root}/image.sh"

submit=0
cpu=8
memory=32
name=release-performance
image=${TORCHSKY_RELEASE_PERFORMANCE_IMAGE}
container_digest=${image##*@}

usage() {
  printf '%s\n' "Usage: launch_canfar.sh [--submit] --output-root /arc/projects/... --harness-sha SHA --baseline-sha SHA|--baseline-wheel PATH --candidate-sha SHA|--candidate-wheel PATH [--name NAME]"
}

output_root=
harness_sha=
baseline_kind=
baseline_value=
candidate_kind=
candidate_value=
while [[ $# -gt 0 ]]; do
  case $1 in
    --submit) submit=1 ;;
    --output-root|--harness-sha|--baseline-sha|--baseline-wheel|--candidate-sha|--candidate-wheel|--name)
      option=$1
      shift
      [[ $# -gt 0 ]] || { printf '%s\n' "${option} requires a value" >&2; exit 2; }
      case ${option} in
        --output-root) output_root=$1 ;;
        --harness-sha) harness_sha=$1 ;;
        --baseline-sha) baseline_kind=sha; baseline_value=$1 ;;
        --baseline-wheel) baseline_kind=wheel; baseline_value=$1 ;;
        --candidate-sha) candidate_kind=sha; candidate_value=$1 ;;
        --candidate-wheel) candidate_kind=wheel; candidate_value=$1 ;;
        --name) name=$1 ;;
      esac
      ;;
    --help|-h) usage; exit 0 ;;
    *) printf '%s\n' "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ ${output_root} == /arc/projects/* ]] || { printf '%s\n' "output root must be under /arc/projects" >&2; exit 2; }
[[ ! -e ${output_root} ]] || { printf '%s\n' "refusing existing destination: ${output_root}" >&2; exit 2; }
[[ ${harness_sha} =~ ^[0-9a-f]{40}$ ]]
[[ -n ${baseline_kind} && -n ${candidate_kind} ]]
[[ ${image} == images.canfar.net/astroai/base@${container_digest} ]]
[[ ${container_digest} =~ ^sha256:[0-9a-f]{64}$ ]]
for value in "${output_root}" "${harness_sha}" "${baseline_kind}" "${baseline_value}" "${candidate_kind}" "${candidate_value}" "${image}"; do
  [[ ${value} =~ ^[A-Za-z0-9_./:@-]+$ ]] || { printf '%s\n' "argument contains unsupported characters: ${value}" >&2; exit 2; }
done

remote="git clone --filter=blob:none --no-checkout https://github.com/astroai/torchsky-harness.git /scratch/code-launcher && git -C /scratch/code-launcher checkout --detach ${harness_sha} && bash /scratch/code-launcher/campaigns/release_performance/run_canfar.sh ${output_root} ${harness_sha} ${baseline_kind} ${baseline_value} ${candidate_kind} ${candidate_value} ${image}"
create=(create headless "${image}" --name "${name}" --cpu "${cpu}" --memory "${memory}" --env PYTHONNOUSERSITE=1 -- bash -lc "${remote}")
printf '%s\n' "image=${image}" "container_digest=${container_digest}" "cpu=${cpu}" "memory_gb=${memory}" "gpu=0" "concurrency=1" "max_attempts=1" "submission=canfar ${create[*]}"
if [[ ${submit} -eq 1 ]]; then
  canfar "${create[@]}"
fi
