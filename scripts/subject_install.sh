#!/usr/bin/env bash
# Install an explicitly selected subject into an evaluation environment.
torchsky_install_subject() {
  local env_name=$1
  local harness_root=$2
  if [[ -n "${TORCHSKY_WHEEL:-}" ]]; then
    test -f "${TORCHSKY_WHEEL}"
    pixi run --manifest-path "${harness_root}/pixi.toml" -e "${env_name}" -- \
      python -m pip install --no-deps --force-reinstall "${TORCHSKY_WHEEL}"
  else
    local subject_root=${TORCHSKY_SUBJECT_ROOT:?set TORCHSKY_SUBJECT_ROOT or TORCHSKY_WHEEL}
    local subject_commit=${TORCHSKY_COMMIT:?set the exact subject SHA}
    if [[ ! -d "${subject_root}/.git" ]]; then
      test ! -e "${subject_root}"
      gh repo clone "${TORCHSKY_SUBJECT_REPO:-astroai/torchsky}" "${subject_root}" -- --filter=blob:none
      git -C "${subject_root}" checkout --detach "${subject_commit}"
    fi
    test "$(git -C "${subject_root}" rev-parse HEAD)" = "${subject_commit}"
    test -z "$(git -C "${subject_root}" status --porcelain)"
    pixi run --manifest-path "${harness_root}/pixi.toml" -e "${env_name}" -- \
      python -m pip install -e "${subject_root}" --no-deps --no-build-isolation
  fi
  pixi run --manifest-path "${harness_root}/pixi.toml" -e "${env_name}" -- \
    python -m pip install -e "${harness_root}" --no-deps --no-build-isolation
  pixi run --manifest-path "${harness_root}/pixi.toml" -e "${env_name}" -- \
    python -c 'import torchsky; import torchsky._cpp'
}
