#!/usr/bin/env bash
# Run the torchsky-harness campaign on a CANFAR headless session (CPU or GPU).
#
# Required env:
#   TSKY_SRC     torchsky source tarball (.tar.gz) on /arc
#   HARNESS_SRC  torchsky-harness source tarball (.tar.gz) on /arc
# Optional env:
#   RUN          output directory name (default harness_<UTC timestamp>)
#   MODE         cpu | gpu (default cpu)
#   SUITES       comma list of correctness,parity,performance,examples (default: all four)
#   TORCHSKY_ARC_ROOT  durable storage root (default /arc/projects/ots)
#   OUT          output directory (default $TORCHSKY_ARC_ROOT/harness-campaigns/$RUN)
#   SCRATCH      fast local scratch (default /scratch)
#   TORCH_VERSION      torch wheel version (default 2.14.0)
#   TORCHFITS_SPEC     torchfits requirement (default torchfits==1.3.1)
#   TSKY_COMMIT        subject git commit to record (tarballs carry none)
#   GATE_TIMEOUT       per-gate timeout in seconds (default 3600)
#   REQUIRE_MAP_COMPAT 1 to make SWarp/Montage gates required (default 0)
# Writes $OUT/summary.md, $OUT/summary.json, $OUT/gates.jsonl, $OUT/gates/*.log.
set -uo pipefail

: "${TSKY_SRC:?set TSKY_SRC to the torchsky source tarball}"
: "${HARNESS_SRC:?set HARNESS_SRC to the torchsky-harness source tarball}"
RUN=${RUN:-harness_$(date -u +%Y%m%dT%H%M%SZ)}
MODE=${MODE:-cpu}
SUITES=${SUITES:-correctness,parity,performance,examples}
ARC_ROOT=${TORCHSKY_ARC_ROOT:-/arc/projects/ots}
OUT=${OUT:-$ARC_ROOT/harness-campaigns/$RUN}
SCRATCH=${SCRATCH:-/scratch}
TORCH_VERSION=${TORCH_VERSION:-2.14.0}
TORCHFITS_SPEC=${TORCHFITS_SPEC:-torchfits==1.3.1}
GATE_TIMEOUT=${GATE_TIMEOUT:-3600}
case "$MODE" in cpu|gpu) ;; *) echo "MODE must be cpu or gpu" >&2; exit 2 ;; esac

mkdir -p "$OUT"
exec > >(tee -a "$OUT/session.log") 2>&1
echo "campaign $RUN mode=$MODE suites=$SUITES start=$(date -u +%FT%TZ)"
[ "$MODE" = gpu ] && { nvidia-smi -L || { echo "no GPU visible in gpu mode" >&2; exit 3; }; }

export PIP_CACHE_DIR="$SCRATCH/pip-cache" TMPDIR="$SCRATCH/tmp" PYTHONNOUSERSITE=1
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
unset PYTHONPATH
export TORCHSKY_HARNESS_REQUIRE_MAP_COMPAT=${REQUIRE_MAP_COMPAT:-0}

# Pick the newest interpreter >= 3.12.
PYTHON_BIN=""
for candidate in ${PYTHON:-} python3.13 python3.12 python3; do
  [ -n "$candidate" ] || continue
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
    PYTHON_BIN=$candidate; break
  fi
done
[ -n "$PYTHON_BIN" ] || { echo "need Python >= 3.12 on the image" >&2; exit 4; }

extract() { # extract <tarball> <dest>: flatten a single top-level directory
  local tmp; tmp=$(mktemp -d "$TMPDIR/extract.XXXXXX")
  tar -xzf "$1" -C "$tmp" || return 1
  mkdir -p "$2"
  local entries; entries=$(ls -A "$tmp")
  if [ "$(printf '%s\n' "$entries" | wc -l)" -eq 1 ] && [ -d "$tmp/$entries" ]; then
    cp -a "$tmp/$entries/." "$2/"
  else
    cp -a "$tmp/." "$2/"
  fi
  rm -rf "$tmp"
}

SRC_DIR="$SCRATCH/torchsky-src"; HARNESS_DIR="$SCRATCH/torchsky-harness"; WHEELS="$SCRATCH/wheels"
rm -rf "$SRC_DIR" "$HARNESS_DIR" "$WHEELS"; mkdir -p "$WHEELS"
extract "$TSKY_SRC" "$SRC_DIR" || { echo "cannot extract TSKY_SRC" >&2; exit 5; }
extract "$HARNESS_SRC" "$HARNESS_DIR" || { echo "cannot extract HARNESS_SRC" >&2; exit 5; }

# 1. Build the subject wheel in a CPU-torch build venv (same torch ABI as the run venv).
BUILD_VENV="$SCRATCH/venv-build"; RUN_VENV="$SCRATCH/venv-run"
"$PYTHON_BIN" -m venv "$BUILD_VENV"
"$BUILD_VENV/bin/pip" install -q "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu || exit 6
"$BUILD_VENV/bin/pip" install -q scikit-build-core nanobind numpy cmake ninja setuptools wheel || exit 6
"$BUILD_VENV/bin/pip" wheel -q --no-build-isolation --no-deps -w "$WHEELS" "$SRC_DIR" || { echo "subject wheel build failed" >&2; exit 7; }
WHEEL=$(ls "$WHEELS"/torchsky-*.whl | head -n1)

# 2. Run venv: torch (cpu or cu130) + comparators + subject + harness.
"$PYTHON_BIN" -m venv "$RUN_VENV"
TORCH_INDEX=cpu; [ "$MODE" = gpu ] && TORCH_INDEX=cu130
"$RUN_VENV/bin/pip" install -q "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$TORCH_INDEX" || exit 6
"$RUN_VENV/bin/pip" install -q numpy scipy astropy "$TORCHFITS_SPEC" pyarrow polars pandas requests psutil \
  pytest pytest-xdist pytest-benchmark matplotlib setuptools wheel tomli || exit 6
# Comparators: install one by one so a single failure is recorded, not fatal.
: > "$OUT/comparator-install.log"
for pkg in healpy hpgeom astropy-healpix mocpy spherical-geometry healsparse reproject drizzle dfreproject \
           dust_extinction speclite synphot pyerfa MontagePy; do
  if "$RUN_VENV/bin/pip" install -q "$pkg" >> "$OUT/comparator-install.log" 2>&1; then
    echo "comparator ok: $pkg"
  else
    echo "comparator FAILED: $pkg (see comparator-install.log)"
  fi
done
command -v swarp >/dev/null 2>&1 && echo "comparator ok: SWarp ($(command -v swarp))" || echo "comparator absent: SWarp (optional)"

"$RUN_VENV/bin/pip" install -q --no-deps --force-reinstall "$WHEEL" || { echo "subject install failed" >&2; exit 7; }
"$RUN_VENV/bin/pip" install -q --no-deps --no-build-isolation -e "$HARNESS_DIR" || { echo "harness install failed" >&2; exit 7; }

# 3. Record the subject identity next to the venv (read by torchsky_harness.subject).
"$RUN_VENV/bin/python" - "$TSKY_SRC" "$WHEEL" "${TSKY_COMMIT:-}" <<'PY'
import hashlib, json, sys
from pathlib import Path
src, wheel, commit = sys.argv[1:4]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
record = {"requested": src, "kind": "sdist", "sha256": sha(src), "wheel": Path(wheel).name,
          "wheel_sha256": sha(wheel), "revision": commit or None, "dirty": None}
Path(sys.prefix, "torchsky-subject-install.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY
cd "$SCRATCH" && "$RUN_VENV/bin/python" -I -c "
import torch, torchsky
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'torchsky', torchsky.__file__)
assert '$SRC_DIR' not in torchsky.__file__, 'subject imported from the source tree'
" || exit 8
[ "$MODE" = gpu ] && { "$RUN_VENV/bin/python" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "CUDA unavailable in gpu mode" >&2; exit 3; }; }

# 4. Suites.
cd "$HARNESS_DIR" || exit 9
RC=0
IFS=',' read -r -a WANTED <<< "$SUITES"
for suite in "${WANTED[@]}"; do
  case "$suite" in
    correctness|parity|performance|examples) ;;
    *) echo "unknown suite: $suite" >&2; RC=2; continue ;;
  esac
  echo "=== suite $suite ($(date -u +%FT%TZ)) ==="
  extra=()
  [ "$suite" = examples ] && extra=(--examples-dir "$SRC_DIR/examples")
  "$RUN_VENV/bin/python" scripts/canfar/run_gates.py --suite "$suite" --mode "$MODE" --out "$OUT" \
    --timeout "$GATE_TIMEOUT" ${extra[@]+"${extra[@]}"} || RC=1
done

# 5. Summary and raw outputs.
[ -d benchmark_results ] && cp -a benchmark_results "$OUT/benchmark_results"
"$RUN_VENV/bin/pip" freeze > "$OUT/pip-freeze.txt" 2>/dev/null
"$RUN_VENV/bin/python" scripts/canfar/summarize.py --out "$OUT" --run "$RUN" --mode "$MODE" || RC=1
echo "campaign $RUN done rc=$RC $(date -u +%FT%TZ)"
exit "$RC"
