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
#   REQUIRE_MAP_COMPAT 1 to make SWarp/Montage gates required (default 0; 1 under IMAGE_PROFILE=improc)
#   REQUIRE_TOOLS      1 to require the external binaries (SWarp, SCAMP, SExtractor, PSFEx, solve-field,
#                      Montage, WCSTools) and the astrometry index dir (default 0; 1 under improc)
#   IMAGE_PROFILE      "improc" for images.canfar.net/astroai/improc:latest (ships those binaries);
#                      auto-selected when swarp, mProject and solve-field are all on PATH
#   CATALOG_REAL_DIR   pre-staged real Gaia/2MASS catalogues (gaia.fits, twomass.fits, manifest.json); default: fetched once
#                      into $OUT/real-catalogs by scripts/fetch_real_catalogs.py (cached; recorded in the summary)
#   ASTROMETRY_INDEX_DIR  astrometry.net index files (default /arc/projects/uvickbos/astrometry.net/data)
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
if [ -z "${IMAGE_PROFILE:-}" ] && command -v swarp >/dev/null 2>&1 && command -v mProject >/dev/null 2>&1 \
   && command -v solve-field >/dev/null 2>&1; then
  IMAGE_PROFILE=improc
fi
IMAGE_PROFILE=${IMAGE_PROFILE:-base}
if [ "$IMAGE_PROFILE" = improc ]; then
  REQUIRE_MAP_COMPAT=${REQUIRE_MAP_COMPAT:-1}
  REQUIRE_TOOLS=${REQUIRE_TOOLS:-1}
  export ASTROMETRY_INDEX_DIR=${ASTROMETRY_INDEX_DIR:-/arc/projects/uvickbos/astrometry.net/data}
fi
echo "image profile: $IMAGE_PROFILE"
export TORCHSKY_HARNESS_REQUIRE_MAP_COMPAT=${REQUIRE_MAP_COMPAT:-0}
export TORCHSKY_HARNESS_REQUIRE_TOOLS=${REQUIRE_TOOLS:-0}

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
# Unpinned: long-standing parity comparators. Pinned (==): comparators added with the claims-vs-gates
# audit, versions as verified locally (see docs/coverage-matrix.md).
for pkg in healpy hpgeom astropy-healpix mocpy spherical-geometry healsparse reproject drizzle dfreproject \
           dust_extinction speclite synphot pyerfa MontagePy \
           gwcs==1.0.3 sgp4==2.27 jplephem==2.24 mhealpy==0.3.7 cdshealpix==0.9.0 specutils==2.4.0 \
           torch-harmonics==0.8.0 's2fft==1.5.0' 'jax<0.8' 'jaxlib<0.8'; do  # jax>=0.12 breaks s2fft 1.5.0 HEALPix FFT
  if "$RUN_VENV/bin/pip" install -q "$pkg" >> "$OUT/comparator-install.log" 2>&1; then
    echo "comparator ok: $pkg"
  else
    echo "comparator FAILED: $pkg (see comparator-install.log)"
  fi
done
# torch-harmonics must not replace the pinned torch wheel.
"$RUN_VENV/bin/pip" install -q --no-deps "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$TORCH_INDEX" >> "$OUT/comparator-install.log" 2>&1 || true
for tool in swarp scamp source-extractor psfex solve-field mProject mProjExec mAdd xy2sky sky2xy stilts; do
  if command -v "$tool" >/dev/null 2>&1; then echo "binary ok: $tool ($(command -v "$tool"))"
  else echo "binary absent: $tool ($([ "${REQUIRE_TOOLS:-0}" = 1 ] && echo REQUIRED || echo optional))"; fi
done
# DE440s kernel for the ephemeris parity gate (fail-soft here; the gate fails closed if it is missing).
if [ -z "${TORCHSKY_HARNESS_DE_KERNEL:-}" ]; then
  if curl -fsSL --retry 3 -o "$SCRATCH/de440s.bsp" https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp; then
    export TORCHSKY_HARNESS_DE_KERNEL="$SCRATCH/de440s.bsp"; echo "kernel ok: de440s"
  else echo "kernel FAILED: de440s (ephemeris gate will fail)"; fi
fi

# Out-of-process oracles: PyAST and AstroDrizzle in their own venv (any Python/numpy they build on).
# tests/_oracle.py runs them through
# scripts/oracle/*.py with NPZ exchange. A genuine provisioning failure exports
# TORCHSKY_HARNESS_ORACLE_<ENGINE>_UNPROVISIONED=<reason>; its gate then fails with "NOT PROVISIONED: <reason>".
# Note: the run venv (summary "Comparators" table) never holds these packages; the oracle versions are listed
# under "Out-of-process oracles".
: > "$OUT/oracle-install.log"
echo "oracle interpreter: $("$PYTHON_BIN" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
provision_venv() { # provision_venv <venv> <log> : create a venv from $PYTHON_BIN with pip; prints reason on failure
  "$PYTHON_BIN" -m venv "$1" >> "$2" 2>&1 || { echo "venv creation failed with $PYTHON_BIN (python3-venv missing?)"; return 1; }
  "$1/bin/python" -m pip --version >> "$2" 2>&1 || "$1/bin/python" -m ensurepip --upgrade >> "$2" 2>&1 \
    || { echo "no pip in the oracle venv"; return 1; }
  "$1/bin/python" -m pip install -q --upgrade pip setuptools wheel numpy astropy >> "$2" 2>&1 \
    || { echo "pip could not install numpy/astropy (index unreachable?)"; return 1; }
}
oracle_install() { # oracle_install <ENGINE> <venv> <pip requirement...> -- <import check>
  local engine=$1 venv=$2; shift 2
  local reqs=() check=""
  while [ $# -gt 0 ]; do [ "$1" = "--" ] && { check=$2; break; }; reqs+=("$1"); shift; done
  if "$venv/bin/python" -m pip install -q "${reqs[@]}" >> "$OUT/oracle-install.log" 2>&1; then
    if "$venv/bin/python" -c "$check" >> "$OUT/oracle-install.log" 2>&1; then
      export "TORCHSKY_HARNESS_ORACLE_${engine}_PYTHON=$venv/bin/python"; echo "oracle ok: $engine ($("$venv/bin/python" -V))"
      return 0
    fi
    reason="installed ${reqs[*]} but '$check' fails"
  else reason="pip install ${reqs[*]} failed: $(tail -n 1 "$OUT/oracle-install.log" | cut -c1-160)"; fi
  export "TORCHSKY_HARNESS_ORACLE_${engine}_UNPROVISIONED=$reason"
  echo "oracle NOT PROVISIONED: $engine: $reason (see oracle-install.log)"
}
ORACLE_VENV="$SCRATCH/venv-oracle"
if reason=$(provision_venv "$ORACLE_VENV" "$OUT/oracle-install.log"); then
  oracle_install PYAST "$ORACLE_VENV" "starlink-pyast==4.2.0" -- "import starlink.Ast"
  oracle_install DRIZZLEPAC "$ORACLE_VENV" "drizzlepac==3.11.0" -- "import drizzlepac.cdriz"
else
  export TORCHSKY_HARNESS_ORACLE_PYAST_UNPROVISIONED="$reason" TORCHSKY_HARNESS_ORACLE_DRIZZLEPAC_UNPROVISIONED="$reason"
  echo "oracle NOT PROVISIONED: venv-oracle: $reason"
fi
# Real catalogues for the STILTS gate: pre-staged ($CATALOG_REAL_DIR) or fetched once per campaign.
if [ -z "${CATALOG_REAL_DIR:-}" ]; then
  if "$RUN_VENV/bin/python" "$HARNESS_DIR/scripts/fetch_real_catalogs.py" --out "$OUT/real-catalogs" >> "$OUT/oracle-install.log" 2>&1; then
    export CATALOG_REAL_DIR="$OUT/real-catalogs"; echo "real catalogues ok: $CATALOG_REAL_DIR"
  else echo "real catalogues FAILED to fetch (the STILTS real-data gate fails closed under REQUIRE_TOOLS)"; fi
else echo "real catalogues pre-staged: $CATALOG_REAL_DIR"; fi

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
