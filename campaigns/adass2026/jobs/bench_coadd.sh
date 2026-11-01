#!/usr/bin/env bash
# Speed benchmark: torchsky SWarp-like / Drizzle-like (CPU, GPU) vs SWarp 2.41 and STScI drizzle. Image astroai/improc (has SWarp).
set -uo pipefail
A=/arc/projects/ots/adass2026
RUN=${RUN:-bench_coadd}
OUT=$A/$RUN; mkdir -p "$OUT/provenance"
exec > >(tee -a "$OUT/session.log") 2>&1
publish() {
  V=/scratch/venvG/bin; [ -x $V/vcp ] || V=$(dirname "$(command -v vcp)")
  $V/vmkdir -p vos:sfabbro/adass2026/$RUN 2>/dev/null; $V/vmkdir -p vos:sfabbro/adass2026/$RUN/provenance 2>/dev/null
  (cd "$OUT" && find . -type f | while read f; do $V/vcp "$f" "vos:sfabbro/adass2026/$RUN/${f#./}" || echo "publish failed $f"; done)
}
trap publish EXIT
echo "start $(date -u +%FT%TZ) host=$(hostname) cpus=$(nproc)"; nvidia-smi -L || true
P=$OUT/provenance
cp $A/scripts/bench_coadd.py $A/scripts/bench_coadd.sh "$P/"; printf '%s\n' "TSKY_SRC=${TSKY_SRC}" "THREADS=${THREADS:-8}" > "$P/launch.env"
sha256sum $A/scripts/bench_coadd.py "$A/scripts/${TSKY_SRC}" > "$P/sha256.txt"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH; mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv /scratch/venvC
/scratch/venvC/bin/pip install -q "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
/scratch/venvC/bin/pip install -q scikit-build-core nanobind numpy cmake ninja
mkdir -p /scratch/torchsky-src /scratch/wheels && tar -xzf "$A/scripts/${TSKY_SRC}" -C /scratch/torchsky-src
/scratch/venvC/bin/pip wheel -q --no-build-isolation --no-deps -w /scratch/wheels /scratch/torchsky-src || exit 4
python3 -m venv /scratch/venvG
/scratch/venvG/bin/pip install -q "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu126
/scratch/venvG/bin/pip install -q numpy astropy "torchfits==1.3.1" drizzle sip_tpv vos
/scratch/venvG/bin/pip install -q --no-deps /scratch/wheels/torchsky-*.whl
/scratch/venvG/bin/pip freeze > "$P/pip-freeze.txt"; swarp -v > "$P/swarp-version.txt" 2>&1; nproc > "$P/nproc.txt"
/scratch/venvG/bin/python -u $A/scripts/bench_coadd.py --work /scratch/bench --out "$OUT" --threads ${THREADS:-8}
echo "exit=$? done $(date -u +%FT%TZ)"
