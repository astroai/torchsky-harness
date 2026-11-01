#!/usr/bin/env bash
# UNIONS tile astrometry vs Gaia DR3 (CPU). env: TSKY_SRC, COADD_DIR (default unions_coadd_v3/CFIS.000.229), TILE, BANDS
set -uo pipefail
A=/arc/projects/ots/adass2026
RUN=${RUN:-unions_gaia_astrom}
OUT=$A/$RUN; mkdir -p "$OUT/provenance"
exec > >(tee -a "$OUT/session.log") 2>&1
publish() {
  V=/scratch/venvC/bin; [ -x $V/vcp ] || V=$(dirname "$(command -v vcp)")
  $V/vmkdir -p vos:sfabbro/adass2026/$RUN 2>/dev/null; $V/vmkdir -p vos:sfabbro/adass2026/$RUN/provenance 2>/dev/null
  (cd "$OUT" && find . -type f | while read f; do $V/vcp "$f" "vos:sfabbro/adass2026/$RUN/${f#./}" || echo "publish failed $f"; done)
  echo "published to vos:sfabbro/adass2026/$RUN"
}
trap publish EXIT
echo "start $(date -u +%FT%TZ) host=$(hostname)"
P=$OUT/provenance
cp $A/scripts/unions_gaia_astrom.py $A/scripts/unions_gaia.sh "$P/"
printf '%s\n' "TSKY_SRC=${TSKY_SRC}" "COADD_DIR=${COADD_DIR:-}" "TILE=${TILE:-}" "BANDS=${BANDS:-}" > "$P/launch.env"
sha256sum $A/scripts/unions_gaia_astrom.py "$A/scripts/${TSKY_SRC}" > "$P/sha256.txt"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
unset PYTHONPATH
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv /scratch/venvC
/scratch/venvC/bin/pip install -q "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
/scratch/venvC/bin/pip install -q scikit-build-core nanobind numpy cmake ninja
mkdir -p /scratch/torchsky-src /scratch/wheels && tar -xzf "$A/scripts/${TSKY_SRC}" -C /scratch/torchsky-src
/scratch/venvC/bin/pip wheel -q --no-build-isolation --no-deps -w /scratch/wheels /scratch/torchsky-src || exit 4
/scratch/venvC/bin/pip install -q --no-deps /scratch/wheels/torchsky-*.whl
/scratch/venvC/bin/pip install -q "torchfits==1.3.1" astroquery astropy matplotlib vos
/scratch/venvC/bin/pip freeze > "$P/pip-freeze.txt"
export PATH=/scratch/venvC/bin:$PATH
python -c "import torch, torchfits, torchsky; print('torch', torch.__version__, 'torchfits', torchfits.__version__, 'torchsky', torchsky.__version__)" | tee "$P/versions.txt"
python -u $A/scripts/unions_gaia_astrom.py --tile ${TILE:-CFIS.000.229} --bands ${BANDS:-r,u} \
  --coadd-dir ${COADD_DIR:-$A/unions_coadd_v3/CFIS.000.229} --work /scratch/g --out "$OUT"
echo "exit=$? done $(date -u +%FT%TZ)"
