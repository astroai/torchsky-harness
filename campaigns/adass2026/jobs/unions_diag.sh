#!/usr/bin/env bash
# UNIONS CPU coadd timing diagnosis: same prepared CCDs, torchsky 9749c80 (v5 baseline) vs NEW, fixed vs auto tiles.
set -uo pipefail
A=/arc/projects/ots/adass2026
OUT=$A/${OUTNAME:-unions_diag}; mkdir -p "$OUT"
exec > >(tee -a "$OUT/session.log") 2>&1
echo "start $(date -u +%FT%TZ) host=$(hostname) cpus=$(nproc) mem=$(free -g | awk '/Mem/{print $2}')G"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${THREADS:-$(nproc)}
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv /scratch/venvB && /scratch/venvB/bin/pip install -q "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu \
  && /scratch/venvB/bin/pip install -q scikit-build-core nanobind numpy cmake ninja || exit 3
for v in OLD NEW; do
  src=$([ $v = OLD ] && echo ${OLD_SRC:-torchsky-src-9749c80.tar.gz} || echo ${NEW_SRC:-torchsky-src-82522bf.tar.gz})
  mkdir -p /scratch/src$v /scratch/wh$v && tar -xzf $A/scripts/$src -C /scratch/src$v
  /scratch/venvB/bin/pip wheel -q --no-build-isolation --no-deps -w /scratch/wh$v /scratch/src$v || { echo "build $v failed"; exit 4; }
  python3 -m venv /scratch/venv$v
  /scratch/venv$v/bin/pip install -q "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
  /scratch/venv$v/bin/pip install -q numpy scipy astropy "torchfits==1.3.1" vos weightmask
  /scratch/venv$v/bin/pip install -q --no-deps /scratch/wh$v/torchsky-*.whl
  echo "$v = $src"
done
export PATH=/scratch/venvNEW/bin:$PATH
W=/scratch/diag
python -u $A/scripts/unions_diag.py prep --work $W --max-exposures ${NEXP:-2} || exit 5
for cfg in ${CONFIGS:-OLD:tile2500 NEW:tile2500 NEW:auto}; do
  set -- ${cfg/:/ }
  flag=$([ "$2" = auto ] && echo --auto || echo "--tile-edge ${2#tile}")
  /scratch/venv$1/bin/python -u $A/scripts/unions_diag.py run --work $W --out "$OUT" --label "${1}_$2" $flag || echo "$cfg failed"
done
python -u $A/scripts/unions_diag.py cmp --work $W --out "$OUT"
echo "done $(date -u +%FT%TZ)"
