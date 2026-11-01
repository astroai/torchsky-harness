#!/usr/bin/env bash
# UNIONS tile coadd v3: weightmask (flat-based masks, cosmics, object-masked sky, inverse variance) per exposure,
# then torchsky Lanczos3 weighted coadd on the official tile grid. torch 2.14 lane + torchfits 1.3.1, CPU.
set -uo pipefail
A=/arc/projects/ots/adass2026
OUT=$A/${OUTNAME:-unions_coadd_v5}/${TILE:-CFIS.000.229}; mkdir -p "$OUT"
exec > >(tee -a "$OUT/session.log") 2>&1
echo "start $(date -u +%FT%TZ) host=$(hostname) cpus=$(nproc) mem=$(free -g | awk '/Mem/{print $2}')G"
export PIP_CACHE_DIR=/scratch/pip-cache TMPDIR=/scratch/tmp PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$(nproc)
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv /scratch/venvB || { echo "venv failed"; exit 3; }
/scratch/venvB/bin/pip install -q "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
/scratch/venvB/bin/pip install -q scikit-build-core nanobind numpy cmake ninja
mkdir -p /scratch/torchsky-src /scratch/wheels && tar -xzf $A/scripts/${TSKY_SRC:-torchsky-src.tar.gz} -C /scratch/torchsky-src
/scratch/venvB/bin/pip wheel -q --no-build-isolation --no-deps -w /scratch/wheels /scratch/torchsky-src || { echo "torchsky build failed"; exit 4; }
python3 -m venv /scratch/venvU || { echo "venv failed"; exit 3; }
P=/scratch/venvU/bin
if nvidia-smi -L >/dev/null 2>&1; then IDX=https://download.pytorch.org/whl/cu126; else IDX=https://download.pytorch.org/whl/cpu; fi
$P/pip install -q "torch==2.14.0" --index-url $IDX
$P/pip install -q numpy scipy astropy matplotlib "torchfits==1.3.1" vos weightmask
$P/pip install -q --no-deps /scratch/wheels/torchsky-*.whl
$P/python -c "import torch, torchfits, torchsky, weightmask; print('torch', torch.__version__, 'torchfits', torchfits.__version__, 'torchsky', torchsky.__version__, 'weightmask', weightmask.__version__)"
export PATH=$P:$PATH
for b in ${BANDS:-r u}; do
  python -u $A/scripts/unions_coadd_v5.py --tile ${TILE:-CFIS.000.229} --band $b --work /scratch/unions --out "$OUT" ${UARGS:-} || echo "band $b failed"
done
vmkdir -p vos:sfabbro/adass2026/${OUTNAME:-unions_coadd_v5}/$(basename $OUT) 2>/dev/null
cp $A/scripts/unions_coadd_v5.py $A/scripts/unions_coadd_v5.sh "$OUT"/ 2>/dev/null
for f in "$OUT"/*; do vcp "$f" vos:sfabbro/adass2026/${OUTNAME:-unions_coadd_v5}/$(basename $OUT)/ || echo "publish failed $f"; done
echo "published to vos:sfabbro/adass2026/${OUTNAME:-unions_coadd_v5}/$(basename $OUT)"
echo "done $(date -u +%FT%TZ)"
