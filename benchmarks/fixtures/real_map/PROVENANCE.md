# Real-image map fixture

- Source: `/Users/fabbros/data/CFHT/MegaPipe.120.258/g/mosaic_g.fits` (2096101440 bytes, sha256 `adbb4e0c729c738b3159af5e140acec093a7f03b8524e65c2af50b2ce3ce2c1f`)
- Derived: CFHT MegaPipe g-band mosaic, contiguous 256x256 cutout centred on the mosaic, full resolution
- WCS: the original TAN header recentred for the cutout
- Generator: `scripts/derive_real_map_fixture.py` (1.1s)

The fixture stores one real cutout plus its WCS keywords; raw survey pixels are not redistributed beyond this small cutout.
