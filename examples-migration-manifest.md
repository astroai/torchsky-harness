# Example notebooks and tweakreg campaign moved in second pass

The experiment notebooks and tweakreg comparison helper/test moved from the
torchsky checkout into this harness staging tree. Notebook launch instructions
now assume the harness root and an explicitly installed torchsky subject. The
tweakreg helper now names the harness SMACS comparison and its `smacs` Pixi
environment; its wrapper test resolves that staged helper by harness root.

| Original path | Harness path | Original SHA-256 | Staged SHA-256 |
| --- | --- | --- | --- |
| `examples/notebooks/01_field_fit_synthetic.ipynb` | `notebooks/01_field_fit_synthetic.ipynb` | `02e0eb79a0c463e8eff7fa37c1159907b3b58980261126ea2bb281ac3c0a0234` | `9f941f4107581ac9d1c8eae535f0769219b906a224ac131de4a3a399fa4eaf7f` |
| `examples/notebooks/02_multiband_hyperspectral_synthetic.ipynb` | `notebooks/02_multiband_hyperspectral_synthetic.ipynb` | `5581c9ae8e9c0b7a4d54eef67e061f1f09ba024a25b7d9ae685584cef0830fe0` | `43c724afc726c76a4b6c4f1f407670b988e5a0597e2dca6e4989fbfb84be4e99` |
| `examples/notebooks/03_smacs0723_real_minicutout.ipynb` | `notebooks/03_smacs0723_real_minicutout.ipynb` | `9316bc1cb856d75f1ecf1280c3bbfbb8e82aa1e8036b0e294eb2b4e36c40f86f` | `b2487bfe87a66ef53cc3472f5870257656293e10f67a576afd8ae7d84edef9f2` |
| `scripts/tweakreg_align_exposures.py` | `scripts/tweakreg_align_exposures.py` | `987f235a3bf36e8e5535171c71dc8029c8fcc05ed95a61802c1bb98f9bb139cd` | `1c73a848dcc002d165a9b948dfe08419676a4f48fe9cf0546f0500e8dfc8a8b5` |
| `tests/test_tweakreg_wrapper.py` | `tests/test_tweakreg_wrapper.py` | `51a065b0c556c842c462f879ce6e86aa867da37602b1363887e5ee1f0c0eed98` | `2f772e7ca3213111004b3602206ae0b455079c4024571d08bc127428291155d7` |
