# Real-world WCS fixtures

These files come from the Astropy WCS test suite, which uses real instrument
headers and small real images. They are vendored unmodified (452 KB total) so
the harness can compare torchsky against astropy on real-world WCS metadata
without a network dependency.

- Source: <https://github.com/astropy/astropy/tree/main/astropy/wcs/tests/data>
- License: BSD-3-Clause (Astropy); see <https://github.com/astropy/astropy/blob/main/LICENSE.rst>
- Fetched: 2026-10-19 from `main`.

The bounded AstroDrizzle campaign also vendors
`sample_ipppssoot_flt.fits` from
`spacetelescope/drizzlepac@635834581ea2b226a3f3028b6ac54efb5e180dfc`
(`tests/hap/sample_ipppssoot_flt.fits`, blob
`6ed6fb53ce2c8fb3039465a7f22dca7df4a6051e`). It is an unmodified compact
HST/WFC3-IR FLT sample distributed under DrizzlePac's BSD-3-Clause license.

| File | What it is |
|---|---|
| `2wcses.hdr` | Isaac Newton Telescope WFC mosaic header with ZPN projection (2048x4096). |
| `irac_sip.hdr` | Spitzer IRAC 256x256 exposure header with SIP distortion. |
| `tpvonly.hdr` | Palomar Transient Factory exposure header with TPV distortion. |
| `siponly.hdr` | JWST/HST-style TAN-SIP header. |
| `zpn-hole.hdr` | ZPN header whose sky coverage has a hole (partial validity). |
| `j94f05bgq_flt.fits` | Real HST WFPC2 FLT exposure with SCI extension, SIP + distortion tables. |
| `sample_ipppssoot_flt.fits` | Compact real HST WFC3/IR FLT sample for bounded AstroDrizzle execution. |
| `ie6d07ujq_wcs.fits` | Real HST WFC3-style WCS header with D2IM lookup extensions. |
| `dist_lookup.fits.gz` | SIP + CPDIS/D2IM/WCSDVARR lookup-distortion exposure. |
| `dist.fits` | Paper-IV distortion lookup arrays (D2IMARR extension, no celestial WCS). |
| `sip.fits`, `sip2.fits` | SIP distortion headers used by astropy's WCS tests. |
| `header_newlines.fits` | TAN-SIP header with long-string concatenation. |
| `validate.fits` | TPV/SIP headers from the WCS validation suite. |
| `tab-time-last-axis.fits` | 3D (RA, Dec, TIME) header with a real `-TAB` BINTABLE. |
| `dss.14.29.56-62.41.05.fits.gz` | DSS cutout; kept for reference, skipped because astropy rejects its PC matrix. |
| `3d_cd.hdr`, `nonstandard_units.hdr`, `header_with_time.fits` | Header edge cases; skipped until a matching construction path exists. |

The torchsky-harness `bench_wcs_real_headers` benchmark records a row per
fixture and reports `SKIPPED` with a reason when a construction path is not
supported yet.
