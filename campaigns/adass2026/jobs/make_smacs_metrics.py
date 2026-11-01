"""Build deck/assets/smacs_metrics.json from the v7/v8 runs (cross-validation protocol: validation exposures
for early stopping, separate test exposures for every reported number)."""
import json
from pathlib import Path

ROOT = Path("/home/claude/adass")
V7, V8 = ROOT / "v7", ROOT / "v8"


def load(d):
    return json.load(open(d / "metrics.json"))


RUNS = {
    "e04": load(V7 / "v7_e04_huber"), "e04g": load(V7 / "v7_e04_gauss"), "i04": load(V7 / "v7_i04_huber"),
    "s04": load(V7 / "v7_s04_siren_huber"), "e02": load(V7 / "v7_e02_huber_tv010"), "e02b": load(V7 / "v7_e02_huber_tv002"),
    "i02": load(V7 / "v7_i02_huber"), "e3d": load(V8 / "v8_e3d_04"), "n3d": load(V8 / "v8_n3d_04"),
    "e3dfix": load(V8 / "v8_e3d_04_fixedwcs"), "e3d02": load(V8 / "v8_e3d_02"),
}
for extra, d in (("e3dk12", ROOT / "v10" / "v10_e3d_04_k12"), ("n3dfix", ROOT / "v10" / "v10_n3d_04_fixedbasis"),
                 ("e3dnos", ROOT / "v10" / "v10_e3d_04_nosmooth"), ("n3dnos", ROOT / "v10" / "v10_n3d_04_fixedbasis_nosmooth"),
                 ("e3drel", ROOT / "v11" / "v11_e3d_04_relsmooth"), ("n3drel", ROOT / "v11" / "v11_n3d_04_fixedbasis_relsmooth"),
                 ("e3dk12rel", ROOT / "v12" / "v12_e3d_04_k12_rel")):
    if (d / "metrics.json").exists():
        RUNS[extra] = load(d)


RESCORE = {"e04": "rescore14b_v7_e04_huber", "e04g": "rescore14_v7_e04_gauss", "i04": "rescore14b_v7_i04_huber",
           "s04": "rescore14b_v7_s04_siren_huber", "e3drel": "rescore14_v11_e3d_04_relsmooth",
           "n3drel": "rescore14b_v11_n3d_04_fixedbasis_relsmooth", "e3dk12rel": "rescore14_v12_e3d_04_k12_rel",
           "e02": "rescore14_v7_e02_huber_tv010"}
PH = {}
for k, d in RESCORE.items():
    f = ROOT / "rescore14" / d / "metrics.json"
    if f.exists():
        PH[k] = json.load(open(f))["metrics"].get("heldout_photometry", {})


def phot(run, key):
    """Aperture photometry on the test exposures (0.2" radius, sources S/N > 20): NMAD of (data - prediction)/sigma,
    median and NMAD of the magnitude difference (mmag), centroid NMAD (mas). None when not re-scored."""
    a = PH.get(run, {}).get(key, {}).get("all")
    if not a:
        return None
    return dict(pull=a["pull_nmad"], dm=a["dm_median_mmag"], scat=a["dm_nmad_mmag"], cen=a["centroid_nmad_mas"], n=a["n"])


def test(run, key):
    m = RUNS[run]["metrics"]
    return (m["heldout_chi2"][key], m["heldout_chi2_sources"][key], m["heldout_robust"][key], m["heldout_robust_sources"][key])


methods = []
for key, run, label, cube, best in [("swarp", "e04", "SWarp-like", "swarp", False),
                                    ("drizzle", "e04", "Drizzle-like", "drizzle", False),
                                    ("explicit_chrom", "e04", "Explicit model", "explicit", True),
                                    ("implicit_chrom", "i04", "Implicit neural field", "neural", False)]:
    a, s, ra, rs = test(run, key)
    methods.append(dict(key=key, label=label, cube=cube, resid=key, cv=a, cvs=s, rob=ra, robs=rs, best=best, phot=phot(run, key)))

E3 = "e3drel"  # slides: K=8 relative prior (smooth spectra); K=12 relative (v12) fits sources slightly better but its compact-source spectra wiggle (backup table)
rows = [("SWarp-like", "e04", "swarp"), ("Drizzle-like", "e04", "drizzle"),
        ("Explicit, Gaussian loss", "e04g", "explicit_chrom"), ("Explicit, Huber loss", "e04", "explicit_chrom"),
        ("Explicit, one field s(α,δ,λ)", E3, "explicit_chrom_3d"), ("Implicit neural field", "i04", "implicit_chrom"),
        ("Implicit neural field, s(α,δ,λ)", "n3drel", "implicit_chrom_3d"), ("SIREN, Huber", "s04", "implicit_siren_chrom")]
chart = dict(labels=[r[0] for r in rows], all=[round(test(r[1], r[2])[0], 2) for r in rows],
             src=[round(test(r[1], r[2])[1], 2) for r in rows],
             pull=[round((phot(r[1], r[2]) or {"pull": 0.0})["pull"], 2) for r in rows])

variants = []
for lab, run, key, grid in [("Drizzle-like", "e04", "drizzle", "0.04″"),
                            ("Explicit, Huber (slides)", "e04", "explicit_chrom", "0.04″"),
                            ("Explicit, Gaussian loss", "e04g", "explicit_chrom", "0.04″"),
                            ("Explicit, s(α,δ,λ), K=8, absolute spectral prior", "e3d", "explicit_chrom_3d", "0.04″"),
                            ("Explicit, s(α,δ,λ), K=8, no spectral prior", "e3dnos", "explicit_chrom_3d", "0.04″"),
                            ("Explicit, s(α,δ,λ), K=8, relative prior (slides)", "e3drel", "explicit_chrom_3d", "0.04″"),
                            ("Explicit, s(α,δ,λ), K=12, weak prior", "e3dk12", "explicit_chrom_3d", "0.04″"),
                            ("Explicit, s(α,δ,λ), K=12, relative prior", "e3dk12rel", "explicit_chrom_3d", "0.04″"),
                            ("Explicit, s(x,y,λ), WCS only (no learned geometry)", "e3dfix", "explicit_fixedgeom_chrom_3d", "0.04″"),
                            ("Implicit neural field, Huber", "i04", "implicit_chrom", "0.04″"),
                            ("Implicit field, s(α,δ,λ), learned basis", "n3d", "implicit_chrom_3d", "0.04″"),
                            ("Implicit field, s(α,δ,λ), fixed basis, relative prior", "n3drel", "implicit_chrom_3d", "0.04″"),
                            ("SIREN, Huber", "s04", "implicit_siren_chrom", "0.04″"),
                            ("Drizzle-like", "e02", "drizzle", "0.02″"),
                            ("Explicit, Huber, TV 0.10", "e02", "explicit_chrom", "0.02″"),
                            ("Explicit, Huber, TV 0.02", "e02b", "explicit_chrom", "0.02″"),
                            ("Explicit, s(α,δ,λ), absolute prior", "e3d02", "explicit_chrom_3d", "0.02″"),
                            ("Implicit neural field, Huber", "i02", "implicit_chrom", "0.02″")]:
    if run not in RUNS:
        continue
    a, s, ra, rs = test(run, key)
    variants.append(dict(label=lab, grid=grid, cv=a, cvs=s, rob=ra, robs=rs, phot=phot(run, key)))

c = {m["key"]: m for m in methods}
e3 = test(E3, "explicit_chrom_3d")
n3 = test("n3drel", "implicit_chrom_3d")
bd = test("e04", "explicit_chrom")
dz = test("e04", "drizzle")
pb3 = RUNS[E3]["metrics"]["heldout_per_band"]["explicit_chrom_3d"]
pbb = RUNS["e04"]["metrics"]["heldout_per_band"]["explicit_chrom"]
out = dict(
    methods=methods, chart=chart, variants=variants,
    spectral=dict(e3d=e3, n3d=n3, band=bd, drizzle=dz, run=E3, phot=dict(e3d=phot(E3, "explicit_chrom_3d"), band=phot("e04", "explicit_chrom"), drizzle=phot("e04", "drizzle"))),
    variants_caption=("⟨r²/σ²⟩ on test exposures (one per band, never used for fitting or stopping); validation exposures (one per band) "
                      "chose the stopping point. S/N > 5: pixels more than 5σ above sky in the SWarp-like coadd. Robust: pseudo-Huber at 3σ. Flux error: NMAD of (data − prediction)/σ in 0.2″ apertures; centroid: NMAD in mas. Each grid has its own coadds under the same protocol."),
    caption=("SMACS J0723, NIRCam F090W–F444W, 0.04″ grid, torchfits RGB (top: giant arc; bottom: faint lensed sources across a detector seam). "
             "On exposures never used in the fit: ⟨r²/σ²⟩ (1 = noise only) over all pixels and pixels with S/N > 5; flux error: NMAD of (data − prediction)/σ in 0.2″ apertures."),
    notes=("~80 s. The same JWST exposures of SMACS J0723, six bands, the same grid, four estimators, all implemented in torchsky: a SWarp-like and a Drizzle-like coadd, "
           "and two forward models fitted directly to the detector pixels: an explicit pixel-grid sky, and an implicit neural field. "
           "Both share one PSF field across all six bands, smooth in wavelength and detector position, and use a robust Huber loss. "
           "Top row the giant arc, bottom row faint lensed sources in the cluster light, rendered with torchfits RGB at the same stretch. Notice the detector seam, blue against orange, in both coadds: it disappears when the background is part of the model. "
           "How do we score an inverse problem without ground truth? We leave exposures out: one per band to decide when to stop, and a different one per band that we only use to report. "
           "The number is the mean of residual squared over sigma squared on those unseen exposures; it is not a reduced chi-square per degree of freedom, nothing was fitted to those pixels except the same few nuisance terms for every method, registration, gain, background and stripes. One would be pure noise. "
           f"Both forward models predict the unseen exposures better than the coadds: on pixels with S/N above 5 {c['explicit_chrom']['cvs']:.2f} and {c['implicit_chrom']['cvs']:.2f} "
           f"against {c['swarp']['cvs']:.2f} SWarp-like and {c['drizzle']['cvs']:.2f} Drizzle-like; over all pixels {c['explicit_chrom']['cv']:.2f} against {c['drizzle']['cv']:.2f}. "
           "S/N above 5 means pixels more than 5 sigma above the sky in the SWarp-like coadd, the same mask for everybody. "
           + (f"In photometric terms: the flux in 0.2 arcsec apertures on the unseen exposures, about 300 source measurements, is predicted to within {c['explicit_chrom']['phot']['pull']:.1f} sigma (NMAD) by the explicit model, "
              f"{c['drizzle']['phot']['pull']:.1f} sigma by Drizzle-like and {c['swarp']['phot']['pull']:.1f} by SWarp-like; one would be photon noise. Centroids are predicted to about 2 mas by everybody. "
              if c['explicit_chrom'].get('phot') and c['drizzle'].get('phot') and c['swarp'].get('phot') else "") +
           "Caveats if asked: bright saturated stars are still the weak point of every forward model (square PSF truncation, ringing); the floor above one is model error common to all methods, "
           "dominated by F277W at about 15 for every method; the ERR arrays are right per pixel (first-difference test)."),
    learn_notes=("~50 s. All variants under the same test protocol, coadds at the top, lower is better. "
                 "Grey: all pixels, orange: pixels with S/N above 5, blue: flux error of 0.2 arcsec aperture photometry in units of the noise. The robust loss mainly helps on the bright pixels; the one-field models s(alpha, delta, lambda) tie one image per band over all pixels and beat the coadds on bright pixels, but not yet one image per band there; I come back to that. "
                 "On the right, what the explicit model learns besides the sky: one PSF network evaluated from 0.9 to 4.4 microns, broadening with wavelength as diffraction says; "
                 "a static distortion per detector of a few milliarcseconds on top of the WCS, plus per-exposure registration at about 4 mas; backgrounds, 1/f stripes and gains."),
    spectral_notes=("~60 s. So far each band had its own image. Here the sky is one continuous function of position on the sky, alpha and delta in the calibrated frame, and of wavelength: "
                    f"{RUNS[E3]['args']['spec_k']} smooth basis functions in log lambda. Each exposure sees it through its own filter: the SVO transmission curve, sampled at six photon-weighted wavelengths, each convolved with the PSF at that wavelength. "
                    "On the left, the same lensed arc rendered at five wavelengths from one model. On the right, spectra read off the model: the brightest cluster galaxy peaks near 2 microns, "
                    "the old-star bump, while the arc is redder and flat. Dots are the model's band averages, circles the drizzled values; for compact sources they differ because drizzle keeps the PSF. "
                    "The spectral prior penalises curvature relative to the local brightness, so bright and faint sources are smoothed alike. "
                    f"In prediction of unseen exposures it ties one image per band over all pixels, {e3[0]:.2f} against {bd[0]:.2f}, and beats both coadds on sources, {e3[1]:.2f} against {dz[1]:.2f} Drizzle-like; "
                    f"one image per band is still better on sources, {bd[1]:.2f}, and the gap sits in F277W and F356W ({pb3['F277W']['chi2_src']:.1f} and {pb3['F356W']['chi2_src']:.1f} against {pbb['F277W']['chi2_src']:.1f} and {pbb['F356W']['chi2_src']:.1f}), "
                    "the long-wavelength bands with nine exposures each. This is the representation that lets radio, optical and X-ray data, spectra and images, constrain the same sky."),
)
dst = ROOT / "deck" / "assets" / "smacs_metrics.json"
json.dump(out, open(dst, "w"), indent=1)
print("wrote", dst)
for m in methods:
    print(m["label"], round(m["cv"], 2), round(m["cvs"], 2))
print(chart)
