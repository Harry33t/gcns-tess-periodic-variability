"""Measure per-sector and consensus TESS periods for the candidate sample."""
from __future__ import annotations

from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import detrend
from scipy.stats import median_abs_deviation


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "work" / "data" / "interim" / "tess_download_manifest.csv"
GAIA = ROOT / "work" / "data" / "processed" / "gaia_dr3_gcns_halo_dwarfs_enriched.parquet"
OUTDIR = ROOT / "work" / "data" / "processed"


def as_float_array(column) -> np.ndarray:
    values = getattr(column, "value", column)
    return np.asarray(values, dtype=float)


def analyze_one(path: Path) -> dict:
    lc = lk.read(path)
    crowdsap = float(lc.meta.get("CROWDSAP", np.nan))
    t = as_float_array(lc.time)
    y = as_float_array(lc.flux)
    ye = as_float_array(lc.flux_err) if "flux_err" in lc.colnames else np.full_like(y, np.nan)
    good = np.isfinite(t) & np.isfinite(y)
    if "quality" in lc.colnames:
        q = np.asarray(lc["quality"])
        good &= (q == 0)
    t, y, ye = t[good], y[good], ye[good]
    order = np.argsort(t)
    t, y, ye = t[order], y[order], ye[order]
    if len(t) < 500:
        return {"status": "too_few_cadences", "n_cadences": len(t)}
    baseline = float(t.max() - t.min())
    cadence = float(np.nanmedian(np.diff(t)))
    if baseline < 10:
        return {"status": "baseline_too_short", "n_cadences": len(t), "baseline_days": baseline}

    med = np.nanmedian(y)
    if not np.isfinite(med) or med == 0:
        return {"status": "invalid_flux", "n_cadences": len(t), "baseline_days": baseline}
    y = y / med - 1.0
    # Iterative robust clipping before the period search. Eclipses/outbursts are
    # counted separately through the clipped fraction rather than driving the
    # sinusoidal periodogram.
    resid0 = detrend(y, type="linear")
    scale0 = 1.4826 * median_abs_deviation(resid0, nan_policy="omit")
    keep = np.isfinite(resid0)
    if np.isfinite(scale0) and scale0 > 0:
        keep &= np.abs(resid0 - np.nanmedian(resid0)) < 6 * scale0
    clipped_fraction = 1.0 - float(keep.mean())
    t2, y2 = t[keep], y[keep]
    y2 = detrend(y2, type="linear")

    min_period = max(0.10, 3.0 * cadence)
    max_period = min(13.0, baseline / 3.0)
    if max_period <= min_period:
        return {"status": "invalid_period_range", "n_cadences": len(t), "baseline_days": baseline}
    ls = LombScargle(t2, y2, normalization="standard")
    freq, power = ls.autopower(
        minimum_frequency=1.0 / max_period,
        maximum_frequency=1.0 / min_period,
        samples_per_peak=12,
    )
    j = int(np.nanargmax(power))
    best_frequency = float(freq[j])
    best_period = 1.0 / best_frequency
    best_power = float(power[j])
    try:
        fap = float(ls.false_alarm_probability(best_power, method="baluev"))
    except Exception:
        fap = float(ls.false_alarm_probability(best_power, method="naive"))
    model = ls.model(t2, best_frequency)
    semi_amplitude = 0.5 * (np.nanpercentile(model, 95) - np.nanpercentile(model, 5))
    residual = y2 - model
    noise = 1.4826 * median_abs_deviation(residual, nan_policy="omit")
    # Coherent sinusoid S/N.  The semi-amplitude can be below the point-to-
    # point scatter yet highly significant after thousands of cadences.
    amplitude_snr = (
        float(semi_amplitude * np.sqrt(len(t2) / 2.0) / noise)
        if np.isfinite(noise) and noise > 0 else np.nan
    )
    cycles = baseline / best_period
    tess_orbit_aliases = np.array([13.7 / n for n in range(1, 6)])
    tess_alias = np.any(
        np.abs(best_period - tess_orbit_aliases)
        < np.maximum(0.04, 0.025 * tess_orbit_aliases)
    )
    alias_flag = bool(
        abs(best_period - 0.5) < 0.02
        or abs(best_period - 1.0) < 0.03
        or abs(best_period - 2.0) < 0.05
        or tess_alias
    )
    boundary_flag = bool(best_period <= 1.02 * min_period or best_period >= 0.98 * max_period)
    significant = bool(
        fap < 1e-3 and cycles >= 3 and amplitude_snr >= 3
        and not boundary_flag and not alias_flag
    )
    return {
        "status": "ok",
        "n_cadences": len(t),
        "baseline_days": baseline,
        "cadence_days": cadence,
        "best_period_days": best_period,
        "best_power": best_power,
        "fap": fap,
        "semi_amplitude_frac": float(semi_amplitude),
        "amplitude_snr": amplitude_snr,
        "cycles": cycles,
        "clipped_fraction": clipped_fraction,
        "crowdsap": crowdsap,
        "alias_flag": alias_flag,
        "boundary_flag": boundary_flag,
        "sector_significant": significant,
    }


def harmonic_distance(p: float, base: float) -> tuple[float, float]:
    options = np.array([p / 2.0, p, p * 2.0])
    rel = np.abs(options - base) / base
    j = int(np.argmin(rel))
    return float(rel[j]), float(options[j])


def target_consensus(group: pd.DataFrame) -> pd.Series:
    ok = group[(group["status"] == "ok") & group["sector_significant"]].copy()
    n_products = len(group)
    crowd = float(pd.to_numeric(group.get("crowdsap"), errors="coerce").median())
    if ok.empty:
        return pd.Series({
            "n_tess_products": n_products, "n_significant_sectors": 0,
            "period_days": np.nan, "period_support": 0,
            "period_support_fraction": 0.0,
            "period_scatter_frac": np.nan, "period_class": "no_detection",
            "semi_amplitude_pct": np.nan, "best_fap": np.nan,
            "max_amplitude_snr": np.nan,
            "median_crowdsap": crowd,
        })
    periods = ok["best_period_days"].to_numpy(float)
    if len(periods) == 1:
        strongest = ok.iloc[0]
        cls = (
            "strong_single_sector"
            if n_products == 1
            and strongest["fap"] < 1e-8
            and strongest["amplitude_snr"] >= 6
            and not strongest["alias_flag"]
            else "candidate_single_sector"
        )
        return pd.Series({
            "n_tess_products": n_products,
            "n_significant_sectors": 1,
            "period_days": float(periods[0]),
            "period_support": 1,
            "period_support_fraction": 1.0,
            "period_scatter_frac": np.nan,
            "period_class": cls,
            "semi_amplitude_pct": float(100 * strongest["semi_amplitude_frac"]),
            "best_fap": float(strongest["fap"]),
            "max_amplitude_snr": float(strongest["amplitude_snr"]),
            "median_crowdsap": crowd,
        })
    candidate_bases = np.unique(periods)
    best = None
    for base in candidate_bases:
        mapped = np.array([harmonic_distance(p, base)[1] for p in periods])
        rel = np.abs(mapped - base) / base
        support = int((rel <= 0.10).sum())
        # Prefer the longer fundamental when P and P/2 are equally supported;
        # spot modulation often has two similar minima per rotation.
        score = (support, -float(np.nanmedian(rel)), float(base))
        if best is None or score > best[0]:
            best = (score, base, mapped, rel)
    _, base, mapped, rel = best
    supported = mapped[rel <= 0.10]
    consensus = float(np.nanmedian(supported))
    scatter = float(np.nanstd(supported) / consensus) if len(supported) > 1 else np.nan
    support = len(supported)
    strongest = ok.sort_values("fap").iloc[0]
    if support >= 2:
        cls = "multi_sector"
    else:
        cls = "inconsistent_multi_sector"
    return pd.Series({
        "n_tess_products": n_products,
        "n_significant_sectors": len(ok),
        "period_days": consensus,
        "period_support": support,
        "period_support_fraction": float(support / len(ok)),
        "period_scatter_frac": scatter,
        "period_class": cls,
        "semi_amplitude_pct": float(100 * ok["semi_amplitude_frac"].median()),
        "best_fap": float(ok["fap"].min()),
        "max_amplitude_snr": float(ok["amplitude_snr"].max()),
        "median_crowdsap": crowd,
    })


def main() -> int:
    manifest = pd.read_csv(MANIFEST, dtype={"source_id": "string"})
    products = manifest[
        manifest["local_path"].notna()
        & (pd.to_numeric(manifest["distance_arcsec"], errors="coerce") <= 3.0)
    ].copy()
    rows = []
    for _, row in products.iterrows():
        path = ROOT / str(row["local_path"])
        base = row.to_dict()
        try:
            base.update(analyze_one(path))
        except Exception as exc:
            base.update({"status": "read_error", "error": repr(exc)})
        rows.append(base)
        print(f"{row['source_id']} sector={row['sector']} -> {base['status']}", flush=True)
    sector = pd.DataFrame(rows)
    sector.to_csv(OUTDIR / "tess_sector_periods.csv", index=False)
    target = sector.groupby("source_id", sort=False).apply(target_consensus, include_groups=False).reset_index()
    gaia = pd.read_parquet(GAIA)
    gaia["source_id"] = gaia["source_id"].astype(str)
    target = gaia.merge(target, on="source_id", how="left")
    target["has_tess_product"] = target["n_tess_products"].fillna(0) > 0
    target["robust_period"] = target["period_class"].isin(["multi_sector", "strong_single_sector"])
    target["fast_rotator_candidate"] = target["robust_period"] & (target["period_days"] < 2.0)
    target.to_parquet(OUTDIR / "tess_target_periods_enriched.parquet", index=False)
    csv = target.copy()
    csv["source_id"] = csv["source_id"].astype(str)
    csv.to_csv(OUTDIR / "tess_target_periods_enriched.csv", index=False)
    print(f"Products analyzed: {len(sector)}")
    print(f"Targets with products: {target['has_tess_product'].sum()}")
    print(f"Robust periods: {target['robust_period'].sum()}")
    print(f"Fast-rotator candidates: {target['fast_rotator_candidate'].sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
