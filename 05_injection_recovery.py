"""Empirical sinusoid injection/recovery on representative TESS light curves."""
from __future__ import annotations

from pathlib import Path

import lightkurve as lk
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import detrend
from scipy.stats import median_abs_deviation


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work" / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
PERIODS = np.array([0.25, 0.7, 1.3, 2.3, 5.0])
AMPS = np.array([0.00001, 0.00003, 0.0001, 0.0003, 0.001])
N_PHASE = 5
SEED = 20260811


def load_clean(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lc = lk.read(path)
    t = np.asarray(getattr(lc.time, "value", lc.time), float)
    y = np.asarray(getattr(lc.flux, "value", lc.flux), float)
    good = np.isfinite(t) & np.isfinite(y)
    if "quality" in lc.colnames:
        good &= np.asarray(lc["quality"]) == 0
    t, y = t[good], y[good]
    order = np.argsort(t); t, y = t[order], y[order]
    y = y / np.nanmedian(y) - 1
    y = detrend(y)
    scale = 1.4826 * median_abs_deviation(y, nan_policy="omit")
    keep = np.abs(y - np.nanmedian(y)) < 6 * scale
    return t[keep], y[keep]


def matched(found: float, injected: float) -> bool:
    return min(abs(found / injected - r) for r in (0.5, 1.0, 2.0)) <= 0.10


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    sector = pd.read_csv(DATA / "tess_sector_periods.csv", dtype={"source_id": "string"})
    targets = pd.read_parquet(DATA / "tess_target_periods_enriched.parquet")
    targets["source_id"] = targets.source_id.astype(str)
    ok = sector[sector.status.eq("ok")].sort_values("n_cadences", ascending=False).drop_duplicates("source_id")
    ok = ok.merge(targets[["source_id", "phot_g_mean_mag"]], on="source_id", how="left").sort_values("phot_g_mean_mag")
    # Span the brightness range with deterministic quantile representatives.
    take = np.unique(np.linspace(0, len(ok)-1, min(8, len(ok))).round().astype(int))
    reps = ok.iloc[take]
    rng = np.random.default_rng(SEED)
    rows = []
    for _, rec in reps.iterrows():
        t, noise = load_clean(ROOT / rec.local_path)
        baseline = t.max() - t.min()
        cadence = np.nanmedian(np.diff(t))
        min_period, max_period = max(.1, 3*cadence), min(13, baseline/3)
        for period in PERIODS[(PERIODS >= min_period) & (PERIODS <= max_period)]:
            for amp in AMPS:
                for trial in range(N_PHASE):
                    phase = rng.uniform(0, 2*np.pi)
                    # Circularly shift the real residuals to retain red-noise structure.
                    shifted = np.roll(noise, int(rng.integers(0, len(noise))))
                    injected = shifted + amp * np.sin(2*np.pi*t/period + phase)
                    ls = LombScargle(t, injected, normalization="standard")
                    freq, power = ls.autopower(minimum_frequency=1/max_period,
                                               maximum_frequency=1/min_period,
                                               samples_per_peak=8)
                    j = int(np.argmax(power)); found = float(1/freq[j])
                    try:
                        fap = float(ls.false_alarm_probability(power[j], method="baluev"))
                    except Exception:
                        fap = float(ls.false_alarm_probability(power[j], method="naive"))
                    rows.append({"source_id": rec.source_id, "g_mag": rec.phot_g_mean_mag,
                                 "injected_period_days": period, "semi_amplitude_frac": amp,
                                 "trial": trial, "recovered_period_days": found, "fap": fap,
                                 "recovered": bool(fap < 1e-3 and matched(found, period))})
        print(f"completed injection grid for {rec.source_id}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(TAB / "injection_recovery_trials.csv", index=False)
    grid = result.groupby(["semi_amplitude_frac", "injected_period_days"]).recovered.mean().unstack()
    grid.to_csv(TAB / "injection_recovery_matrix.csv")

    mpl.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True})
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(grid.to_numpy(), origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(grid.columns)), [f"{x:g}" for x in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f"{100*x:g}" for x in grid.index])
    ax.set(xlabel="Injected period (d)", ylabel="Injected semi-amplitude (%)")
    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            val = grid.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < .55 else "black", fontsize=9)
    cb = fig.colorbar(im, ax=ax, pad=.02); cb.set_label("Recovery fraction")
    fig.tight_layout()
    fig.savefig(FIG / "fig06_injection_recovery.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig06_injection_recovery.pdf", bbox_inches="tight")
    print(grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
