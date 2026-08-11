"""Attach Gaia DR3 astrometry, photometry, RV and multiplicity diagnostics."""
from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import pyvo
from astropy.coordinates import SkyCoord


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "work" / "data" / "processed"
RAW = ROOT / "work" / "data" / "raw"
INFILE = PROCESSED / "gcns_halo_dwarf_candidates.parquet"
RAW_DR3 = RAW / "gcns_candidates_gaia_dr3.parquet"
OUTFILE = PROCESSED / "gaia_dr3_gcns_halo_dwarfs_enriched.parquet"
TAP_URL = "https://gaia.aip.de/tap"

FIELDS = [
    "source_id", "ra", "dec", "parallax", "parallax_error",
    "pmra", "pmra_error", "pmdec", "pmdec_error",
    "radial_velocity", "radial_velocity_error", "rv_nb_transits",
    "rv_chisq_pvalue", "rv_renormalised_gof",
    "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
    "ruwe", "visibility_periods_used", "astrometric_params_solved",
    "ipd_frac_multi_peak", "ipd_gof_harmonic_amplitude",
    "phot_bp_rp_excess_factor", "duplicated_source", "non_single_star",
    "phot_variable_flag",
]


def fetch(source_ids: list[int]) -> pd.DataFrame:
    if RAW_DR3.exists():
        return pd.read_parquet(RAW_DR3)
    service = pyvo.dal.TAPService(TAP_URL)
    parts = []
    for i in range(0, len(source_ids), 75):
        ids = source_ids[i:i + 75]
        id_list = ",".join(str(int(x)) for x in ids)
        query = (
            "SELECT " + ",".join(FIELDS)
            + " FROM gaiadr3.gaia_source WHERE source_id IN (" + id_list + ")"
        )
        tab = service.search(query).to_table().to_pandas()
        print(f"Gaia DR3 chunk {i//75+1}: {len(tab)} rows", flush=True)
        parts.append(tab)
    out = pd.concat(parts, ignore_index=True)
    out["source_id"] = pd.to_numeric(out["source_id"], errors="raise").astype("int64")
    out.to_parquet(RAW_DR3, index=False)
    return out


def main() -> int:
    gcns = pd.read_parquet(INFILE)
    gcns["source_id"] = gcns["source_id"].astype("int64")
    dr3 = fetch(gcns["source_id"].tolist())
    # Prefix old GCNS columns only where a DR3 homonym exists.
    overlap = (set(gcns.columns) & set(dr3.columns)) - {"source_id"}
    gcns = gcns.rename(columns={c: f"gcns_{c}" for c in overlap})
    df = gcns.merge(dr3, on="source_id", how="left", validate="one_to_one")

    df["bp_rp_dr3"] = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]
    df["distance_pc_dr3"] = 1000.0 / df["parallax"]
    df["abs_g_dr3"] = df["phot_g_mean_mag"] + 5 * np.log10(df["parallax"]) - 10
    valid6d = (
        df[["ra", "dec", "parallax", "pmra", "pmdec", "radial_velocity"]]
        .notna().all(axis=1) & (df["parallax"] > 0)
    )
    df["u_lsr_dr3"] = np.nan
    df["v_lsr_component_dr3"] = np.nan
    df["w_lsr_dr3"] = np.nan
    if valid6d.any():
        v = df.loc[valid6d]
        c = SkyCoord(
            ra=v["ra"].to_numpy() * u.deg,
            dec=v["dec"].to_numpy() * u.deg,
            distance=v["distance_pc_dr3"].to_numpy() * u.pc,
            pm_ra_cosdec=v["pmra"].to_numpy() * u.mas / u.yr,
            pm_dec=v["pmdec"].to_numpy() * u.mas / u.yr,
            radial_velocity=v["radial_velocity"].to_numpy() * u.km / u.s,
        ).galactic
        df.loc[valid6d, "u_lsr_dr3"] = c.velocity.d_x.to_value(u.km/u.s) + 11.1
        df.loc[valid6d, "v_lsr_component_dr3"] = c.velocity.d_y.to_value(u.km/u.s) + 12.24
        df.loc[valid6d, "w_lsr_dr3"] = c.velocity.d_z.to_value(u.km/u.s) + 7.25
    df["v_lsr_dr3"] = np.sqrt(
        df["u_lsr_dr3"]**2 + df["v_lsr_component_dr3"]**2 + df["w_lsr_dr3"]**2
    )
    df["toomre_perp_dr3"] = np.hypot(df["u_lsr_dr3"], df["w_lsr_dr3"])
    df["rv_variable_gaia"] = (
        (df["rv_nb_transits"] >= 10)
        & (df["rv_chisq_pvalue"] < 0.01)
        & (df["rv_renormalised_gof"] > 4)
    )
    df["nss_gaia"] = df["non_single_star"].fillna(0).astype(float) > 0
    df["multiplicity_indicator"] = (
        df["nss_gaia"] | df["rv_variable_gaia"] | (df["ruwe"] > 1.4)
    )
    df["primary_dr3"] = (
        valid6d
        & (df["parallax"] / df["parallax_error"] >= 10)
        & (df["radial_velocity_error"] < 10)
        & (df["v_lsr_dr3"] >= 180)
        & (df["ruwe"] < 2.0)
    )
    df.to_parquet(OUTFILE, index=False)
    csv = df.copy()
    csv["source_id"] = csv["source_id"].astype(str)
    csv.to_csv(PROCESSED / "gaia_dr3_gcns_halo_dwarfs_enriched.csv", index=False)
    print(f"GCNS candidates: {len(df)}")
    print(f"Resolved in DR3: {int(valid6d.sum())}")
    print(f"DR3 v_LSR >= 180 km/s: {int(df['primary_dr3'].sum())}")
    print(f"Gaia multiplicity indicators: {int(df.loc[df['primary_dr3'], 'multiplicity_indicator'].sum())}")
    print(f"Gaia RV-variable flags: {int(df.loc[df['primary_dr3'], 'rv_variable_gaia'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
