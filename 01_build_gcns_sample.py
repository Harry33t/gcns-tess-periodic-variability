"""Build a nearby high-velocity dwarf sample from the official GCNS table.

The catalogue supplies posterior distances and Galactic velocities for an
independently defined nearby-star parent sample. Gaia DR3 diagnostics are
attached by source_id in the next stage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.vizier import Vizier


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "work" / "data" / "raw"
PROCESSED = ROOT / "work" / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

RAW_FILE = RAW / "gcns_table1c_selected.parquet"
CATALOG = "J/A+A/649/A6/table1c"
COLUMNS = [
    "GaiaEDR3", "RA_ICRS", "DE_ICRS", "Plx", "e_Plx",
    "pmRA", "e_pmRA", "pmDE", "e_pmDE", "Gmag", "BPmag", "RPmag",
    "RUWE", "IPDfmp", "RV", "e_RV", "GCNSprob", "WDprob", "Dist50",
    "Uvel50", "Vvel50", "Wvel50",
]


def fetch() -> pd.DataFrame:
    if RAW_FILE.exists():
        return pd.read_parquet(RAW_FILE)
    viz = Vizier(columns=COLUMNS, row_limit=-1, timeout=600)
    tables = viz.get_catalogs(CATALOG)
    table = tables[0]
    df = table.to_pandas()
    df.to_parquet(RAW_FILE, index=False)
    return df


def main() -> int:
    df = fetch().copy()
    rename = {
        "GaiaEDR3": "source_id", "RA_ICRS": "ra", "DE_ICRS": "dec",
        "Plx": "parallax", "e_Plx": "parallax_error",
        "pmRA": "pmra", "e_pmRA": "pmra_error",
        "pmDE": "pmdec", "e_pmDE": "pmdec_error",
        "Gmag": "phot_g_mean_mag", "BPmag": "phot_bp_mean_mag",
        "RPmag": "phot_rp_mean_mag", "RUWE": "ruwe",
        "IPDfmp": "ipd_frac_multi_peak", "RV": "radial_velocity",
        "e_RV": "radial_velocity_error", "GCNSprob": "gcns_prob",
        "WDprob": "wd_prob", "Dist50": "distance_kpc_gcns",
        "Uvel50": "u_gcns", "Vvel50": "v_gcns", "Wvel50": "w_gcns",
    }
    df = df.rename(columns=rename)
    df["source_id"] = pd.to_numeric(df["source_id"], errors="raise").astype("int64")
    df["bp_rp"] = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]
    df["abs_g"] = df["phot_g_mean_mag"] - 5 * np.log10(
        df["distance_kpc_gcns"] * 1000.0
    ) + 5
    df["v_gcns_total"] = np.sqrt(
        df["u_gcns"] ** 2 + df["v_gcns"] ** 2 + df["w_gcns"] ** 2
    )
    df["q_basic"] = (
        (df["gcns_prob"] >= 0.9)
        & (df["wd_prob"].fillna(0) < 0.5)
        & df["radial_velocity"].notna()
        & (df["radial_velocity_error"] < 10)
        & df["phot_g_mean_mag"].between(6, 14)
        & df["bp_rp"].between(0.5, 1.8)
    )
    lower = 2.2 + 2.0 * df["bp_rp"]
    upper = 4.8 + 3.6 * df["bp_rp"]
    df["q_dwarf_broad"] = (df["abs_g"] > lower) & (df["abs_g"] < upper)
    df["q_astrometry_inclusive"] = (df["ruwe"] < 2.0) & (df["ipd_frac_multi_peak"] <= 20)
    df["q_astrometry_strict"] = df["q_astrometry_inclusive"] & (df["ruwe"] < 1.4)
    df["halo_candidate_gcns"] = df["v_gcns_total"] >= 180.0
    df["primary_sample"] = (
        df["q_basic"]
        & df["q_dwarf_broad"]
        & df["q_astrometry_inclusive"]
        & df["halo_candidate_gcns"]
    )

    df.to_parquet(PROCESSED / "gcns_derived_all.parquet", index=False)
    primary = df.loc[df["primary_sample"]].copy()
    primary = primary.sort_values(["phot_g_mean_mag", "v_gcns_total"], ascending=[True, False])
    primary.to_parquet(PROCESSED / "gcns_halo_dwarf_candidates.parquet", index=False)
    csv = primary.copy()
    csv["source_id"] = csv["source_id"].astype(str)
    csv.to_csv(PROCESSED / "gcns_halo_dwarf_candidates.csv", index=False)

    funnel = pd.DataFrame(
        [
            ("GCNS reliable catalogue", len(df)),
            ("Photometry/RV/basic quality", int(df["q_basic"].sum())),
            ("Broad dwarf locus", int((df["q_basic"] & df["q_dwarf_broad"]).sum())),
            ("Inclusive astrometry", int((df["q_basic"] & df["q_dwarf_broad"] & df["q_astrometry_inclusive"]).sum())),
            ("GCNS speed >= 180 km/s", len(primary)),
            ("Strict RUWE < 1.4", int(primary["q_astrometry_strict"].sum())),
        ],
        columns=["stage", "n"],
    )
    funnel.to_csv(PROCESSED / "gcns_sample_funnel.csv", index=False)
    print(funnel.to_string(index=False))
    print(primary[["source_id", "phot_g_mean_mag", "bp_rp", "abs_g", "v_gcns_total", "ruwe"]].head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
