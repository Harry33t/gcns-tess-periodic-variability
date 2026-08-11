"""Search and download cached TESS light curves for the Gaia/GCNS sample."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import astropy.units as u
import lightkurve as lk
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations


ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "work" / "data" / "processed" / "gaia_dr3_gcns_halo_dwarfs_enriched.parquet"
RAW_TESS = ROOT / "work" / "data" / "raw" / "tess"
MANIFEST = ROOT / "work" / "data" / "interim" / "tess_download_manifest.csv"
RAW_TESS.mkdir(parents=True, exist_ok=True)
MANIFEST.parent.mkdir(parents=True, exist_ok=True)

AUTHOR_PRIORITY = {
    "SPOC": 0,
    "TESS-SPOC": 1,
    "QLP": 2,
    "TASOC": 3,
    "TARS": 4,
}


def choose_products(table: pd.DataFrame, max_sectors: int) -> pd.DataFrame:
    if table.empty:
        return table
    tab = table.copy()
    tab["author"] = tab["author"].astype(str)
    tab["priority"] = tab["author"].map(AUTHOR_PRIORITY).fillna(99)
    tab["sector"] = pd.to_numeric(tab["sequence_number"], errors="coerce")
    tab["distance_arcsec"] = pd.to_numeric(tab["distance"], errors="coerce")
    # TIC positions are Gaia-tied for these bright nearby stars.  A 3 arcsec
    # product match keeps legitimate epoch differences while rejecting nearby
    # TIC light curves returned by the wider discovery cone.
    tab = tab[tab["distance_arcsec"].fillna(999) <= 3.0]
    tab = tab.sort_values(["sector", "priority", "exptime", "distance_arcsec"])
    tab = tab.drop_duplicates("sector", keep="first")
    # Spread a capped set across the observing baseline rather than selecting
    # only adjacent sectors.
    if len(tab) > max_sectors:
        idx = sorted(set(round(x) for x in pd.Series(range(max_sectors)) * (len(tab)-1) / (max_sectors-1)))
        tab = tab.iloc[idx]
    return tab


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-targets", type=int, default=80)
    parser.add_argument("--max-sectors", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.8)
    args = parser.parse_args()

    sample = pd.read_parquet(INFILE)
    sample = sample.loc[sample["primary_dr3"]].copy()
    sample = sample.sort_values(["phot_g_mean_mag", "v_lsr_dr3"], ascending=[True, False])
    sample = sample.head(args.max_targets)

    old = pd.read_csv(MANIFEST, dtype={"source_id": "string"}) if MANIFEST.exists() else pd.DataFrame()
    if not old.empty:
        dist = pd.to_numeric(old.get("distance_arcsec"), errors="coerce")
        valid_product = old["search_status"].isin(["downloaded", "available"]) & (dist <= 3.0)
        definitive_no_hit = old["search_status"].eq("no_hits")
        completed_ids = set(old.loc[valid_product | definitive_no_hit, "source_id"].astype(str))
    else:
        completed_ids = set()
    rows = old.to_dict("records") if not old.empty else []

    for k, (_, star) in enumerate(sample.iterrows(), start=1):
        sid = str(int(star["source_id"]))
        if sid in completed_ids:
            continue
        coord = SkyCoord(float(star["ra"]) * u.deg, float(star["dec"]) * u.deg)
        try:
            result = lk.search_lightcurve(coord, mission="TESS", radius=20 * u.arcsec)
            tab = result.table.to_pandas() if len(result) else pd.DataFrame()
            selected = choose_products(tab, args.max_sectors)
            print(f"[{k}/{len(sample)}] {sid}: hits={len(tab)}, selected={len(selected)}", flush=True)
            if selected.empty:
                rows.append({
                    "source_id": sid, "search_status": "no_hits", "sector": None,
                    "author": None, "exptime": None, "distance_arcsec": None,
                    "data_uri": None, "local_path": None,
                })
            for _, product in selected.iterrows():
                target_dir = RAW_TESS / sid
                target_dir.mkdir(parents=True, exist_ok=True)
                filename = str(product["productFilename"])
                local_path = target_dir / filename
                status = "available"
                if not local_path.exists():
                    dl = Observations.download_file(
                        str(product["dataURI"]), local_path=str(local_path), cache=True
                    )
                    status = "downloaded" if str(dl).upper() in {"COMPLETE", "LOCAL", "SKIPPED"} or local_path.exists() else str(dl)
                rows.append({
                    "source_id": sid,
                    "search_status": status,
                    "sector": int(product["sector"]) if pd.notna(product["sector"]) else None,
                    "author": str(product["author"]),
                    "exptime": float(product["exptime"]),
                    "distance_arcsec": float(product["distance_arcsec"]),
                    "data_uri": str(product["dataURI"]),
                    "local_path": str(local_path.relative_to(ROOT)),
                })
        except Exception as exc:
            print(f"  ERROR {sid}: {exc}", flush=True)
            rows.append({
                "source_id": sid, "search_status": "error", "sector": None,
                "author": None, "exptime": None, "distance_arcsec": None,
                "data_uri": None, "local_path": None, "error": repr(exc),
            })
        pd.DataFrame(rows).to_csv(MANIFEST, index=False)
        time.sleep(args.sleep)

    final = pd.DataFrame(rows)
    final.to_csv(MANIFEST, index=False)
    usable = final[final["local_path"].notna()] if "local_path" in final else pd.DataFrame()
    print(f"Targets requested: {len(sample)}")
    print(f"Targets with products: {usable['source_id'].nunique() if not usable.empty else 0}")
    print(f"Downloaded/available products: {len(usable)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
