# GCNS--Gaia--TESS periodic-variability pipeline

This repository contains the compact, reproducible core of the analysis used
for *TESS periodic variability of kinematically selected nearby high-velocity
dwarfs from GCNS and Gaia DR3*.

The release intentionally contains only the five scientific pipeline stages:

1. build the GCNS parent sample;
2. attach Gaia DR3 measurements and recompute LSR velocities;
3. retrieve position-matched TESS light curves;
4. search individual sectors and form target-level period-consistency tiers;
5. run the sinusoidal injection--recovery experiment.

Manuscript preparation, cached archive products, downloaded FITS files, and
locally generated figures are not included.

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Reproducing the core analysis

Run the scripts from the repository root in numerical order:

```bash
python 01_build_gcns_sample.py
python 02_enrich_gaia_dr3.py
python 03_download_tess.py --max-targets 100 --max-sectors 4
python 04_search_periods.py
python 05_injection_recovery.py
```

Archive queries and TESS downloads require an internet connection. Intermediate
files are written below `work/data/`; final tabular products are written below
`work/data/processed/` and `outputs/tables/`. Re-running a stage uses cached
products when they are already present.

The code implements the thresholds reported in the manuscript, including
$v_{\rm LSR}\geq180$ km s$^{-1}$, the 0.1--13 d Lomb--Scargle search,
spacecraft-timescale screening, 10 per cent cross-sector period agreement, and
the fixed injection grid.

## Published data table

`data/tableA1_periodic_candidates.csv` is the machine-readable counterpart of
Table A1 of the paper: the full catalogue of all 24 periodic-variability
candidates (14 primary, 10 secondary), one row per target. Column definitions
are in `data/README.md`.

## Data sources

- Gaia Catalogue of Nearby Stars (GCNS), VizieR catalogue `J/A+A/649/A6`
- Gaia DR3 source table through a public TAP service
- TESS light-curve products through MAST

No proprietary observations or credentials are required.

## License

The source code is released under the MIT License. Catalogue and mission data
remain subject to the policies of their respective archives.
