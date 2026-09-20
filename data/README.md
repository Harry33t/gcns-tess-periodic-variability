# Published data table

`tableA1_periodic_candidates.csv` is the machine-readable counterpart of
Table A1 of the paper: the complete catalogue of all 24 periodic-variability
candidates, comprising the 14 primary (multi-sector) and 10 secondary
(strong-single-sector) candidates. One row per target, 24 rows.

## Columns

| Column | Unit | Description |
|---|---|---|
| `source_id` | | Gaia DR3 source identifier |
| `ra`, `dec` | deg | Gaia DR3 position (ICRS) |
| `phot_g_mean_mag` | mag | Gaia DR3 mean *G* magnitude |
| `bp_rp_dr3` | mag | Gaia DR3 colour *G*_BP - *G*_RP |
| `abs_g_dr3` | mag | Absolute *G* magnitude |
| `v_lsr_dr3` | km/s | Total speed relative to the local standard of rest |
| `period_days` | d | Adopted periodic-variability period |
| `period_support` | | Number of sectors supporting the adopted period |
| `period_scatter_frac` | | Fractional scatter of the supporting sector periods |
| `period_support_fraction` | | Fraction of a target's significant sectors that agree |
| `period_class` | | `multi_sector` = primary tier; `strong_single_sector` = secondary tier |
| `semi_amplitude_pct` | per cent | Fitted semi-amplitude of the modulation |
| `best_fap` | | Lowest Lomb-Scargle false-alarm probability across sectors |
| `max_amplitude_snr` | | Highest coherent-amplitude signal-to-noise ratio across sectors |
| `median_crowdsap` | | Median CROWDSAP where supplied by the pipeline |
| `ruwe` | | Gaia DR3 renormalised unit weight error |
| `rv_variable_gaia` | | Gaia radial-velocity variability flag |
| `nss_gaia` | | Gaia DR3 non-single-star solution flag |
| `gcns_resolved_binary` | | GCNS resolved-pair flag |
| `multiplicity_indicator` | | True if any of the three multiplicity flags is set |

## Notes

Periods are reported as periodic photometric variability of the light within
the TESS aperture. They identify rotation candidates, not confirmed rotation
periods. The two tiers carry different levels of observational confirmation;
see the paper for the tier definitions and for the alias, neighbour, and
injection-recovery diagnostics.

The shortest-period target, Gaia DR3 2112434245261736192, shares its TESS
pixel with a GCNS-bound companion at 2.76 arcsec, so its signal belongs to the
unresolved pair.
