# Result-to-source index

All performance cells in this repository are own reruns. External papers are
method references, not the source of the numbers below. `source_tables/`
includes current table inputs and additional development outputs; use the
paths below when verifying the paper's main numerical statements.

| Result family | Input cells | Check or generating code |
|---|---|---|
| Primary 10-target ChEMBL, five seeds, B=50 | `source_tables/fifth_primary/fifth_raw/` and `fifth_method_mean_target_sd.csv` | `tests/verify_fifth.py`; `scripts/probe_meta_band_v2_20261003.py` |
| Same-supervision feature ablation | `source_tables/pr_revision_20261003/` | `scripts/pr_revision_experiments.py` |
| Later 30-target, five-scorer-seed matched selectors | `source_tables/chembl_eighth_fixed_selector_results_20261005/`, `pr_eighth_fixed_75_25_fusion_20261005/`, `pr_eighth_modern_selector_controls_20261005/` | `tests/audit_eighth_fixed_fusion.py`; `scripts/eighth_fixed_fusion_20261005.py` |
| Later-cohort curation and eligible-target inventory | `source_tables/chembl_eighth_screening_manifest_20261005.csv`, `data/chembl_modern_eighth_20261005/technical_eligibility.csv` | `scripts/eighth_cohort_builder_20261005.py`; `scripts/eighth_cohort_scorer_20261005.py` |
| Natural-score transfer, four datasets, five seeds | `source_tables/pr_natural_fixed_fusion_20261007/`, `pr_natural_modern_selectors_20261004/` | `tests/audit_natural_fixed_fusion.py`; `scripts/natural_fixed_fusion_20261007.py` |
| Modern selectors, scorer sensitivity, target-specific checks | `source_tables/pr_modern_selectors_20261004/`, `pr_lightgbm_fifth_20261004/`, `pr_target_specific_20261003/` | corresponding scripts in `scripts/` |
| Seventh/eighth and swap analyses | `source_tables/chembl_seventh_fixed_selector_results_20261005/`, `pr_selector_swap_audit_20261005/` | corresponding seventh-cohort and swap-audit scripts |
| Measured-assay and follow-up panels | `source_tables/pr_welqrate_litmus_20261004/`, `pubchem_primary_welqrate_20261004/`, `pubchem_confirmatory_failures_20261004/`, `pubchem_four_untouched_20261004/`, `mfpcba_twenty_20261004/`, `mfpcba_twelve_new_20261004/` | source-specific build, download, evaluation, and audit scripts |

The fifth and eighth CSVs preserve target/seed/perturbation keys. The current
paper treats the ten-target fifth cohort as the primary matched selector test;
the later 30-target comparison assesses a development-selected fixed fusion.
Natural-score results are a separate transfer setting. Do not pool these
families or treat later controls as preregistered fifth-cohort comparators.

Some older development tables are retained for traceability but are not
substitutes for the current main-text comparisons. Large source datasets,
third-party code snapshots, and local environments are intentionally omitted.
