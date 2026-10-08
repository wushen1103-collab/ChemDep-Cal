#!/usr/bin/env python3
"""Audit fixed eighth-cohort fusion against all archived matched-cell controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp


ROOT = Path(__file__).resolve().parents[1] / "source_tables"
FUSION = ROOT / "pr_eighth_fixed_75_25_fusion_20261005"
BASELINE = ROOT / "chembl_eighth_fixed_selector_results_20261005"
EXTRA = ROOT / "pr_eighth_modern_selector_controls_20261005"
KEY = ["target", "scorer_seed", "rep", "fraction", "boost"]
METHODS = {
    "score_cap1": "scorecap_hits",
    "chemdep_cal": "chemdep_hits",
    "multi_fp_rf": "rf_hits",
    "augmented_13view_rf": "augmented_rf_hits",
}


def read_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    fusion = pd.concat(
        [pd.read_csv(FUSION / f"seed{seed}_cells.csv") for seed in range(35801, 35806)],
        ignore_index=True,
    )
    baseline = pd.concat(
        [pd.read_csv(BASELINE / f"seed{seed}.csv") for seed in range(35801, 35806)],
        ignore_index=True,
    )
    if len(fusion) != 6000 or fusion.duplicated(KEY).any():
        raise AssertionError("Incomplete or duplicate fusion cells")
    if fusion.target.nunique() != 30 or fusion.scorer_seed.nunique() != 5:
        raise AssertionError("Target/seed inventory changed")
    if not fusion.selected.eq(50).all() or not fusion.scaffolds.eq(50).all():
        raise AssertionError("Fusion budget or cap changed")
    if fusion.groupby(["target", "scorer_seed"]).size().nunique() != 1:
        raise AssertionError("Unbalanced target-seed cells")
    wide = baseline.pivot(index=KEY, columns="method", values="hits").reset_index()
    if len(wide) != 6000 or set(METHODS) - set(wide.columns):
        raise AssertionError("Incomplete archived control cells")
    merged = fusion.merge(wide, on=KEY, validate="one_to_one")
    if len(merged) != 6000:
        raise AssertionError("Unmatched archived cells")
    for method, col in METHODS.items():
        if col == "augmented_rf_hits":
            merged[col] = merged[method]
        elif not merged[col].eq(merged[method]).all():
            raise AssertionError(f"Original {method} replay mismatch")
    extra = pd.concat(
        [pd.read_csv(EXTRA / f"seed{seed}_cells.csv") for seed in range(35801, 35806)],
        ignore_index=True,
    )
    if len(extra) != 12000 or extra.duplicated(KEY + ["method"]).any():
        raise AssertionError("Incomplete or duplicate extra-control cells")
    if not extra.selected.eq(50).all() or not extra.scaffolds.eq(50).all():
        raise AssertionError("Extra-control budget or cap changed")
    extra_wide = extra.pivot(index=KEY, columns="method", values="hits").reset_index()
    if len(extra_wide) != 6000:
        raise AssertionError("Extra controls do not cover all original cells")
    merged = merged.merge(extra_wide, on=KEY, validate="one_to_one")
    if len(merged) != 6000:
        raise AssertionError("Unmatched extra-control cells")
    return fusion, merged


def main() -> None:
    fusion, merged = read_rows()
    cols = ["fusion_hits", *METHODS.values(),
            "xgb_pairwise_cap1", "mmr_module_l095_cap1"]
    target_seed = merged.groupby(["target", "scorer_seed"], sort=True)[cols].mean()
    targets = target_seed.groupby(level="target")[cols].mean()
    target_seed_sd = target_seed.groupby(level="target")[cols].std(ddof=1)
    seeds = target_seed.groupby(level="scorer_seed")[cols].mean()
    overall = pd.DataFrame({
        "method": cols,
        "target_macro_hits": targets[cols].mean().to_numpy(),
        "target_sd": targets[cols].std(ddof=1).to_numpy(),
        "scorer_seed_macro_sd": seeds[cols].std(ddof=1).to_numpy(),
    })
    saved = pd.read_csv(FUSION / "overall.csv")
    if not np.allclose(saved.target_macro_hits, overall.target_macro_hits):
        raise AssertionError("Published eighth-cohort means differ from archived cells")
    diff = targets.fusion_hits.to_numpy() - targets.rf_hits.to_numpy()
    rng = np.random.default_rng(20261005)
    bootstrap = diff[rng.integers(len(diff), size=(100000, len(diff)))].mean(axis=1)
    uncertainty = {
        "comparison": "fixed 75:25 fusion minus clean multi-fingerprint RF",
        "unit": "30 target means; each averages 5 scorer seeds and 40 cells per seed",
        "mean_delta_hits": float(diff.mean()),
        "target_sd_of_delta": float(diff.std(ddof=1)),
        "bootstrap_95ci": [float(x) for x in np.quantile(bootstrap, [.025, .975])],
        "positive_targets": int((diff > 0).sum()),
        "negative_targets": int((diff < 0).sum()),
        "tied_targets": int((diff == 0).sum()),
        "sign_test_two_sided_p": float(binomtest(int((diff > 0).sum()),
                                                  int((diff != 0).sum())).pvalue),
        "paired_t_two_sided_p": float(ttest_1samp(diff, 0).pvalue),
        "seed_macro_mean_sd": {
            col: {"mean": float(seeds[col].mean()),
                  "sd": float(seeds[col].std(ddof=1))} for col in cols
        },
        "source_files_sha256": {
            f"seed{seed}_cells.csv": hashlib.sha256(
                (FUSION / f"seed{seed}_cells.csv").read_bytes()).hexdigest()
            for seed in range(35801, 35806)
        },
        "all_6000_cells_archived_baselines_exactly_replayed": True,
        "cohort_eighth_results_seen_before_fusion_run": True,
        "extra_controls_added_after_fusion_result_seen": True,
    }
    print(overall.to_string(index=False))
    print(json.dumps({key: uncertainty[key] for key in (
        "mean_delta_hits", "bootstrap_95ci", "positive_targets",
        "negative_targets", "tied_targets", "sign_test_two_sided_p",
        "paired_t_two_sided_p")}, indent=2))


if __name__ == "__main__":
    main()
