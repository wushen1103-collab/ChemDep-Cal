#!/usr/bin/env python3
"""Audit fixed eighth-cohort selector outputs and target-level uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("score_cap1", "chemdep_cal", "multi_fp_rf", "augmented_13view_rf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    ledger = pd.read_csv(source / "technical_eligibility.csv")
    if len(ledger) != 100 or ledger.target_chembl_id.duplicated().any():
        raise AssertionError("Frozen 100-target eligibility ledger incomplete")
    eligible = set(ledger.loc[ledger.eligible, "target_chembl_id"])
    if not eligible:
        raise AssertionError("No eligible targets")
    paths = [source / "fixed_selector_results" / f"seed{seed}.csv"
             for seed in range(35801, 35806)]
    data = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    key = ["target", "scorer_seed", "rep", "fraction", "boost", "method"]
    if len(data) != len(eligible) * 5 * 40 * len(METHODS):
        raise AssertionError("Incomplete fixed selector grid")
    if data.duplicated(key).any() or set(data.target) != eligible or set(data.method) != set(METHODS):
        raise AssertionError("Duplicate or unexpected selector cell")
    if set(data.scorer_seed) != set(range(35801, 35806)):
        raise AssertionError("Missing scorer seed")
    if set(data.rep) != set(range(10)) or set(data.fraction) != {.2, .3} or set(data.boost) != {4., 6.}:
        raise AssertionError("Fixed scenario grid changed")
    if not data.hits.between(0, 50).all() or not data.selected.eq(50).all() or not data.scaffolds.eq(50).all():
        raise AssertionError("Hit range or B50/cap1 violated")
    seed_means = data.groupby(["target", "scorer_seed", "method"], as_index=False).hits.mean()
    target = seed_means.groupby(["target", "method"]).hits.agg(["mean", "std"]).reset_index()
    target_wide = target.pivot(index="target", columns="method", values="mean")
    if target_wide.isna().any().any():
        raise AssertionError("Incomplete method comparison")
    target.to_csv(source / "fixed_selector_results" / "target_mean_seed_sd.csv", index=False)
    macro = {method: float(target_wide[method].mean()) for method in METHODS}
    comparisons = {}
    rng = np.random.default_rng(20261005)
    for control in ("multi_fp_rf", "chemdep_cal", "score_cap1"):
        diff = (target_wide["augmented_13view_rf"] - target_wide[control]).to_numpy()
        samples = rng.choice(diff, (20000, len(diff)), replace=True).mean(axis=1)
        comparisons[control] = {
            "mean": float(diff.mean()),
            "target_bootstrap_95ci": [float(v) for v in np.quantile(samples, [.025, .975])],
            "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
            "losses": int((diff < 0).sum()),
        }
    report = {
        "status": "fixed before eighth activity retrieval; label-aware synthetic simulation",
        "provenance": "all numerical method results are own reruns, not original-paper numbers",
        "inventory_targets": len(ledger), "eligible_targets": len(eligible),
        "scorer_seeds": 5, "method_cells": len(data),
        "target_macro_hits_at_50": macro,
        "augmented_13view_paired_effects": comparisons,
        "cell_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    (source / "fixed_selector_results" / "audit_summary.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(target.to_string(index=False))


if __name__ == "__main__":
    main()
