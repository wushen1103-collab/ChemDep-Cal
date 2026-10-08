#!/usr/bin/env python3
"""Audit exposed fifth-cohort augmented-RF development at target level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    directory = root / "data/pr_fifth_augmented_multifp_development_20261005"
    files = [directory / f"seed{seed}_cells.csv" for seed in range(35501, 35506)]
    cells = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    if len(cells) != 2000 or cells.duplicated(
            ["target", "seed", "rep", "fraction", "boost"]).any():
        raise AssertionError("Expected exactly 2,000 distinct matched cells")
    methods = ["score_cap1_hits", "multifp_rf_hits", "augmented_score_rf_hits",
               "augmented_context_rf_hits"]
    if cells[methods].isna().any().any():
        raise AssertionError("Missing hits")
    for method in methods:
        if not cells[method].between(0, 50).all():
            raise AssertionError(f"Invalid hit count: {method}")
    per_seed = cells.groupby(["target", "seed"], as_index=False)[methods].mean()
    if per_seed.groupby("target").size().ne(5).any():
        raise AssertionError("Five scorer seeds required for each target")
    target = per_seed.groupby("target")[methods].agg(["mean", "std"])
    target.columns = [f"{method}_{stat}" for method, stat in target.columns]
    target = target.reset_index()
    macro = target[[f"{method}_mean" for method in methods]].mean().to_dict()
    rng = np.random.default_rng(20261005)
    effects = {}
    for candidate in methods[2:]:
        for control in (methods[1], methods[2]):
            if candidate == control:
                continue
            diff = (target[f"{candidate}_mean"] - target[f"{control}_mean"]).to_numpy()
            boot = rng.choice(diff, size=(20000, len(diff)), replace=True).mean(axis=1)
            effects[f"{candidate}_minus_{control}"] = {
                "mean": float(diff.mean()),
                "target_bootstrap_95ci": [float(x) for x in np.quantile(boot, [.025, .975])],
                "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
                "losses": int((diff < 0).sum()),
            }
    report = {
        "status": "post hoc exposed-cohort development; not independent confirmation",
        "cells": len(cells), "targets": len(target), "scorer_seeds": 5,
        "target_macro": macro, "paired_target_effects": effects,
    }
    target.to_csv(directory / "target_mean_seed_sd.csv", index=False)
    (directory / "audit_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(target.to_string(index=False))


if __name__ == "__main__":
    main()
