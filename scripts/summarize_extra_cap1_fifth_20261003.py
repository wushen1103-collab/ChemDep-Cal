#!/usr/bin/env python3
"""Descriptive, post hoc fifth-cohort cap-one sensitivity table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_meta_fourth_20261003 import exact_signflip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--extra-dir", type=Path, required=True)
    p.add_argument("--target-means", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.extra_dir.glob("extra_cap1_seed*.csv"))
    if len(paths) != 5:
        raise AssertionError(f"Expected 5 extra-control seed files, found {len(paths)}")
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if raw.duplicated(["target", "rep", "fraction", "boost", "method"]).sum() == 0:
        raise AssertionError("Seed was omitted from the raw table; repeated cells are expected")
    extra = raw.groupby(["target", "method"], as_index=False).agg(
        hits=("hits", "mean"), fdp=("fdp", "mean"),
        selected=("selected", "mean"), blocks=("blocks", "mean"),
        before_fill=("selected_before_fill", "mean"),
    )
    methods = extra["method"].nunique()
    targets = extra["target"].nunique()
    if methods != 4 or targets != 10 or len(raw) != 10 * 5 * 4 * 10 * 4:
        raise AssertionError("Extra-control cells incomplete")
    summary = extra.groupby("method", as_index=False).agg(
        hits=("hits", "mean"), hits_target_sd=("hits", "std"),
        fdp=("fdp", "mean"), before_fill=("before_fill", "mean"),
    ).sort_values("hits", ascending=False)
    summary.to_csv(args.outdir / "extra_cap1_target_mean_sd.csv", index=False)
    core = pd.read_csv(args.target_means)
    candidate = core[core["method"].eq("meta_calibration")].set_index("target")["hits"]
    rows = []
    for method, group in extra.groupby("method"):
        comparator = group.set_index("target")["hits"].reindex(candidate.index)
        if comparator.isna().any():
            raise AssertionError(f"Target mismatch for {method}")
        diff = (candidate - comparator).to_numpy(float)
        rng = np.random.default_rng(355701)
        draws = rng.integers(0, len(diff), size=(100000, len(diff)))
        ci = np.quantile(diff[draws].mean(axis=1), [.025, .975])
        rows.append({
            "comparator": method, "meta_minus_control_hits": float(diff.mean()),
            "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "positive_targets": int((diff > 1e-10).sum()),
            "negative_targets": int((diff < -1e-10).sum()),
            "exact_signflip_p_two_sided_descriptive": exact_signflip(diff),
            "status": "post hoc descriptive, not preregistered confirmatory",
        })
    paired = pd.DataFrame(rows)
    paired.to_csv(args.outdir / "extra_cap1_paired_descriptive.csv", index=False)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
