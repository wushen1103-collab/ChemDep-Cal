#!/usr/bin/env python3
"""Rank exploratory stress scenarios by target-level paired effects.

The output is for protocol selection only, not confirmatory inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    df = pd.read_csv(args.input)
    df = df[
        (df["block_perturbation"] == "none")
        & (df["method"].isin(["chemdeprc_cap1", "score_cap1"]))
        & (df["artifact_fraction"] > 0)
        & (df["q"] == .7)
    ].copy()
    keys = ["stress_label", "artifact_fraction", "logit_boost", "budget_label", "target_chembl_id", "rep"]
    paired = df.pivot(index=keys, columns="method", values=["true_hits", "fdp", "selected_count"])
    paired = paired.dropna()
    if len(paired) * 2 != len(df):
        raise AssertionError("Grid contains missing or duplicate method pairs")
    diff = pd.DataFrame({
        "hits_diff": paired[("true_hits", "chemdeprc_cap1")] - paired[("true_hits", "score_cap1")],
        "fdp_diff": paired[("fdp", "chemdeprc_cap1")] - paired[("fdp", "score_cap1")],
        "fill": paired[("selected_count", "chemdeprc_cap1")],
        "scorecap_hits": paired[("true_hits", "score_cap1")],
    }).reset_index()
    grouped = diff.groupby(["stress_label", "artifact_fraction", "logit_boost", "budget_label", "target_chembl_id"], as_index=False).agg(
        hits_diff=("hits_diff", "mean"), fdp_diff=("fdp_diff", "mean"),
        fill=("fill", "mean"), scorecap_hits=("scorecap_hits", "mean"), n_reps=("rep", "size"),
    )
    out = grouped.groupby(["stress_label", "artifact_fraction", "logit_boost", "budget_label"], as_index=False).agg(
        n_targets=("target_chembl_id", "nunique"),
        n_positive_targets=("hits_diff", lambda x: int((x > 0).sum())),
        hits_diff=("hits_diff", "mean"),
        target_sd=("hits_diff", "std"),
        fdp_diff=("fdp_diff", "mean"),
        fill=("fill", "mean"),
        scorecap_hits=("scorecap_hits", "mean"),
        min_reps=("n_reps", "min"),
    )
    out["t_exploratory"] = out["hits_diff"] / (out["target_sd"] / np.sqrt(out["n_targets"]))
    out = out.sort_values(["t_exploratory", "hits_diff"], ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    grouped.to_csv(args.out.with_name(args.out.stem + "_target_means.csv"), index=False)
    print(out.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
