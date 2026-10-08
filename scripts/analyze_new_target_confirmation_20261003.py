#!/usr/bin/env python3
"""Target-cluster analysis for the frozen new-target high-artifact band."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


METHOD = "chemdeprc_cap1"
CORE = ["score_cap1", "opdiv_tanimoto0.7"]
METRICS = ["true_hits", "fdp", "selected_count", "unique_blocks"]


def paired_stats(data: pd.DataFrame, method: str, reference: str, rng: np.random.Generator) -> dict:
    left = data.loc[data["method"] == method].set_index("target_chembl_id")
    right = data.loc[data["method"] == reference].set_index("target_chembl_id")
    if not left.index.equals(right.index):
        raise AssertionError(f"Target mismatch: {method} vs {reference}")
    out: dict = {"reference": reference, "n_targets": len(left)}
    for metric in METRICS:
        delta = (left[metric] - right[metric]).to_numpy(float)
        samples = rng.integers(0, len(delta), size=(50000, len(delta)))
        means = delta[samples].mean(axis=1)
        out[f"{metric}_delta"] = float(delta.mean())
        out[f"{metric}_ci_low"] = float(np.quantile(means, .025))
        out[f"{metric}_ci_high"] = float(np.quantile(means, .975))
        out[f"{metric}_positive_targets"] = int(np.sum(delta > 0))
        out[f"{metric}_negative_targets"] = int(np.sum(delta < 0))
        if metric == "true_hits":
            observed = abs(float(delta.mean()))
            null = [
                abs(float(np.mean(np.asarray(signs) * delta)))
                for signs in itertools.product((-1., 1.), repeat=len(delta))
            ]
            out["hits_exact_signflip_p_two_sided"] = float(np.mean(np.asarray(null) >= observed - 1e-12))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--opdiv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()
    df = pd.read_csv(args.grid)
    df = df[
        df["artifact_fraction"].isin([.2, .3])
        & df["logit_boost"].isin([4., 6.])
        & df["budget_label"].astype(str).eq("10")
        & df["q"].eq(.7)
    ].copy()
    op = pd.read_csv(args.opdiv)
    if not op["selected_count"].eq(10).all():
        raise AssertionError("OPDiv underfilled at least one confirmation cell")
    if not op["status"].eq("optimal").all():
        raise AssertionError("OPDiv lacks optimal certificate at least one confirmation cell")
    keys = ["target_chembl_id", "rep", "artifact_fraction", "logit_boost"]
    sanity = df[df["method"].eq("score_cap1")][keys + ["true_hits"]]
    aligned = op[keys + ["score_cap1_sanity_hits"]].merge(sanity, on=keys, validate="one_to_one")
    if len(aligned) != 1600 or not np.array_equal(
        aligned["score_cap1_sanity_hits"].to_numpy(), aligned["true_hits"].to_numpy()
    ):
        raise AssertionError("OPDiv and grid score-cap cells do not match")
    op["budget_label"] = 10
    df = pd.concat([df, op], ignore_index=True)
    methods = sorted(df["method"].unique())
    targets = sorted(df["target_chembl_id"].unique())
    if len(targets) != 8:
        raise AssertionError(f"Expected 8 qualified targets, found {len(targets)}")
    counts = df.groupby(["target_chembl_id", "method"]).size()
    if not counts.eq(200).all():
        raise AssertionError("Incomplete four-condition, 50-repeat grid")
    if not df["selected_count"].le(10).all():
        raise AssertionError("Selection exceeded budget")
    if not df["true_hits"].le(df["selected_count"]).all():
        raise AssertionError("True hits exceed selected count")
    target_means = df.groupby(["target_chembl_id", "method"], as_index=False)[METRICS].mean()
    summary = target_means.groupby("method", as_index=False).agg(
        n_targets=("target_chembl_id", "nunique"),
        hits_mean=("true_hits", "mean"), hits_target_sd=("true_hits", "std"),
        fdp_mean=("fdp", "mean"), fdp_target_sd=("fdp", "std"),
        selected_mean=("selected_count", "mean"), blocks_mean=("unique_blocks", "mean"),
    ).sort_values("hits_mean", ascending=False)
    rng = np.random.default_rng(353600)
    pairs = pd.DataFrame([
        paired_stats(target_means, METHOD, other, rng)
        for other in methods if other != METHOD
    ]).sort_values("true_hits_delta", ascending=False)
    scenario = df.groupby(["artifact_fraction", "logit_boost", "method"], as_index=False).agg(
        hits=("true_hits", "mean"), fdp=("fdp", "mean"),
        selected=("selected_count", "mean"), blocks=("unique_blocks", "mean"),
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    target_means.to_csv(args.outdir / "target_means.csv", index=False)
    summary.to_csv(args.outdir / "publication_mean_target_sd.csv", index=False)
    pairs.to_csv(args.outdir / "paired_target_bootstrap.csv", index=False)
    scenario.to_csv(args.outdir / "scenario_summary.csv", index=False)
    print("MAIN SUMMARY")
    print(summary.to_string(index=False))
    print("PRIMARY PAIRS")
    print(pairs[pairs["reference"].isin(CORE)].to_string(index=False))
    print("ALL HIT COMPARISONS")
    print(pairs[["reference", "true_hits_delta", "true_hits_ci_low", "true_hits_ci_high", "true_hits_positive_targets"]].to_string(index=False))


if __name__ == "__main__":
    main()
