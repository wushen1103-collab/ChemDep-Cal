#!/usr/bin/env python3
"""Integrity and target-cluster inference for frozen third-cohort B50 test."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["target", "rep", "fraction", "boost"]
METRICS = ["hits", "fdp", "selected", "blocks"]


def paired(target_means: pd.DataFrame, reference: str, rng: np.random.Generator) -> dict:
    left = target_means[target_means["method"].eq("meta_gate")].set_index("target").sort_index()
    right = target_means[target_means["method"].eq(reference)].set_index("target").sort_index()
    if not left.index.equals(right.index):
        raise AssertionError(f"Target mismatch: {reference}")
    result: dict = {"reference": reference, "n_targets": len(left)}
    for metric in METRICS:
        diff = (left[metric] - right[metric]).to_numpy(float)
        samples = rng.integers(0, len(diff), size=(100000, len(diff)))
        ci = np.quantile(diff[samples].mean(axis=1), [.025, .975])
        result[f"{metric}_delta"] = float(diff.mean())
        result[f"{metric}_ci_low"] = float(ci[0])
        result[f"{metric}_ci_high"] = float(ci[1])
        result[f"{metric}_positive_targets"] = int((diff > 0).sum())
        result[f"{metric}_negative_targets"] = int((diff < 0).sum())
        if metric == "hits":
            observed = abs(float(diff.mean()))
            p_exact = np.mean([
                abs(np.mean(np.asarray(signs) * diff)) >= observed - 1e-12
                for signs in itertools.product((-1., 1.), repeat=len(diff))
            ])
            result["hits_exact_signflip_p_two_sided"] = float(p_exact)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--opdiv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()
    raw = pd.read_csv(args.gate)
    raw = raw[raw["budget"].eq(50)].copy()
    targets = sorted(raw["target"].unique())
    if len(targets) < 5:
        raise AssertionError("Fewer than five qualified targets")
    counts = raw.groupby([*KEYS, "method"]).size()
    if not counts.eq(1).all() or len(counts) != len(targets) * 50 * 4 * 6:
        raise AssertionError("Missing/duplicate gate or comparator cells")
    if not raw.groupby("method")["target"].nunique().eq(len(targets)).all():
        raise AssertionError("A method omits at least one target")
    base = raw[raw["method"].isin(["score_cap1", "minp_block_bh_cap1"])][KEYS + ["method", "hits"]]
    grid = pd.read_csv(args.grid)
    grid = grid[
        grid["budget_label"].astype(str).eq("50")
        & grid["artifact_fraction"].isin([.2, .3])
        & grid["logit_boost"].isin([4., 6.])
        & grid["method"].isin(["score_cap1", "minp_block_bh_cap1"])
    ][["target_chembl_id", "rep", "artifact_fraction", "logit_boost", "method", "true_hits"]]
    grid = grid.rename(columns={
        "target_chembl_id": "target", "artifact_fraction": "fraction",
        "logit_boost": "boost", "true_hits": "grid_hits",
    })
    matched = base.merge(grid, on=[*KEYS, "method"], validate="one_to_one")
    if len(matched) != len(base) or not np.array_equal(matched["hits"], matched["grid_hits"]):
        raise AssertionError("Gate and independent grid baselines disagree")
    op = pd.read_csv(args.opdiv)
    if len(op) != len(targets) * 50 * 4:
        raise AssertionError("Incomplete OPDiv grid")
    sanity = op[KEYS + ["score_cap1_sanity_hits"]].merge(
        base[base["method"].eq("score_cap1")][KEYS + ["hits"]], on=KEYS, validate="one_to_one"
    )
    if not np.array_equal(sanity["score_cap1_sanity_hits"], sanity["hits"]):
        raise AssertionError("OPDiv stress seed differs from gate/grid seed")
    full_opdiv = op["selected"].eq(50).all()
    all_optimal = op["status"].eq("optimal").all()
    if full_opdiv:
        raw = pd.concat([raw, op[raw.columns]], ignore_index=True)
    target_means = raw.groupby(["target", "method"], as_index=False)[METRICS].mean()
    summary = target_means.groupby("method", as_index=False).agg(
        n_targets=("target", "nunique"),
        hits_mean=("hits", "mean"), hits_target_sd=("hits", "std"),
        fdp_mean=("fdp", "mean"), fdp_target_sd=("fdp", "std"),
        selected_mean=("selected", "mean"), blocks_mean=("blocks", "mean"),
    ).sort_values("hits_mean", ascending=False)
    rng = np.random.default_rng(353700)
    pairs = pd.DataFrame([
        paired(target_means, method, rng)
        for method in summary["method"] if method != "meta_gate"
    ]).sort_values("hits_delta", ascending=False)
    args.outdir.mkdir(parents=True, exist_ok=True)
    target_means.to_csv(args.outdir / "target_means.csv", index=False)
    summary.to_csv(args.outdir / "publication_mean_target_sd.csv", index=False)
    pairs.to_csv(args.outdir / "paired_target_bootstrap.csv", index=False)
    audit = {
        "qualified_targets": targets,
        "n_targets": len(targets),
        "repetitions_per_condition": 50,
        "conditions": [[.2, 4.], [.2, 6.], [.3, 4.], [.3, 6.]],
        "grid_gate_baseline_cells_matched": len(matched),
        "opdiv_scorecap_cells_matched": len(sanity),
        "opdiv_full_cells": int(op["selected"].eq(50).sum()),
        "opdiv_optimal_cells": int(op["status"].eq("optimal").sum()),
        "opdiv_cells": len(op),
        "source_sha256": {
            "gate": hashlib.sha256(args.gate.read_bytes()).hexdigest(),
            "grid": hashlib.sha256(args.grid.read_bytes()).hexdigest(),
            "opdiv": hashlib.sha256(args.opdiv.read_bytes()).hexdigest(),
        },
        "warning": "Artificial label-informed score corruption, not prospective screening. Experimental unit is target, not injection repeat.",
    }
    (args.outdir / "integrity_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(pairs[["reference", "hits_delta", "hits_ci_low", "hits_ci_high", "hits_positive_targets", "hits_negative_targets", "hits_exact_signflip_p_two_sided"]].to_string(index=False))
    print(f"OPDiv full={full_opdiv}, all optimal={all_optimal}")


if __name__ == "__main__":
    main()
