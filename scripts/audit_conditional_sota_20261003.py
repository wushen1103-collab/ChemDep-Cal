#!/usr/bin/env python3
"""Audit a conditional screening claim against frozen policies by target."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["true_hits", "fdp", "unique_blocks", "max_block_share"]


def bootstrap_interval(values: np.ndarray, draws: np.ndarray, alpha: float) -> tuple[float, float]:
    sampled = values[draws].mean(axis=1)
    return tuple(np.quantile(sampled, [alpha / 2, 1 - alpha / 2]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    args = parser.parse_args()

    root = args.root.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "scripts"))
    import make_pr_adaptive_policy_tables_fast as source

    data_path = root / "data/jctc_budget_phase/latest_results.csv"
    methods = source.DEFAULT_METHODS
    needed = list(dict.fromkeys(source.CELL_KEYS + [
        "method", "n_testpool", "pvalue_min", "pvalue_median",
        "mean_pairwise_tanimoto", *source.METRICS,
    ]))
    df = pd.read_csv(data_path, usecols=lambda col: col in needed)
    df["budget_label"] = df["budget_label"].astype(str)
    df = df[
        np.isclose(df["q"], 0.7)
        & (df["block_perturbation"] == "none")
        & df["method"].isin(methods)
    ].copy()
    df["cell_id"] = df.groupby(source.CELL_KEYS, sort=False).ngroup()
    cell_meta, matrices = source.metric_matrices(df, methods)
    pareto = source.pareto_matrix(matrices)
    utility = source.policy_utility(matrices)
    oracle_idx = np.argmax(np.where(pareto, utility, -np.inf), axis=1)
    features = source.label_free_features(df).sort_values("cell_id").reset_index(drop=True)
    feature_cols = [
        "topB_block_concentration", "topB_unique_ratio", "topB_coverage_gap",
        "budget_to_topB_unique_blocks", "budget_to_pool",
        "mean_pairwise_similarity_topB", "neglog10_min_pvalue",
        "neglog10_median_pvalue", "log_budget",
    ]
    adaptive_idx, _, _ = source.train_leave_target_out(
        features, np.asarray(methods, dtype=object)[oracle_idx], feature_cols,
        methods, seed=353535, max_depth=3, min_samples_leaf=50,
    )
    rows = source.build_policy_rows(
        cell_meta, matrices, pareto, oracle_idx, adaptive_idx, methods,
    )
    rows["stress_regime"] = np.where(
        (rows["artifact_fraction"] >= 0.1) & (rows["logit_boost"] >= 4.0),
        "hard_artifact", "mild_or_none",
    )
    deployed_b50 = rows[
        (rows["budget_label"] == "50")
        & (rows["stress_regime"] == "hard_artifact")
        & (rows["policy"] == "label_free_adaptive_tree")
    ].copy()
    if len(deployed_b50) != 3000:
        raise AssertionError(f"Expected 3000 deployed B=50 stress cells, found {len(deployed_b50)}")
    deployed_b50.to_csv(outdir / "deployed_adaptive_B50_hard_cells.csv", index=False)

    published = pd.read_csv(root / "tables/pr_adaptive_policy_by_regime_q0p7.csv")
    check = rows.groupby(["stress_regime", "policy"], as_index=False)[METRICS].mean()
    check = check.merge(
        published[["stress_regime", "policy", *METRICS]],
        on=["stress_regime", "policy"], suffixes=("_recomputed", "_published"),
        validate="one_to_one",
    )
    max_abs_error = max(
        np.max(np.abs(check[f"{metric}_recomputed"] - check[f"{metric}_published"]))
        for metric in METRICS
    )
    if len(check) != len(published) or max_abs_error > 1e-8:
        raise AssertionError(f"Cannot reproduce existing summary: {max_abs_error}")

    by_target = rows.groupby(
        ["stress_regime", "target_chembl_id", "policy"], as_index=False,
    )[METRICS].mean()
    by_target.to_csv(outdir / "target_policy_means.csv", index=False)

    rng = np.random.default_rng(350103)
    comparisons = []
    for regime in ["hard_artifact", "mild_or_none"]:
        part = by_target[by_target["stress_regime"] == regime]
        targets = sorted(part["target_chembl_id"].unique())
        draws = rng.integers(0, len(targets), size=(args.n_bootstrap, len(targets)))
        wide = {
            metric: part.pivot(index="target_chembl_id", columns="policy", values=metric)
            .reindex(targets)
            for metric in METRICS
        }
        for method in methods:
            comparator = f"fixed_{method}"
            for metric in METRICS:
                delta = (wide[metric]["label_free_adaptive_tree"] - wide[metric][comparator]).to_numpy()
                lo95, hi95 = bootstrap_interval(delta, draws, 0.05)
                lo_sim, hi_sim = bootstrap_interval(delta, draws, 0.05 / (len(methods) * len(METRICS)))
                comparisons.append({
                    "stress_regime": regime,
                    "comparator": comparator,
                    "metric": metric,
                    "direction": "higher_better" if metric in {"true_hits", "unique_blocks"} else "lower_better",
                    "n_targets": len(targets),
                    "mean_adaptive_minus_comparator": float(delta.mean()),
                    "target_sd_delta": float(delta.std(ddof=1)),
                    "targets_delta_positive": int(np.sum(delta > 0)),
                    "targets_delta_negative": int(np.sum(delta < 0)),
                    "ci95_low": float(lo95),
                    "ci95_high": float(hi95),
                    "bonferroni32_ci_low": float(lo_sim),
                    "bonferroni32_ci_high": float(hi_sim),
                })
    pd.DataFrame(comparisons).to_csv(outdir / "paired_target_bootstrap.csv", index=False)

    digest = hashlib.sha256()
    with data_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest = {
        "source": str(data_path),
        "source_sha256": digest.hexdigest(),
        "source_policy_script_sha256": hashlib.sha256(
            (root / "scripts/make_pr_adaptive_policy_tables_fast.py").read_bytes()
        ).hexdigest(),
        "reproduced_max_abs_error": float(max_abs_error),
        "unit_of_inference": "target_chembl_id (10 clusters), not scenario cell",
        "hard_artifact_rule": "artifact_fraction >= 0.1 and logit_boost >= 4.0",
        "methods": methods,
        "bootstrap_draws": args.n_bootstrap,
        "bootstrap_seed": 350103,
        "multiplicity": "Bonferroni over 8 fixed comparators x 4 metrics, per regime",
        "warning": "Stress regime and comparator family were analyzed after results; confidence intervals do not correct this selection.",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "n_cells": len(cell_meta), "n_targets": by_target["target_chembl_id"].nunique(),
        "summary_max_error": max_abs_error, "outdir": str(outdir),
    }))


if __name__ == "__main__":
    main()
