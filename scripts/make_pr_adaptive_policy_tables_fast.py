#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


CELL_KEYS = [
    "target_chembl_id",
    "pref_name",
    "dataset",
    "scenario",
    "rep",
    "stress_label",
    "artifact_fraction",
    "logit_boost",
    "block_perturbation",
    "q",
    "budget_label",
    "budget",
]

DEFAULT_METHODS = [
    "raw_top_b",
    "bh",
    "weighted_bh",
    "block_bh",
    "score_cap1",
    "chemdeprc_cap1",
    "chemdeprc_soft75",
    "chemdeprc_scoresoft50",
]

METRICS = ["true_hits", "fdp", "unique_blocks", "max_block_share", "selected_count"]


def write_pair(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(df.to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8")


def method_family(method: str) -> str:
    if method.startswith("chemdeprc"):
        return "adaptive-ChemDep-family"
    if method.startswith("score_cap"):
        return "score-cap"
    if method in {"bh", "weighted_bh", "block_bh"}:
        return "FDR-ranking"
    if method == "raw_top_b":
        return "raw-score"
    return "other"


def metric_matrices(df: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    cell_meta = df[CELL_KEYS + ["cell_id"]].drop_duplicates("cell_id").sort_values("cell_id").reset_index(drop=True)
    matrices: dict[str, np.ndarray] = {}
    for metric in METRICS:
        wide = df.pivot(index="cell_id", columns="method", values=metric).reindex(columns=methods)
        matrices[metric] = wide.to_numpy(dtype=float)
    return cell_meta, matrices


def pareto_matrix(matrices: dict[str, np.ndarray]) -> np.ndarray:
    hits = matrices["true_hits"]
    unique = matrices["unique_blocks"]
    selected = matrices["selected_count"]
    fdp = matrices["fdp"]
    share = matrices["max_block_share"]
    n_cells, n_methods = hits.shape
    dominated = np.zeros((n_cells, n_methods), dtype=bool)
    for i in range(n_methods):
        for j in range(n_methods):
            if i == j:
                continue
            no_worse = (
                (hits[:, j] >= hits[:, i] - 1e-12)
                & (unique[:, j] >= unique[:, i] - 1e-12)
                & (selected[:, j] >= selected[:, i] - 1e-12)
                & (fdp[:, j] <= fdp[:, i] + 1e-12)
                & (share[:, j] <= share[:, i] + 1e-12)
            )
            strict = (
                (hits[:, j] > hits[:, i] + 1e-12)
                | (unique[:, j] > unique[:, i] + 1e-12)
                | (selected[:, j] > selected[:, i] + 1e-12)
                | (fdp[:, j] < fdp[:, i] - 1e-12)
                | (share[:, j] < share[:, i] - 1e-12)
            )
            dominated[:, i] |= no_worse & strict
    return ~dominated


def policy_utility(matrices: dict[str, np.ndarray]) -> np.ndarray:
    return (
        matrices["true_hits"]
        - 25.0 * matrices["fdp"]
        + 0.10 * matrices["unique_blocks"]
        - 10.0 * matrices["max_block_share"]
    )


def label_free_features(df: pd.DataFrame) -> pd.DataFrame:
    raw = df[df["method"] == "raw_top_b"].sort_values("cell_id").copy()
    budget = np.maximum(raw["budget"].astype(float).to_numpy(), 1.0)
    n_test = np.maximum(raw["n_testpool"].astype(float).to_numpy(), 1.0)
    unique = np.maximum(raw["unique_blocks"].astype(float).to_numpy(), 1.0)
    p_min = np.maximum(raw["pvalue_min"].astype(float).to_numpy(), 1e-12)
    p_median = np.maximum(raw["pvalue_median"].astype(float).to_numpy(), 1e-12)
    out = raw[CELL_KEYS + ["cell_id"]].copy()
    out["topB_block_concentration"] = raw["max_block_share"].astype(float).to_numpy()
    out["topB_unique_ratio"] = unique / budget
    out["topB_coverage_gap"] = 1.0 - out["topB_unique_ratio"]
    out["budget_to_topB_unique_blocks"] = budget / unique
    out["budget_to_pool"] = budget / n_test
    out["mean_pairwise_similarity_topB"] = raw["mean_pairwise_tanimoto"].astype(float).to_numpy()
    out["neglog10_min_pvalue"] = -np.log10(p_min)
    out["neglog10_median_pvalue"] = -np.log10(p_median)
    out["log_budget"] = np.log1p(budget)
    return out


def train_leave_target_out(
    features: pd.DataFrame,
    oracle_method: np.ndarray,
    feature_cols: list[str],
    methods: list[str],
    *,
    seed: int,
    max_depth: int,
    min_samples_leaf: int,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    pred_idx = np.zeros(len(features), dtype=int)
    rules: list[str] = []
    importances: list[np.ndarray] = []
    method_to_idx = {m: i for i, m in enumerate(methods)}
    features = features.reset_index(drop=True)
    y = np.asarray([method_to_idx[m] for m in oracle_method], dtype=int)
    for target in sorted(features["target_chembl_id"].astype(str).unique()):
        test_mask = features["target_chembl_id"].astype(str).to_numpy() == target
        train_mask = ~test_mask
        y_train = y[train_mask]
        if len(np.unique(y_train)) == 1:
            pred_idx[test_mask] = y_train[0]
            rules.append(f"target={target}: constant {methods[y_train[0]]}")
            importances.append(np.zeros(len(feature_cols), dtype=float))
            continue
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=seed,
        )
        clf.fit(features.loc[train_mask, feature_cols], y_train)
        pred_idx[test_mask] = clf.predict(features.loc[test_mask, feature_cols])
        rules.append(f"target={target}\n" + export_text(clf, feature_names=feature_cols, decimals=3))
        importances.append(clf.feature_importances_)
    return pred_idx, rules, np.mean(np.vstack(importances), axis=0)


def build_policy_rows(
    cell_meta: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    pareto: np.ndarray,
    oracle_idx: np.ndarray,
    adaptive_idx: np.ndarray,
    methods: list[str],
) -> pd.DataFrame:
    n = len(cell_meta)
    method_to_idx = {m: i for i, m in enumerate(methods)}
    policies: list[tuple[str, np.ndarray]] = [
        ("oracle_label_aware", oracle_idx),
        ("label_free_adaptive_tree", adaptive_idx),
    ]
    policies.extend((f"fixed_{m}", np.full(n, method_to_idx[m], dtype=int)) for m in methods)
    rows = []
    row_index = np.arange(n)
    oracle_metrics = {metric: matrices[metric][row_index, oracle_idx] for metric in METRICS}
    for policy, idx in policies:
        part = cell_meta.copy()
        part["policy"] = policy
        part["chosen_method"] = [methods[i] for i in idx]
        part["oracle_method"] = [methods[i] for i in oracle_idx]
        part["pareto_member"] = pareto[row_index, idx].astype(float)
        for metric in METRICS:
            part[metric] = matrices[metric][row_index, idx]
            part[f"oracle_{metric}"] = oracle_metrics[metric]
        part["hit_regret_vs_oracle"] = part["oracle_true_hits"] - part["true_hits"]
        part["fdp_regret_vs_oracle"] = part["fdp"] - part["oracle_fdp"]
        part["unique_block_regret_vs_oracle"] = part["oracle_unique_blocks"] - part["unique_blocks"]
        part["max_share_regret_vs_oracle"] = part["max_block_share"] - part["oracle_max_block_share"]
        part["matches_oracle"] = (part["chosen_method"] == part["oracle_method"]).astype(float)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def summarize_policy(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metrics = [
        "true_hits",
        "fdp",
        "unique_blocks",
        "max_block_share",
        "hit_regret_vs_oracle",
        "fdp_regret_vs_oracle",
        "unique_block_regret_vs_oracle",
        "max_share_regret_vs_oracle",
        "matches_oracle",
        "pareto_member",
    ]
    mean = rows.groupby(keys, as_index=False)[metrics].mean()
    std = rows.groupby(keys, as_index=False)[metrics].std().rename(columns={m: f"{m}_std" for m in metrics})
    n = rows.groupby(keys, as_index=False).size().rename(columns={"size": "n_cells"})
    return mean.merge(std, on=keys, how="left").merge(n, on=keys, how="left")


def phase_regime(features: pd.DataFrame, oracle_methods: np.ndarray) -> pd.DataFrame:
    use = features.copy()
    use["oracle_method"] = oracle_methods
    use["oracle_family"] = use["oracle_method"].map(method_family)
    use["dependency_pressure"] = (
        use["topB_block_concentration"]
        + use["topB_coverage_gap"]
        + use["mean_pairwise_similarity_topB"].fillna(0.0)
        + np.minimum(use["budget_to_topB_unique_blocks"], 5.0) / 5.0
    )
    use["budget_ratio"] = use["budget_to_topB_unique_blocks"]
    use["pressure_bin"] = pd.qcut(use["dependency_pressure"], 3, labels=["low", "mid", "high"], duplicates="drop")
    use["budget_ratio_bin"] = pd.qcut(use["budget_ratio"], 3, labels=["small", "mid", "large"], duplicates="drop")
    counts = (
        use.groupby(["pressure_bin", "budget_ratio_bin", "oracle_family", "oracle_method"], observed=False)
        .size()
        .reset_index(name="n_cells")
    )
    total = counts.groupby(["pressure_bin", "budget_ratio_bin"], observed=False)["n_cells"].transform("sum")
    counts["cell_share"] = counts["n_cells"] / np.maximum(total, 1)
    return counts.sort_values(["pressure_bin", "budget_ratio_bin", "n_cells"], ascending=[True, True, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/jctc_budget_phase/latest_results.csv")
    parser.add_argument("--outdir", default="tables")
    parser.add_argument("--q", type=float, default=0.7)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seed", type=int, default=353535)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    needed = list(dict.fromkeys(CELL_KEYS + ["method", "n_testpool", "pvalue_min", "pvalue_median", "mean_pairwise_tanimoto"] + METRICS))
    df = pd.read_csv(root / args.results, usecols=lambda col: col in needed)
    df["budget_label"] = df["budget_label"].astype(str)
    df = df[(np.isclose(df["q"], args.q)) & (df["block_perturbation"] == "none") & (df["method"].isin(args.methods))].copy()
    df["cell_id"] = df.groupby(CELL_KEYS, sort=False).ngroup()

    cell_meta, matrices = metric_matrices(df, args.methods)
    pareto = pareto_matrix(matrices)
    util = policy_utility(matrices)
    oracle_scores = np.where(pareto, util, -np.inf)
    oracle_idx = np.argmax(oracle_scores, axis=1)
    oracle_methods = np.asarray(args.methods, dtype=object)[oracle_idx]

    features = label_free_features(df).sort_values("cell_id").reset_index(drop=True)
    feature_cols = [
        "topB_block_concentration",
        "topB_unique_ratio",
        "topB_coverage_gap",
        "budget_to_topB_unique_blocks",
        "budget_to_pool",
        "mean_pairwise_similarity_topB",
        "neglog10_min_pvalue",
        "neglog10_median_pvalue",
        "log_budget",
    ]
    adaptive_idx, rules, importances = train_leave_target_out(
        features,
        oracle_methods,
        feature_cols,
        args.methods,
        seed=args.seed,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    rows = build_policy_rows(cell_meta, matrices, pareto, oracle_idx, adaptive_idx, args.methods)

    q_label = str(args.q).replace(".", "p")
    write_pair(summarize_policy(rows, ["policy"]), outdir, f"pr_adaptive_policy_overall_q{q_label}")
    write_pair(summarize_policy(rows, ["budget_label", "policy"]), outdir, f"pr_adaptive_policy_by_budget_q{q_label}")
    rows["stress_regime"] = np.where(
        (rows["artifact_fraction"] >= 0.1) & (rows["logit_boost"] >= 4.0), "hard_artifact", "mild_or_none"
    )
    write_pair(summarize_policy(rows, ["stress_regime", "policy"]), outdir, f"pr_adaptive_policy_by_regime_q{q_label}")

    frequency = (
        rows[rows["policy"] == "label_free_adaptive_tree"]
        .groupby(["budget_label", "chosen_method"], as_index=False)
        .size()
        .rename(columns={"size": "n_cells"})
    )
    frequency["cell_share"] = frequency.groupby("budget_label")["n_cells"].transform(lambda s: s / s.sum())
    write_pair(frequency, outdir, f"pr_adaptive_policy_choice_frequency_q{q_label}")

    feature_table = pd.DataFrame({"feature": feature_cols, "leave_target_out_tree_importance": importances})
    write_pair(feature_table.sort_values("leave_target_out_tree_importance", ascending=False), outdir, f"pr_adaptive_policy_features_q{q_label}")
    write_pair(phase_regime(features, oracle_methods), outdir, f"pr_phase_regime_table_q{q_label}")

    (outdir / f"pr_adaptive_policy_rules_q{q_label}.txt").write_text("\n\n".join(rules) + "\n", encoding="utf-8")
    metadata = {
        "source_results": args.results,
        "q": args.q,
        "candidate_methods": args.methods,
        "policy": "leave-target-out DecisionTreeClassifier using raw_top_b label-free structure features",
        "heldout_labels_used_by_selector": False,
        "oracle_definition": "Pareto-aware phase utility for regret reference only; raw metrics are reported separately.",
        "n_cells": int(len(cell_meta)),
        "feature_columns": feature_cols,
    }
    (outdir / f"pr_adaptive_policy_metadata_q{q_label}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
