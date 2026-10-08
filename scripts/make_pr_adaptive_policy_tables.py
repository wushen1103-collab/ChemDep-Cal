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
HIGHER_IS_BETTER = ["true_hits", "unique_blocks", "selected_count"]
LOWER_IS_BETTER = ["fdp", "max_block_share"]


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


def dominates(a: pd.Series, b: pd.Series) -> bool:
    strict = False
    for metric in HIGHER_IS_BETTER:
        if a[metric] < b[metric] - 1e-12:
            return False
        strict = strict or a[metric] > b[metric] + 1e-12
    for metric in LOWER_IS_BETTER:
        if a[metric] > b[metric] + 1e-12:
            return False
        strict = strict or a[metric] < b[metric] - 1e-12
    return strict


def add_cell_frontier(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby(CELL_KEYS, sort=False):
        group = group.copy().reset_index(drop=True)
        high = group[HIGHER_IS_BETTER].to_numpy(dtype=float)
        low = group[LOWER_IS_BETTER].to_numpy(dtype=float)
        n = len(group)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            no_worse_high = np.all(high >= high[i] - 1e-12, axis=1)
            no_worse_low = np.all(low <= low[i] + 1e-12, axis=1)
            strict_high = np.any(high > high[i] + 1e-12, axis=1)
            strict_low = np.any(low < low[i] - 1e-12, axis=1)
            candidate = no_worse_high & no_worse_low & (strict_high | strict_low)
            candidate[i] = False
            dominated[i] = bool(np.any(candidate))
        group["pareto_member"] = ~dominated
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def utility(frame: pd.DataFrame) -> pd.Series:
    # Transparent policy-training utility only. Manuscript-facing claims use the raw metric regrets below.
    return (
        frame["true_hits"]
        - 25.0 * frame["fdp"]
        + 0.10 * frame["unique_blocks"]
        - 10.0 * frame["max_block_share"]
    )


def choose_oracle(df: pd.DataFrame) -> pd.DataFrame:
    use = df.copy()
    use["policy_utility"] = utility(use)
    use = use.sort_values(
        CELL_KEYS + ["pareto_member", "policy_utility", "true_hits", "fdp", "unique_blocks", "max_block_share"],
        ascending=[True] * len(CELL_KEYS) + [False, False, False, True, False, True],
    )
    oracle = use.groupby(CELL_KEYS, as_index=False).head(1).copy()
    oracle = oracle.rename(columns={"method": "oracle_method"})
    keep = CELL_KEYS + ["oracle_method", "policy_utility"] + METRICS
    return oracle[keep].rename(columns={m: f"oracle_{m}" for m in METRICS})


def label_free_features(raw_top: pd.DataFrame) -> pd.DataFrame:
    features = raw_top[CELL_KEYS].copy()
    budget = np.maximum(raw_top["budget"].astype(float).to_numpy(), 1.0)
    n_test = np.maximum(raw_top["n_testpool"].astype(float).to_numpy(), 1.0)
    unique = np.maximum(raw_top["unique_blocks"].astype(float).to_numpy(), 1.0)
    max_share = raw_top["max_block_share"].astype(float).to_numpy()
    p_min = np.maximum(raw_top["pvalue_min"].astype(float).to_numpy(), 1e-12)
    p_median = np.maximum(raw_top["pvalue_median"].astype(float).to_numpy(), 1e-12)
    features["topB_block_concentration"] = max_share
    features["topB_unique_ratio"] = unique / budget
    features["topB_coverage_gap"] = 1.0 - features["topB_unique_ratio"]
    features["budget_to_topB_unique_blocks"] = budget / unique
    features["budget_to_pool"] = budget / n_test
    features["mean_pairwise_similarity_topB"] = raw_top["mean_pairwise_tanimoto"].astype(float)
    features["neglog10_min_pvalue"] = -np.log10(p_min)
    features["neglog10_median_pvalue"] = -np.log10(p_median)
    features["log_budget"] = np.log1p(budget)
    return features


def train_leave_target_out(
    merged: pd.DataFrame,
    feature_cols: list[str],
    *,
    seed: int,
    max_depth: int,
    min_samples_leaf: int,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    preds = []
    importances = []
    rules = []
    targets = sorted(merged["target_chembl_id"].astype(str).unique())
    for target in targets:
        train = merged[merged["target_chembl_id"].astype(str) != target].copy()
        test = merged[merged["target_chembl_id"].astype(str) == target].copy()
        if train.empty or test.empty:
            continue
        classes = train["oracle_method"].astype(str)
        if classes.nunique() == 1:
            pred = np.asarray([classes.iloc[0]] * len(test), dtype=object)
            tree_rules = f"target={target}: constant {classes.iloc[0]}"
            imp = np.zeros(len(feature_cols), dtype=float)
        else:
            clf = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight="balanced",
                random_state=seed,
            )
            clf.fit(train[feature_cols], classes)
            pred = clf.predict(test[feature_cols])
            tree_rules = f"target={target}\n" + export_text(clf, feature_names=feature_cols, decimals=3)
            imp = clf.feature_importances_
        tmp = test[CELL_KEYS + ["oracle_method"]].copy()
        tmp["adaptive_method"] = pred
        preds.append(tmp)
        rules.append(tree_rules)
        importances.append(imp)
    pred_df = pd.concat(preds, ignore_index=True)
    mean_importance = np.mean(np.vstack(importances), axis=0) if importances else np.zeros(len(feature_cols))
    return pred_df, rules, mean_importance


def policy_rows(
    results: pd.DataFrame,
    oracle: pd.DataFrame,
    adaptive_choice: pd.DataFrame,
    methods: list[str],
) -> pd.DataFrame:
    rows = []
    metric_cols = CELL_KEYS + ["method", "pareto_member"] + METRICS
    method_results = results[metric_cols].copy()

    oracle_rows = oracle[CELL_KEYS + ["oracle_method"] + [f"oracle_{m}" for m in METRICS]].copy()
    oracle_rows["policy"] = "oracle_label_aware"
    oracle_rows["chosen_method"] = oracle_rows["oracle_method"]
    for metric in METRICS:
        oracle_rows[metric] = oracle_rows[f"oracle_{metric}"]
    oracle_rows["pareto_member"] = True
    rows.append(oracle_rows[CELL_KEYS + ["policy", "chosen_method", "pareto_member"] + METRICS])

    adaptive = adaptive_choice.merge(
        method_results,
        left_on=CELL_KEYS + ["adaptive_method"],
        right_on=CELL_KEYS + ["method"],
        how="left",
    )
    adaptive["policy"] = "label_free_adaptive_tree"
    adaptive["chosen_method"] = adaptive["adaptive_method"]
    rows.append(adaptive[CELL_KEYS + ["policy", "chosen_method", "pareto_member"] + METRICS])

    for method in methods:
        fixed = method_results[method_results["method"] == method].copy()
        fixed["policy"] = f"fixed_{method}"
        fixed["chosen_method"] = method
        rows.append(fixed[CELL_KEYS + ["policy", "chosen_method", "pareto_member"] + METRICS])

    out = pd.concat(rows, ignore_index=True)
    out = out.merge(oracle, on=CELL_KEYS, how="left")
    out["hit_regret_vs_oracle"] = out["oracle_true_hits"] - out["true_hits"]
    out["fdp_regret_vs_oracle"] = out["fdp"] - out["oracle_fdp"]
    out["unique_block_regret_vs_oracle"] = out["oracle_unique_blocks"] - out["unique_blocks"]
    out["max_share_regret_vs_oracle"] = out["max_block_share"] - out["oracle_max_block_share"]
    out["matches_oracle"] = (out["chosen_method"] == out["oracle_method"]).astype(float)
    return out


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


def phase_regime(features: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    use = features.merge(oracle[CELL_KEYS + ["oracle_method"]], on=CELL_KEYS, how="inner")
    use["oracle_family"] = use["oracle_method"].map(method_family)
    pressure_raw = (
        use["topB_block_concentration"]
        + use["topB_coverage_gap"]
        + use["mean_pairwise_similarity_topB"].fillna(0.0)
        + np.minimum(use["budget_to_topB_unique_blocks"], 5.0) / 5.0
    )
    use["dependency_pressure"] = pressure_raw
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
    results = pd.read_csv(root / args.results)
    results["budget_label"] = results["budget_label"].astype(str)
    results = results[(np.isclose(results["q"], args.q)) & (results["block_perturbation"] == "none")].copy()
    results = results[results["method"].isin(args.methods)].copy()
    results = add_cell_frontier(results)

    raw_top = results[results["method"] == "raw_top_b"].copy()
    features = label_free_features(raw_top)
    oracle = choose_oracle(results)
    train_frame = features.merge(oracle[CELL_KEYS + ["oracle_method"]], on=CELL_KEYS, how="inner")
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
    adaptive_choice, rules, importances = train_leave_target_out(
        train_frame,
        feature_cols,
        seed=args.seed,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    rows = policy_rows(results, oracle, adaptive_choice, args.methods)

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
    write_pair(phase_regime(features, oracle), outdir, f"pr_phase_regime_table_q{q_label}")

    (outdir / f"pr_adaptive_policy_rules_q{q_label}.txt").write_text("\n\n".join(rules) + "\n", encoding="utf-8")
    metadata = {
        "source_results": args.results,
        "q": args.q,
        "candidate_methods": args.methods,
        "policy": "leave-target-out DecisionTreeClassifier using only raw_top_b structural score/block features",
        "heldout_labels_used_by_selector": False,
        "oracle_definition": "cellwise Pareto-aware phase utility used only for regret reference; raw metrics are reported separately",
        "feature_columns": feature_cols,
        "n_policy_cells": int(rows[rows["policy"] == "label_free_adaptive_tree"].shape[0]),
    }
    (outdir / f"pr_adaptive_policy_metadata_q{q_label}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
