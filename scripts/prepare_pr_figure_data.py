"""Extract auditable, target-level figure data from the frozen remote runs."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "pr_figure_data"
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import make_pr_adaptive_policy_tables_fast as adaptive  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cluster_interval(target_values: pd.DataFrame, metrics: list[str], n_boot: int = 20000) -> dict:
    rng = np.random.default_rng(350035)
    result = {}
    for metric in metrics:
        values = target_values[metric].to_numpy(dtype=float)
        if len(values) != 10 or not np.isfinite(values).all():
            raise ValueError(f"{metric}: expected ten finite target effects")
        samples = values[rng.integers(0, 10, size=(n_boot, 10))].mean(axis=1)
        result[metric] = {
            "mean": float(values.mean()),
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "n_targets": 10,
            "resampling_unit": "target_chembl_id",
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stress_file = ROOT / "data" / "pr_figure_reproduction" / "latest_results.csv"
    budget_file = ROOT / "data" / "jctc_budget_phase" / "latest_results.csv"
    misspec_file = ROOT / "data" / "jctc_block_misspec" / "latest_results.csv"
    image_file = ROOT / "data" / "pr_native_retrieval" / "pr_native_retrieval_results_20260930T081340Z.csv"
    replay_file = ROOT / "data" / "jctc_active_learning_replay" / "latest_results.csv"
    for path in (stress_file, budget_file, misspec_file, image_file, replay_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    needed = list(
        dict.fromkeys(
            adaptive.CELL_KEYS
            + [
                "method",
                "n_testpool",
                "pvalue_min",
                "pvalue_median",
                "mean_pairwise_tanimoto",
                "fdp_exceeds_q",
                *adaptive.METRICS,
            ]
        )
    )
    df = pd.read_csv(stress_file, usecols=lambda col: col in needed)
    df = df[np.isclose(df["q"], 0.7)].copy()
    metrics = [
        "true_hits",
        "fdp",
        "unique_blocks",
        "max_block_share",
        "mean_pairwise_tanimoto",
        "selected_count",
    ]
    target_keys = [
        "target_chembl_id",
        "artifact_fraction",
        "logit_boost",
        "block_perturbation",
        "budget_label",
        "method",
    ]
    target_grid = df.groupby(target_keys, dropna=False, as_index=False)[metrics].mean()
    target_grid.to_csv(OUT / "stress_target_condition.csv", index=False)
    misspec = pd.read_csv(misspec_file, usecols=lambda col: col in needed)
    misspec = misspec[np.isclose(misspec["q"], 0.7)]
    misspec_target = misspec.groupby(
        ["target_chembl_id", "block_perturbation", "budget_label", "method"],
        as_index=False,
    )[metrics].mean()
    misspec_target.to_csv(OUT / "misspec_target_condition.csv", index=False)

    main_cell = df[
        (df["artifact_fraction"] == 0.1)
        & (df["logit_boost"] == 4.0)
        & (df["block_perturbation"] == "none")
        & (df["budget_label"].astype(str) == "50")
    ].copy()
    main_target = main_cell.groupby(["target_chembl_id", "method"], as_index=False)[metrics].mean()
    main_target.to_csv(OUT / "main_stress_target_method.csv", index=False)
    paired = main_cell[main_cell["method"].isin(["score_cap1", "chemdeprc_cap1"])].pivot(
        index=["target_chembl_id", "rep"], columns="method", values=metrics
    )
    paired = paired.dropna()
    delta = pd.DataFrame(
        {metric: paired[(metric, "chemdeprc_cap1")] - paired[(metric, "score_cap1")] for metric in metrics}
    ).reset_index()
    if len(delta) != 500:
        raise ValueError(f"Main stress has {len(delta)} paired target-replicates, not 500")
    target_delta = delta.groupby("target_chembl_id", as_index=False)[metrics].mean()
    target_delta.to_csv(OUT / "main_mechanism_target_delta.csv", index=False)
    mechanism_ci = cluster_interval(target_delta, metrics)

    budget_df = pd.read_csv(budget_file, usecols=lambda col: col in needed)
    work = budget_df[
        np.isclose(budget_df["q"], 0.7)
        & (budget_df["block_perturbation"] == "none")
        & budget_df["method"].isin(adaptive.DEFAULT_METHODS)
    ].copy()
    work["budget_label"] = work["budget_label"].astype(str)
    work["cell_id"] = work.groupby(adaptive.CELL_KEYS, sort=False).ngroup()
    cell_meta, matrices = adaptive.metric_matrices(work, adaptive.DEFAULT_METHODS)
    pareto = adaptive.pareto_matrix(matrices)
    utility = adaptive.policy_utility(matrices)
    oracle_idx = np.argmax(np.where(pareto, utility, -np.inf), axis=1)
    oracle_methods = np.asarray(adaptive.DEFAULT_METHODS, dtype=object)[oracle_idx]
    features = adaptive.label_free_features(work).sort_values("cell_id").reset_index(drop=True)
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
    adaptive_idx, _, importances = adaptive.train_leave_target_out(
        features,
        oracle_methods,
        feature_cols,
        adaptive.DEFAULT_METHODS,
        seed=353535,
        max_depth=3,
        min_samples_leaf=50,
    )
    rows = adaptive.build_policy_rows(
        cell_meta, matrices, pareto, oracle_idx, adaptive_idx, adaptive.DEFAULT_METHODS
    )
    policy_metrics = [
        "true_hits",
        "fdp",
        "selected_count",
        "unique_blocks",
        "max_block_share",
        "hit_regret_vs_oracle",
        "fdp_regret_vs_oracle",
        "matches_oracle",
        "pareto_member",
    ]
    target_policy = rows.groupby(["target_chembl_id", "policy"], as_index=False)[policy_metrics].mean()
    target_policy.to_csv(OUT / "adaptive_target_policy.csv", index=False)

    features["dependency_pressure"] = (
        features["topB_block_concentration"]
        + features["topB_coverage_gap"]
        + features["mean_pairwise_similarity_topB"].fillna(0.0)
        + np.minimum(features["budget_to_topB_unique_blocks"], 5.0) / 5.0
    )
    features["pressure_bin"] = pd.qcut(
        features["dependency_pressure"], 3, labels=["low", "mid", "high"], duplicates="drop"
    )
    features["budget_ratio_bin"] = pd.qcut(
        features["budget_to_topB_unique_blocks"], 3, labels=["small", "mid", "large"], duplicates="drop"
    )
    grouping = features[["cell_id", "pressure_bin", "budget_ratio_bin"]]
    regime = rows[rows["policy"].isin(["fixed_bh", "fixed_weighted_bh", "fixed_chemdeprc_cap1", "label_free_adaptive_tree"])]
    regime = regime.merge(grouping, on="cell_id", validate="many_to_one")
    target_regime = regime.groupby(
        ["target_chembl_id", "pressure_bin", "budget_ratio_bin", "policy"],
        observed=True,
        as_index=False,
    )[policy_metrics].mean()
    target_regime.to_csv(OUT / "adaptive_target_regime.csv", index=False)

    image = pd.read_csv(image_file)
    if image["seed"].nunique() != 5:
        raise ValueError("Image evaluation must contain five seeds")
    image.to_csv(OUT / "image_task_seed_raw.csv", index=False)
    replay = pd.read_csv(replay_file)
    if len(replay) != 440 or sorted(replay["round"].unique()) != list(range(11)):
        raise ValueError("Active replay must contain 440 rows spanning rounds 0-10")
    replay.to_csv(OUT / "active_replay_target_round.csv", index=False)
    adaptive_check = target_policy.groupby("policy", as_index=False)["true_hits"].mean().set_index("policy")
    if abs(adaptive_check.loc["label_free_adaptive_tree", "true_hits"] - 81.5088) > 0.01:
        raise ValueError("Adaptive extraction disagrees with frozen summary")
    if abs(main_cell.loc[main_cell["method"] == "chemdeprc_cap1", "true_hits"].mean() - 42.454) > 1e-6:
        raise ValueError("Main stress extraction disagrees with frozen summary")

    audit = {
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (stress_file, budget_file, misspec_file, image_file, replay_file)
        },
        "revision_source_rows_q0p7": int(len(df)),
        "budget_source_rows_q0p7": int(len(budget_df[np.isclose(budget_df["q"], 0.7)])),
        "stress_target_condition_rows": int(len(target_grid)),
        "misspec_target_condition_rows": int(len(misspec_target)),
        "main_stress_targets": int(main_target["target_chembl_id"].nunique()),
        "main_stress_target_rep_pairs": int(len(delta)),
        "adaptive_cells": int(len(cell_meta)),
        "adaptive_target_policy_rows": int(len(target_policy)),
        "image_rows": int(len(image)),
        "image_seed_count": int(image["seed"].nunique()),
        "replay_rows": int(len(replay)),
        "replay_rounds": sorted(map(int, replay["round"].unique())),
        "mechanism_target_cluster_bootstrap": mechanism_ci,
        "adaptive_feature_importance": dict(zip(feature_cols, map(float, importances))),
        "notes": [
            "Bootstrap draws target IDs; all repetitions of a target remain together.",
            "Oracle is retrospective and is not a deployable comparator.",
            "Pareto membership is five-objective, not a 2D plot frontier.",
        ],
    }
    (OUT / "extraction_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
