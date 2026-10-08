#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


CHEMDEP_METHODS = [
    "chemdeprc_cap1",
    "chemdeprc_soft50",
    "chemdeprc_soft75",
    "chemdeprc_scoresoft50",
]
KEY_METHODS = [
    "raw_top_b",
    "bh",
    "weighted_bh",
    "block_bh",
    "score_cap1",
    "score_mmr25",
    "score_dpp",
    *CHEMDEP_METHODS,
]
METRICS = ["true_hits", "fdp", "fdp_exceeds_q", "unique_blocks", "max_block_share", "mean_pairwise_tanimoto"]
SUMMARY_METRICS = [
    "selected_count",
    "true_hits",
    "false_hits",
    "fdp",
    "fdp_exceeds_q",
    "precision",
    "power",
    "unique_blocks",
    "max_block_share",
    "mean_pairwise_tanimoto",
]


def write_pair(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(df.to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8")


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1 or n_boot <= 0:
        value = float(values.mean())
        return value, value, value
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_bootstrap_table(
    results: pd.DataFrame,
    *,
    baseline: str,
    methods: list[str],
    metrics: list[str],
    group_cols: list[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    unit_cols = ["target_chembl_id"]
    if "rep" in results.columns:
        unit_cols.append("rep")
    for key, group in results.groupby(group_cols, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        key_map = dict(zip(group_cols, key_values))
        base = group[group["method"] == baseline][unit_cols + metrics]
        for method in methods:
            if method == baseline:
                continue
            comp = group[group["method"] == method][unit_cols + metrics]
            paired = comp.merge(base, on=unit_cols, suffixes=("", "_baseline"))
            if paired.empty:
                continue
            for metric in metrics:
                diff = paired[metric].to_numpy(dtype=float) - paired[f"{metric}_baseline"].to_numpy(dtype=float)
                mean, lo, hi = bootstrap_ci(diff, rng, n_boot)
                rows.append(
                    {
                        **key_map,
                        "baseline": baseline,
                        "method": method,
                        "metric": metric,
                        "mean_delta": mean,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n_pairs": int(len(diff)),
                    }
                )
    return pd.DataFrame(rows)


def dominates(a: pd.Series, b: pd.Series) -> bool:
    ge_hits = a["true_hits"] >= b["true_hits"]
    le_fdp = a["fdp"] <= b["fdp"]
    ge_blocks = a["unique_blocks"] >= b["unique_blocks"]
    le_share = a["max_block_share"] <= b["max_block_share"]
    strict = (
        a["true_hits"] > b["true_hits"]
        or a["fdp"] < b["fdp"]
        or a["unique_blocks"] > b["unique_blocks"]
        or a["max_block_share"] < b["max_block_share"]
    )
    return bool(ge_hits and le_fdp and ge_blocks and le_share and strict)


def frontier_probabilities(
    results: pd.DataFrame,
    *,
    group_cols: list[str],
    methods: list[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    unit_cols = ["target_chembl_id"]
    if "rep" in results.columns:
        unit_cols.append("rep")
    rows = []
    for key, group in results[results["method"].isin(methods)].groupby(group_cols, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        key_map = dict(zip(group_cols, key_values))
        pivot = group.pivot_table(index=unit_cols, columns="method", values=["true_hits", "fdp", "unique_blocks", "max_block_share"])
        if pivot.empty:
            continue
        complete_methods = [method for method in methods if ("true_hits", method) in pivot.columns]
        if len(complete_methods) < 2:
            continue
        unit_index = np.arange(len(pivot))
        counts = {method: 0 for method in complete_methods}
        for _ in range(max(1, n_boot)):
            sample = rng.choice(unit_index, size=len(unit_index), replace=True)
            means = {}
            for method in complete_methods:
                means[method] = pd.Series(
                    {
                        "true_hits": float(pivot[("true_hits", method)].iloc[sample].mean()),
                        "fdp": float(pivot[("fdp", method)].iloc[sample].mean()),
                        "unique_blocks": float(pivot[("unique_blocks", method)].iloc[sample].mean()),
                        "max_block_share": float(pivot[("max_block_share", method)].iloc[sample].mean()),
                    }
                )
            for method, row in means.items():
                if not any(dominates(other, row) for other_method, other in means.items() if other_method != method):
                    counts[method] += 1
        for method in complete_methods:
            rows.append({**key_map, "method": method, "frontier_probability": counts[method] / max(1, n_boot)})
    return pd.DataFrame(rows)


def add_delta_vs_bh(summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["stress_label", "artifact_fraction", "logit_boost", "block_perturbation", "q", "budget_label"]
    base = summary[summary["method"] == "bh"][keys + METRICS]
    merged = summary.merge(base, on=keys, suffixes=("", "_bh"), how="left")
    for metric in METRICS:
        merged[f"delta_{metric}_vs_bh"] = merged[metric] - merged[f"{metric}_bh"]
    return merged


def add_delta_vs_baseline(summary: pd.DataFrame, *, keys: list[str], baseline: str, metrics: list[str]) -> pd.DataFrame:
    base = summary[summary["method"] == baseline][keys + metrics]
    merged = summary.merge(base, on=keys, suffixes=("", f"_{baseline}"), how="left")
    for metric in metrics:
        merged[f"delta_{metric}_vs_{baseline}"] = merged[metric] - merged[f"{metric}_{baseline}"]
    return merged


def phase_strategy(summary: pd.DataFrame) -> pd.DataFrame:
    df = add_delta_vs_bh(summary)
    mask = (
        (df["q"] == 0.7)
        & (df["block_perturbation"] == "none")
        & (df["method"].isin(CHEMDEP_METHODS + ["weighted_bh", "score_mmr25", "score_dpp"]))
    )
    candidates = df[mask].copy()
    if candidates.empty:
        return pd.DataFrame()
    # A compact utility score for table selection: reward hits and blocks, penalize FDP and concentration.
    candidates["phase_score"] = (
        candidates["delta_true_hits_vs_bh"]
        - 25.0 * candidates["delta_fdp_vs_bh"]
        + 0.10 * candidates["delta_unique_blocks_vs_bh"]
        - 10.0 * candidates["delta_max_block_share_vs_bh"]
    )
    idx = candidates.groupby(["artifact_fraction", "logit_boost", "budget_label"])["phase_score"].idxmax()
    cols = [
        "artifact_fraction",
        "logit_boost",
        "budget_label",
        "method",
        "true_hits",
        "fdp",
        "unique_blocks",
        "max_block_share",
        "delta_true_hits_vs_bh",
        "delta_fdp_vs_bh",
        "delta_unique_blocks_vs_bh",
        "delta_max_block_share_vs_bh",
        "phase_score",
        "n_target_reps",
    ]
    return candidates.loc[idx, cols].sort_values(["artifact_fraction", "logit_boost", "budget_label"])


def parse_external_dataset_name(dataset: object) -> tuple[str, str]:
    text = str(dataset)
    if ":" in text:
        left, right = text.split(":", 1)
        return left, right
    return text, ""


def external_vs_summary(root: Path, dirs: list[str], *, q: float) -> pd.DataFrame:
    rows = []
    for entry in dirs:
        path = root / entry / "latest_results.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        parsed = df["dataset"].map(parse_external_dataset_name)
        df["external_dataset"] = [item[0] for item in parsed]
        df["source_backbone"] = [item[1] for item in parsed]
        df["selection_dir"] = entry
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    all_results = pd.concat(rows, ignore_index=True)
    all_results["budget_label"] = all_results["budget_label"].astype(str)
    all_results = all_results[all_results["q"] == q].copy()
    keys = ["external_dataset", "source_backbone", "q", "budget_label", "method"]
    mean = all_results.groupby(keys, as_index=False)[SUMMARY_METRICS].mean()
    std = all_results.groupby(keys, as_index=False)[SUMMARY_METRICS].std().rename(
        columns={metric: f"{metric}_std" for metric in SUMMARY_METRICS}
    )
    counts = all_results.groupby(keys, as_index=False)["target_chembl_id"].nunique().rename(
        columns={"target_chembl_id": "n_targets"}
    )
    summary = mean.merge(std, on=keys, how="left").merge(counts, on=keys, how="left")
    summary = add_delta_vs_baseline(
        summary,
        keys=["external_dataset", "source_backbone", "q", "budget_label"],
        baseline="raw_top_b",
        metrics=["true_hits", "fdp", "unique_blocks", "max_block_share", "mean_pairwise_tanimoto"],
    )
    summary = add_delta_vs_baseline(
        summary,
        keys=["external_dataset", "source_backbone", "q", "budget_label"],
        baseline="bh",
        metrics=["true_hits", "fdp", "unique_blocks", "max_block_share", "mean_pairwise_tanimoto"],
    )
    keep_methods = [
        "raw_top_b",
        "bh",
        "by",
        "weighted_bh",
        "block_bh",
        "score_cap1",
        "score_mmr25",
        "score_dpp",
        "chemdeprc_cap1",
        "chemdeprc_soft75",
        "chemdeprc_scoresoft50",
    ]
    summary = summary[summary["method"].isin(keep_methods)].copy()
    return summary.sort_values(["external_dataset", "source_backbone", "budget_label", "method"])


def dependency_selection_summary(root: Path, dirs: list[str], *, q: float) -> pd.DataFrame:
    rows = []
    for entry in dirs:
        path = root / entry / "latest_summary.csv"
        meta_path = root / entry / "metadata.json"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        definition = entry.rstrip("/").split("/")[-1].replace("jctc_dependency_", "")
        df["definition"] = definition
        df["selection_block_col"] = metadata.get("selection_block_col", "")
        df["artifact_block_col"] = metadata.get("artifact_block_col", "")
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    summary = pd.concat(rows, ignore_index=True)
    summary["budget_label"] = summary["budget_label"].astype(str)
    keys = [
        "definition",
        "selection_block_col",
        "artifact_block_col",
        "stress_label",
        "artifact_fraction",
        "logit_boost",
        "block_perturbation",
        "q",
        "budget_label",
    ]
    summary = add_delta_vs_baseline(
        summary,
        keys=keys,
        baseline="bh",
        metrics=["true_hits", "fdp", "unique_blocks", "max_block_share", "mean_pairwise_tanimoto"],
    )
    keep_methods = ["bh", "weighted_bh", "chemdeprc_cap1", "chemdeprc_soft75", "chemdeprc_scoresoft50"]
    cols = [
        "definition",
        "selection_block_col",
        "artifact_fraction",
        "logit_boost",
        "budget_label",
        "method",
        "true_hits",
        "fdp",
        "unique_blocks",
        "max_block_share",
        "mean_pairwise_tanimoto",
        "delta_true_hits_vs_bh",
        "delta_fdp_vs_bh",
        "delta_unique_blocks_vs_bh",
        "delta_max_block_share_vs_bh",
        "delta_mean_pairwise_tanimoto_vs_bh",
        "n_target_reps",
    ]
    filtered = summary[
        (summary["q"] == q)
        & (summary["block_perturbation"] == "none")
        & (summary["artifact_fraction"].isin([0.0, 0.1]))
        & (summary["logit_boost"].isin([0.0, 4.0]))
        & (summary["budget_label"].isin(["50", "100"]))
        & (summary["method"].isin(keep_methods))
    ].copy()
    return filtered[cols].sort_values(["artifact_fraction", "logit_boost", "budget_label", "definition", "method"])


def make_audit(root: Path) -> pd.DataFrame:
    checks = []
    table_exists = lambda p: (root / p).exists()
    data_exists = lambda p: (root / p).exists()
    checks.append(
        {
            "jctc_requirement": "LIT-PCBA/DUD-E or structure-oriented public VS benchmark",
            "status": "partial" if table_exists("tables/jctc_external_vs_score_backbone_q0.7.csv") else "missing",
            "evidence": "LIT-PCBA and DUD-E external score-backbone panels are converted and selected under one protocol; no full claim-safe docking benchmark yet",
        }
    )
    checks.append(
        {
            "jctc_requirement": "multiple scoring backbones",
            "status": "partial" if table_exists("tables/jctc_external_vs_score_backbone_q0.7.csv") else "missing",
            "evidence": "ECFP-XGBoost plus external full and ligand-only logits; docking-score backbone remains outside the claim boundary",
        }
    )
    checks.append(
        {
            "jctc_requirement": "multiple dependence definitions",
            "status": "pass" if table_exists("tables/jctc_dependency_definition_selection_q0.7.csv") else "partial",
            "evidence": "Murcko, Tanimoto40/50/60/70/80, hybrid75, and analog-series proxy definitions with block stats and selection outcomes",
        }
    )
    checks.append(
        {
            "jctc_requirement": "budget x dependence x cap-strength phase diagram tables",
            "status": "pass" if table_exists("tables/jctc_phase_strategy_q0.7.csv") else "missing",
            "evidence": "continuous budgets and artifact strength grid from JCTC stress runner",
        }
    )
    checks.append(
        {
            "jctc_requirement": "block misspecification robustness",
            "status": "pass" if table_exists("tables/jctc_block_misspec_q0.7.csv") else "missing",
            "evidence": "split, merge, missing-edge, and false-edge perturbations evaluated against true Murcko blocks",
        }
    )
    checks.append(
        {
            "jctc_requirement": "ALBF / modern finite-budget competitor",
            "status": "partial" if table_exists("tables/jctc_active_learning_replay_q0.7.csv") else "missing",
            "evidence": "one-shot diversity/FDR competitors complete; sequential ALBF remains a separate protocol unless active-learning replay table exists",
        }
    )
    checks.append(
        {
            "jctc_requirement": "runtime / scalability",
            "status": "pass" if table_exists("tables/jctc_runtime_scalability.csv") else "missing",
            "evidence": "selection runtime across synthetic N and ChEMBL panel sizes",
        }
    )
    checks.append(
        {
            "jctc_requirement": "paired bootstrap CI and frontier probability",
            "status": "pass" if table_exists("tables/jctc_paired_bootstrap_ci_q0.7.csv") else "missing",
            "evidence": "paired target/rep bootstrap CIs and P(non-dominated) tables",
        }
    )
    checks.append(
        {
            "jctc_requirement": "formal properties",
            "status": "documented" if table_exists("literature/jctc_experiment_package_notes.md") else "missing",
            "evidence": "independence reduction, hard-cap feasible set, and soft-cap limiting behavior noted for manuscript methods/SI",
        }
    )
    return pd.DataFrame(checks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-summary", default="data/jctc_budget_phase/latest_summary.csv")
    parser.add_argument("--stress-results", default="data/jctc_budget_phase/latest_results.csv")
    parser.add_argument("--misspec-summary", default="data/jctc_block_misspec/latest_summary.csv")
    parser.add_argument("--misspec-results", default="data/jctc_block_misspec/latest_results.csv")
    parser.add_argument("--al-summary", default="data/jctc_active_learning_replay/latest_summary.csv")
    parser.add_argument("--al-results", default="data/jctc_active_learning_replay/latest_results.csv")
    parser.add_argument("--runtime-summary", default="data/jctc_scalability/latest_summary.csv")
    parser.add_argument("--block-stat-dirs", nargs="*", default=[])
    parser.add_argument("--dependency-selection-dirs", nargs="*", default=[])
    parser.add_argument("--external-selection-dirs", nargs="*", default=[])
    parser.add_argument("--outdir", default="tables")
    parser.add_argument("--q", type=float, default=0.7)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3579)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    rng_seed = args.seed

    stress_summary_path = root / args.stress_summary
    stress_results_path = root / args.stress_results
    if stress_summary_path.exists():
        stress_summary = pd.read_csv(stress_summary_path)
        stress_summary["budget_label"] = stress_summary["budget_label"].astype(str)
        phase = phase_strategy(stress_summary)
        write_pair(phase, outdir, f"jctc_phase_strategy_q{args.q:g}")
        stress_delta = add_delta_vs_bh(stress_summary)
        key_budget = stress_delta[
            (stress_delta["q"] == args.q)
            & (stress_delta["block_perturbation"] == "none")
            & (stress_delta["method"].isin(KEY_METHODS))
            & (
                stress_delta["budget_label"].astype(str).isin(["10", "20", "50", "100", "200", "500"])
            )
            & (
                (stress_delta["artifact_fraction"].isin([0.0, 0.1, 0.2, 0.3]))
                & (stress_delta["logit_boost"].isin([0.0, 2.0, 4.0, 6.0]))
            )
        ].copy()
        cols = [
            "stress_label",
            "artifact_fraction",
            "logit_boost",
            "budget_label",
            "method",
            "true_hits",
            "fdp",
            "fdp_exceeds_q",
            "unique_blocks",
            "max_block_share",
            "mean_pairwise_tanimoto",
            "delta_true_hits_vs_bh",
            "delta_fdp_vs_bh",
            "delta_unique_blocks_vs_bh",
            "delta_max_block_share_vs_bh",
            "n_target_reps",
        ]
        write_pair(key_budget[cols].sort_values(["artifact_fraction", "logit_boost", "budget_label", "method"]), outdir, f"jctc_budget_dependence_grid_q{args.q:g}")

    if stress_results_path.exists():
        stress_results = pd.read_csv(stress_results_path)
        stress_results["budget_label"] = stress_results["budget_label"].astype(str)
        focus = stress_results[
            (stress_results["q"] == args.q)
            & (stress_results["block_perturbation"] == "none")
            & (stress_results["artifact_fraction"].isin([0.1, 0.2]))
            & (stress_results["logit_boost"].isin([4.0, 6.0]))
            & (stress_results["budget_label"].astype(str).isin(["50", "100", "200"]))
            & (stress_results["method"].isin(["bh", "weighted_bh", "score_mmr25", "score_dpp"] + CHEMDEP_METHODS))
        ].copy()
        ci = paired_bootstrap_table(
            focus,
            baseline="bh",
            methods=["weighted_bh", "score_mmr25", "score_dpp"] + CHEMDEP_METHODS,
            metrics=METRICS,
            group_cols=["artifact_fraction", "logit_boost", "budget_label"],
            n_boot=args.n_boot,
            seed=rng_seed,
        )
        write_pair(ci, outdir, f"jctc_paired_bootstrap_ci_q{args.q:g}")
        frontier = frontier_probabilities(
            focus,
            group_cols=["artifact_fraction", "logit_boost", "budget_label"],
            methods=["bh", "weighted_bh", "score_mmr25", "score_dpp"] + CHEMDEP_METHODS,
            n_boot=args.n_boot,
            seed=rng_seed + 1,
        )
        write_pair(frontier, outdir, f"jctc_frontier_probability_q{args.q:g}")

    misspec_summary_path = root / args.misspec_summary
    misspec_results_path = root / args.misspec_results
    if misspec_summary_path.exists():
        misspec_summary = pd.read_csv(misspec_summary_path)
        misspec_summary["budget_label"] = misspec_summary["budget_label"].astype(str)
        misspec_delta = add_delta_vs_bh(misspec_summary)
        misspec = misspec_delta[
            (misspec_delta["q"] == args.q)
            & (misspec_delta["artifact_fraction"] == 0.1)
            & (misspec_delta["logit_boost"] == 4.0)
            & (misspec_delta["budget_label"].astype(str).isin(["50", "100"]))
            & (misspec_delta["method"].isin(["weighted_bh", "chemdeprc_cap1", "chemdeprc_soft75", "chemdeprc_scoresoft50"]))
        ].copy()
        cols = [
            "block_perturbation",
            "budget_label",
            "method",
            "true_hits",
            "fdp",
            "unique_blocks",
            "max_block_share",
            "delta_true_hits_vs_bh",
            "delta_fdp_vs_bh",
            "delta_unique_blocks_vs_bh",
            "delta_max_block_share_vs_bh",
            "n_target_reps",
        ]
        write_pair(misspec[cols].sort_values(["block_perturbation", "budget_label", "method"]), outdir, f"jctc_block_misspec_q{args.q:g}")
    if misspec_results_path.exists():
        misspec_results = pd.read_csv(misspec_results_path)
        misspec_results["budget_label"] = misspec_results["budget_label"].astype(str)
        misspec_focus = misspec_results[
            (misspec_results["q"] == args.q)
            & (misspec_results["artifact_fraction"] == 0.1)
            & (misspec_results["logit_boost"] == 4.0)
            & (misspec_results["budget_label"].astype(str).isin(["50", "100"]))
            & (misspec_results["method"].isin(["bh", "weighted_bh", "chemdeprc_cap1", "chemdeprc_soft75", "chemdeprc_scoresoft50"]))
        ].copy()
        misspec_ci = paired_bootstrap_table(
            misspec_focus,
            baseline="bh",
            methods=["weighted_bh", "chemdeprc_cap1", "chemdeprc_soft75", "chemdeprc_scoresoft50"],
            metrics=METRICS,
            group_cols=["block_perturbation", "budget_label"],
            n_boot=args.n_boot,
            seed=rng_seed + 2,
        )
        write_pair(misspec_ci, outdir, f"jctc_block_misspec_bootstrap_ci_q{args.q:g}")

    block_rows = []
    for entry in args.block_stat_dirs:
        path = root / entry / "block_stats.csv"
        meta_path = root / entry / "metadata.json"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        df["definition_dir"] = entry
        df["block_definition"] = meta.get("block_definition", entry)
        block_rows.append(df)
    if block_rows:
        blocks = pd.concat(block_rows, ignore_index=True)
        testpool = blocks[blocks["split"] == "testpool"].copy()
        summary = (
            testpool.groupby(["definition_dir", "block_col", "mode", "block_definition"], dropna=False)
            .agg(
                n_targets=("target_chembl_id", "nunique"),
                mean_n_blocks=("n_blocks", "mean"),
                mean_max_block_size=("max_block_size", "mean"),
                mean_singleton_rate=("singleton_rate", "mean"),
                mean_median_block_size=("median_block_size", "mean"),
            )
            .reset_index()
        )
        write_pair(summary, outdir, "jctc_dependency_definition_summary")

    dependency_selection = dependency_selection_summary(root, args.dependency_selection_dirs, q=args.q)
    if not dependency_selection.empty:
        write_pair(dependency_selection, outdir, f"jctc_dependency_definition_selection_q{args.q:g}")

    external_selection = external_vs_summary(root, args.external_selection_dirs, q=args.q)
    if not external_selection.empty:
        write_pair(external_selection, outdir, f"jctc_external_vs_score_backbone_q{args.q:g}")

    al_summary_path = root / args.al_summary
    if al_summary_path.exists():
        al = pd.read_csv(al_summary_path)
        final_round = int(al["round"].max())
        final = al[al["round"] == final_round].copy()
        base = final[final["strategy"] == "score"][
            [
                "batch_size",
                "cumulative_budget",
                "true_hits",
                "fdp",
                "unique_blocks",
                "max_block_share",
                "mean_pairwise_tanimoto",
            ]
        ]
        merged = final.merge(base, on=["batch_size", "cumulative_budget"], suffixes=("", "_score"), how="left")
        for metric in ["true_hits", "fdp", "unique_blocks", "max_block_share", "mean_pairwise_tanimoto"]:
            merged[f"delta_{metric}_vs_score"] = merged[metric] - merged[f"{metric}_score"]
        cols = [
            "strategy",
            "round",
            "batch_size",
            "cumulative_budget",
            "true_hits",
            "fdp",
            "unique_blocks",
            "max_block_share",
            "mean_pairwise_tanimoto",
            "delta_true_hits_vs_score",
            "delta_fdp_vs_score",
            "delta_unique_blocks_vs_score",
            "delta_max_block_share_vs_score",
            "n_targets",
        ]
        write_pair(merged[cols].sort_values("strategy"), outdir, f"jctc_active_learning_replay_q{args.q:g}")

    runtime_summary_path = root / args.runtime_summary
    if runtime_summary_path.exists():
        runtime = pd.read_csv(runtime_summary_path)
        runtime["seconds_per_100k"] = runtime["seconds_mean"] / runtime["n_candidates"] * 100000.0
        write_pair(runtime.sort_values(["n_candidates", "method"]), outdir, "jctc_runtime_scalability")

    audit = make_audit(root)
    write_pair(audit, outdir, "jctc_minimum_package_audit")
    print(f"Wrote JCTC tables to {outdir}")


if __name__ == "__main__":
    main()
