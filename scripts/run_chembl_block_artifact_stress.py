#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from run_chembl_selection import (
    DEFAULT_METHODS,
    FP_COL,
    block_array,
    conformal_right_tail_pvalues,
    evaluate_real_selection,
    fingerprint_matrix,
    parse_budget,
    select_real_method,
    summarize,
)


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def expit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def choose_artifact_blocks(
    testpool: pd.DataFrame,
    *,
    artifact_block_col: str,
    rng: np.random.Generator,
    artifact_fraction: float,
    max_artifact_blocks: int,
    min_inactive_per_block: int,
) -> list[str]:
    if artifact_fraction <= 0:
        return []
    if artifact_block_col not in testpool.columns:
        raise ValueError(f"Artifact block column {artifact_block_col!r} is missing from scores file")
    block_stats = (
        testpool.assign(is_inactive=(testpool["label"] == 0).astype(int))
        .groupby(artifact_block_col, as_index=False)
        .agg(block_size=("label", "size"), inactive_count=("is_inactive", "sum"))
    )
    candidates = block_stats[block_stats["inactive_count"] >= min_inactive_per_block].copy()
    if candidates.empty:
        candidates = block_stats[block_stats["inactive_count"] > 0].copy()
    if candidates.empty:
        return []
    n_blocks = int(math.ceil(len(candidates) * artifact_fraction))
    n_blocks = max(1, min(n_blocks, max_artifact_blocks, len(candidates)))
    weights = candidates["inactive_count"].to_numpy(dtype=float)
    weights = weights / weights.sum()
    chosen = rng.choice(candidates[artifact_block_col].to_numpy(), size=n_blocks, replace=False, p=weights)
    return [str(item) for item in chosen]


def inject_artifact(
    testpool: pd.DataFrame,
    base_scores: np.ndarray,
    artifact_blocks: list[str],
    *,
    artifact_block_col: str,
    logit_boost: float,
) -> tuple[np.ndarray, int, int]:
    if not artifact_blocks or logit_boost <= 0:
        return base_scores.copy(), 0, 0
    if artifact_block_col not in testpool.columns:
        raise ValueError(f"Artifact block column {artifact_block_col!r} is missing from scores file")
    block_mask = testpool[artifact_block_col].astype(str).isin(artifact_blocks).to_numpy()
    inactive_mask = testpool["label"].to_numpy(dtype=np.int8) == 0
    artifact_mask = block_mask & inactive_mask
    perturbed = base_scores.copy()
    perturbed[artifact_mask] = expit(logit(perturbed[artifact_mask]) + logit_boost)
    return perturbed, int(block_mask.sum()), int(artifact_mask.sum())


def run_target(task: tuple) -> list[dict]:
    (
        scores_path,
        q_values,
        budget_labels,
        methods,
        reps,
        artifact_fractions,
        logit_boosts,
        max_artifact_blocks,
        min_inactive_per_block,
        selection_block_col,
        artifact_block_col,
        seed,
    ) = task
    scores_path = Path(scores_path)
    scores = pd.read_csv(scores_path)
    target_id = str(scores["target_chembl_id"].iloc[0])
    pref_name = str(scores["pref_name"].iloc[0])
    dataset = str(scores["dataset"].iloc[0]) if "dataset" in scores.columns else "ChEMBL"
    scenario = str(scores["scenario"].iloc[0]) if "scenario" in scores.columns else "block_artifact_stress"
    panel_seed = scores["panel_seed"].iloc[0] if "panel_seed" in scores.columns else ""
    calibration = scores[scores["split"] == "calibration"].copy()
    testpool = scores[scores["split"] == "testpool"].copy().reset_index(drop=True)
    null_scores = calibration.loc[calibration["label"] == 0, "score"].to_numpy(dtype=float)
    if len(null_scores) == 0 or len(testpool) == 0:
        return []

    labels = testpool["label"].to_numpy(dtype=np.int8)
    base_scores = testpool["score"].to_numpy(dtype=float)
    blocks = block_array(testpool, selection_block_col)
    bits = fingerprint_matrix(testpool[FP_COL])
    rows: list[dict] = []
    target_seed = seed + int(hashlib.sha256(target_id.encode("utf-8")).hexdigest()[:12], 16) % 1_000_000
    for rep in range(reps):
        for artifact_fraction in artifact_fractions:
            for logit_boost in logit_boosts:
                if artifact_fraction == 0.0 and logit_boost != 0.0:
                    continue
                if artifact_fraction > 0.0 and logit_boost == 0.0:
                    continue
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(artifact_fraction * 10_000))
                artifact_blocks = choose_artifact_blocks(
                    testpool,
                    artifact_block_col=artifact_block_col,
                    rng=rng,
                    artifact_fraction=artifact_fraction,
                    max_artifact_blocks=max_artifact_blocks,
                    min_inactive_per_block=min_inactive_per_block,
                )
                perturbed_scores, artifact_molecules, artifact_inactive_molecules = inject_artifact(
                    testpool,
                    base_scores,
                    artifact_blocks,
                    artifact_block_col=artifact_block_col,
                    logit_boost=logit_boost,
                )
                pvalues = conformal_right_tail_pvalues(perturbed_scores, null_scores)
                stress_label = (
                    "no_artifact"
                    if artifact_fraction == 0.0
                    else f"frac{artifact_fraction:g}_boost{logit_boost:g}"
                )
                for q in q_values:
                    for budget_label in budget_labels:
                        budget = parse_budget(budget_label, len(testpool))
                        for method in methods:
                            result = select_real_method(
                                method,
                                scores=perturbed_scores,
                                pvalues=pvalues,
                                blocks=blocks,
                                bits=bits,
                                q=q,
                                budget=budget,
                                seed_parts=(target_id, dataset, scenario, panel_seed, rep, stress_label, q, budget_label),
                            )
                            metrics = evaluate_real_selection(
                                labels=labels,
                                scores=perturbed_scores,
                                blocks=blocks,
                                bits=bits,
                                selected=result.selected,
                                budget=budget,
                                q=q,
                            )
                            rows.append(
                                {
                                    "target_chembl_id": target_id,
                                    "pref_name": pref_name,
                                    "dataset": dataset,
                                    "scenario": scenario,
                                    "panel_seed": panel_seed,
                                    "rep": rep,
                                    "stress_label": stress_label,
                                    "artifact_fraction": artifact_fraction,
                                    "logit_boost": logit_boost,
                                    "artifact_block_count": len(artifact_blocks),
                                    "artifact_molecules": artifact_molecules,
                                    "artifact_inactive_molecules": artifact_inactive_molecules,
                                    "q": q,
                                    "budget_label": budget_label,
                                    "budget": budget,
                                    "method": method,
                                    "selection_block_col": selection_block_col,
                                    "artifact_block_col": artifact_block_col,
                                    "certificate": result.certificate,
                                    "n_testpool": int(len(testpool)),
                                    "n_testpool_active": int(np.sum(labels)),
                                    "n_testpool_inactive": int(len(labels) - np.sum(labels)),
                                    "n_calibration": int(len(calibration)),
                                    "n_calib_null": int(len(null_scores)),
                                    "pvalue_floor": float(1.0 / (len(null_scores) + 1.0)),
                                    "pvalue_min": float(np.min(pvalues)),
                                    "pvalue_median": float(np.median(pvalues)),
                                    **metrics,
                                }
                            )
    return rows


def summarize_stress(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "budget",
        "artifact_block_count",
        "artifact_molecules",
        "artifact_inactive_molecules",
        "selected_count",
        "true_hits",
        "false_hits",
        "fdp",
        "fdp_exceeds_q",
        "precision",
        "power",
        "enrichment_factor",
        "unique_blocks",
        "max_block_share",
        "mean_pairwise_tanimoto",
        "empty",
        "pool_active_rate",
        "n_calib_null",
        "pvalue_floor",
        "pvalue_min",
    ]
    keys = ["stress_label", "artifact_fraction", "logit_boost", "q", "budget_label", "method"]
    grouped = results.groupby(keys, as_index=False)[metrics]
    mean = grouped.mean()
    std = grouped.std().rename(columns={m: f"{m}_std" for m in metrics})
    target_counts = results.groupby(keys, as_index=False)["target_chembl_id"].nunique().rename(
        columns={"target_chembl_id": "n_targets"}
    )
    unit_counts = results.groupby(keys, as_index=False).size().rename(columns={"size": "n_target_reps"})
    return mean.merge(std, on=keys, how="left").merge(target_counts, on=keys, how="left").merge(
        unit_counts, on=keys, how="left"
    )


def append_log(root: Path, metadata: dict, summary_path: Path, results_path: Path) -> None:
    log_path = root / "EXPERIMENT_LOG.md"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now} ChEMBL block-artifact stress\n\n")
        f.write("Metadata:\n\n")
        f.write("```json\n")
        f.write(json.dumps(metadata, indent=2, sort_keys=True))
        f.write("\n```\n\n")
        f.write(f"- Raw target-replicate results: `{results_path}`\n")
        f.write(f"- Aggregate summary: `{summary_path}`\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", default="data/chembl_ecfp_xgb_potent_a7_i6/scores")
    parser.add_argument("--outdir", default="data/chembl_block_artifact_potent_a7_i6")
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--artifact-fractions", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.1])
    parser.add_argument("--logit-boosts", nargs="+", type=float, default=[0.0, 2.0, 4.0])
    parser.add_argument("--max-artifact-blocks", type=int, default=25)
    parser.add_argument("--min-inactive-per-block", type=int, default=2)
    parser.add_argument("--q-values", nargs="+", type=float, default=[0.5, 0.7, 0.8, 0.9])
    parser.add_argument("--budgets", nargs="+", default=["25", "50", "100"])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--selection-block-col", default="murcko_scaffold")
    parser.add_argument("--artifact-block-col", default="murcko_scaffold")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--leave-cpus-free", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3535)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    scores_dir = root / args.scores_dir
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    score_paths = sorted(scores_dir.glob("*_scores.csv"))
    if not score_paths:
        raise RuntimeError(f"No score files found under {scores_dir}")
    usable_cpus = max(1, (os.cpu_count() or 4) - args.leave_cpus_free)
    workers = args.workers or min(len(score_paths), usable_cpus)
    workers = max(1, min(workers, len(score_paths), usable_cpus))
    artifact_fractions = sorted(set(args.artifact_fractions))
    logit_boosts = sorted(set(args.logit_boosts))
    if 0.0 not in artifact_fractions:
        artifact_fractions = [0.0] + artifact_fractions
    if 0.0 not in logit_boosts:
        logit_boosts = [0.0] + logit_boosts

    tasks = [
        (
            path,
            args.q_values,
            args.budgets,
            args.methods,
            args.reps,
            artifact_fractions,
            logit_boosts,
            args.max_artifact_blocks,
            args.min_inactive_per_block,
            args.selection_block_col,
            args.artifact_block_col,
            args.seed + i * 100_000,
        )
        for i, path in enumerate(score_paths)
    ]
    rows: list[dict] = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_target, task) for task in tasks]
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="block artifact"):
            rows.extend(fut.result())
    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No block-artifact stress rows were produced")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = outdir / f"block_artifact_results_{stamp}.csv"
    summary_path = outdir / f"block_artifact_summary_{stamp}.csv"
    latest_results = outdir / "latest_results.csv"
    latest_summary = outdir / "latest_summary.csv"
    results.to_csv(results_path, index=False)
    results.to_csv(latest_results, index=False)
    summary_df = summarize_stress(results)
    summary_df.to_csv(summary_path, index=False)
    summary_df.to_csv(latest_summary, index=False)
    metadata = {
        "scores_dir": args.scores_dir,
        "outdir": args.outdir,
        "reps": args.reps,
        "targets": len(score_paths),
        "artifact_fractions": artifact_fractions,
        "logit_boosts": logit_boosts,
        "max_artifact_blocks": args.max_artifact_blocks,
        "min_inactive_per_block": args.min_inactive_per_block,
        "q_values": args.q_values,
        "budgets": args.budgets,
        "methods": args.methods,
        "selection_block_col": args.selection_block_col,
        "artifact_block_col": args.artifact_block_col,
        "workers": workers,
        "stress_design": "random inactive artifact-block molecules receive a positive logit boost in the testpool only",
        "formal_warning": "stress uses retrospective labels to inject errors; it evaluates mechanism robustness, not a deployable unsupervised detector",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_log(root, metadata, summary_path.relative_to(root), results_path.relative_to(root))
    print(summary_df.to_markdown(index=False, floatfmt=".4f"))
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
