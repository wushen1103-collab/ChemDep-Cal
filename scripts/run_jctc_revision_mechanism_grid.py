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

from run_chembl_block_artifact_stress import (
    choose_artifact_blocks,
    conformal_right_tail_pvalues,
    fingerprint_matrix,
    inject_artifact,
    parse_budget,
    select_real_method,
)
from run_jctc_stress_grid import evaluate_selection_fast, pairwise_tanimoto_matrix
from run_chembl_selection import FP_COL, block_array


DEFAULT_METHODS = (
    "bh",
    "by",
    "weighted_bh",
    "block_bh",
    "score_cap1",
    "bh_cap1",
    "block_by_cap1",
    "minp_block_bh_cap1",
    "hier_bh_cap1",
    "chemdeprc_cap1",
    "chemdeprc_scorecap1",
    "chemdeprc_soft75",
)


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
        max_tanimoto_pairs,
        seed,
    ) = task
    scores_path = Path(scores_path)
    scores = pd.read_csv(scores_path)
    target_id = str(scores["target_chembl_id"].iloc[0])
    pref_name = str(scores["pref_name"].iloc[0])
    dataset = str(scores["dataset"].iloc[0]) if "dataset" in scores.columns else "ChEMBL"
    scenario = str(scores["scenario"].iloc[0]) if "scenario" in scores.columns else "jctc_revision_mechanism"
    panel_seed = scores["panel_seed"].iloc[0] if "panel_seed" in scores.columns else ""
    calibration = scores[scores["split"] == "calibration"].copy()
    testpool = scores[scores["split"] == "testpool"].copy().reset_index(drop=True)
    null_scores = calibration.loc[calibration["label"] == 0, "score"].to_numpy(dtype=float)
    if len(null_scores) == 0 or len(testpool) == 0:
        return []

    labels = testpool["label"].to_numpy(dtype=np.int8)
    base_scores = testpool["score"].to_numpy(dtype=float)
    eval_blocks = block_array(testpool, artifact_block_col)
    selection_blocks = block_array(testpool, selection_block_col)
    bits = fingerprint_matrix(testpool[FP_COL])
    sim = pairwise_tanimoto_matrix(bits)
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
                                blocks=selection_blocks,
                                bits=bits,
                                q=q,
                                budget=budget,
                                seed_parts=(target_id, dataset, scenario, panel_seed, rep, stress_label, q, budget_label),
                            )
                            metrics = evaluate_selection_fast(
                                labels=labels,
                                scores=perturbed_scores,
                                eval_blocks=eval_blocks,
                                selection_blocks=selection_blocks,
                                sim=sim,
                                selected=result.selected,
                                budget=budget,
                                q=q,
                                max_tanimoto_pairs=max_tanimoto_pairs,
                                seed_parts=(target_id, rep, stress_label, q, budget_label, method),
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
                                    "block_perturbation": "none",
                                    "artifact_block_count": len(artifact_blocks),
                                    "artifact_molecules": artifact_molecules,
                                    "artifact_inactive_molecules": artifact_inactive_molecules,
                                    "q": q,
                                    "budget_label": str(budget_label),
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


def count_csv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def run_target_to_file(task: tuple) -> tuple[str, int]:
    task_id, core_task, partial_dir, resume_existing = task
    partial_dir = Path(partial_dir)
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partial_dir / f"target_{int(task_id):03d}.csv"
    if resume_existing and partial_path.exists() and partial_path.stat().st_size > 0:
        return str(partial_path), count_csv_data_rows(partial_path)
    rows = run_target(core_task)
    pd.DataFrame(rows).to_csv(partial_path, index=False)
    return str(partial_path), len(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
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
        "selection_unique_blocks",
        "selection_max_block_share",
        "mean_pairwise_tanimoto",
        "empty",
        "pool_active_rate",
        "n_calib_null",
        "pvalue_floor",
        "pvalue_min",
    ]
    keys = ["stress_label", "artifact_fraction", "logit_boost", "block_perturbation", "q", "budget_label", "method"]
    mean = results.groupby(keys, as_index=False)[metrics].mean()
    std = results.groupby(keys, as_index=False)[metrics].std().rename(columns={m: f"{m}_std" for m in metrics})
    counts = results.groupby(keys, as_index=False).size().rename(columns={"size": "n_target_reps"})
    targets = results.groupby(keys, as_index=False)["target_chembl_id"].nunique().rename(
        columns={"target_chembl_id": "n_targets"}
    )
    return mean.merge(std, on=keys, how="left").merge(counts, on=keys, how="left").merge(targets, on=keys, how="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", default="data/chembl_ecfp_xgb_potent_a7_i6/scores")
    parser.add_argument("--outdir", default="data/jctc_revision_mechanism")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--artifact-fractions", nargs="+", type=float, default=[0.0, 0.025, 0.05, 0.1, 0.2])
    parser.add_argument("--logit-boosts", nargs="+", type=float, default=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0])
    parser.add_argument("--q-values", nargs="+", type=float, default=[0.01, 0.025, 0.05, 0.1, 0.2, 0.7])
    parser.add_argument("--budgets", nargs="+", default=["20", "30", "50", "75", "100"])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--selection-block-col", default="murcko_scaffold")
    parser.add_argument("--artifact-block-col", default="murcko_scaffold")
    parser.add_argument("--max-artifact-blocks", type=int, default=25)
    parser.add_argument("--min-inactive-per-block", type=int, default=2)
    parser.add_argument("--max-tanimoto-pairs", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--leave-cpus-free", type=int, default=30)
    parser.add_argument("--seed", type=int, default=353536)
    parser.add_argument("--resume-existing-partials", action="store_true")
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

    core_tasks = [
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
            args.max_tanimoto_pairs,
            args.seed + i * 100_000,
        )
        for i, path in enumerate(score_paths)
    ]
    partial_dir = outdir / "partials"
    tasks = [(i, task, partial_dir, args.resume_existing_partials) for i, task in enumerate(core_tasks)]
    partial_paths: list[str] = []
    row_count = 0
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_target_to_file, task) for task in tasks]
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="jctc revision mechanism"):
            partial_path, n_rows = fut.result()
            partial_paths.append(partial_path)
            row_count += n_rows

    frames = [pd.read_csv(path) for path in sorted(partial_paths)]
    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if results.empty:
        raise RuntimeError("No revision mechanism rows were produced")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = outdir / f"jctc_revision_mechanism_results_{stamp}.csv"
    summary_path = outdir / f"jctc_revision_mechanism_summary_{stamp}.csv"
    results.to_csv(results_path, index=False)
    results.to_csv(outdir / "latest_results.csv", index=False)
    summary = summarize(results)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(outdir / "latest_summary.csv", index=False)
    metadata = {
        "purpose": "Direct JCTC revision tests: ChemDep-RC vs score_cap1, nominal alpha calibration, structured FDR competitors, and frozen budget policy support.",
        "scores_dir": args.scores_dir,
        "outdir": args.outdir,
        "targets": len(score_paths),
        "reps": args.reps,
        "artifact_fractions": artifact_fractions,
        "logit_boosts": logit_boosts,
        "q_values": args.q_values,
        "budgets": args.budgets,
        "methods": args.methods,
        "workers": workers,
        "resume_existing_partials": bool(args.resume_existing_partials),
        "partial_rows": int(row_count),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary.head(80).to_string(index=False))
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
