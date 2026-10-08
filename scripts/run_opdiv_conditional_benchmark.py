#!/usr/bin/env python3
"""Run the author's OPDiv implementation on the frozen ChEMBL stress grid."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from opdiv import select


def one_target(task: tuple) -> list[dict]:
    (
        path, target_index, reps, thresholds, time_limit,
        expected, root, fraction, boost, budget_label, base_seed, sanity_method,
    ) = task
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from run_chembl_selection import (
        FP_COL, block_array, fingerprint_matrix, parse_budget, select_real_method,
    )
    from run_jctc_stress_grid import evaluate_selection_fast, pairwise_tanimoto_matrix

    frame = pd.read_csv(path)
    target = str(frame["target_chembl_id"].iloc[0])
    testpool = frame[frame["split"] == "testpool"].copy().reset_index(drop=True)
    labels = testpool["label"].to_numpy(dtype=np.int8)
    base_scores = testpool["score"].to_numpy(dtype=float)
    blocks = block_array(testpool, "murcko_scaffold")
    bits = fingerprint_matrix(testpool[FP_COL])
    sim = pairwise_tanimoto_matrix(bits)
    budget = parse_budget(str(budget_label), len(testpool))
    conflicts = {}
    for threshold in thresholds:
        matrix = sim > threshold
        np.fill_diagonal(matrix, False)
        conflicts[threshold] = matrix

    target_seed = base_seed + target_index * 100_000 + (
        int(hashlib.sha256(target.encode("utf-8")).hexdigest()[:12], 16) % 1_000_000
    )
    rows = []
    for rep in range(reps):
        rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
        artifact_blocks = choose_artifact_blocks(
            testpool, artifact_block_col="murcko_scaffold", rng=rng,
            artifact_fraction=fraction, max_artifact_blocks=25,
            min_inactive_per_block=2,
        )
        scores, _, _ = inject_artifact(
            testpool, base_scores, artifact_blocks,
            artifact_block_col="murcko_scaffold", logit_boost=boost,
        )

        key = (target, rep)
        exp = expected.get(key)
        if exp is None:
            raise AssertionError(f"Missing existing grid row {key}")
        if sanity_method == "raw_top_b":
            raw_idx = np.argsort(-scores, kind="mergesort")[:budget]
        else:
            raw_idx = select_real_method(
                sanity_method, scores=scores, pvalues=np.ones(len(scores)),
                blocks=blocks, bits=bits, q=0.7, budget=budget,
                seed_parts=(target, rep, fraction, boost, budget_label),
            ).selected
        raw_hits = float(labels[raw_idx].sum())
        if raw_hits != exp["true_hits"]:
            raise AssertionError(f"Artifact mismatch {key}: {raw_hits} != {exp['true_hits']}")

        for threshold in thresholds:
            started = time.monotonic()
            answer = select(
                scores, k=budget, conflicts=conflicts[threshold],
                time_limit=time_limit,
            )
            elapsed = time.monotonic() - started
            full = len(answer.indices) == budget
            metrics = {}
            if full:
                metrics = evaluate_selection_fast(
                    labels=labels, scores=scores, eval_blocks=blocks,
                    selection_blocks=blocks, sim=sim,
                    selected=np.asarray(answer.indices, dtype=int),
                    budget=budget, q=0.7, max_tanimoto_pairs=5000,
                    seed_parts=(target, rep, fraction, boost, budget_label, threshold),
                )
            rows.append({
                "target_chembl_id": target,
                "rep": rep,
                "artifact_fraction": fraction,
                "logit_boost": boost,
                "q": 0.7,
                "budget_label": str(budget_label),
                "budget": budget,
                "method": f"opdiv_tanimoto{threshold:g}",
                "max_similarity": threshold,
                "status": answer.status,
                "full_selection": full,
                "solver_gap": answer.gap,
                "solver_upper_bound": answer.upper_bound,
                "selected_score_mean": answer.value,
                "elapsed_seconds": elapsed,
                "n_testpool": len(testpool),
                "raw_sanity_hits": raw_hits,
                **metrics,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.7, 0.8, 0.9])
    parser.add_argument("--time-limit", type=float, default=2.0)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--boost", type=float, default=4.0)
    parser.add_argument("--budget", default="50")
    parser.add_argument("--seed", type=int, default=353500)
    parser.add_argument("--source-results", default="data/jctc_budget_phase/latest_results.csv")
    parser.add_argument("--sanity-method", choices=["raw_top_b", "score_cap1"], default="raw_top_b")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    root = args.root.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    score_paths = sorted((root / "data/chembl_ecfp_xgb_potent_a7_i6/scores").glob("*_scores.csv"))
    if len(score_paths) != 10:
        raise AssertionError(f"Expected 10 ChEMBL targets, found {len(score_paths)}")
    if args.workers > (os.cpu_count() or 1) - 30:
        raise ValueError("Reserve at least 30 CPU cores")
    original = pd.read_csv(
        root / args.source_results,
        usecols=["target_chembl_id", "rep", "artifact_fraction", "logit_boost",
                 "block_perturbation", "q", "budget_label", "method", "true_hits"],
    )
    use = original[
        np.isclose(original["artifact_fraction"], args.fraction)
        & np.isclose(original["logit_boost"], args.boost)
        & np.isclose(original["q"], 0.7)
        & (original["block_perturbation"] == "none")
        & (original["budget_label"].astype(str) == str(args.budget))
        & (original["method"] == args.sanity_method)
        & (original["rep"] < args.reps)
    ]
    expected = {
        (str(row.target_chembl_id), int(row.rep)): {"true_hits": float(row.true_hits)}
        for row in use.itertuples(index=False)
    }
    if len(expected) != 10 * args.reps:
        raise AssertionError(f"Expected {10 * args.reps} sanity rows; found {len(expected)}")

    tasks = [
        (path, i, args.reps, args.thresholds, args.time_limit,
         expected, root, args.fraction, args.boost, str(args.budget),
         args.seed, args.sanity_method)
        for i, path in enumerate(score_paths)
    ]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        for part in pool.map(one_target, tasks):
            rows.extend(part)
    results = pd.DataFrame(rows)
    results.to_csv(outdir / "opdiv_raw.csv", index=False)
    summary = results.groupby("method", as_index=False).agg(
        n_rows=("rep", "size"), n_full=("full_selection", "sum"),
        n_optimal=("status", lambda s: int((s == "optimal").sum())),
        true_hits=("true_hits", "mean"), fdp=("fdp", "mean"),
        unique_blocks=("unique_blocks", "mean"),
        max_block_share=("max_block_share", "mean"),
        elapsed_seconds=("elapsed_seconds", "mean"),
    )
    summary.to_csv(outdir / "opdiv_summary.csv", index=False)
    source = root / args.source_results
    manifest = {
        "protocol": "ChEMBL stress-grid score files and deterministic artifact injection, q=0.7",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_results": args.source_results,
        "seed": args.seed,
        "sanity_method": args.sanity_method,
        "selection_implementation": "opdiv package 0.1.0, author's official implementation",
        "ortools_version": "9.15.6755",
        "thresholds": args.thresholds,
        "time_limit_per_solve_seconds": args.time_limit,
        "reps_per_target": args.reps,
        "targets": len(score_paths),
        "fraction": args.fraction,
        "boost": args.boost,
        "budget": str(args.budget),
        "n_expected_sanity_rows": len(expected),
        "note": "A time-limited feasible answer is not certified optimal; underfilled outputs have NaN metrics and are not counted as losses.",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
