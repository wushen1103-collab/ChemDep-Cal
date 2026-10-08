#!/usr/bin/env python3
"""Independent public-target stress validation with same-score selectors."""

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
from opdiv import select as opdiv_select
from rdkit import DataStructs


DEFAULT_METHODS = [
    "raw_top_b", "bh", "by", "weighted_bh", "score_cap1",
    "chemdeprc_cap1", "chemdeprc_soft75", "score_mmr25", "score_dpp",
]


def full_tanimoto_matrix(hex_values: pd.Series) -> np.ndarray:
    fingerprints = [DataStructs.CreateFromBinaryText(bytes.fromhex(value)) for value in hex_values.astype(str)]
    n = len(fingerprints)
    matrix = np.empty((n, n), dtype=np.float32)
    for i, fingerprint in enumerate(fingerprints):
        matrix[i] = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints)
    return matrix


def one_target(task: tuple) -> list[dict]:
    path, target_index, reps, root, fraction, boost, budget_label, methods, threshold, time_limit = task
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from run_chembl_selection import (
        FP_COL, block_array, conformal_right_tail_pvalues,
        evaluate_real_selection, fingerprint_matrix, parse_budget,
        select_real_method,
    )
    from run_jctc_stress_grid import pairwise_tanimoto_matrix

    frame = pd.read_csv(path)
    target = str(frame["target_chembl_id"].iloc[0])
    dataset = str(frame["external_dataset"].iloc[0])
    score_backbone = str(frame["source_backbone"].iloc[0])
    calibration = frame[frame["split"] == "calibration"].copy()
    testpool = frame[frame["split"] == "testpool"].copy().reset_index(drop=True)
    if calibration.empty or testpool.empty:
        raise AssertionError(f"Empty calibration/test pool for {target}")
    null_scores = calibration.loc[calibration["label"] == 0, "score"].to_numpy(dtype=float)
    if len(null_scores) == 0:
        raise AssertionError(f"No inactive calibration scores for {target}")
    labels = testpool["label"].to_numpy(dtype=np.int8)
    base_scores = testpool["score"].to_numpy(dtype=float)
    blocks = block_array(testpool, "murcko_scaffold")
    bits = fingerprint_matrix(testpool[FP_COL])
    budget = parse_budget(str(budget_label), len(testpool))
    sim = full_tanimoto_matrix(testpool[FP_COL])
    check_n = min(20, len(testpool))
    reference = pairwise_tanimoto_matrix(bits[:check_n])
    if not np.allclose(sim[:check_n, :check_n], reference, atol=1e-7):
        raise AssertionError(f"RDKit similarity differs from original matrix for {target}")
    conflicts = sim > threshold
    np.fill_diagonal(conflicts, False)
    panel_seed = frame["panel_seed"].iloc[0]
    scenario = str(frame["scenario"].iloc[0])
    target_seed = 353500 + target_index * 100_000 + (
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
        scores, artifact_molecules, artifact_inactive = inject_artifact(
            testpool, base_scores, artifact_blocks,
            artifact_block_col="murcko_scaffold", logit_boost=boost,
        )
        pvalues = conformal_right_tail_pvalues(scores, null_scores)
        for method in [*methods, f"opdiv_tanimoto{threshold:g}"]:
            started = time.monotonic()
            if method.startswith("opdiv_"):
                answer = opdiv_select(
                    scores, k=budget, conflicts=conflicts, time_limit=time_limit,
                )
                selected = np.asarray(answer.indices, dtype=int)
                status = answer.status
                gap = answer.gap
            else:
                answer = select_real_method(
                    method, scores=scores, pvalues=pvalues, blocks=blocks,
                    bits=bits, q=0.7, budget=budget,
                    seed_parts=(target, dataset, scenario, panel_seed, rep,
                                f"frac{fraction:g}_boost{boost:g}", 0.7, budget_label),
                )
                selected = answer.selected
                status = answer.certificate
                gap = None
            elapsed = time.monotonic() - started
            full = len(selected) == budget
            metrics = evaluate_real_selection(
                labels=labels, scores=scores, blocks=blocks, bits=bits,
                selected=selected, budget=budget, q=0.7,
            ) if (full or not method.startswith("opdiv_")) else {}
            rows.append({
                "external_dataset": dataset,
                "score_backbone": score_backbone,
                "target_chembl_id": target,
                "rep": rep,
                "method": method,
                "artifact_fraction": fraction,
                "logit_boost": boost,
                "artifact_molecules": artifact_molecules,
                "artifact_inactive_molecules": artifact_inactive,
                "budget": budget,
                "q": 0.7,
                "status": status,
                "solver_gap": gap,
                "full_selection": full,
                "elapsed_seconds": elapsed,
                "n_testpool": len(testpool),
                **metrics,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--boost", type=float, default=4.0)
    parser.add_argument("--budget", default="50")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--time-limit", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--max-targets", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    paths = sorted((root / args.scores_dir).glob("*_scores.csv"))
    if args.max_targets:
        paths = paths[:args.max_targets]
    if not paths:
        raise RuntimeError("No score files")
    if args.workers > (os.cpu_count() or 1) - 30:
        raise ValueError("Reserve at least 30 CPU cores")
    tasks = [
        (path, i, args.reps, root, args.fraction, args.boost, str(args.budget),
         args.methods, args.threshold, args.time_limit)
        for i, path in enumerate(paths)
    ]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        for part in pool.map(one_target, tasks):
            rows.extend(part)
    results = pd.DataFrame(rows)
    results.to_csv(outdir / "raw.csv", index=False)
    summary = results.groupby(["external_dataset", "method"], as_index=False).agg(
        n_rows=("rep", "size"), n_full=("full_selection", "sum"),
        n_optimal=("status", lambda s: int((s == "optimal").sum())),
        true_hits=("true_hits", "mean"), fdp=("fdp", "mean"),
        unique_blocks=("unique_blocks", "mean"),
        max_block_share=("max_block_share", "mean"),
        seconds=("elapsed_seconds", "mean"),
    )
    summary.to_csv(outdir / "summary.csv", index=False)
    manifest = {
        "score_dir": args.scores_dir, "targets": len(paths),
        "reps": args.reps, "fraction": args.fraction, "boost": args.boost,
        "budget": str(args.budget), "q": 0.7, "methods": args.methods,
        "opdiv_version": "0.1.0", "opdiv_threshold": args.threshold,
        "opdiv_time_limit": args.time_limit,
        "score_file_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "warning": "Artificial score corruption is label-informed by construction; this is a simulated stress test, not a natural prospective benchmark.",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
