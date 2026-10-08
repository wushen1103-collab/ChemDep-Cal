#!/usr/bin/env python3
"""Official OPDiv comparison on frozen new-target B10 stress band."""

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


def one_target(task: tuple) -> list[dict]:
    path, idx, root, reps, seed, limit, expected = task
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from run_chembl_selection import FP_COL, block_array, conformal_right_tail_pvalues, fingerprint_matrix
    from run_jctc_stress_grid import pairwise_tanimoto_matrix
    from chemdeprc.selection import select

    frame = pd.read_csv(path)
    target = str(frame["target_chembl_id"].iloc[0])
    cal = frame[frame["split"] == "calibration"]
    test = frame[frame["split"] == "testpool"].reset_index(drop=True)
    null = cal.loc[cal["label"] == 0, "score"].to_numpy(float)
    labels = test["label"].to_numpy(int)
    base = test["score"].to_numpy(float)
    blocks = block_array(test, "murcko_scaffold")
    bits = fingerprint_matrix(test[FP_COL])
    sim = pairwise_tanimoto_matrix(bits)
    conflicts = sim > .7
    np.fill_diagonal(conflicts, False)
    target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
    target_seed = seed + idx * 100_000 + target_hash
    rows = []
    for rep in range(reps):
        for fraction in (.2, .3):
            rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
            artifact_blocks = choose_artifact_blocks(
                test, artifact_block_col="murcko_scaffold", rng=rng,
                artifact_fraction=fraction, max_artifact_blocks=25,
                min_inactive_per_block=2,
            )
            for boost in (4., 6.):
                scores, _, _ = inject_artifact(
                    test, base, artifact_blocks,
                    artifact_block_col="murcko_scaffold", logit_boost=boost,
                )
                pvalues = conformal_right_tail_pvalues(scores, null)
                sanity = select(
                    "score_cap1", scores=scores, pvalues=pvalues,
                    blocks=blocks, q=.7, budget=10,
                ).selected
                key = (target, rep, fraction, boost)
                if key not in expected:
                    raise AssertionError(f"Grid missing score-cap cell: {key}")
                if int(labels[sanity].sum()) != expected[key]:
                    raise AssertionError(f"Mismatched artifact/score-cap cell: {key}")
                started = time.monotonic()
                answer = opdiv_select(scores, k=10, conflicts=conflicts, time_limit=limit)
                elapsed = time.monotonic() - started
                selected = np.asarray(answer.indices, dtype=int)
                if len(selected) != 10:
                    hits = np.nan
                    fdp = np.nan
                    n_blocks = np.nan
                else:
                    hits = int(labels[selected].sum())
                    fdp = (10 - hits) / 10
                    n_blocks = len(np.unique(blocks[selected]))
                rows.append({
                    "target_chembl_id": target, "rep": rep,
                    "artifact_fraction": fraction, "logit_boost": boost,
                    "budget": 10, "method": "opdiv_tanimoto0.7",
                    "status": answer.status, "solver_gap": answer.gap,
                    "selected_count": len(selected), "true_hits": hits,
                    "fdp": fdp, "unique_blocks": n_blocks,
                    "elapsed_seconds": elapsed,
                    "score_cap1_sanity_hits": expected[key],
                })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--seed", type=int, default=353600)
    p.add_argument("--time-limit", type=float, default=2.0)
    p.add_argument("--workers", type=int, default=5)
    args = p.parse_args()
    root = args.root.resolve()
    paths = sorted((root / args.scores_dir).glob("*_scores.csv"))
    if not paths:
        raise RuntimeError("No score files")
    if args.workers > (os.cpu_count() or 1) - 30:
        raise ValueError("Reserve at least 30 CPU cores")
    grid = pd.read_csv(args.grid)
    mask = (
        grid["method"].eq("score_cap1")
        & grid["budget_label"].astype(str).eq("10")
        & grid["q"].eq(.7)
        & grid["artifact_fraction"].isin([.2, .3])
        & grid["logit_boost"].isin([4., 6.])
        & grid["rep"].lt(args.reps)
    )
    reference = grid.loc[mask, ["target_chembl_id", "rep", "artifact_fraction", "logit_boost", "true_hits"]]
    if reference.duplicated(["target_chembl_id", "rep", "artifact_fraction", "logit_boost"]).any():
        raise AssertionError("Duplicate reference cells")
    expected = {
        (str(r.target_chembl_id), int(r.rep), float(r.artifact_fraction), float(r.logit_boost)): int(r.true_hits)
        for r in reference.itertuples(index=False)
    }
    if len(expected) != len(paths) * args.reps * 4:
        raise AssertionError(f"Expected {len(paths) * args.reps * 4} score-cap cells, found {len(expected)}")
    tasks = [(path, i, root, args.reps, args.seed, args.time_limit, expected) for i, path in enumerate(paths)]
    with cf.ProcessPoolExecutor(max_workers=min(args.workers, len(paths))) as pool:
        chunks = list(pool.map(one_target, tasks))
    result = pd.DataFrame([row for chunk in chunks for row in chunk])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    summary = result.groupby("method", as_index=False).agg(
        n=("true_hits", "size"), full=("selected_count", lambda x: int((x == 10).sum())),
        optimal=("status", lambda x: int((x == "optimal").sum())),
        hits=("true_hits", "mean"), fdp=("fdp", "mean"),
        blocks=("unique_blocks", "mean"), seconds=("elapsed_seconds", "mean"),
    )
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    args.out.with_name(args.out.stem + "_manifest.json").write_text(json.dumps({
        "score_dir": args.scores_dir, "grid": str(args.grid),
        "seed": args.seed, "reps": args.reps, "threshold": .7,
        "time_limit": args.time_limit, "official_package": "opdiv==0.1.0",
        "score_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
