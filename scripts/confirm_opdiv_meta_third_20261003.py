#!/usr/bin/env python3
"""Official OPDiv same-score audit for the third ChEMBL cohort at B50."""

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
    path, index, root, expected, reps, seed, limit = task
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from chemdeprc.selection import select
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from run_chembl_selection import FP_COL, block_array, conformal_right_tail_pvalues, fingerprint_matrix
    from run_jctc_stress_grid import pairwise_tanimoto_matrix

    frame = pd.read_csv(path)
    target = str(frame["target_chembl_id"].iloc[0])
    cal = frame[frame["split"] == "calibration"]
    test = frame[frame["split"] == "testpool"].reset_index(drop=True)
    null = cal.loc[cal["label"] == 0, "score"].to_numpy(float)
    base = test["score"].to_numpy(float)
    labels = test["label"].to_numpy(int)
    blocks = block_array(test, "murcko_scaffold")
    sim = pairwise_tanimoto_matrix(fingerprint_matrix(test[FP_COL]))
    conflicts = sim > .7
    np.fill_diagonal(conflicts, False)
    target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
    target_seed = seed + index * 100_000 + target_hash
    rows = []
    for rep in range(reps):
        for fraction in (.2, .3):
            rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
            artifacts = choose_artifact_blocks(
                test, artifact_block_col="murcko_scaffold", rng=rng,
                artifact_fraction=fraction, max_artifact_blocks=25, min_inactive_per_block=2,
            )
            for boost in (4., 6.):
                scores, _, _ = inject_artifact(
                    test, base, artifacts, artifact_block_col="murcko_scaffold", logit_boost=boost,
                )
                pvalues = conformal_right_tail_pvalues(scores, null)
                sanity = select(
                    "score_cap1", scores=scores, pvalues=pvalues,
                    blocks=blocks, q=.7, budget=50,
                ).selected
                key = (target, rep, fraction, boost)
                if key not in expected or int(labels[sanity].sum()) != expected[key]:
                    raise AssertionError(f"Score-cap mismatch or missing grid cell: {key}")
                started = time.monotonic()
                answer = opdiv_select(scores, k=50, conflicts=conflicts, time_limit=limit)
                elapsed = time.monotonic() - started
                selected = np.asarray(answer.indices, dtype=int)
                if len(selected) == 50:
                    hits = int(labels[selected].sum())
                    fdp = (50 - hits) / 50
                    n_blocks = len(np.unique(blocks[selected]))
                else:
                    hits = np.nan
                    fdp = np.nan
                    n_blocks = np.nan
                rows.append({
                    "target": target, "rep": rep, "fraction": fraction,
                    "boost": boost, "budget": 50,
                    "method": "opdiv_tanimoto0.7", "status": answer.status,
                    "solver_gap": answer.gap, "selected": len(selected),
                    "hits": hits, "fdp": fdp, "blocks": n_blocks,
                    "seconds": elapsed, "score_cap1_sanity_hits": expected[key],
                })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--seed", type=int, default=353700)
    p.add_argument("--time-limit", type=float, default=2.)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    root = args.root.resolve()
    paths = sorted((root / args.scores_dir).glob("*_scores.csv"))
    if len(paths) < 5:
        raise AssertionError("Fewer than five qualified targets")
    if args.workers > (os.cpu_count() or 1) - 30:
        raise ValueError("Must leave at least 30 CPUs")
    grid = pd.read_csv(args.grid)
    if {"target", "rep", "fraction", "boost", "budget", "method", "hits"}.issubset(grid.columns):
        use = grid[
            grid["method"].eq("score_cap1")
            & grid["budget"].eq(50)
            & grid["fraction"].isin([.2, .3])
            & grid["boost"].isin([4., 6.])
            & grid["rep"].lt(args.reps)
        ].rename(columns={
            "target": "target_chembl_id", "fraction": "artifact_fraction",
            "boost": "logit_boost", "hits": "true_hits",
        })
    else:
        use = grid[
            grid["method"].eq("score_cap1")
            & grid["budget_label"].astype(str).eq("50")
            & grid["artifact_fraction"].isin([.2, .3])
            & grid["logit_boost"].isin([4., 6.])
            & grid["rep"].lt(args.reps)
        ]
    keys = ["target_chembl_id", "rep", "artifact_fraction", "logit_boost"]
    if use.duplicated(keys).any():
        raise AssertionError("Duplicate score-cap grid cell")
    expected = {
        (str(r.target_chembl_id), int(r.rep), float(r.artifact_fraction), float(r.logit_boost)): int(r.true_hits)
        for r in use.itertuples(index=False)
    }
    if len(expected) != len(paths) * args.reps * 4:
        raise AssertionError("Incomplete score-cap reference grid")
    tasks = [(path, i, root, expected, args.reps, args.seed, args.time_limit) for i, path in enumerate(paths)]
    with cf.ProcessPoolExecutor(max_workers=min(len(paths), args.workers)) as pool:
        chunks = list(pool.map(one_target, tasks))
    data = pd.DataFrame([r for chunk in chunks for r in chunk])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    data.groupby("method", as_index=False).agg(
        cells=("hits", "size"), full=("selected", lambda x: int((x == 50).sum())),
        optimal=("status", lambda x: int((x == "optimal").sum())),
        hits=("hits", "mean"), fdp=("fdp", "mean"),
        blocks=("blocks", "mean"), seconds=("seconds", "mean"),
    ).to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    args.out.with_name(args.out.stem + "_manifest.json").write_text(json.dumps({
        "score_dir": args.scores_dir, "seed": args.seed,
        "reps": args.reps, "threshold": .7, "time_limit": args.time_limit,
        "package": "official opdiv==0.1.0",
        "score_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
    }, indent=2) + "\n", encoding="utf-8")
    print(pd.read_csv(args.out.with_name(args.out.stem + "_summary.csv")).to_string(index=False))


if __name__ == "__main__":
    main()
