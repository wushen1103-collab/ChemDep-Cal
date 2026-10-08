#!/usr/bin/env python3
"""Exploratory cap-one ranking/fill audit on matched score files.

This script does not certify SOTA: all existing target panels have been inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def block_leaders(scores: np.ndarray, blocks: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    unique, inverse = np.unique(blocks, return_inverse=True)
    groups = [np.flatnonzero(inverse == k) for k in range(len(unique))]
    leaders = np.asarray([g[np.argmax(scores[g])] for g in groups], dtype=int)
    return leaders, groups


def blended_order(
    scores: np.ndarray, pvalues: np.ndarray, blocks: np.ndarray, alpha: float
) -> np.ndarray:
    from chemdeprc.selection import _simes_pvalue

    leaders, groups = block_leaders(scores, blocks)
    block_p = np.asarray([_simes_pvalue(pvalues[g]) for g in groups])
    score_order = np.argsort(-scores[leaders], kind="mergesort")
    p_order = np.argsort(block_p, kind="mergesort")
    score_rank = np.empty(len(leaders), dtype=int)
    p_rank = np.empty(len(leaders), dtype=int)
    score_rank[score_order] = np.arange(len(leaders))
    p_rank[p_order] = np.arange(len(leaders))
    rank = (1 - alpha) * score_rank + alpha * p_rank
    order = np.lexsort((-scores[leaders], rank))
    return leaders[order]


def one_target(path: Path, mode: str, target_index: int, reps: int, root: Path) -> list[dict]:
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from chemdeprc.selection import select
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from run_chembl_selection import block_array, conformal_right_tail_pvalues

    frame = pd.read_csv(path)
    target = str(frame["target_chembl_id"].iloc[0])
    calibration = frame[frame["split"] == "calibration"]
    testpool = frame[frame["split"] == "testpool"].reset_index(drop=True)
    null_scores = calibration.loc[calibration["label"] == 0, "score"].to_numpy(float)
    if len(null_scores) == 0 or testpool.empty:
        raise ValueError(f"Missing calibration nulls or testpool: {target}")
    base_scores = testpool["score"].to_numpy(float)
    labels = testpool["label"].to_numpy(int)
    blocks = block_array(testpool, "murcko_scaffold")
    if mode == "chembl":
        target_seed = 353536 + target_index * 100_000 + (
            int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        )
        dataset = "ChEMBL"
    else:
        target_seed = 353500 + target_index * 100_000 + (
            int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        )
        dataset = str(frame["external_dataset"].iloc[0]) + "/" + str(frame["source_backbone"].iloc[0])
    rows = []
    for rep in range(reps):
        rng = np.random.default_rng(target_seed + rep * 10_003 + 1000)
        artifact_blocks = choose_artifact_blocks(
            testpool, artifact_block_col="murcko_scaffold", rng=rng,
            artifact_fraction=0.1, max_artifact_blocks=25, min_inactive_per_block=2,
        )
        scores, _, _ = inject_artifact(
            testpool, base_scores, artifact_blocks,
            artifact_block_col="murcko_scaffold", logit_boost=4.0,
        )
        pvalues = conformal_right_tail_pvalues(scores, null_scores)
        scorecap = select(
            "score_cap1", scores=scores, pvalues=pvalues, blocks=blocks, q=0.7, budget=50
        ).selected
        chemcap = select(
            "chemdeprc_cap1", scores=scores, pvalues=pvalues, blocks=blocks, q=0.7, budget=50
        ).selected
        ranked = blended_order(scores, pvalues, blocks, 0.0)
        chem_blocks = set(blocks[chemcap])
        fill = np.asarray(
            list(chemcap) + [int(i) for i in ranked if blocks[i] not in chem_blocks][:50 - len(chemcap)],
            dtype=int,
        )
        methods = {"score_cap1": scorecap, "chemdeprc_cap1": chemcap, "risk_fill": fill}
        for alpha in (0.25, 0.5, 0.75, 1.0):
            methods[f"blend_{alpha:g}"] = blended_order(scores, pvalues, blocks, alpha)[:50]
        for method, selected in methods.items():
            selected = np.asarray(selected, dtype=int)
            if len(selected) != len(np.unique(selected)) or len(np.unique(blocks[selected])) != len(selected):
                raise AssertionError(f"Cap-one violation: {target} {rep} {method}")
            hits = int(labels[selected].sum())
            rows.append({
                "dataset": dataset, "target": target, "rep": rep,
                "method": method, "selected": len(selected), "hits": hits,
                "fdp": (len(selected) - hits) / len(selected) if len(selected) else 0.0,
                "empty": int(len(selected) == 0),
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--mode", choices=["chembl", "external"], required=True)
    p.add_argument("--reps", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    paths = sorted((root / args.scores_dir).glob("*_scores.csv"))
    if not paths:
        raise RuntimeError("No score files")
    rows = []
    for i, path in enumerate(paths):
        rows.extend(one_target(path, args.mode, i, args.reps, root))
        print(f"Finished {path.stem}", flush=True)
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = df.groupby(["dataset", "method"], as_index=False).agg(
        n=("hits", "size"), hits=("hits", "mean"), fdp=("fdp", "mean"),
        selected=("selected", "mean"), empty=("empty", "sum")
    )
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False))
    args.out.with_name(args.out.stem + "_manifest.json").write_text(json.dumps({
        "mode": args.mode, "scores_dir": args.scores_dir, "reps": args.reps,
        "fraction": 0.1, "boost": 4.0, "q": 0.7, "budget": 50,
        "warning": "Exploratory target panels already examined; no confirmatory SOTA claim.",
        "score_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
