#!/usr/bin/env python3
"""Explore a fixed interpolation between Bonferroni-min and Simes block p-values."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def select_mix(
    scores: np.ndarray, pvalues: np.ndarray, blocks: np.ndarray,
    budget: int, lam: float, q: float = .7,
) -> np.ndarray:
    from chemdeprc.selection import (
        _bh, _members_by_block, _rank_blocks, _rank_members, _simes_pvalue,
    )

    _, members = _members_by_block(blocks)
    simes = np.asarray([_simes_pvalue(pvalues[g]) for g in members])
    minp = np.asarray([min(1., len(g) * float(np.min(pvalues[g]))) for g in members])
    block_p = np.exp((1 - lam) * np.log(np.maximum(minp, 1e-12))
                     + lam * np.log(np.maximum(simes, 1e-12)))
    block_scores = np.asarray([float(np.max(scores[g])) for g in members])
    chosen_blocks = _bh(block_p, q, scores=block_scores)
    ordered = _rank_blocks(chosen_blocks, block_p=block_p,
                           block_score=block_scores, rank_by="pvalue")[:budget]
    answer = []
    for pos in ordered:
        ranked = _rank_members(members[int(pos)], pvalues=pvalues,
                               scores=scores, rank_by="pvalue")
        answer.append(int(ranked[0]))
    return np.asarray(answer, dtype=int)


def one_target(path: Path, idx: int, seed: int, reps: int) -> list[dict]:
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from run_chembl_selection import block_array, conformal_right_tail_pvalues

    frame = pd.read_csv(path)
    target = str(frame["target_chembl_id"].iloc[0])
    cal = frame[frame["split"] == "calibration"]
    test = frame[frame["split"] == "testpool"].reset_index(drop=True)
    null = cal.loc[cal["label"] == 0, "score"].to_numpy(float)
    labels = test["label"].to_numpy(int)
    base = test["score"].to_numpy(float)
    blocks = block_array(test, "murcko_scaffold")
    hash_part = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
    target_seed = seed + idx * 100_000 + hash_part
    rows = []
    for rep in range(reps):
        for fraction in (.2, .3):
            rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
            artifacts = choose_artifact_blocks(
                test, artifact_block_col="murcko_scaffold", rng=rng,
                artifact_fraction=fraction, max_artifact_blocks=25,
                min_inactive_per_block=2,
            )
            for boost in (4., 6.):
                scores, _, _ = inject_artifact(
                    test, base, artifacts, artifact_block_col="murcko_scaffold",
                    logit_boost=boost,
                )
                pvalues = conformal_right_tail_pvalues(scores, null)
                for budget in (10, 50, 75):
                    for lam in (0., .25, .5, .75, 1.):
                        chosen = select_mix(scores, pvalues, blocks, budget, lam)
                        hits = int(labels[chosen].sum())
                        rows.append({
                            "target": target, "rep": rep, "fraction": fraction,
                            "boost": boost, "budget": budget, "lambda": lam,
                            "selected": len(chosen), "hits": hits,
                            "fdp": (len(chosen) - hits) / len(chosen) if len(chosen) else 0.,
                            "blocks": len(np.unique(blocks[chosen])),
                        })
    print(f"Finished {target}", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    paths = sorted((root / args.scores_dir).glob("*_scores.csv"))
    if not paths:
        raise RuntimeError("No scores")
    df = pd.DataFrame([
        row for idx, path in enumerate(paths)
        for row in one_target(path, idx, args.seed, args.reps)
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    target_means = df.groupby(["target", "budget", "lambda"], as_index=False).agg(
        hits=("hits", "mean"), fdp=("fdp", "mean"), selected=("selected", "mean"),
    )
    summary = target_means.groupby(["budget", "lambda"], as_index=False).agg(
        hits=("hits", "mean"), hits_target_sd=("hits", "std"),
        fdp=("fdp", "mean"), selected=("selected", "mean"),
    )
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    args.out.with_name(args.out.stem + "_manifest.json").write_text(json.dumps({
        "scores_dir": args.scores_dir, "seed": args.seed, "reps": args.reps,
        "fraction": [.2, .3], "boost": [4, 6], "budget": [10, 50, 75],
        "lambda": [0, .25, .5, .75, 1],
        "warning": "Exploratory; interpolation is not a certified FDR procedure.",
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
