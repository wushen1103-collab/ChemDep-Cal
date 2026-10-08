#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from chemdeprc.diversity_selection import select_diversity
from chemdeprc.selection import select


DIVERSITY_METHODS = {"score_mmr25", "score_dpp", "score_maxmin"}


def make_synthetic_pool(
    n: int,
    n_blocks: int,
    seed: int,
    *,
    need_bits: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = rng.beta(2.0, 5.0, size=n)
    hot = rng.choice(n, size=max(1, n // 20), replace=False)
    scores[hot] = rng.beta(6.0, 2.0, size=len(hot))
    blocks = np.asarray([f"b{idx:07d}" for idx in rng.integers(0, n_blocks, size=n)], dtype=object)
    null_scores = rng.beta(2.0, 5.0, size=max(2000, min(50000, n // 2)))
    sorted_null = np.sort(null_scores)
    counts_ge = len(sorted_null) - np.searchsorted(sorted_null, scores, side="left")
    pvalues = (1.0 + counts_ge) / (len(sorted_null) + 1.0)
    if need_bits:
        bits = rng.random((min(n, 20000), 2048)) < 0.03
        if n > len(bits):
            bits = np.vstack([bits, rng.random((n - len(bits), 2048)) < 0.03])
    else:
        bits = np.empty((0, 0), dtype=np.bool_)
    return scores.astype(np.float64), pvalues.astype(np.float64), blocks, bits.astype(np.bool_)


def bench_once(
    *,
    n: int,
    n_blocks: int,
    method: str,
    scores: np.ndarray,
    pvalues: np.ndarray,
    blocks: np.ndarray,
    bits: np.ndarray,
    budget: int,
    q: float,
    seed: int,
) -> dict:
    t0 = time.perf_counter()
    if method in DIVERSITY_METHODS:
        result = select_diversity(method, scores=scores, bits=bits, budget=budget, seed=seed)
    else:
        result = select(method, scores=scores, pvalues=pvalues, blocks=blocks, q=q, budget=budget)
    seconds = time.perf_counter() - t0
    return {
        "n_candidates": n,
        "n_blocks": n_blocks,
        "budget": budget,
        "q": q,
        "method": method,
        "seconds": seconds,
        "selected_count": int(len(result.selected)),
        "certificate": result.certificate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data/jctc_scalability")
    parser.add_argument("--sizes", nargs="+", type=int, default=[10000, 100000, 1000000])
    parser.add_argument("--methods", nargs="+", default=["raw_top_b", "bh", "weighted_bh", "block_bh", "chemdeprc_cap1", "chemdeprc_soft75"])
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--q", type=float, default=0.7)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3591)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    needs_bits = any(method in DIVERSITY_METHODS for method in args.methods)
    for n in args.sizes:
        n_blocks = max(100, n // 20)
        budget = min(args.budget, n)
        for rep in range(args.reps):
            pool_seed = args.seed + rep * 1009 + n
            scores, pvalues, blocks, bits = make_synthetic_pool(
                n,
                n_blocks,
                pool_seed,
                need_bits=needs_bits,
            )
            for method in args.methods:
                rows.append(
                    bench_once(
                        n=n,
                        n_blocks=n_blocks,
                        method=method,
                        scores=scores,
                        pvalues=pvalues,
                        blocks=blocks,
                        bits=bits,
                        budget=budget,
                        q=args.q,
                        seed=pool_seed,
                    )
                )
    results = pd.DataFrame(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results.to_csv(outdir / f"scalability_results_{stamp}.csv", index=False)
    results.to_csv(outdir / "latest_results.csv", index=False)
    summary = (
        results.groupby(["n_candidates", "n_blocks", "budget", "q", "method"], as_index=False)
        .agg(seconds_mean=("seconds", "mean"), seconds_std=("seconds", "std"), selected_count_mean=("selected_count", "mean"), reps=("seconds", "size"))
        .sort_values(["n_candidates", "method"])
    )
    summary.to_csv(outdir / f"scalability_summary_{stamp}.csv", index=False)
    summary.to_csv(outdir / "latest_summary.csv", index=False)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sizes": args.sizes,
        "methods": args.methods,
        "budget": args.budget,
        "q": args.q,
        "reps": args.reps,
        "note": "Measures selection-layer runtime after scores, p-values, and blocks are available; this is the relevant overhead relative to docking/scoring.",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary.to_markdown(index=False, floatfmt=".4f"))


if __name__ == "__main__":
    main()
