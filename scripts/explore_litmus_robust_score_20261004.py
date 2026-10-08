#!/usr/bin/env python3
"""Post hoc diagnostic for a scaffold-coherence score correction.

This is development on exposed AIDs, never a confirmatory SOTA result.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


AIDS = ("AID1798", "AID435034", "AID463087")
ALPHAS = (0.0, 0.25, 0.5, 1.0)


def corrected_score(scores: np.ndarray, blocks: np.ndarray,
                    null: np.ndarray, alpha: float) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    null_clipped = np.clip(null, 1e-6, 1 - 1e-6)
    null_logit = np.log(null_clipped / (1 - null_clipped))
    threshold = float(np.quantile(null_logit, .95))
    grouped = pd.DataFrame({"block": blocks, "logit": logits})
    stats = grouped.groupby("block", sort=False).logit.agg(["size", "median"])
    stats["penalty"] = np.where(stats["size"] >= 2,
                                np.maximum(stats["median"] - threshold, 0.), 0.)
    return logits - alpha * stats.loc[blocks, "penalty"].to_numpy(float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from run_chembl_selection import block_array
    from train_artifact_meta_ranker_20261003 import cap_one

    base = root / "data/pr_welqrate_litmus_20261004"
    rows = []
    for seed in range(1, 6):
        for index, aid in enumerate(AIDS):
            frame = pd.read_parquet(base / "prep" / f"{aid}_molecules.parquet")
            with np.load(base / "prep" / f"{aid}_seed{seed}_indices.npz") as split:
                cal_idx, test_idx = split["calibration"], split["test"]
            with np.load(base / "scores" / f"{aid}_seed{seed}_scores.npz") as scored:
                cal_scores, base_scores = scored["calibration"], scored["test"]
            cal = frame.iloc[cal_idx].reset_index(drop=True).copy()
            test = frame.iloc[test_idx].reset_index(drop=True).copy()
            test["molecule_chembl_id"] = [f"{aid}:{row}" for row in test_idx]
            blocks = block_array(test, "murcko_scaffold")
            labels = test.label.to_numpy(int)
            null = cal_scores[cal.label.to_numpy(int) == 0]

            def record(scenario: str, rep: int, fraction: float,
                       boost: float, scores: np.ndarray) -> None:
                for alpha in ALPHAS:
                    rank = corrected_score(scores, blocks, null, alpha)
                    chosen = cap_one(rank, blocks, 50)
                    rows.append({"aid": aid, "split_seed": seed,
                                 "scenario": scenario, "rep": rep,
                                 "fraction": fraction, "boost": boost,
                                 "alpha": alpha, "hits": int(labels[chosen].sum())})

            record("natural", -1, 0., 0., base_scores)
            target_hash = int(hashlib.sha256(aid.encode()).hexdigest()[:12], 16) % 1_000_000
            target_seed = 355700 + index * 100_000 + target_hash
            for rep in range(10):
                for fraction in (.2, .3):
                    rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
                    artifacts = choose_artifact_blocks(
                        test, artifact_block_col="murcko_scaffold", rng=rng,
                        artifact_fraction=fraction, max_artifact_blocks=25,
                        min_inactive_per_block=2)
                    for boost in (4., 6.):
                        changed = inject_artifact(
                            test, base_scores, artifacts,
                            artifact_block_col="murcko_scaffold", logit_boost=boost)[0]
                        record("synthetic_high_artifact", rep, fraction, boost, changed)
            print(aid, seed, "done", flush=True)
    output = base / "selectors" / "robust_score_posthoc_cells.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output, len(rows), flush=True)


if __name__ == "__main__":
    main()
