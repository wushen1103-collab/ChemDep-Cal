#!/usr/bin/env python3
"""Exploratory fair-capacity weighted-BH cap-one control."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=354700)
    p.add_argument("--reps", type=int, default=10)
    args = p.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from chemdeprc.selection import _inverse_block_size_weights, select
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from run_chembl_selection import block_array, conformal_right_tail_pvalues

    ref = pd.read_csv(args.meta)
    ref = ref[(ref["budget"] == 50) & (ref["method"] == "weighted_bh")]
    expected = {
        (r.target, int(r.rep), float(r.fraction), float(r.boost)): int(r.hits)
        for r in ref.itertuples(index=False)
    }
    paths = sorted((root / args.scores_dir).glob("*_scores.csv"))
    if len(expected) != len(paths) * args.reps * 4:
        raise AssertionError("Incomplete reference weighted BH cells")
    rows = []
    for idx, path in enumerate(paths):
        frame = pd.read_csv(path)
        target = str(frame["target_chembl_id"].iloc[0])
        cal = frame[frame["split"] == "calibration"]
        test = frame[frame["split"] == "testpool"].reset_index(drop=True)
        null = cal.loc[cal["label"] == 0, "score"].to_numpy(float)
        base = test["score"].to_numpy(float)
        labels = test["label"].to_numpy(int)
        blocks = block_array(test, "murcko_scaffold")
        weights = _inverse_block_size_weights(blocks)
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = args.seed + idx * 100_000 + target_hash
        for rep in range(args.reps):
            for fraction in (.2, .3):
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
                artifacts = choose_artifact_blocks(
                    test, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2,
                )
                for boost in (4., 6.):
                    scores, _, _ = inject_artifact(
                        test, base, artifacts,
                        artifact_block_col="murcko_scaffold", logit_boost=boost,
                    )
                    pv = conformal_right_tail_pvalues(scores, null)
                    raw = select("weighted_bh", scores=scores, pvalues=pv,
                                 blocks=blocks, q=.7, budget=50).selected
                    key = (target, rep, fraction, boost)
                    if int(labels[raw].sum()) != expected[key]:
                        raise AssertionError(f"Weighted BH replay mismatch: {key}")
                    rejected = select("weighted_bh", scores=scores, pvalues=pv,
                                      blocks=blocks, q=.7, budget=len(scores)).selected
                    adjusted = pv / np.maximum(weights, 1e-12)
                    ordered = rejected[np.lexsort((-scores[rejected], adjusted[rejected]))]
                    chosen = []
                    seen = set()
                    for member in ordered:
                        block = blocks[member]
                        if block not in seen:
                            chosen.append(int(member))
                            seen.add(block)
                            if len(chosen) == 50:
                                break
                    bh_count = len(chosen)
                    if len(chosen) < 50:
                        for member in np.argsort(-scores, kind="mergesort"):
                            block = blocks[member]
                            if block not in seen:
                                chosen.append(int(member))
                                seen.add(block)
                                if len(chosen) == 50:
                                    break
                    if len(chosen) != 50:
                        raise AssertionError(f"Cap-one budget failure: {key}")
                    hits = int(labels[chosen].sum())
                    rows.append({
                        "target": target, "rep": rep, "fraction": fraction,
                        "boost": boost, "budget": 50, "method": "weighted_bh_cap1_fill",
                        "hits": hits, "fdp": (50 - hits) / 50,
                        "selected": 50, "blocks": 50, "bh_selected_before_fill": bh_count,
                    })
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.groupby("target")["hits"].mean().to_string())


if __name__ == "__main__":
    main()
