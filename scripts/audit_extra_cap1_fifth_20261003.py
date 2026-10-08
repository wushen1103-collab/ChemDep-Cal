#!/usr/bin/env python3
"""Post-confirmation stress audit of additional cap-one selectors."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("bh_cap1", "hier_bh_cap1", "block_by_cap1", "chemdeprc_scorecap1")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=355700)
    p.add_argument("--reps", type=int, default=10)
    args = p.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from chemdeprc.selection import select
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from run_chembl_selection import block_array, conformal_right_tail_pvalues

    meta = pd.read_csv(args.meta)
    ref = meta[(meta["budget"] == 50) & (meta["method"] == "score_cap1")]
    expected = {
        (r.target, int(r.rep), float(r.fraction), float(r.boost)): int(r.hits)
        for r in ref.itertuples(index=False)
    }
    paths = sorted((root / args.scores_dir).glob("*_scores.csv"))
    if len(expected) != len(paths) * args.reps * 4:
        raise AssertionError("Incomplete score-cap reference")
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
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = args.seed + idx * 100_000 + target_hash
        for rep in range(args.reps):
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
                    pv = conformal_right_tail_pvalues(scores, null)
                    key = (target, rep, fraction, boost)
                    sanity = select("score_cap1", scores=scores, pvalues=pv,
                                    blocks=blocks, q=.7, budget=50).selected
                    if int(labels[sanity].sum()) != expected[key]:
                        raise AssertionError(f"Different score/corruption cell: {key}")
                    for method in METHODS:
                        chosen = list(map(int, select(method, scores=scores, pvalues=pv,
                            blocks=blocks, q=.7, budget=50).selected))
                        before_fill = len(chosen)
                        if len(set(blocks[chosen])) != before_fill:
                            raise AssertionError(f"Non-cap-one {method}: {key}")
                        if before_fill < 50:
                            seen = set(blocks[chosen])
                            for candidate in np.argsort(-scores, kind="mergesort"):
                                if blocks[candidate] not in seen:
                                    chosen.append(int(candidate))
                                    seen.add(blocks[candidate])
                                    if len(chosen) == 50:
                                        break
                        if len(chosen) != 50:
                            raise AssertionError(f"Could not fill {method}: {key}")
                        hits = int(labels[chosen].sum())
                        rows.append({
                            "target": target, "rep": rep, "fraction": fraction,
                            "boost": boost, "budget": 50, "method": method + "_score_fill",
                            "hits": hits, "fdp": (50 - hits) / 50,
                            "selected": 50, "blocks": 50,
                            "selected_before_fill": before_fill,
                        })
    output = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(output.groupby("method")["hits"].mean().to_string())


if __name__ == "__main__":
    main()
