#!/usr/bin/env python3
"""Develop a calibration-trained artifact ranker on a fixed high-stress band."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from train_artifact_meta_ranker_20261003 import cap_one, features, fit_ranker, load_panel


FRACTIONS = (.2, .3)
BOOSTS = (4., 6.)
BUDGETS = (10, 50, 75)


def training_matrix(panels: list[dict], repeats: int) -> tuple[np.ndarray, np.ndarray]:
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact

    xs, ys = [], []
    for panel in panels:
        cal = panel["cal"]
        labels = cal["label"].to_numpy(int)
        base = cal["score"].to_numpy(float)
        blocks = panel["cal_blocks"]
        null = panel["null"]
        xs.append(features(base, blocks, null))
        ys.append(labels)
        target_hash = int(hashlib.sha256(panel["target"].encode()).hexdigest()[:12], 16) % 1_000_000
        for rep in range(repeats):
            for fraction in FRACTIONS:
                rng = np.random.default_rng(852026 + target_hash + rep * 1009 + int(fraction * 10000))
                artifacts = choose_artifact_blocks(
                    cal, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2,
                )
                for boost in BOOSTS:
                    changed, _, _ = inject_artifact(
                        cal, base, artifacts,
                        artifact_block_col="murcko_scaffold", logit_boost=boost,
                    )
                    xs.append(features(changed, blocks, null))
                    ys.append(labels)
    return np.concatenate(xs), np.concatenate(ys)


def evaluate(panels: list[dict], model, seed: int, reps: int) -> pd.DataFrame:
    from chemdeprc.selection import select
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from run_chembl_selection import conformal_right_tail_pvalues

    rows = []
    for idx, panel in enumerate(panels):
        test = panel["test"]
        base = test["score"].to_numpy(float)
        labels = test["label"].to_numpy(int)
        blocks = panel["test_blocks"]
        null = panel["null"]
        target = panel["target"]
        hash_part = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = seed + idx * 100_000 + hash_part
        for rep in range(reps):
            for fraction in FRACTIONS:
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
                artifacts = choose_artifact_blocks(
                    test, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2,
                )
                for boost in BOOSTS:
                    scores, _, _ = inject_artifact(
                        test, base, artifacts,
                        artifact_block_col="murcko_scaffold", logit_boost=boost,
                    )
                    pvalues = conformal_right_tail_pvalues(scores, null)
                    prob = model.predict_proba(features(scores, blocks, null))[:, 1]
                    for budget in BUDGETS:
                        result = {
                            "meta_calibration": cap_one(prob, blocks, budget),
                        }
                        for method in ("chemdeprc_cap1", "minp_block_bh_cap1", "score_cap1", "weighted_bh"):
                            result[method] = select(
                                method, scores=scores, pvalues=pvalues,
                                blocks=blocks, q=.7, budget=budget,
                            ).selected
                        for method, chosen in result.items():
                            hits = int(labels[chosen].sum())
                            rows.append({
                                "target": target, "rep": rep, "fraction": fraction,
                                "boost": boost, "budget": budget, "method": method,
                                "selected": len(chosen), "hits": hits,
                                "fdp": (len(chosen) - hits) / len(chosen) if len(chosen) else 0.,
                                "blocks": len(np.unique(blocks[chosen])),
                            })
        print(f"Evaluated {target}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--train-repeats", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    paths = sorted((root / args.scores_dir).glob("*_scores.csv"))
    panels = [load_panel(path) for path in paths]
    x, y = training_matrix(panels, args.train_repeats)
    print(f"Calibration meta-train: n={len(y)}, active={int(y.sum())}", flush=True)
    model = fit_ranker(x, y, 862026)
    raw = evaluate(panels, model, args.seed, args.reps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out, index=False)
    target_means = raw.groupby(["target", "budget", "method"], as_index=False).agg(
        hits=("hits", "mean"), fdp=("fdp", "mean"),
        selected=("selected", "mean"), blocks=("blocks", "mean"),
    )
    summary = target_means.groupby(["budget", "method"], as_index=False).agg(
        hits=("hits", "mean"), hits_target_sd=("hits", "std"),
        fdp=("fdp", "mean"), selected=("selected", "mean"), blocks=("blocks", "mean"),
    )
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    model.save_model(args.out.with_suffix(".model.json"))
    args.out.with_name(args.out.stem + "_manifest.json").write_text(json.dumps({
        "scores_dir": args.scores_dir, "seed": args.seed, "reps": args.reps,
        "train_repeats": args.train_repeats, "fractions": FRACTIONS,
        "boosts": BOOSTS, "budgets": BUDGETS,
        "meta_training": "calibration labels only; testpool labels only for evaluation",
        "status": "exploratory new method, not original ChemDep-RC",
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
