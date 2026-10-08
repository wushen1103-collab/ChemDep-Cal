#!/usr/bin/env python3
"""Train and evaluate calibration-supervised scaffold-capped ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def features(scores: np.ndarray, blocks: np.ndarray, null_scores: np.ndarray) -> np.ndarray:
    from run_chembl_selection import conformal_right_tail_pvalues

    eps = 1e-6
    pvalues = conformal_right_tail_pvalues(scores, null_scores)
    _, inverse, counts = np.unique(blocks, return_inverse=True, return_counts=True)
    block_max = np.full(len(counts), -np.inf)
    block_sum = np.bincount(inverse, weights=scores, minlength=len(counts))
    np.maximum.at(block_max, inverse, scores)
    block_mean = block_sum / counts
    block_pmin = np.full(len(counts), 1.0)
    np.minimum.at(block_pmin, inverse, pvalues)
    calib_med = float(np.median(null_scores))
    calib_iqr = max(float(np.percentile(null_scores, 75) - np.percentile(null_scores, 25)), .02)
    logits = np.log(np.clip(scores, eps, 1 - eps) / np.clip(1 - scores, eps, 1 - eps))
    columns = [
        scores,
        logits,
        -np.log10(np.maximum(pvalues, eps)),
        (scores - calib_med) / calib_iqr,
        np.log1p(counts[inverse]),
        block_max[inverse],
        block_mean[inverse],
        block_max[inverse] - block_mean[inverse],
        scores - block_mean[inverse],
        -np.log10(np.maximum(block_pmin[inverse], eps)),
    ]
    return np.column_stack(columns).astype(np.float32)


def cap_one(prob: np.ndarray, blocks: np.ndarray, budget: int) -> np.ndarray:
    order = np.argsort(-prob, kind="mergesort")
    chosen = []
    seen = set()
    for idx in order:
        block = blocks[idx]
        if block not in seen:
            chosen.append(int(idx))
            seen.add(block)
            if len(chosen) == budget:
                break
    return np.asarray(chosen, dtype=int)


def load_panel(path: Path) -> dict:
    from run_chembl_selection import block_array

    frame = pd.read_csv(path)
    cal = frame[frame["split"] == "calibration"].reset_index(drop=True)
    test = frame[frame["split"] == "testpool"].reset_index(drop=True)
    null = cal.loc[cal["label"] == 0, "score"].to_numpy(float)
    if len(null) < 10 or len(test) < 50:
        raise ValueError(f"Insufficient panel: {path}")
    return {
        "target": str(frame["target_chembl_id"].iloc[0]),
        "cal": cal, "test": test, "null": null,
        "cal_blocks": block_array(cal, "murcko_scaffold"),
        "test_blocks": block_array(test, "murcko_scaffold"),
    }


def train_rows(panels: list[dict], n_augments: int, boost: float) -> tuple[np.ndarray, np.ndarray]:
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact

    xs, ys = [], []
    for panel in panels:
        cal, null, blocks = panel["cal"], panel["null"], panel["cal_blocks"]
        base = cal["score"].to_numpy(float)
        labels = cal["label"].to_numpy(int)
        xs.append(features(base, blocks, null))
        ys.append(labels)
        target_hash = int(hashlib.sha256(panel["target"].encode()).hexdigest()[:12], 16) % 1_000_000
        for rep in range(n_augments):
            rng = np.random.default_rng(840000 + target_hash + rep * 1009)
            artifacts = choose_artifact_blocks(
                cal, artifact_block_col="murcko_scaffold", rng=rng,
                artifact_fraction=.1, max_artifact_blocks=25,
                min_inactive_per_block=2,
            )
            changed, _, _ = inject_artifact(
                cal, base, artifacts, artifact_block_col="murcko_scaffold",
                logit_boost=boost,
            )
            xs.append(features(changed, blocks, null))
            ys.append(labels)
    return np.concatenate(xs), np.concatenate(ys)


def fit_ranker(x: np.ndarray, y: np.ndarray, seed: int) -> XGBClassifier:
    if len(np.unique(y)) != 2:
        raise ValueError("Calibration meta-training requires both active and inactive labels")
    positives = max(int(y.sum()), 1)
    scale_pos_weight = min(20.0, max(1.0, (len(y) - positives) / positives))
    model = XGBClassifier(
        n_estimators=160, max_depth=3, learning_rate=.05,
        subsample=.9, colsample_bytree=.9, min_child_weight=8,
        reg_lambda=4.0, tree_method="hist", eval_metric="logloss",
        n_jobs=4, random_state=seed, scale_pos_weight=scale_pos_weight,
    )
    model.fit(x, y)
    return model


def evaluate(panels: list[dict], model: XGBClassifier, mode: str, reps: int) -> pd.DataFrame:
    from chemdeprc.selection import select
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from run_chembl_selection import conformal_right_tail_pvalues

    rows = []
    for i, panel in enumerate(panels):
        target = panel["target"]
        frame = panel["test"]
        blocks = panel["test_blocks"]
        labels = frame["label"].to_numpy(int)
        base = frame["score"].to_numpy(float)
        null = panel["null"]
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        seed = (353536 if mode == "chembl" else 353500) + i * 100_000 + target_hash
        for rep in range(reps):
            rng = np.random.default_rng(seed + rep * 10_003 + 1000)
            artifacts = choose_artifact_blocks(
                frame, artifact_block_col="murcko_scaffold", rng=rng,
                artifact_fraction=.1, max_artifact_blocks=25,
                min_inactive_per_block=2,
            )
            scores, _, _ = inject_artifact(
                frame, base, artifacts, artifact_block_col="murcko_scaffold", logit_boost=4.0,
            )
            pvalues = conformal_right_tail_pvalues(scores, null)
            prob = model.predict_proba(features(scores, blocks, null))[:, 1]
            results = {
                "meta_augmented": cap_one(prob, blocks, 50),
                "score_cap1": select(
                    "score_cap1", scores=scores, pvalues=pvalues,
                    blocks=blocks, q=.7, budget=50,
                ).selected,
                "chemdeprc_cap1": select(
                    "chemdeprc_cap1", scores=scores, pvalues=pvalues,
                    blocks=blocks, q=.7, budget=50,
                ).selected,
            }
            for method, selected in results.items():
                hits = int(labels[selected].sum())
                rows.append({
                    "target": target, "rep": rep, "method": method,
                    "selected": len(selected), "hits": hits,
                    "fdp": (len(selected) - hits) / len(selected) if len(selected) else 0.0,
                    "empty": int(len(selected) == 0),
                })
        print(f"Evaluated {target}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--mode", choices=["chembl", "external"], required=True)
    p.add_argument("--reps", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--augments", type=int, default=12)
    p.add_argument("--boost", type=float, default=4.0)
    args = p.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    paths = sorted((root / args.scores_dir).glob("*_scores.csv"))
    panels = [load_panel(path) for path in paths]
    x, y = train_rows(panels, args.augments, args.boost)
    print(f"Meta-training calibration rows: {len(y)}, active: {int(y.sum())}", flush=True)
    model = fit_ranker(x, y, 842026)
    raw = evaluate(panels, model, args.mode, args.reps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out, index=False)
    raw.groupby("method", as_index=False).agg(
        n=("hits", "size"), hits=("hits", "mean"), fdp=("fdp", "mean"),
        selected=("selected", "mean"), empty=("empty", "sum"),
    ).to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    model.save_model(args.out.with_suffix(".model.json"))
    args.out.with_name(args.out.stem + "_manifest.json").write_text(json.dumps({
        "mode": args.mode, "scores_dir": args.scores_dir, "reps": args.reps,
        "augments": args.augments, "boost": args.boost,
        "meta_training": "calibration labels only; test labels used for evaluation",
        "warning": "Exploratory; all current target panels previously inspected.",
        "score_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
    }, indent=2) + "\n", encoding="utf-8")
    print(pd.read_csv(args.out.with_name(args.out.stem + "_summary.csv")).to_string(index=False))


if __name__ == "__main__":
    main()
