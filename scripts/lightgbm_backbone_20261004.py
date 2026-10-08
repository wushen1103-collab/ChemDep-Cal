#!/usr/bin/env python3
"""Post hoc fifth-cohort scorer-backbone sensitivity: ECFP4-LightGBM."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight


SEEDS = (35501, 35502, 35503, 35504, 35505)
METHODS = ("chemdep_cal", "score_cap1")


def read_manifest(root: Path) -> pd.DataFrame:
    path = root / "data/chembl_meta_ungated_fifth_cohort_20261003/screening_manifest.csv"
    frame = pd.read_csv(path, dtype={"passes_smoke_threshold": "string"})
    eligible = frame[frame.passes_smoke_threshold.str.lower().eq("true")].copy()
    if len(eligible) != 10 or not eligible.target_chembl_id.is_unique:
        raise AssertionError("Frozen fifth-cohort target eligibility changed")
    return eligible.sort_values("target_chembl_id")


def score_one(task: tuple[str, str, str, int, int]) -> dict:
    root_s, path_s, expected_sha, seed, index = task
    root = Path(root_s)
    path = root / path_s
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise AssertionError(f"Panel hash mismatch: {path}")
    sys.path.insert(0, str(root / "scripts"))
    from train_chembl_ecfp_xgb import fingerprint_matrix

    frame = pd.read_csv(path)
    train = frame[frame["split"].eq("train")]
    y_train = train.label.to_numpy(np.int8)
    x_train = fingerprint_matrix(train.ecfp4_radius2_nbits2048)
    model = LGBMClassifier(
        n_estimators=350, max_depth=4, num_leaves=15, learning_rate=0.04,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=20, reg_lambda=1.0, objective="binary",
        deterministic=True, force_col_wise=True, n_jobs=4,
        random_state=seed + index, verbosity=-1,
    )
    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(x_train, y_train, sample_weight=weights)
    scored = frame.copy()
    metrics = {"target": path.stem.replace("_panel", ""), "seed": seed}
    for split in ("train", "calibration", "testpool"):
        mask = frame["split"].eq(split).to_numpy()
        scores = model.predict_proba(fingerprint_matrix(frame.loc[mask, "ecfp4_radius2_nbits2048"]))[:, 1]
        scored.loc[mask, "score"] = scores
        truth = frame.loc[mask, "label"].to_numpy(np.int8)
        metrics[f"{split}_auroc"] = float(roc_auc_score(truth, scores))
        metrics[f"{split}_auprc"] = float(average_precision_score(truth, scores))
    scored["scorer"] = "ecfp4_lightgbm_4.6.0"
    scored["score_rank_in_split"] = scored.groupby("split").score.rank(
        method="first", ascending=False).astype(int)
    out = root / "data/pr_lightgbm_fifth_20261004" / f"seed{seed}"
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    scored.to_csv(out / "scores" / f"{metrics['target']}_scores.csv", index=False)
    model.booster_.save_model(str(out / "models" / f"{metrics['target']}_lightgbm.txt"))
    return metrics


def score_all(root: Path) -> None:
    manifest = read_manifest(root)
    tasks = [(str(root), str(row.panel_path), str(row.panel_sha256), seed, index)
             for seed in SEEDS for index, row in enumerate(manifest.itertuples(index=False))]
    with futures.ProcessPoolExecutor(max_workers=24) as pool:
        rows = list(pool.map(score_one, tasks))
    result = pd.DataFrame(rows).sort_values(["seed", "target"])
    if len(result) != 50 or result.duplicated(["seed", "target"]).any():
        raise AssertionError("Incomplete LightGBM scorer grid")
    out = root / "data/pr_lightgbm_fifth_20261004"
    result.to_csv(out / "scorer_metrics.csv", index=False)
    (out / "scorer_manifest.json").write_text(json.dumps({
        "status": "post hoc alternative upstream scorer on inspected fifth cohort",
        "library": "lightgbm==4.6.0", "seeds": SEEDS,
        "workers": 24, "threads_per_scorer": 4, "maximum_threads": 96,
        "panel_sha256": dict(zip(manifest.target_chembl_id, manifest.panel_sha256)),
        "parameters": "350 trees, depth 4, 15 leaves, learning rate .04, balanced training weights; no tuning on selector hits",
    }, indent=2) + "\n", encoding="utf-8")
    print(result.groupby("seed").testpool_auprc.mean().to_string(), flush=True)


def evaluate_one(root_s: str, seed: int) -> tuple[int, int]:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    import probe_meta_band_v2_20261003  # Registers the 16-feature map.
    from probe_meta_band_20261003 import training_matrix
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import cap_one, fit_ranker, load_panel

    directory = root / "data/pr_lightgbm_fifth_20261004" / f"seed{seed}"
    paths = sorted((directory / "scores").glob("*_scores.csv"))
    if len(paths) != 10:
        raise AssertionError(f"Expected ten LightGBM score files, found {len(paths)}")
    panels = [load_panel(path) for path in paths]
    x, y = training_matrix(panels, repeats=3)
    ranker = fit_ranker(x, y, 862026)
    ranker.save_model(directory / "chemdep_ranker.json")
    rows = []
    for index, panel in enumerate(panels):
        test = panel["test"]
        target = panel["target"]
        labels = test.label.to_numpy(int)
        blocks = panel["test_blocks"]
        base = test.score.to_numpy(float)
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = 355700 + index * 100_000 + target_hash
        for rep in range(10):
            for fraction in (.2, .3):
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
                artifacts = choose_artifact_blocks(
                    test, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2,
                )
                for boost in (4., 6.):
                    scores = inject_artifact(
                        test, base, artifacts, artifact_block_col="murcko_scaffold",
                        logit_boost=boost,
                    )[0]
                    features = evidence_features(scores, blocks, panel["null"])
                    selections = {
                        "chemdep_cal": cap_one(ranker.predict_proba(features)[:, 1], blocks, 50),
                        "score_cap1": cap_one(scores, blocks, 50),
                    }
                    for method, chosen in selections.items():
                        if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                            raise AssertionError("Selection budget or scaffold capacity changed")
                        rows.append({"target": target, "scorer_seed": seed, "rep": rep,
                                     "fraction": fraction, "boost": boost,
                                     "method": method, "hits": int(labels[chosen].sum()),
                                     "selected": 50, "scaffolds": 50})
        print(f"seed{seed} {target}", flush=True)
    result = pd.DataFrame(rows)
    if len(result) != 800 or result.duplicated(
            ["target", "scorer_seed", "rep", "fraction", "boost", "method"]).any():
        raise AssertionError("Incomplete fifth-cohort LightGBM selector grid")
    result.to_csv(directory / "selector_cells.csv", index=False)
    (directory / "selector_manifest.json").write_text(json.dumps({
        "status": "post hoc backbone sensitivity; not an independent cohort confirmation",
        "scorer_seed": seed, "ranker_training": "one pooled ranker over ten labelled calibration panels per scorer seed",
        "training_views": 13, "budget": 50, "scaffold_capacity": 1,
        "score_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }, indent=2) + "\n", encoding="utf-8")
    return seed, len(result)


def evaluate_all(root: Path) -> None:
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        completed = list(pool.map(evaluate_one, [str(root)] * 5, SEEDS))
    print(completed, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("score", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "score":
        score_all(args.root.resolve())
    else:
        evaluate_all(args.root.resolve())


if __name__ == "__main__":
    main()
