#!/usr/bin/env python3
"""Matched pooled, augmented molecular comparator for inspected fifth cohort."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBClassifier

from evaluate_mfpcba_twenty_20261004 import molecular_features


def run_seed(root_s: str, seed: int) -> int:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from probe_meta_band_20261003 import BOOSTS, FRACTIONS
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import cap_one, load_panel

    RDLogger.DisableLog("rdApp.*")
    paths = sorted((root / "data/chembl_meta_ungated_fifth_xgb_20261003"
                    / f"seed{seed}" / "scores").glob("*_scores.csv"))
    if len(paths) != 10:
        raise AssertionError("Expected original ten targets")
    panels = [load_panel(path) for path in paths]
    prior = pd.read_csv(root / "data/pr_fifth_molecular_confirmation_posthoc_20261004"
                        / f"seed{seed}_cells.csv")
    expected = {(r.target, int(r.rep), float(r.fraction), float(r.boost)):
                (int(r.chemdep_hits), int(r.scorecap_hits), int(r.ecfp_score_hits))
                for r in prior.itertuples(index=False)}
    if len(expected) != 400:
        raise AssertionError("Prior molecular comparison incomplete")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_x, train_y, test_fingerprints = [], [], {}
    target_names = [panel["target"] for panel in panels]
    for target_index, panel in enumerate(panels):
        target = panel["target"]
        cal = panel["cal"]
        fp_cal = molecular_features(cal.canonical_smiles, generator)
        test_fingerprints[target] = molecular_features(panel["test"].canonical_smiles, generator)
        labels = cal.label.to_numpy(int)
        scores = cal.score.to_numpy(float)
        marker = np.zeros((len(cal), 10), dtype=np.float32)
        marker[:, target_index] = 1.
        views = [scores]
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        for rep in range(3):
            for fraction in FRACTIONS:
                rng = np.random.default_rng(852026 + target_hash + rep * 1009 + int(fraction * 10000))
                artifacts = choose_artifact_blocks(
                    cal, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2)
                for boost in BOOSTS:
                    changed = inject_artifact(
                        cal, scores, artifacts,
                        artifact_block_col="murcko_scaffold", logit_boost=boost)[0]
                    views.append(changed)
        if len(views) != 13:
            raise AssertionError("Augmentation view mismatch")
        for view in views:
            train_x.append(np.column_stack([fp_cal, view, marker]).astype(np.float32))
            train_y.append(labels)
    x = np.concatenate(train_x)
    y = np.concatenate(train_y)
    positives = int(y.sum())
    model = XGBClassifier(
        n_estimators=160, max_depth=3, learning_rate=.05,
        subsample=.9, colsample_bytree=.9, min_child_weight=8,
        reg_lambda=4., tree_method="hist", eval_metric="logloss",
        n_jobs=4, random_state=862026,
        scale_pos_weight=min(20., max(1., (len(y) - positives) / positives)),
    ).fit(x, y)
    output = root / "data/pr_fifth_pooled_molecular_posthoc_20261004"
    output.mkdir(parents=True, exist_ok=True)
    model.save_model(output / f"seed{seed}.pooled_ecfp_score.json")
    rows = []
    for panel_index, panel in enumerate(panels):
        target = panel["target"]
        test = panel["test"]
        fp_test = test_fingerprints[target]
        labels = test.label.to_numpy(int)
        base = test.score.to_numpy(float)
        blocks = panel["test_blocks"]
        marker = np.zeros((len(test), 10), dtype=np.float32)
        marker[:, panel_index] = 1.
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = 355700 + panel_index * 100_000 + target_hash
        for rep in range(10):
            for fraction in (.2, .3):
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
                artifacts = choose_artifact_blocks(
                    test, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2)
                for boost in (4., 6.):
                    scores = inject_artifact(
                        test, base, artifacts,
                        artifact_block_col="murcko_scaffold", logit_boost=boost)[0]
                    key = (target, rep, fraction, boost)
                    chemdep, scorecap, target_ecfp = expected[key]
                    if int(labels[cap_one(scores, blocks, 50)].sum()) != scorecap:
                        raise AssertionError(f"Frozen score-cap mismatch {key}")
                    x_test = np.column_stack([fp_test, scores, marker]).astype(np.float32)
                    chosen = cap_one(model.predict_proba(x_test)[:, 1], blocks, 50)
                    if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                        raise AssertionError("Capacity violation")
                    rows.append({"target": target, "seed": seed, "rep": rep,
                                 "fraction": fraction, "boost": boost,
                                 "chemdep_hits": chemdep, "scorecap_hits": scorecap,
                                 "target_ecfp_hits": target_ecfp,
                                 "pooled_ecfp_hits": int(labels[chosen].sum())})
        print(seed, target, "completed", flush=True)
    result = pd.DataFrame(rows)
    if len(result) != 400:
        raise AssertionError("Incomplete augmented molecular comparison")
    result.to_csv(output / f"seed{seed}_cells.csv", index=False)
    (output / f"seed{seed}_manifest.json").write_text(json.dumps({
        "status": "post hoc on inspected fifth cohort; no new independent confirmation",
        "seed": seed, "train_rows": len(y), "train_views_per_target": 13,
        "features": "ECFP4 2048, perturbed score, 10 target indicators",
        "score_file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "frozen_scorecap_cells_replayed": 400,
        "test_label_use": "original label-aware injection and final hit evaluation",
    }, indent=2) + "\n", encoding="utf-8")
    return len(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        print(list(pool.map(run_seed, [str(args.root.resolve())] * 5,
                            range(35501, 35506))), flush=True)


if __name__ == "__main__":
    main()
