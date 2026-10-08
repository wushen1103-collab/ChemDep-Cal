#!/usr/bin/env python3
"""Post hoc training-geometry diagnosis on exposed confirmation cohorts."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from xgboost import XGBClassifier


AIDS = ("AID1798", "AID435034", "AID463087")
SIGNS = {"AID1798": 1, "AID435034": -1, "AID463087": -1}


def fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None) -> XGBClassifier:
    positive = max(int(y.sum()), 1)
    scale = 1. if weights is not None else min(20., max(1., (len(y) - positive) / positive))
    model = XGBClassifier(
        n_estimators=160, max_depth=3, learning_rate=.05,
        subsample=.9, colsample_bytree=.9, min_child_weight=8,
        reg_lambda=4., tree_method="hist", eval_metric="logloss",
        n_jobs=4, random_state=862026, scale_pos_weight=scale)
    model.fit(x, y, sample_weight=weights)
    return model


def run_seed(root_s: str, seed: int) -> tuple[int, int]:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    import probe_meta_band_v2_20261003  # Register 16 features.
    from probe_meta_band_20261003 import training_matrix
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_selection import block_array
    from train_artifact_meta_ranker_20261003 import cap_one

    base = root / "data/pubchem_primary_welqrate_20261004/confirmatory_failures"
    panels = []
    for aid in AIDS:
        frame = pd.read_parquet(base / f"{aid}_cohort.parquet")
        with np.load(base / f"{aid}_seed{seed}_indices.npz") as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        cal = frame.iloc[cal_idx].reset_index(drop=True).copy()
        test = frame.iloc[test_idx].reset_index(drop=True).copy()
        sign = SIGNS[aid]
        oriented = sign * cal.bscore.to_numpy(float)
        center = float(np.median(oriented))
        iqr = float(np.percentile(oriented, 75) - np.percentile(oriented, 25))
        cal["score"] = expit((oriented - center) / iqr)
        test["score"] = expit((sign * test.bscore.to_numpy(float) - center) / iqr)
        cal["molecule_chembl_id"] = [f"{aid}:{row}" for row in cal_idx]
        test["molecule_chembl_id"] = [f"{aid}:{row}" for row in test_idx]
        panels.append({"target": aid, "cal": cal, "test": test,
                       "null": cal.loc[cal.label.eq(0), "score"].to_numpy(float),
                       "cal_blocks": block_array(cal, "murcko_scaffold"),
                       "test_blocks": block_array(test, "murcko_scaffold")})

    natural_x, natural_y = training_matrix(panels, repeats=0)
    natural_model = fit(natural_x, natural_y, None)
    local_models = {}
    for panel in panels:
        x, y = training_matrix([panel], repeats=0)
        local_models[panel["target"]] = fit(x, y, None)
    augmented_x, augmented_y = training_matrix(panels, repeats=3)
    weights = []
    for panel in panels:
        labels = panel["cal"].label.to_numpy(int)
        n_pos = max(int(labels.sum()), 1)
        n_neg = max(int((labels == 0).sum()), 1)
        base_weight = np.where(labels == 1, 1. / n_pos, 1. / n_neg)
        weights.extend([base_weight * 6.5] + [base_weight * (6.5 / 12)] * 12)
    sample_weight = np.concatenate(weights)
    sample_weight /= sample_weight.mean()
    if len(sample_weight) != len(augmented_y):
        raise AssertionError("Augmentation-weight geometry mismatch")
    balanced_model = fit(augmented_x, augmented_y, sample_weight)

    rows = []
    for panel in panels:
        aid = panel["target"]
        test = panel["test"]
        blocks = panel["test_blocks"]
        labels = test.label.to_numpy(int)
        scores = test.score.to_numpy(float)
        features = evidence_features(scores, blocks, panel["null"])
        models = {"pooled_natural_only": natural_model,
                  "target_natural_only": local_models[aid],
                  "task_class_scenario_balanced": balanced_model}
        for name, model in models.items():
            chosen = cap_one(model.predict_proba(features)[:, 1], blocks, 50)
            rows.append({"aid": aid, "split_seed": seed, "method": name,
                         "hits": int(labels[chosen].sum()),
                         "selected": len(chosen), "scaffolds": len(set(blocks[chosen]))})
    output = base / "selectors" / f"seed{seed}_training_diagnosis_posthoc.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(seed, "training diagnosis done", flush=True)
    return seed, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        print(list(pool.map(run_seed, [str(args.root.resolve())] * 5, range(1, 6))), flush=True)


if __name__ == "__main__":
    main()
