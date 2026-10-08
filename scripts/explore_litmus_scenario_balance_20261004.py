#!/usr/bin/env python3
"""Post hoc diagnosis of natural/augmented training imbalance on exposed AIDs."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


AIDS = ("AID1798", "AID435034", "AID463087")


def model_fit(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> XGBClassifier:
    positives = max(int(y.sum()), 1)
    scale = min(20., max(1., (len(y) - positives) / positives))
    model = XGBClassifier(
        n_estimators=160, max_depth=3, learning_rate=.05,
        subsample=.9, colsample_bytree=.9, min_child_weight=8,
        reg_lambda=4., tree_method="hist", eval_metric="logloss",
        n_jobs=4, random_state=862026, scale_pos_weight=scale)
    model.fit(x, y, sample_weight=weight)
    return model


def run_seed(root_s: str, seed: int) -> int:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    import probe_meta_band_v2_20261003  # Registers the 16-feature map.
    from probe_meta_band_20261003 import training_matrix
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from run_chembl_selection import block_array
    from train_artifact_meta_ranker_20261003 import cap_one

    base = root / "data/pr_welqrate_litmus_20261004"
    panels = []
    for aid in AIDS:
        frame = pd.read_parquet(base / "prep" / f"{aid}_molecules.parquet")
        with np.load(base / "prep" / f"{aid}_seed{seed}_indices.npz") as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        with np.load(base / "scores" / f"{aid}_seed{seed}_scores.npz") as scored:
            cal_score, test_score = scored["calibration"], scored["test"]
        cal = frame.iloc[cal_idx].reset_index(drop=True).copy()
        test = frame.iloc[test_idx].reset_index(drop=True).copy()
        cal["molecule_chembl_id"] = [f"{aid}:{row}" for row in cal_idx]
        test["molecule_chembl_id"] = [f"{aid}:{row}" for row in test_idx]
        cal["score"], test["score"] = cal_score, test_score
        panels.append({"target": aid, "cal": cal, "test": test,
                       "null": cal.loc[cal.label.eq(0), "score"].to_numpy(float),
                       "cal_blocks": block_array(cal, "murcko_scaffold"),
                       "test_blocks": block_array(test, "murcko_scaffold")})

    x, y = training_matrix(panels, repeats=3)
    pieces, natural_indices = [], []
    offset = 0
    for panel in panels:
        n = len(panel["cal"])
        pieces.append(np.concatenate([np.full(n, 6.5), np.full(n * 12, .5 * 13 / 12)]))
        natural_indices.extend(range(offset, offset + n))
        offset += n * 13
    weights = np.concatenate(pieces)
    natural_indices = np.asarray(natural_indices, dtype=int)
    if len(weights) != len(y):
        raise AssertionError("Training geometry mismatch")
    models = {
        "natural_only": model_fit(x[natural_indices], y[natural_indices],
                                  np.ones(len(natural_indices))),
        "balanced_50_50": model_fit(x, y, weights),
    }
    rows = []
    for index, panel in enumerate(panels):
        aid = panel["target"]
        test = panel["test"]
        blocks = panel["test_blocks"]
        labels = test.label.to_numpy(int)
        original = test.score.to_numpy(float)

        def record(scenario: str, rep: int, fraction: float, boost: float,
                   scores: np.ndarray) -> None:
            features = evidence_features(scores, blocks, panel["null"])
            for name, model in models.items():
                prob = model.predict_proba(features)[:, 1]
                chosen = cap_one(prob, blocks, 50)
                rows.append({"aid": aid, "split_seed": seed, "scenario": scenario,
                             "rep": rep, "fraction": fraction, "boost": boost,
                             "method": name, "hits": int(labels[chosen].sum())})

        record("natural", -1, 0., 0., original)
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
                        test, original, artifacts, artifact_block_col="murcko_scaffold",
                        logit_boost=boost)[0]
                    record("synthetic_high_artifact", rep, fraction, boost, changed)
        print(seed, aid, "done", flush=True)
    out = base / "selectors" / f"seed{seed}_scenario_balance_posthoc.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        print(list(pool.map(run_seed, [str(args.root.resolve())] * 5, range(1, 6))), flush=True)


if __name__ == "__main__":
    main()
