#!/usr/bin/env python3
"""Train fixed LightGBM scorers on the eligible WelQrate/Litmus panels."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight


def score_one(task: tuple[str, str, int]) -> dict:
    root_s, aid, seed = task
    root = Path(root_s)
    prep = root / "data/pr_welqrate_litmus_20261004/prep"
    output = root / "data/pr_welqrate_litmus_20261004/scores"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(prep / f"{aid}_molecules.parquet")
    labels = frame.label.to_numpy(np.int8)
    packed = np.load(prep / f"{aid}_ecfp4_packed.npy", mmap_mode="r")
    with np.load(prep / f"{aid}_seed{seed}_indices.npz") as indices:
        train, cal, test = (indices[name].copy() for name in ("train", "calibration", "test"))
    if min(labels[train].sum(), labels[cal].sum(), labels[test].sum()) < 5:
        raise AssertionError(f"{aid}/seed{seed}: active-count eligibility failed")
    x_train = np.unpackbits(packed[train], axis=1, bitorder="little")
    y_train = labels[train]
    model = LGBMClassifier(
        n_estimators=350, max_depth=4, num_leaves=15, learning_rate=.04,
        subsample=.9, subsample_freq=1, colsample_bytree=.8,
        min_child_samples=20, reg_lambda=1., objective="binary",
        deterministic=True, force_col_wise=True, n_jobs=8,
        random_state=20261004 + seed + int(aid[3:]), verbosity=-1,
    )
    model.fit(x_train, y_train,
              sample_weight=compute_sample_weight(class_weight="balanced", y=y_train))
    score_cal = model.predict_proba(np.unpackbits(packed[cal], axis=1, bitorder="little"))[:, 1]
    score_test = model.predict_proba(np.unpackbits(packed[test], axis=1, bitorder="little"))[:, 1]
    score_path = output / f"{aid}_seed{seed}_scores.npz"
    np.savez_compressed(score_path, calibration=score_cal, test=score_test)
    model.booster_.save_model(str(output / f"{aid}_seed{seed}_lightgbm.txt"))
    return {"aid": aid, "seed": seed, "n_train": len(train), "n_calibration": len(cal),
            "n_test": len(test), "active_train": int(y_train.sum()),
            "active_calibration": int(labels[cal].sum()), "active_test": int(labels[test].sum()),
            "test_auroc": float(roc_auc_score(labels[test], score_test)),
            "test_auprc": float(average_precision_score(labels[test], score_test)),
            "score_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    prep = root / "data/pr_welqrate_litmus_20261004/prep"
    audit = pd.read_csv(prep / "split_audit.csv")
    if len(audit) != 15 or not audit.eligible.all():
        raise AssertionError("The frozen 15-panel eligibility rule is not met")
    tasks = [(str(root), str(row.aid), int(row.seed)) for row in audit.itertuples(index=False)]
    with futures.ProcessPoolExecutor(max_workers=9) as pool:
        rows = list(pool.map(score_one, tasks))
    result = pd.DataFrame(rows).sort_values(["aid", "seed"])
    if len(result) != 15 or result.duplicated(["aid", "seed"]).any():
        raise AssertionError("Incomplete scorer grid")
    output = root / "data/pr_welqrate_litmus_20261004/scores"
    result.to_csv(output / "scorer_metrics.csv", index=False)
    (output / "scorer_manifest.json").write_text(json.dumps({
        "status": "post hoc third-party Litmus-derived WelQrate split; no test-based scorer tuning",
        "scorer": "ECFP4 radius2 2048 + LightGBM 4.6.0",
        "parameters": {"trees": 350, "max_depth": 4, "num_leaves": 15,
                       "learning_rate": .04, "subsample": .9, "colsample_bytree": .8,
                       "min_child_samples": 20, "reg_lambda": 1., "balanced_train_weights": True},
        "parallel_workers": 9, "threads_per_model": 8,
        "split_audit_sha256": hashlib.sha256((prep / "split_audit.csv").read_bytes()).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")
    print(result[["aid", "seed", "active_test", "test_auroc", "test_auprc"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
