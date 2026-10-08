#!/usr/bin/env python3
"""Post hoc top-50 ranking-objective development on exposed assay chains."""

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
from scipy.special import expit
from xgboost import XGBClassifier, XGBRanker

from develop_mfpcba_multifp_rf_20261004 import fingerprints
from develop_mfpcba_readout_context_20261004 import quality_frame
from evaluate_mfpcba_twenty_20261004 import choose


def run_task(root_s: str, name: str) -> list[dict]:
    repo = Path(root_s)
    sys.path[:0] = [str(repo / "src"), str(repo / "scripts")]
    import probe_meta_band_v2_20261003  # noqa: F401; install 16-feature map
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_selection import block_array

    directory = repo / "data/mfpcba_twelve_new_20261004"
    output = repo / "data/mfpcba_decision_ranker_development_20261004"
    output.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")
    cohort = pd.read_parquet(directory / "cohorts" / f"{name}_cohort.parquet")
    fp = fingerprints(cohort.canonical_smiles)
    quality, columns = quality_frame(directory, name, cohort)
    rf = pd.read_csv(repo / "data/mfpcba_multifp_rf_development_20261004/cells.csv")
    rf = rf.loc[rf.task.eq(name) & rf.method.eq("multifp_score_readout_rf")]
    rows = []
    for seed in range(1, 6):
        with np.load(directory / "cohorts" / f"{name}_seed{seed}_indices.npz") as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        cal = cohort.iloc[cal_idx].reset_index(drop=True).copy()
        test = cohort.iloc[test_idx].reset_index(drop=True).copy()
        center = float(np.median(cal.score))
        scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        for frame in (cal, test):
            frame["score"] = expit((frame.score.to_numpy(float) - center) / scale)
            frame["molecule_chembl_id"] = frame.canonical_smiles
            frame["target_chembl_id"] = name
        labels_cal = cal.label.to_numpy(int)
        labels_test = test.label.to_numpy(int)
        x_cal = np.column_stack([fp[cal_idx], cal.score.to_numpy(float), quality[cal_idx]])
        x_test = np.column_stack([fp[test_idx], test.score.to_numpy(float), quality[test_idx]])
        cal_blocks = block_array(cal, "murcko_scaffold")
        test_blocks = block_array(test, "murcko_scaffold")
        null = cal.loc[cal.label.eq(0), "score"].to_numpy(float)
        context_cal = evidence_features(cal.score.to_numpy(float), cal_blocks, null)
        context_test = evidence_features(test.score.to_numpy(float), test_blocks, null)
        keys = np.asarray([int.from_bytes(hashlib.sha256(
            f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        methods = (
            ("multifp_readout_xgb_logloss", x_cal, x_test),
            ("multifp_readout_ndcg50", x_cal, x_test),
            ("multifp_readout_context_ndcg50",
             np.column_stack([x_cal, context_cal]),
             np.column_stack([x_test, context_test])),
        )
        for method, train_x, eval_x in methods:
            if method.endswith("logloss"):
                positives = max(int(labels_cal.sum()), 1)
                model = XGBClassifier(
                    n_estimators=200, max_depth=3, learning_rate=.05,
                    subsample=.9, colsample_bytree=.9, min_child_weight=8,
                    reg_lambda=4., tree_method="hist", eval_metric="logloss",
                    n_jobs=4, random_state=862026,
                    scale_pos_weight=min(20., max(1., (len(labels_cal) - positives) / positives)),
                ).fit(train_x, labels_cal)
                prediction = model.predict_proba(eval_x)[:, 1]
            else:
                model = XGBRanker(
                    objective="rank:ndcg", eval_metric="ndcg@50",
                    lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=50,
                    n_estimators=200, max_depth=3, learning_rate=.05,
                    subsample=.9, colsample_bytree=.9, min_child_weight=8,
                    reg_lambda=4., tree_method="hist", n_jobs=4,
                    random_state=862026,
                ).fit(train_x, labels_cal, qid=np.zeros(len(labels_cal), dtype=int))
                prediction = model.predict(eval_x)
            selected = choose(prediction, test_blocks, keys)
            if len(selected) != 50 or len(set(test_blocks[selected])) != 50:
                raise AssertionError("Budget or scaffold-cap violation")
            model.save_model(output / f"{name}.seed{seed}.{method}.json")
            rows.append({"task": name, "seed": seed, "method": method,
                         "hits": int(labels_test[selected].sum()),
                         "rf_readout_hits": int(rf.loc[rf.seed.eq(seed), "hits"].iloc[0]),
                         "quality_dimensions": len(columns), "test_n": len(test),
                         "test_pos": int(labels_test.sum())})
        print(name, seed, "completed", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.root.resolve()
    manifest = json.loads((repo / "data/mfpcba_twelve_new_20261004/cohort_manifest.json").read_text())
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        groups = list(pool.map(run_task, [str(repo)] * len(manifest["eligible_tasks"]),
                               manifest["eligible_tasks"]))
    output = repo / "data/mfpcba_decision_ranker_development_20261004"
    cells = pd.DataFrame([row for group in groups for row in group])
    cells.to_csv(output / "cells.csv", index=False)
    assay = cells.groupby(["task", "method"]).hits.agg(["mean", "std"]).reset_index()
    assay.to_csv(output / "assay_mean_sd.csv", index=False)
    print(assay.to_string(index=False), flush=True)
    print(assay.groupby("method")["mean"].mean().to_string(), flush=True)
    print("RF readout macro", cells.groupby("task").rf_readout_hits.mean().mean(), flush=True)


if __name__ == "__main__":
    main()
