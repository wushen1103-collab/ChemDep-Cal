#!/usr/bin/env python3
"""Post hoc multi-view selector development on already exposed assay chains."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, log_evaluation
from rdkit import RDLogger
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier

from develop_mfpcba_multifp_rf_20261004 import fingerprints
from develop_mfpcba_readout_context_20261004 import quality_frame
from evaluate_mfpcba_twenty_20261004 import choose


def run_task(root_s: str, campaign: str, name: str) -> list[dict]:
    repo = Path(root_s)
    directory = repo / f"data/mfpcba_{campaign}_20261004"
    cohort_file = (directory / f"{name}_cohort.parquet" if campaign == "twenty"
                   else directory / "cohorts" / f"{name}_cohort.parquet")
    cohort = pd.read_parquet(cohort_file)
    RDLogger.DisableLog("rdApp.*")
    fp = fingerprints(cohort.canonical_smiles)
    quality, columns = ((np.empty((len(cohort), 0)), []) if campaign == "twenty"
                        else quality_frame(directory, name, cohort))
    rows = []
    for seed in range(1, 6):
        split_file = (directory / f"{name}_seed{seed}_indices.npz" if campaign == "twenty"
                      else directory / "cohorts" / f"{name}_seed{seed}_indices.npz")
        with np.load(split_file) as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        cal, test = cohort.iloc[cal_idx], cohort.iloc[test_idx]
        center = float(np.median(cal.score))
        scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        cal_score = expit((cal.score.to_numpy(float) - center) / scale)
        test_score = expit((test.score.to_numpy(float) - center) / scale)
        cal_q, test_q = quality[cal_idx], quality[test_idx]
        y_cal, y_test = cal.label.to_numpy(int), test.label.to_numpy(int)
        x_cal = np.column_stack([fp[cal_idx], cal_score, cal_q])
        x_test = np.column_stack([fp[test_idx], test_score, test_q])
        score_cal = np.column_stack([cal_score, cal_q])
        score_test = np.column_stack([test_score, test_q])
        keys = np.asarray([int.from_bytes(hashlib.sha256(
            f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        blocks = test.murcko_scaffold.where(
            test.murcko_scaffold.ne(""), "singleton:" + test.canonical_smiles).to_numpy(str)

        rf = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, max_features=.2,
            class_weight="balanced_subsample", n_jobs=8,
            random_state=862026 + seed).fit(x_cal, y_cal)
        rf_pred = rf.predict_proba(x_test)[:, 1]
        boosted_cal = np.column_stack([fp[cal_idx], np.tile(score_cal, (1, 20))])
        boosted_test = np.column_stack([fp[test_idx], np.tile(score_test, (1, 20))])
        rf_boosted = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, max_features=.2,
            class_weight="balanced_subsample", n_jobs=8,
            random_state=862026 + seed).fit(boosted_cal, y_cal)
        boosted_pred = rf_boosted.predict_proba(boosted_test)[:, 1]
        lgbm = LGBMClassifier(
            n_estimators=300, learning_rate=.035, num_leaves=15,
            min_child_samples=20, colsample_bytree=.8, subsample=.9,
            subsample_freq=1, reg_lambda=5., verbosity=-1,
            n_jobs=8, random_state=862026 + seed,
            class_weight="balanced").fit(x_cal, y_cal, callbacks=[log_evaluation(0)])
        lgbm_pred = lgbm.predict_proba(x_test)[:, 1]
        methods = {
            "multifp_readout_rf_replay": rf_pred,
            "multifp_readout_rf_scoreweighted": boosted_pred,
            "multifp_readout_lgbm": lgbm_pred,
            "multifp_readout_rf_lgbm_8020": .8 * rf_pred + .2 * lgbm_pred,
            "multifp_readout_rf_weightedrf_5050": .5 * rf_pred + .5 * boosted_pred,
        }
        for method, prediction in methods.items():
            selected = choose(prediction, blocks, keys)
            if len(selected) != 50 or len(set(blocks[selected])) != 50:
                raise AssertionError("Budget or scaffold-cap violation")
            rows.append({"task": name, "seed": seed, "method": method,
                         "hits": int(y_test[selected].sum()), "test_n": len(test),
                         "test_pos": int(y_test.sum()), "quality_dimensions": len(columns)})
        print(name, seed, "completed", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", choices=("twelve_new", "twenty"),
                        default="twelve_new")
    args = parser.parse_args()
    repo = args.root.resolve()
    directory = repo / f"data/mfpcba_{args.campaign}_20261004"
    manifest = json.loads((directory / "cohort_manifest.json").read_text())
    names = manifest["eligible_tasks"]
    with futures.ProcessPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(run_task, [str(repo)] * len(names),
                               [args.campaign] * len(names), names))
    output = repo / f"data/mfpcba_multiview_{args.campaign}_development_20261004"
    output.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame([row for group in groups for row in group])
    cells.to_csv(output / "cells.csv", index=False)
    assay = cells.groupby(["task", "method"]).hits.agg(["mean", "std"]).reset_index()
    assay.to_csv(output / "assay_mean_sd.csv", index=False)
    print(assay.groupby("method")["mean"].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
