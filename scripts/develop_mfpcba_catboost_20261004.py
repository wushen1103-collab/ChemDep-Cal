#!/usr/bin/env python3
"""Matched CatBoost molecular selector on exposed MF-PCBA assay chains."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from rdkit import RDLogger
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier

from develop_mfpcba_multifp_rf_20261004 import fingerprints
from develop_mfpcba_readout_context_20261004 import quality_frame
from evaluate_mfpcba_twenty_20261004 import choose


def run_task(root_s: str, campaign: str, name: str) -> tuple[list[dict], list[dict]]:
    repo = Path(root_s)
    directory = repo / f"data/mfpcba_{campaign}_20261004"
    cohort_file = (directory / f"{name}_cohort.parquet" if campaign == "twenty"
                   else directory / "cohorts" / f"{name}_cohort.parquet")
    RDLogger.DisableLog("rdApp.*")
    cohort = pd.read_parquet(cohort_file)
    fp = fingerprints(cohort.canonical_smiles)
    quality, columns = ((np.empty((len(cohort), 0)), []) if campaign == "twenty"
                        else quality_frame(directory, name, cohort))
    refdir = ("mfpcba_multifp_rf_twenty_development_20261004" if campaign == "twenty"
              else "mfpcba_multifp_rf_development_20261004")
    reference = pd.read_csv(repo / "data" / refdir / "cells.csv")
    ref_method = "multifp_score_rf" if campaign == "twenty" else "multifp_score_readout_rf"
    cells, predictions = [], []
    for seed in range(1, 6):
        split_file = (directory / f"{name}_seed{seed}_indices.npz" if campaign == "twenty"
                      else directory / "cohorts" / f"{name}_seed{seed}_indices.npz")
        with np.load(split_file) as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        cal, test = cohort.iloc[cal_idx], cohort.iloc[test_idx]
        center = float(np.median(cal.score))
        scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        x_cal = np.column_stack([fp[cal_idx],
                                 expit((cal.score.to_numpy(float) - center) / scale),
                                 quality[cal_idx]])
        x_test = np.column_stack([fp[test_idx],
                                  expit((test.score.to_numpy(float) - center) / scale),
                                  quality[test_idx]])
        y_cal, y_test = cal.label.to_numpy(int), test.label.to_numpy(int)
        rf = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, max_features=.2,
            class_weight="balanced_subsample", n_jobs=8,
            random_state=862026 + seed).fit(x_cal, y_cal)
        rf_score = rf.predict_proba(x_test)[:, 1]
        cat = CatBoostClassifier(
            iterations=600, depth=5, learning_rate=.04, l2_leaf_reg=5,
            loss_function="Logloss", auto_class_weights="Balanced",
            random_seed=862026 + seed, thread_count=8,
            verbose=False, allow_writing_files=False).fit(x_cal, y_cal)
        cat_score = cat.predict_proba(x_test)[:, 1]
        keys = np.asarray([int.from_bytes(hashlib.sha256(
            f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        blocks = test.murcko_scaffold.where(
            test.murcko_scaffold.ne(""), "singleton:" + test.canonical_smiles).to_numpy(str)
        for method, prediction in (("multifp_rf_replay", rf_score),
                                   ("multifp_catboost", cat_score)):
            selected = choose(prediction, blocks, keys)
            if len(selected) != 50 or len(set(blocks[selected])) != 50:
                raise AssertionError("Budget or scaffold-cap violation")
            hits = int(y_test[selected].sum())
            cells.append({"task": name, "seed": seed, "method": method,
                          "hits": hits, "test_n": len(test),
                          "test_pos": int(y_test.sum()),
                          "quality_dimensions": len(columns)})
            if method == "multifp_rf_replay":
                expected = int(reference.loc[reference.task.eq(name)
                                             & reference.seed.eq(seed)
                                             & reference.method.eq(ref_method), "hits"].iloc[0])
                if hits != expected:
                    raise AssertionError(f"RF replay mismatch: {name}/{seed}")
            predictions.extend({"task": name, "seed": seed, "method": method,
                                "canonical_smiles": smiles, "score": float(score)}
                               for smiles, score in zip(test.canonical_smiles, prediction))
        print(name, seed, "completed", flush=True)
    return cells, predictions


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
    out = repo / f"data/mfpcba_catboost_{args.campaign}_development_20261004"
    out.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame([row for group in groups for row in group[0]])
    predictions = pd.DataFrame([row for group in groups for row in group[1]])
    cells.to_csv(out / "cells.csv", index=False)
    predictions.to_parquet(out / "test_predictions.parquet", index=False)
    assay = cells.groupby(["task", "method"]).hits.agg(["mean", "std"]).reset_index()
    assay.to_csv(out / "assay_mean_sd.csv", index=False)
    print(assay.groupby("method")["mean"].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
