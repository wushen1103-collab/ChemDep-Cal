#!/usr/bin/env python3
"""Post hoc matched 128-confirmation-label development on exposed assays."""

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
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from develop_mfpcba_multifp_rf_20261004 import fingerprints
from evaluate_mfpcba_twenty_20261004 import choose, molecular_features


def run_seed(root_s: str, seed: int, budget: int) -> tuple[list[dict], list[dict]]:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    import probe_meta_band_v2_20261003  # noqa: F401; install 16-feature map
    from probe_meta_band_20261003 import training_matrix
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_selection import block_array
    from train_artifact_meta_ranker_20261003 import fit_ranker

    RDLogger.DisableLog("rdApp.*")
    directory = root / "data/mfpcba_twenty_20261004"
    manifest = json.loads((directory / "cohort_manifest.json").read_text())
    panels, ledger = [], []
    for name in manifest["eligible_tasks"]:
        cohort = pd.read_parquet(directory / f"{name}_cohort.parquet")
        with np.load(directory / f"{name}_seed{seed}_indices.npz") as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        if len(cal_idx) < budget:
            ledger.append({"task": name, "seed": seed, "budget": budget,
                           "eligible": False, "reason": "insufficient_calibration_pool"})
            continue
        task_hash = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
        rng = np.random.default_rng(20261004 + seed * 100003 + task_hash)
        sparse_idx = rng.choice(cal_idx, size=budget, replace=False)
        cal = cohort.iloc[sparse_idx].reset_index(drop=True).copy()
        test = cohort.iloc[test_idx].reset_index(drop=True).copy()
        pos = int(cal.label.sum())
        if pos < 5 or budget - pos < 5:
            ledger.append({"task": name, "seed": seed, "budget": budget,
                           "eligible": False, "reason": "fewer_than_five_in_either_class",
                           "cal_pos": pos})
            continue
        center = float(np.median(cal.score))
        scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        if scale <= 0:
            ledger.append({"task": name, "seed": seed, "budget": budget,
                           "eligible": False, "reason": "zero_score_iqr", "cal_pos": pos})
            continue
        for frame in (cal, test):
            frame["score"] = expit((frame.score.to_numpy(float) - center) / scale)
            frame["molecule_chembl_id"] = frame.canonical_smiles
            frame["target_chembl_id"] = name
        panels.append({"target": name, "cal": cal, "test": test,
                       "null": cal.loc[cal.label.eq(0), "score"].to_numpy(float),
                       "cal_blocks": block_array(cal, "murcko_scaffold"),
                       "test_blocks": block_array(test, "murcko_scaffold"),
                       "cohort": cohort, "sparse_idx": sparse_idx, "test_idx": test_idx})
        ledger.append({"task": name, "seed": seed, "budget": budget,
                       "eligible": True, "reason": "", "cal_pos": pos})
    if not panels:
        raise AssertionError("No eligible task at sparse budget")
    x, y = training_matrix(panels, repeats=1)
    chemdep = fit_ranker(x, y, 862026)
    out = root / "data/mfpcba_low_label_development_20261004"
    out.mkdir(parents=True, exist_ok=True)
    chemdep.save_model(out / f"seed{seed}.budget{budget}.chemdep.json")
    cells = []
    for panel in panels:
        name, cal, test = panel["target"], panel["cal"], panel["test"]
        fp = fingerprints(panel["cohort"].canonical_smiles)
        ecfp_fp = molecular_features(
            panel["cohort"].canonical_smiles,
            rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048))
        cal_idx, test_idx = panel["sparse_idx"], panel["test_idx"]
        x_cal = np.column_stack([fp[cal_idx], cal.score.to_numpy(float)])
        x_test = np.column_stack([fp[test_idx], test.score.to_numpy(float)])
        y_cal, y_test = cal.label.to_numpy(int), test.label.to_numpy(int)
        rf = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, max_features=.2,
            class_weight="balanced_subsample", n_jobs=4,
            random_state=862026 + seed).fit(x_cal, y_cal)
        positives = max(int(y_cal.sum()), 1)
        ecfp = XGBClassifier(
            n_estimators=160, max_depth=3, learning_rate=.05,
            subsample=.9, colsample_bytree=.9, min_child_weight=8,
            reg_lambda=4., tree_method="hist", eval_metric="logloss",
            n_jobs=4, random_state=862026,
            scale_pos_weight=min(20., max(1., (len(y_cal) - positives) / positives)),
        ).fit(np.column_stack([ecfp_fp[cal_idx], cal.score.to_numpy(float)]), y_cal)
        ecfp_test = np.column_stack([ecfp_fp[test_idx], test.score.to_numpy(float)])
        blocks = panel["test_blocks"]
        keys = np.asarray([int.from_bytes(hashlib.sha256(
            f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        values = {
            "chemdep_cal_sparse": chemdep.predict_proba(
                evidence_features(test.score.to_numpy(float), blocks, panel["null"]))[:, 1],
            "score_cap1_sparse": test.score.to_numpy(float),
            "multifp_rf_sparse": rf.predict_proba(x_test)[:, 1],
            "ecfp_score_xgb_sparse": ecfp.predict_proba(ecfp_test)[:, 1],
        }
        for method, prediction in values.items():
            selected = choose(prediction, blocks, keys)
            if len(selected) != 50 or len(set(blocks[selected])) != 50:
                raise AssertionError("Capacity or scaffold violation")
            cells.append({"task": name, "seed": seed, "budget": budget,
                          "method": method, "hits": int(y_test[selected].sum()),
                          "cal_pos": int(y_cal.sum()), "test_n": len(test),
                          "test_pos": int(y_test.sum())})
        print(name, seed, budget, "completed", flush=True)
    return cells, ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=128)
    args = parser.parse_args()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        groups = list(pool.map(run_seed, [str(args.root.resolve())] * 5,
                               range(1, 6), [args.budget] * 5))
    out = args.root.resolve() / "data/mfpcba_low_label_development_20261004"
    cells = pd.DataFrame([row for group in groups for row in group[0]])
    ledger = pd.DataFrame([row for group in groups for row in group[1]])
    cells.to_csv(out / f"budget{args.budget}_cells.csv", index=False)
    ledger.to_csv(out / f"budget{args.budget}_ledger.csv", index=False)
    complete = set(ledger.groupby("task").eligible.all().loc[lambda s: s].index)
    assay = cells.loc[cells.task.isin(complete)].groupby(["task", "method"]).hits.agg(
        ["mean", "std"]).reset_index()
    assay.to_csv(out / f"budget{args.budget}_assay_mean_sd.csv", index=False)
    print("Complete tasks", len(complete), flush=True)
    print(assay.groupby("method")["mean"].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
