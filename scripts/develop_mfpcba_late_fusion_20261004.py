#!/usr/bin/env python3
"""Development-only cross-fitted structure/readout expert fusion."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from develop_mfpcba_readout_context_20261004 import classifier, quality_frame
from evaluate_mfpcba_twenty_20261004 import choose, molecular_features


def fit_cross_fusion(x_struct: np.ndarray, x_readout: np.ndarray,
                     labels: np.ndarray, groups: np.ndarray, seed: int):
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=47000 + seed)
    out_of_fold = np.full((len(labels), 2), np.nan)
    for train, holdout in splitter.split(x_struct, labels, groups):
        if len(np.unique(labels[train])) != 2:
            raise ValueError("Cross-fit fold lacks one class")
        for index, matrix in enumerate((x_struct, x_readout)):
            model = classifier(labels[train]).fit(matrix[train], labels[train])
            out_of_fold[holdout, index] = model.predict_proba(matrix[holdout])[:, 1]
    if not np.isfinite(out_of_fold).all():
        raise AssertionError("Incomplete OOF predictions")
    logits = logit(np.clip(out_of_fold, 1e-4, 1. - 1e-4))
    meta = LogisticRegression(C=.1, class_weight="balanced", max_iter=1000,
                              random_state=862026).fit(logits, labels)
    return meta, out_of_fold


def run_task(root_s: str, name: str) -> list[dict]:
    repo = Path(root_s)
    directory = repo / "data/mfpcba_twelve_new_20261004"
    output = repo / "data/mfpcba_late_fusion_development_20261004"
    output.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")
    cohort = pd.read_parquet(directory / "cohorts" / f"{name}_cohort.parquet")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = molecular_features(cohort.canonical_smiles, generator)
    quality, columns = quality_frame(directory, name, cohort)
    reference = pd.read_csv(repo / "data/mfpcba_readout_development_20261004/cells.csv")
    reference = reference.loc[reference.task.eq(name)]
    rows = []
    for seed in range(1, 6):
        with np.load(directory / "cohorts" / f"{name}_seed{seed}_indices.npz") as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        cal = cohort.iloc[cal_idx].reset_index(drop=True)
        test = cohort.iloc[test_idx].reset_index(drop=True)
        center = float(np.median(cal.score))
        scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        cal_score = expit((cal.score.to_numpy(float) - center) / scale)
        test_score = expit((test.score.to_numpy(float) - center) / scale)
        x_struct_cal = np.column_stack([fingerprints[cal_idx], cal_score])
        x_struct_test = np.column_stack([fingerprints[test_idx], test_score])
        x_readout_cal = np.column_stack([cal_score, quality[cal_idx]])
        x_readout_test = np.column_stack([test_score, quality[test_idx]])
        labels_cal = cal.label.to_numpy(int)
        labels_test = test.label.to_numpy(int)
        groups = cal.murcko_scaffold.where(
            cal.murcko_scaffold.ne(""), "singleton:" + cal.canonical_smiles).to_numpy(str)
        meta, oof = fit_cross_fusion(x_struct_cal, x_readout_cal, labels_cal, groups, seed)
        structure = classifier(labels_cal).fit(x_struct_cal, labels_cal)
        readout = classifier(labels_cal).fit(x_readout_cal, labels_cal)
        ps = structure.predict_proba(x_struct_test)[:, 1]
        pr = readout.predict_proba(x_readout_test)[:, 1]
        logits = np.column_stack([logit(np.clip(ps, 1e-4, 1. - 1e-4)),
                                  logit(np.clip(pr, 1e-4, 1. - 1e-4))])
        predictions = {
            "structure_only": ps,
            "readout_only": pr,
            "mean_expert_probability": .5 * ps + .5 * pr,
            "cross_fitted_logit_fusion": meta.predict_proba(logits)[:, 1],
        }
        keys = np.asarray([int.from_bytes(hashlib.sha256(
            f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        blocks = test.murcko_scaffold.where(
            test.murcko_scaffold.ne(""), "singleton:" + test.canonical_smiles).to_numpy(str)
        for method, score in predictions.items():
            selected = choose(score, blocks, keys)
            if len(selected) != 50 or len(set(blocks[selected])) != 50:
                raise AssertionError("Budget or cap violation")
            hits = int(labels_test[selected].sum())
            if method == "structure_only":
                expected = int(reference.loc[reference.seed.eq(seed)
                                             & reference.method.eq("ecfp_score"), "hits"].iloc[0])
                if hits != expected:
                    raise AssertionError(f"Frozen ECFP replay mismatch: {name}/{seed}")
            rows.append({"task": name, "seed": seed, "method": method, "hits": hits,
                         "quality_dimensions": len(columns),
                         "meta_structure_weight": float(meta.coef_[0, 0]),
                         "meta_readout_weight": float(meta.coef_[0, 1]),
                         "oof_structure_mean": float(oof[:, 0].mean()),
                         "oof_readout_mean": float(oof[:, 1].mean())})
        print(name, seed, "completed", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    directory = root / "data/mfpcba_twelve_new_20261004"
    manifest = json.loads((directory / "cohort_manifest.json").read_text())
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        grouped = list(pool.map(run_task, [str(root)] * len(manifest["eligible_tasks"]),
                                manifest["eligible_tasks"]))
    cells = pd.DataFrame([row for group in grouped for row in group])
    output = root / "data/mfpcba_late_fusion_development_20261004"
    cells.to_csv(output / "cells.csv", index=False)
    assay = cells.groupby(["task", "method"]).hits.agg(["mean", "std"]).reset_index()
    assay.to_csv(output / "assay_mean_sd.csv", index=False)
    print(assay.to_string(index=False), flush=True)
    print(assay.groupby("method")["mean"].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
