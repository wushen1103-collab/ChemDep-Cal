#!/usr/bin/env python3
"""Post hoc Tanimoto-kernel local-chemical-evidence control."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.special import expit
from sklearn.svm import SVC

from develop_mfpcba_readout_context_20261004 import quality_frame
from evaluate_mfpcba_twenty_20261004 import choose


def kernel(smiles: pd.Series) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise AssertionError("Invalid frozen SMILES")
        fps.append(generator.GetFingerprint(mol))
    result = np.empty((len(fps), len(fps)), dtype=np.float32)
    for i, fp in enumerate(fps):
        result[i] = DataStructs.BulkTanimotoSimilarity(fp, fps)
    return result


def run_task(root_s: str, campaign: str, name: str) -> list[dict]:
    repo = Path(root_s)
    directory = repo / f"data/mfpcba_{campaign}_20261004"
    cohort_file = (directory / f"{name}_cohort.parquet" if campaign == "twenty"
                   else directory / "cohorts" / f"{name}_cohort.parquet")
    cohort = pd.read_parquet(cohort_file)
    RDLogger.DisableLog("rdApp.*")
    tanimoto = kernel(cohort.canonical_smiles)
    quality, columns = ((np.empty((len(cohort), 0)), []) if campaign == "twenty"
                        else quality_frame(directory, name, cohort))
    rf_dir = ("mfpcba_multifp_rf_twenty_development_20261004" if campaign == "twenty"
              else "mfpcba_multifp_rf_development_20261004")
    source = repo / "data" / rf_dir / "cells.csv"
    rf = pd.read_csv(source)
    rf_method = "multifp_score_rf" if campaign == "twenty" else "multifp_score_readout_rf"
    rows = []
    for seed in range(1, 6):
        split_file = (directory / f"{name}_seed{seed}_indices.npz" if campaign == "twenty"
                      else directory / "cohorts" / f"{name}_seed{seed}_indices.npz")
        with np.load(split_file) as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        cal, test = cohort.iloc[cal_idx], cohort.iloc[test_idx]
        center = float(np.median(cal.score))
        scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        normalized_score = expit((cohort.score.to_numpy(float) - center) / scale)
        raw_low = np.column_stack([normalized_score, quality])
        median = np.nanmedian(raw_low[cal_idx], axis=0)
        spread = (np.nanpercentile(raw_low[cal_idx], 75, axis=0)
                  - np.nanpercentile(raw_low[cal_idx], 25, axis=0))
        spread = np.where(spread > 1e-6, spread, 1.)
        imputed = np.where(np.isfinite(raw_low), raw_low, median)
        low = np.clip((imputed - median) / spread, -5., 5.)
        squared = np.sum((low[:, None, :] - low[None, :, :]) ** 2, axis=2)
        bandwidth = float(np.median(squared[np.triu_indices(len(cohort), 1)]))
        readout_kernel = np.exp(-squared / max(bandwidth, 1e-6))
        y_cal, y_test = cal.label.to_numpy(int), test.label.to_numpy(int)
        keys = np.asarray([int.from_bytes(hashlib.sha256(
            f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        blocks = test.murcko_scaffold.where(
            test.murcko_scaffold.ne(""), "singleton:" + test.canonical_smiles).to_numpy(str)
        for method, weight in (("tanimoto_svm", 0.), ("tanimoto_readout_svm", .2)):
            combined = (1 - weight) * tanimoto + weight * readout_kernel
            model = SVC(kernel="precomputed", C=1., class_weight="balanced",
                        random_state=862026 + seed).fit(combined[np.ix_(cal_idx, cal_idx)], y_cal)
            prediction = model.decision_function(combined[np.ix_(test_idx, cal_idx)])
            selected = choose(prediction, blocks, keys)
            if len(selected) != 50 or len(set(blocks[selected])) != 50:
                raise AssertionError("Budget or scaffold-cap violation")
            rows.append({"task": name, "seed": seed, "method": method,
                         "hits": int(y_test[selected].sum()), "test_n": len(test),
                         "test_pos": int(y_test.sum()), "quality_dimensions": len(columns),
                         "rf_reference_hits": int(rf.loc[rf.task.eq(name) & rf.seed.eq(seed)
                                                   & rf.method.eq(rf_method), "hits"].iloc[0])})
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
    output = repo / f"data/mfpcba_tanimoto_scaled_{args.campaign}_development_20261004"
    output.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame([row for group in groups for row in group])
    cells.to_csv(output / "cells.csv", index=False)
    assay = cells.groupby(["task", "method"]).hits.agg(["mean", "std"]).reset_index()
    assay.to_csv(output / "assay_mean_sd.csv", index=False)
    print(assay.groupby("method")["mean"].mean().to_string(), flush=True)
    print("RF reference", cells.groupby("task").rf_reference_hits.mean().mean(), flush=True)


if __name__ == "__main__":
    main()
