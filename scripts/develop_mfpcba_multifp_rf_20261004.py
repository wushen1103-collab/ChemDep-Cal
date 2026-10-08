#!/usr/bin/env python3
"""Matched multi-fingerprint RF control on exposed confirmatory chains."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier

from develop_mfpcba_readout_context_20261004 import quality_frame
from evaluate_mfpcba_twenty_20261004 import choose


def fingerprints(smiles: pd.Series) -> np.ndarray:
    feature_invariants = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    generators = (
        rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=1024, atomInvariantsGenerator=feature_invariants),
        rdFingerprintGenerator.GetMorganGenerator(
            radius=3, fpSize=1024, atomInvariantsGenerator=feature_invariants),
        rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024),
        rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024),
    )
    output = np.empty((len(smiles), 4263), dtype=np.uint8)
    for i, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise AssertionError("Invalid frozen SMILES")
        parts = [generator.GetFingerprint(mol) for generator in generators]
        parts.insert(3, MACCSkeys.GenMACCSKeys(mol))
        offset = 0
        for fp in parts:
            length = len(fp)
            DataStructs.ConvertToNumpyArray(fp, output[i, offset:offset + length])
            offset += length
        if offset != 4263:
            raise AssertionError(f"Unexpected fingerprint dimension: {offset}")
    return output


def fit_rf(x: np.ndarray, y: np.ndarray, seed: int) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=3, max_features=.2,
        class_weight="balanced_subsample", n_jobs=12, random_state=862026 + seed)
    return model.fit(x, y)


def run_task(root_s: str, name: str, campaign: str) -> list[dict]:
    repo = Path(root_s)
    directory = repo / f"data/mfpcba_{campaign}_20261004"
    RDLogger.DisableLog("rdApp.*")
    cohort_file = (directory / f"{name}_cohort.parquet" if campaign == "twenty"
                   else directory / "cohorts" / f"{name}_cohort.parquet")
    cohort = pd.read_parquet(cohort_file)
    fp = fingerprints(cohort.canonical_smiles)
    quality, columns = ((np.empty((len(cohort), 0)), []) if campaign == "twenty"
                        else quality_frame(directory, name, cohort))
    rows = []
    for seed in range(1, 6):
        split_file = (directory / f"{name}_seed{seed}_indices.npz" if campaign == "twenty"
                      else directory / "cohorts" / f"{name}_seed{seed}_indices.npz")
        with np.load(split_file) as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        cal = cohort.iloc[cal_idx]
        test = cohort.iloc[test_idx]
        center = float(np.median(cal.score))
        scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        cal_score = expit((cal.score.to_numpy(float) - center) / scale)
        test_score = expit((test.score.to_numpy(float) - center) / scale)
        labels_cal = cal.label.to_numpy(int)
        labels_test = test.label.to_numpy(int)
        keys = np.asarray([int.from_bytes(hashlib.sha256(
            f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        blocks = test.murcko_scaffold.where(
            test.murcko_scaffold.ne(""), "singleton:" + test.canonical_smiles).to_numpy(str)
        models = [
                ("multifp_score_rf", np.column_stack([fp[cal_idx], cal_score]),
                 np.column_stack([fp[test_idx], test_score]))]
        if campaign != "twenty":
            models.append(("multifp_score_readout_rf",
                           np.column_stack([fp[cal_idx], cal_score, quality[cal_idx]]),
                           np.column_stack([fp[test_idx], test_score, quality[test_idx]])))
        for method, cal_x, test_x in models:
            model = fit_rf(cal_x, labels_cal, seed)
            selected = choose(model.predict_proba(test_x)[:, 1], blocks, keys)
            if len(selected) != 50 or len(set(blocks[selected])) != 50:
                raise AssertionError("Budget or scaffold-cap violation")
            rows.append({"task": name, "seed": seed, "method": method,
                         "hits": int(labels_test[selected].sum()),
                         "test_n": len(test), "test_pos": int(labels_test.sum()),
                         "quality_dimensions": len(columns)})
        print(name, seed, "completed", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", choices=("twelve_new", "twenty"), default="twelve_new")
    args = parser.parse_args()
    repo = args.root.resolve()
    directory = repo / f"data/mfpcba_{args.campaign}_20261004"
    manifest = json.loads((directory / "cohort_manifest.json").read_text())
    with futures.ProcessPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(run_task, [str(repo)] * len(manifest["eligible_tasks"]),
                               manifest["eligible_tasks"],
                               [args.campaign] * len(manifest["eligible_tasks"])))
    output = repo / f"data/mfpcba_multifp_rf_{args.campaign}_development_20261004"
    output.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame([row for group in groups for row in group])
    cells.to_csv(output / "cells.csv", index=False)
    assay = cells.groupby(["task", "method"]).hits.agg(["mean", "std"]).reset_index()
    assay.to_csv(output / "assay_mean_sd.csv", index=False)
    print(assay.to_string(index=False), flush=True)
    print(assay.groupby("method")["mean"].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
