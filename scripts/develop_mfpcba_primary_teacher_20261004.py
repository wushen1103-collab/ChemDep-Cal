#!/usr/bin/env python3
"""Post hoc primary-screen teacher feature, with no confirmatory test labels."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from develop_mfpcba_multifp_rf_20261004 import fingerprints
from evaluate_mfpcba_twenty_20261004 import choose


def primary_teacher(source: Path, cohort: pd.DataFrame, seed: int) -> tuple[np.ndarray, dict]:
    frame = pd.read_csv(source, usecols=["SMILES", "Primary"])
    active = frame.loc[frame.Primary.eq(1), "SMILES"].dropna().tolist()
    negative = frame.loc[frame.Primary.eq(0), "SMILES"].dropna()
    rng = np.random.default_rng(seed)
    negative = negative.iloc[rng.choice(len(negative), size=min(len(negative), 20 * len(active)),
                                       replace=False)].tolist()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def encode(smiles: list[str], need_scaffold: bool):
        features, canonical, scaffolds = [], [], []
        for value in smiles:
            mol = Chem.MolFromSmiles(value)
            if mol is None:
                continue
            bits = np.empty(2048, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), bits)
            features.append(bits)
            if need_scaffold:
                canonical.append(Chem.MolToSmiles(mol, isomericSmiles=True))
                scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or
                                 "singleton:" + canonical[-1])
        return np.asarray(features), canonical, scaffolds

    positive_x, positive_smiles, groups = encode(active, True)
    negative_x, _, _ = encode(negative, False)
    if len(positive_x) < 100 or len(negative_x) < 1000:
        raise AssertionError("Insufficient primary-screen labels")
    prediction = np.full(len(positive_x), np.nan)
    folds = GroupKFold(n_splits=5)
    for fold, (train_idx, held_idx) in enumerate(folds.split(positive_x, groups=groups)):
        x = np.vstack([positive_x[train_idx], negative_x])
        y = np.r_[np.ones(len(train_idx), dtype=int), np.zeros(len(negative_x), dtype=int)]
        model = LGBMClassifier(
            n_estimators=150, learning_rate=.05, num_leaves=15,
            min_child_samples=30, colsample_bytree=.8, subsample=.9,
            subsample_freq=1, reg_lambda=5., verbosity=-1, n_jobs=8,
            class_weight="balanced", random_state=seed + fold).fit(x, y)
        prediction[held_idx] = model.predict_proba(positive_x[held_idx])[:, 1]
    if not np.isfinite(prediction).all():
        raise AssertionError("Unscored primary actives")
    by_smiles = pd.DataFrame({"canonical_smiles": positive_smiles,
                              "teacher": prediction}).groupby("canonical_smiles").teacher.median()
    aligned = by_smiles.reindex(cohort.canonical_smiles.to_numpy()).to_numpy(float)
    coverage = float(np.isfinite(aligned).mean())
    if coverage < .95:
        raise AssertionError(f"Primary teacher alignment only {coverage:.3f}")
    median = float(np.nanmedian(aligned))
    aligned = np.nan_to_num(aligned, nan=median)
    return aligned, {"source": str(source), "primary_actives": len(positive_x),
                     "sampled_inactives": len(negative_x), "cohort_coverage": coverage}


def run_task(root_s: str, name: str) -> list[dict]:
    repo = Path(root_s)
    RDLogger.DisableLog("rdApp.*")
    source = repo / "data/external_benchmark_sources/AIC_Finder/Datasets" / f"{name}.csv"
    cohort = pd.read_parquet(repo / "data/mfpcba_twenty_20261004" / f"{name}_cohort.parquet")
    teacher, provenance = primary_teacher(source, cohort, 862026)
    fp = fingerprints(cohort.canonical_smiles)
    reference = pd.read_csv(repo / "data/mfpcba_multifp_rf_twenty_development_20261004/cells.csv")
    out = repo / "data/mfpcba_primary_teacher_development_20261004"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(1, 6):
        with np.load(repo / "data/mfpcba_twenty_20261004" /
                     f"{name}_seed{seed}_indices.npz") as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        cal, test = cohort.iloc[cal_idx], cohort.iloc[test_idx]
        center = float(np.median(cal.score))
        scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        cal_score = expit((cal.score.to_numpy(float) - center) / scale)
        test_score = expit((test.score.to_numpy(float) - center) / scale)
        x_cal = np.column_stack([fp[cal_idx], cal_score])
        x_test = np.column_stack([fp[test_idx], test_score])
        y_cal, y_test = cal.label.to_numpy(int), test.label.to_numpy(int)
        rf = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, max_features=.2,
            class_weight="balanced_subsample", n_jobs=8,
            random_state=862026 + seed).fit(x_cal, y_cal)
        base = rf.predict_proba(x_test)[:, 1]
        augmented = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, max_features=.2,
            class_weight="balanced_subsample", n_jobs=8,
            random_state=862026 + seed).fit(
                np.column_stack([x_cal, teacher[cal_idx]]), y_cal)
        informed = augmented.predict_proba(np.column_stack([x_test, teacher[test_idx]]))[:, 1]
        keys = np.asarray([int.from_bytes(hashlib.sha256(
            f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        blocks = test.murcko_scaffold.where(
            test.murcko_scaffold.ne(""), "singleton:" + test.canonical_smiles).to_numpy(str)
        for method, prediction in (("multifp_rf_replay", base),
                                   ("primary_teacher_rf", informed)):
            selected = choose(prediction, blocks, keys)
            if len(selected) != 50 or len(set(blocks[selected])) != 50:
                raise AssertionError("Budget or scaffold-cap violation")
            hits = int(y_test[selected].sum())
            rows.append({"task": name, "seed": seed, "method": method, "hits": hits,
                         "test_n": len(test), "test_pos": int(y_test.sum()),
                         **provenance})
            if method == "multifp_rf_replay":
                expected = int(reference.loc[reference.task.eq(name) & reference.seed.eq(seed),
                                             "hits"].iloc[0])
                if hits != expected:
                    raise AssertionError(f"Frozen RF baseline mismatch {name}/{seed}")
        print(name, seed, "completed", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    repo = args.root.resolve()
    manifest = json.loads((repo / "data/mfpcba_twenty_20261004/cohort_manifest.json").read_text())
    available = {path.stem for path in (repo / "data/external_benchmark_sources/AIC_Finder/Datasets").glob("*.csv")}
    names = args.tasks or sorted(set(manifest["eligible_tasks"]) & available)
    if not set(names) <= set(manifest["eligible_tasks"]) & available:
        raise AssertionError("Task not in exposed eligible source intersection")
    with futures.ProcessPoolExecutor(max_workers=min(4, len(names))) as pool:
        groups = list(pool.map(run_task, [str(repo)] * len(names), names))
    out = repo / "data/mfpcba_primary_teacher_development_20261004"
    cells = pd.DataFrame([row for group in groups for row in group])
    cells.to_csv(out / "cells.csv", index=False)
    assay = cells.groupby(["task", "method"]).hits.agg(["mean", "std"]).reset_index()
    assay.to_csv(out / "assay_mean_sd.csv", index=False)
    print(assay.to_string(index=False), flush=True)
    print(assay.groupby("method")["mean"].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
