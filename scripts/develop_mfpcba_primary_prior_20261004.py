#!/usr/bin/env python3
"""Post hoc multifidelity development on exposed MF-PCBA assays."""

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
from scipy.special import expit
from xgboost import XGBClassifier

from evaluate_mfpcba_twenty_20261004 import choose, molecular_features


SOURCE_COMMIT = "604b8f0769ebe5902336fcf4113962f8f179e88f"


def classifier(labels: np.ndarray) -> XGBClassifier:
    positives = int(labels.sum())
    if positives < 20 or positives >= len(labels):
        raise ValueError(f"Insufficient primary or confirmation classes: {positives}/{len(labels)}")
    return XGBClassifier(
        n_estimators=160, max_depth=3, learning_rate=.05,
        subsample=.9, colsample_bytree=.9, min_child_weight=8,
        reg_lambda=4., tree_method="hist", eval_metric="logloss",
        n_jobs=4, random_state=862026,
        scale_pos_weight=min(20., max(1., (len(labels) - positives) / positives)),
    )


def run_task(root_s: str, task: str) -> dict:
    root = Path(root_s)
    directory = root / "data/mfpcba_twenty_20261004"
    source = (root / "data/external_benchmark_sources/DataValuationPlatform"
              / "DataValuationPlatform/Datasets" / task)
    output = directory / "primary_prior_development"
    output.mkdir(parents=True, exist_ok=True)
    cells_path = output / f"{task}_cells.csv"
    info_path = output / f"{task}_manifest.json"
    if cells_path.exists() and info_path.exists():
        completed = pd.read_csv(cells_path)
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if (len(completed) != 5 or set(completed.seed) != set(range(1, 6))
                or info.get("task") != task):
            raise AssertionError(f"Incomplete existing development result: {task}")
        print(task, "verified existing result", flush=True)
        return info
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    paths = [source / f"{task}_train.csv", source / f"{task}_val.csv"]
    raw = pd.concat([pd.read_csv(path, usecols=["SMILES", "Primary", "Confirmatory"],
                                 low_memory=False) for path in paths], ignore_index=True)
    primary_only = raw.loc[raw.Confirmatory.isna() & raw.Primary.isin([0, 1])].copy()
    if primary_only.SMILES.duplicated().any():
        raise AssertionError(f"Duplicate primary-only structure: {task}")
    cohort = pd.read_parquet(directory / f"{task}_cohort.parquet")
    if set(primary_only.SMILES) & set(cohort.canonical_smiles):
        raise AssertionError(f"Exact primary/confirmation leakage: {task}")
    primary_labels = primary_only.Primary.to_numpy(int)
    reference = pd.concat([pd.read_csv(directory / "selectors" / f"seed{seed}_cells.csv")
                           for seed in range(1, 6)], ignore_index=True)
    reference = reference[(reference.task.eq(task))
                          & reference.method.eq("ecfp_score_xgb_cap1")].set_index("seed")
    if len(reference) != 5:
        raise AssertionError("Incomplete frozen ECFP reference")
    if int(primary_labels.sum()) < 20:
        rows = [{"task": task, "seed": seed,
                 "ecfp_score_hits": int(reference.loc[seed, "hits"]),
                 "ecfp_primary_prior_hits": int(reference.loc[seed, "hits"]),
                 "gain": 0, "fallback": "primary_positive_lt20"}
                for seed in range(1, 6)]
        pd.DataFrame(rows).to_csv(cells_path, index=False)
        info = {"task": task, "source_commit": SOURCE_COMMIT,
                "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in paths},
                "primary_only_n": len(primary_only),
                "primary_only_positives": int(primary_labels.sum()),
                "confirmation_cohort_n": len(cohort),
                "baseline_cells_reused": 5,
                "status": "untrainable primary prior; fixed ECFP baseline fallback, not excluded"}
        info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print(task, info["primary_only_n"], info["primary_only_positives"],
              "baseline fallback", flush=True)
        return info
    primary_model = classifier(primary_labels)
    primary_model.fit(molecular_features(primary_only.SMILES, generator), primary_labels)
    primary_model.save_model(output / f"{task}.primary_only.json")
    fp_confirm = molecular_features(cohort.canonical_smiles, generator)
    prior = primary_model.predict_proba(fp_confirm)[:, 1]
    if not np.isfinite(prior).all():
        raise AssertionError("Nonfinite primary prior")
    rows = []
    for seed in range(1, 6):
        with np.load(directory / f"{task}_seed{seed}_indices.npz") as split:
            cal_index, test_index = split["calibration"], split["test"]
        cal = cohort.iloc[cal_index]
        test = cohort.iloc[test_index]
        center = float(np.median(cal.score))
        scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        cal_score = expit((cal.score.to_numpy(float) - center) / scale)
        test_score = expit((test.score.to_numpy(float) - center) / scale)
        x_cal = np.column_stack([fp_confirm[cal_index], cal_score])
        x_test = np.column_stack([fp_confirm[test_index], test_score])
        labels_cal = cal.label.to_numpy(int)
        base = classifier(labels_cal).fit(x_cal, labels_cal)
        augmented = classifier(labels_cal).fit(
            np.column_stack([x_cal, prior[cal_index]]), labels_cal)
        keys = np.asarray([int.from_bytes(hashlib.sha256(
            f"{task}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        blocks = test.murcko_scaffold.to_numpy(str)
        labels = test.label.to_numpy(int)
        selected_base = choose(base.predict_proba(x_test)[:, 1], blocks, keys)
        base_hits = int(labels[selected_base].sum())
        if base_hits != int(reference.loc[seed, "hits"]):
            raise AssertionError(f"Frozen baseline replay mismatch: {task}/{seed}")
        selected = choose(augmented.predict_proba(
            np.column_stack([x_test, prior[test_index]]))[:, 1], blocks, keys)
        if len(selected) != 50 or len(set(blocks[selected])) != 50:
            raise AssertionError("Budget or scaffold violation")
        augmented.save_model(output / f"{task}.seed{seed}.augmented.json")
        rows.append({"task": task, "seed": seed, "ecfp_score_hits": base_hits,
                     "ecfp_primary_prior_hits": int(labels[selected].sum()),
                     "gain": int(labels[selected].sum()) - base_hits})
    pd.DataFrame(rows).to_csv(cells_path, index=False)
    info = {"task": task, "source_commit": SOURCE_COMMIT,
            "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in paths},
            "primary_only_n": len(primary_only),
            "primary_only_positives": int(primary_labels.sum()),
            "confirmation_cohort_n": len(cohort),
            "baseline_cells_replayed": 5,
            "test_label_use": "final selection hit count only",
            "status": "post hoc development on inspected assays; no confirmatory status"}
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(task, info["primary_only_n"], info["primary_only_positives"],
          pd.DataFrame(rows).gain.mean(), flush=True)
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "data/mfpcba_twenty_20261004/cohort_manifest.json").read_text())
    with futures.ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_task, [str(root)] * len(manifest["eligible_tasks"]),
                                manifest["eligible_tasks"]))
    print("completed", len(results), flush=True)


if __name__ == "__main__":
    main()
