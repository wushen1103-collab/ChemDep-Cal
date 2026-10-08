#!/usr/bin/env python3
"""Post hoc development: add ChemDep scaffold evidence to molecular confirmation XGB."""

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
from xgboost import XGBClassifier

from evaluate_mfpcba_twenty_20261004 import choose, molecular_features


def run_seed(root_s: str, seed: int) -> int:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    import probe_meta_band_v2_20261003  # noqa: F401
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_selection import block_array

    RDLogger.DisableLog("rdApp.*")
    directory = root / "data/mfpcba_twenty_20261004"
    manifest = json.loads((directory / "cohort_manifest.json").read_text())
    source = pd.read_csv(directory / "selectors" / f"seed{seed}_cells.csv")
    source = source[source.method.eq("ecfp_score_xgb_cap1")].set_index("task")
    output = directory / "structure_context_development"
    output.mkdir(parents=True, exist_ok=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows = []
    for name in manifest["eligible_tasks"]:
        cohort = pd.read_parquet(directory / f"{name}_cohort.parquet")
        with np.load(directory / f"{name}_seed{seed}_indices.npz") as split:
            cal = cohort.iloc[split["calibration"]].reset_index(drop=True).copy()
            test = cohort.iloc[split["test"]].reset_index(drop=True).copy()
        center = float(np.median(cal.score))
        scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        for frame in (cal, test):
            frame["score"] = expit((frame.score.to_numpy(float) - center) / scale)
            frame["molecule_chembl_id"] = frame.canonical_smiles
        null = cal.loc[cal.label.eq(0), "score"].to_numpy(float)
        blocks_cal = block_array(cal, "murcko_scaffold")
        blocks_test = block_array(test, "murcko_scaffold")
        ecfp_cal = molecular_features(cal.canonical_smiles, generator)
        ecfp_test = molecular_features(test.canonical_smiles, generator)
        cal_score = cal.score.to_numpy(float)
        test_score = test.score.to_numpy(float)
        base_cal = np.column_stack([ecfp_cal, cal_score])
        base_test = np.column_stack([ecfp_test, test_score])
        context_cal = np.column_stack([ecfp_cal, evidence_features(cal_score, blocks_cal, null)])
        context_test = np.column_stack([ecfp_test, evidence_features(test_score, blocks_test, null)])
        labels_cal = cal.label.to_numpy(int)
        positives = int(labels_cal.sum())
        params = dict(n_estimators=160, max_depth=3, learning_rate=.05,
                      subsample=.9, colsample_bytree=.9, min_child_weight=8,
                      reg_lambda=4., tree_method="hist", eval_metric="logloss",
                      n_jobs=4, random_state=862026,
                      scale_pos_weight=min(20., max(1., (len(labels_cal) - positives) / positives)))
        baseline = XGBClassifier(**params).fit(base_cal, labels_cal)
        context = XGBClassifier(**params).fit(context_cal, labels_cal)
        labels = test.label.to_numpy(int)
        keys = np.asarray([int.from_bytes(hashlib.sha256(
            f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        base_hits = int(labels[choose(baseline.predict_proba(base_test)[:, 1],
                                      blocks_test, keys)].sum())
        if base_hits != int(source.loc[name, "hits"]):
            raise AssertionError(f"Frozen ECFP baseline replay mismatch: {name}/{seed}")
        selected = choose(context.predict_proba(context_test)[:, 1], blocks_test, keys)
        if len(selected) != 50 or len(set(blocks_test[selected])) != 50:
            raise AssertionError("Capacity violation")
        context.save_model(output / f"seed{seed}.{name}.json")
        rows.append({"task": name, "seed": seed, "ecfp_score_hits": base_hits,
                     "ecfp_context_hits": int(labels[selected].sum()),
                     "gain": int(labels[selected].sum()) - base_hits})
        print("development", seed, name, rows[-1]["gain"], flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(output / f"seed{seed}_cells.csv", index=False)
    return len(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        print(list(pool.map(run_seed, [str(args.root.resolve())] * 5, range(1, 6))))


if __name__ == "__main__":
    main()
