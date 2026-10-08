#!/usr/bin/env python3
"""Exploratory redesign on already-inspected explicit-follow-up cohorts."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.special import expit
from xgboost import XGBClassifier


AIDS = ("AID1798", "AID435034", "AID463087")
SIGNS = {"AID1798": 1, "AID435034": -1, "AID463087": -1}
VARIANTS = ("fp_only", "fp_score", "evidence_only", "fp_evidence")


def fingerprints(smiles: pd.Series, generator: object) -> np.ndarray:
    rows = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError("Invalid molecule in frozen cohort")
        rows.append(generator.GetFingerprintAsNumPy(mol))
    return np.asarray(rows, dtype=np.float32)


def evaluate_seed(root_s: str, seed: int) -> int:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    import probe_meta_band_v2_20261003  # noqa: F401; register original feature map
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_selection import block_array
    from train_artifact_meta_ranker_20261003 import cap_one

    base = root / "data/pubchem_primary_welqrate_20261004/confirmatory_failures"
    output = base / "structure_aware_posthoc"
    output.mkdir(parents=True, exist_ok=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    RDLogger.DisableLog("rdApp.*")
    rows = []
    for aid in AIDS:
        cohort = pd.read_parquet(base / f"{aid}_cohort.parquet")
        with np.load(base / f"{aid}_seed{seed}_indices.npz") as split:
            dev_idx = np.concatenate([split["train"], split["calibration"]])
            cal_idx, test_idx = split["calibration"], split["test"]
        dev = cohort.iloc[dev_idx].reset_index(drop=True).copy()
        cal = cohort.iloc[cal_idx].reset_index(drop=True).copy()
        test = cohort.iloc[test_idx].reset_index(drop=True).copy()
        sign = SIGNS[aid]
        orient_cal = sign * cal.bscore.to_numpy(float)
        center = float(np.median(orient_cal))
        iqr = float(np.percentile(orient_cal, 75) - np.percentile(orient_cal, 25))
        if iqr <= 0:
            raise AssertionError("Ineligible calibration IQR")
        for frame in (dev, cal, test):
            frame["score"] = expit((sign * frame.bscore.to_numpy(float) - center) / iqr)
            frame["molecule_chembl_id"] = frame.CID.astype(str)
            frame["target_chembl_id"] = aid
        null = cal.loc[cal.label.eq(0), "score"].to_numpy(float)
        dev_blocks = block_array(dev, "murcko_scaffold")
        test_blocks = block_array(test, "murcko_scaffold")
        dev_evidence = evidence_features(dev.score.to_numpy(float), dev_blocks, null)
        test_evidence = evidence_features(test.score.to_numpy(float), test_blocks, null)
        if dev_evidence.shape[1] != 16 or test_evidence.shape[1] != 16:
            raise AssertionError("Unexpected evidence feature width")
        dev_fp = fingerprints(dev.canonical_smiles, generator)
        test_fp = fingerprints(test.canonical_smiles, generator)
        dev_score = dev.score.to_numpy(np.float32)[:, None]
        test_score = test.score.to_numpy(np.float32)[:, None]
        features = {
            "fp_only": (dev_fp, test_fp),
            "fp_score": (np.column_stack([dev_fp, dev_score]),
                         np.column_stack([test_fp, test_score])),
            "evidence_only": (dev_evidence, test_evidence),
            "fp_evidence": (np.column_stack([dev_fp, dev_evidence]),
                            np.column_stack([test_fp, test_evidence])),
        }
        y = dev.label.to_numpy(int)
        if min(int(y.sum()), int((y == 0).sum())) < 10:
            raise AssertionError("Insufficient development labels")
        for name in VARIANTS:
            x_train, x_test = features[name]
            model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=.03,
                                  min_child_weight=3, subsample=.9, colsample_bytree=.8,
                                  reg_lambda=5, scale_pos_weight=float((y == 0).sum() / y.sum()),
                                  tree_method="hist", n_jobs=4, random_state=20261004 + seed,
                                  eval_metric="logloss")
            model.fit(x_train, y)
            predictions = model.predict_proba(x_test)[:, 1]
            chosen = cap_one(predictions, test_blocks, 50)
            if len(chosen) != 50 or len(set(test_blocks[chosen])) != 50:
                raise AssertionError("Cap mismatch")
            rows.append({"aid": aid, "seed": seed, "method": name,
                         "hits": int(test.label.iloc[chosen].sum()),
                         "selected": len(chosen), "scaffolds": len(set(test_blocks[chosen])),
                         "n_train": len(dev), "train_actives": int(y.sum()),
                         "n_test": len(test), "test_actives": int(test.label.sum())})
        chosen = cap_one(test.score.to_numpy(float), test_blocks, 50)
        rows.append({"aid": aid, "seed": seed, "method": "score_cap1",
                     "hits": int(test.label.iloc[chosen].sum()),
                     "selected": len(chosen), "scaffolds": len(set(test_blocks[chosen])),
                     "n_train": len(dev), "train_actives": int(y.sum()),
                     "n_test": len(test), "test_actives": int(test.label.sum())})
        print(seed, aid, "completed", flush=True)
    result = pd.DataFrame(rows)
    assert len(result) == 15 and not result.duplicated(["aid", "seed", "method"]).any()
    result.to_csv(output / f"seed{seed}_cells.csv", index=False)
    (output / f"seed{seed}_manifest.json").write_text(json.dumps({
        "status": "posthoc method redesign on already-inspected tasks; not SOTA confirmation",
        "seed": seed, "methods": VARIANTS,
        "training": "60% train plus 20% calibration, same labels for all learned variants",
        "test": "frozen 20% scaffold split, B=50 cap1",
    }, indent=2) + "\n", encoding="utf-8")
    return seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        print(list(pool.map(evaluate_seed, [str(args.root.resolve())] * 5, range(1, 6))))


if __name__ == "__main__":
    main()
