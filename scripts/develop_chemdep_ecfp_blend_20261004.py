#!/usr/bin/env python3
"""Post hoc diagnostic blend of two already fitted selectors on exposed assays."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.special import expit
from scipy.stats import rankdata
from xgboost import XGBClassifier

from evaluate_mfpcba_twenty_20261004 import choose


ALPHAS = (0., .05, .1, .2, .35, .5, .75, 1.)


def molecular_features(smiles: pd.Series, generator) -> np.ndarray:
    result = np.empty((len(smiles), 2048), dtype=np.uint8)
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise AssertionError("Invalid frozen molecular structure")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), result[index])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    import sys
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    import probe_meta_band_v2_20261003  # noqa: F401; register 16-feature map
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_selection import block_array

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows = []
    for campaign in ("twenty", "twelve_new"):
        directory = root / f"data/mfpcba_{campaign}_20261004"
        manifest = __import__("json").loads((directory / "cohort_manifest.json").read_text())
        for seed in range(1, 6):
            full = XGBClassifier()
            full.load_model(directory / "selectors" / f"seed{seed}.chemdep.json")
            for name in manifest["eligible_tasks"]:
                cohort_file = (directory / "cohorts" / f"{name}_cohort.parquet"
                               if campaign == "twelve_new" else directory / f"{name}_cohort.parquet")
                split_file = (directory / "cohorts" / f"{name}_seed{seed}_indices.npz"
                              if campaign == "twelve_new" else directory / f"{name}_seed{seed}_indices.npz")
                cohort = pd.read_parquet(cohort_file)
                with np.load(split_file) as split:
                    cal = cohort.iloc[split["calibration"]].reset_index(drop=True).copy()
                    test = cohort.iloc[split["test"]].reset_index(drop=True).copy()
                center = float(np.median(cal.score))
                scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
                for frame in (cal, test):
                    frame["score"] = expit((frame.score.to_numpy(float) - center) / scale)
                    frame["molecule_chembl_id"] = frame.canonical_smiles
                    frame["target_chembl_id"] = name
                blocks = block_array(test, "murcko_scaffold")
                null = cal.loc[cal.label.eq(0), "score"].to_numpy(float)
                chemdep = full.predict_proba(evidence_features(
                    test.score.to_numpy(float), blocks, null))[:, 1]
                ecfp = XGBClassifier()
                ecfp.load_model(directory / "selectors" / f"seed{seed}.{name}.ecfp_score.json")
                x = np.column_stack([molecular_features(test.canonical_smiles, generator),
                                     test.score.to_numpy(float)])
                molecular = ecfp.predict_proba(x)[:, 1]
                keys = np.asarray([int.from_bytes(hashlib.sha256(
                    f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
                    for smiles in test.canonical_smiles], dtype=np.uint64)
                labels = test.label.to_numpy(int)
                frozen = pd.read_csv(directory / "selectors" / f"seed{seed}_cells.csv")
                for method, values in (("chemdep_cal", chemdep),
                                       ("ecfp_score_xgb_cap1", molecular)):
                    recorded = int(frozen.loc[frozen.task.eq(name) & frozen.method.eq(method),
                                              "hits"].iloc[0])
                    replay = int(labels[choose(values, blocks, keys)].sum())
                    if replay != recorded:
                        raise AssertionError(f"Frozen replay mismatch: {campaign} {seed} {name} {method}")
                chemdep_rank = rankdata(chemdep, method="average") / len(chemdep)
                molecular_rank = rankdata(molecular, method="average") / len(molecular)
                for alpha in ALPHAS:
                    blended = (1. - alpha) * molecular_rank + alpha * chemdep_rank
                    hits = int(labels[choose(blended, blocks, keys)].sum())
                    rows.append({"campaign": campaign, "task": name, "seed": seed,
                                 "chemdep_weight": alpha, "hits": hits,
                                 "molecular_frozen_hits": int(labels[choose(molecular, blocks, keys)].sum())})
            print(campaign, seed, "completed", flush=True)
    output = root / "data/mfpcba_blend_development_20261004"
    output.mkdir(parents=True, exist_ok=True)
    cell = pd.DataFrame(rows)
    cell.to_csv(output / "cells.csv", index=False)
    assay = cell.groupby(["campaign", "task", "chemdep_weight"]).agg(
        hits=("hits", "mean"), comparator=("molecular_frozen_hits", "mean")).reset_index()
    assay.to_csv(output / "assay_means.csv", index=False)
    summary = assay.groupby(["campaign", "chemdep_weight"]).agg(
        assay_n=("task", "nunique"), hits=("hits", "mean"), comparator=("comparator", "mean"))
    summary["gain"] = summary.hits - summary.comparator
    summary.to_csv(output / "summary.csv")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
