#!/usr/bin/env python3
"""Post hoc descriptor controls on exposed MF-PCBA assays only."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from develop_mfpcba_multifp_rf_20261004 import fingerprints
from develop_mfpcba_readout_context_20261004 import quality_frame
from evaluate_mfpcba_twenty_20261004 import choose


DESCRIPTOR_NAMES = (
    "MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "AromaticRings",
    "FractionCSP3", "RingCount", "HeavyAtoms", "HeteroAtoms", "FormalCharge",
    "SaturatedRings", "AliphaticRings", "BertzCT", "LabuteASA",
)


def descriptors(smiles: pd.Series) -> np.ndarray:
    functions = (
        Descriptors.MolWt, Crippen.MolLogP, rdMolDescriptors.CalcTPSA,
        Lipinski.NumHDonors, Lipinski.NumHAcceptors, Lipinski.NumRotatableBonds,
        Lipinski.NumAromaticRings, rdMolDescriptors.CalcFractionCSP3,
        Lipinski.RingCount, Lipinski.HeavyAtomCount, Lipinski.NumHeteroatoms,
        Chem.GetFormalCharge, Lipinski.NumSaturatedRings, Lipinski.NumAliphaticRings,
        Descriptors.BertzCT, rdMolDescriptors.CalcLabuteASA,
    )
    values = np.empty((len(smiles), len(functions)), dtype=np.float32)
    for i, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise AssertionError("Invalid frozen SMILES")
        values[i] = [function(mol) for function in functions]
    return values


def run_task(root_s: str, campaign: str, name: str) -> list[dict]:
    repo = Path(root_s)
    directory = repo / f"data/mfpcba_{campaign}_20261004"
    cohort_file = (directory / f"{name}_cohort.parquet" if campaign == "twenty"
                   else directory / "cohorts" / f"{name}_cohort.parquet")
    cohort = pd.read_parquet(cohort_file)
    RDLogger.DisableLog("rdApp.*")
    fp = fingerprints(cohort.canonical_smiles)
    physchem = descriptors(cohort.canonical_smiles)
    quality, quality_columns = ((np.empty((len(cohort), 0)), []) if campaign == "twenty"
                                else quality_frame(directory, name, cohort))
    reference_file = (repo / "data" /
                      ("mfpcba_multifp_rf_twenty_development_20261004" if campaign == "twenty"
                       else "mfpcba_multifp_rf_development_20261004") / "cells.csv")
    reference = pd.read_csv(reference_file)
    reference_method = ("multifp_score_rf" if campaign == "twenty"
                        else "multifp_score_readout_rf")
    rows = []
    for seed in range(1, 6):
        split_file = (directory / f"{name}_seed{seed}_indices.npz" if campaign == "twenty"
                      else directory / "cohorts" / f"{name}_seed{seed}_indices.npz")
        with np.load(split_file) as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        cal, test = cohort.iloc[cal_idx], cohort.iloc[test_idx]
        center = float(np.median(cal.score))
        scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        score = expit((cohort.score.to_numpy(float) - center) / scale)
        median = np.nanmedian(physchem[cal_idx], axis=0)
        extent = np.nanpercentile(physchem[cal_idx], 75, axis=0) - np.nanpercentile(
            physchem[cal_idx], 25, axis=0)
        extent = np.where(extent > 1e-6, extent, 1.0)
        scaled = np.clip((np.where(np.isfinite(physchem), physchem, median) - median)
                         / extent, -10.0, 10.0)
        features = np.column_stack([fp, score, quality, scaled])
        y_cal, y_test = cal.label.to_numpy(int), test.label.to_numpy(int)
        keys = np.asarray([int.from_bytes(hashlib.sha256(
            f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        blocks = test.murcko_scaffold.where(
            test.murcko_scaffold.ne(""), "singleton:" + test.canonical_smiles).to_numpy(str)
        models = (
            ("multifp_descriptor_rf", RandomForestClassifier(
                n_estimators=500, min_samples_leaf=3, max_features=.2,
                class_weight="balanced_subsample", n_jobs=12, random_state=862026 + seed)),
            ("multifp_descriptor_extratrees", ExtraTreesClassifier(
                n_estimators=500, min_samples_leaf=3, max_features=.2,
                class_weight="balanced", n_jobs=12, random_state=862026 + seed)),
        )
        for method, model in models:
            model.fit(features[cal_idx], y_cal)
            selected = choose(model.predict_proba(features[test_idx])[:, 1], blocks, keys)
            if len(selected) != 50 or len(set(blocks[selected])) != 50:
                raise AssertionError("Budget or scaffold-cap violation")
            baseline = reference.loc[reference.task.eq(name) & reference.seed.eq(seed)
                                     & reference.method.eq(reference_method), "hits"]
            if len(baseline) != 1:
                raise AssertionError("Missing matched RF baseline")
            rows.append({"task": name, "seed": seed, "method": method,
                         "hits": int(y_test[selected].sum()), "rf_hits": int(baseline.iloc[0]),
                         "test_n": len(test), "test_pos": int(y_test.sum()),
                         "quality_dimensions": len(quality_columns)})
        print(campaign, name, seed, "completed", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", choices=("twelve_new", "twenty"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    directory = root / f"data/mfpcba_{args.campaign}_20261004"
    manifest = json.loads((directory / "cohort_manifest.json").read_text())
    names = manifest["eligible_tasks"]
    with futures.ProcessPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(run_task, [str(root)] * len(names),
                               [args.campaign] * len(names), names))
    output = root / f"data/mfpcba_descriptors_{args.campaign}_development_20261004"
    output.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame([row for group in groups for row in group])
    cells.to_csv(output / "cells.csv", index=False)
    assay = cells.groupby(["task", "method"]).agg(
        mean=("hits", "mean"), std=("hits", "std"), rf_mean=("rf_hits", "mean")
    ).reset_index()
    assay.to_csv(output / "assay_mean_sd.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "status": "post hoc development; not independent confirmation",
        "source_protocol_sha256": manifest["protocol_sha256"],
        "descriptors": DESCRIPTOR_NAMES,
        "methods": ["multifp_descriptor_rf", "multifp_descriptor_extratrees"],
    }, indent=2) + "\n")
    print(assay.to_string(index=False), flush=True)
    print(assay.groupby("method")["mean"].mean().to_string(), flush=True)
    print("RF reference", assay.groupby("task").rf_mean.mean().mean(), flush=True)


if __name__ == "__main__":
    main()
