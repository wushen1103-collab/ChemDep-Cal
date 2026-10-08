#!/usr/bin/env python3
"""Author-code MVS-A scores replayed under our matched B=50 scaffold cap."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from evaluate_mfpcba_twenty_20261004 import choose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "data/mfpcba_twenty_20261004/cohort_manifest.json").read_text())
    if args.task not in manifest["eligible_tasks"]:
        raise AssertionError("Task is not in fixed eligible Hesse cohort")
    source_dir = root / "data/external_benchmark_sources/AIC_Finder"
    source = source_dir / "Datasets" / f"{args.task}.csv"
    sys.path.insert(0, str(source_dir / "Scripts"))
    from MVS_A import sample_analysis

    RDLogger.DisableLog("rdApp.*")
    frame = pd.read_csv(source, usecols=["SMILES", "Primary"])
    frame = frame.loc[frame.SMILES.notna() & frame.Primary.isin([0, 1])].reset_index(drop=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    features = np.empty((len(frame), 1024), dtype=np.uint8)
    valid = np.ones(len(frame), dtype=bool)
    active_smiles = {}
    for i, (smiles, primary) in enumerate(zip(frame.SMILES, frame.Primary)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            valid[i] = False
            continue
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), features[i])
        if primary == 1:
            active_smiles[i] = Chem.MolToSmiles(mol, isomericSmiles=True)
        if (i + 1) % 50000 == 0:
            print("fingerprints", i + 1, "/", len(frame), flush=True)
    active_positions = np.flatnonzero(valid & frame.Primary.to_numpy().astype(bool))
    x = features[valid]
    y = frame.Primary.to_numpy(int)[valid]
    remap = np.full(len(frame), -1, dtype=int)
    remap[np.flatnonzero(valid)] = np.arange(valid.sum())
    print("primary", len(y), "actives", int(y.sum()), flush=True)
    analysis = sample_analysis(x=x, y=y, params="default", verbose=False, seed=862026)
    analysis.params["num_threads"] = 16
    analysis.get_model()
    importance = analysis.get_importance()
    by_smiles = pd.DataFrame({
        "canonical_smiles": [active_smiles[i] for i in active_positions],
        "mvsa_importance": importance[remap[active_positions]],
    }).groupby("canonical_smiles").mvsa_importance.median()
    cohort = pd.read_parquet(root / "data/mfpcba_twenty_20261004" /
                             f"{args.task}_cohort.parquet")
    aligned = by_smiles.reindex(cohort.canonical_smiles.to_numpy()).to_numpy(float)
    coverage = float(np.isfinite(aligned).mean())
    if coverage < .95:
        raise AssertionError(f"MVS-A cohort alignment only {coverage:.3f}")
    rows = []
    for seed in range(1, 6):
        with np.load(root / "data/mfpcba_twenty_20261004" /
                     f"{args.task}_seed{seed}_indices.npz") as split:
            test_idx = split["test"]
        test = cohort.iloc[test_idx]
        blocks = test.murcko_scaffold.where(
            test.murcko_scaffold.ne(""), "singleton:" + test.canonical_smiles).to_numpy(str)
        keys = np.asarray([int.from_bytes(hashlib.sha256(
            f"{args.task}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        selected = choose(-aligned[test_idx], blocks, keys)
        if len(selected) != 50 or len(set(blocks[selected])) != 50:
            raise AssertionError("Budget or scaffold-cap violation")
        rows.append({"task": args.task, "seed": seed,
                     "method": "mvsa_author_code_primary_only",
                     "hits": int(test.label.to_numpy(int)[selected].sum()),
                     "coverage": coverage, "test_n": len(test),
                     "test_pos": int(test.label.sum())})
    out = root / "data/mfpcba_mvsa_author_audit_20261004"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{args.task}_cells.csv", index=False)
    pd.DataFrame({"canonical_smiles": cohort.canonical_smiles,
                  "mvsa_importance": aligned}).to_parquet(out / f"{args.task}_scores.parquet")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print("mean hits", pd.DataFrame(rows).hits.mean(), flush=True)


if __name__ == "__main__":
    main()
