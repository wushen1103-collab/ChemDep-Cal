#!/usr/bin/env python3
"""Build the frozen, outcome-blind MF-PCBA task and split ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold


EXPOSED = {"GPCR_3", "ion_channel", "ion_channel2", "ion_channel3", "transporter"}
PROTOCOL_SHA256 = "b49197be81c4f7557eaecfabeafa74a0027c4c7fae63f8b7950089f1f0f3af39"
SEEDS = range(36001, 36006)
COLUMNS = ["SMILES", "Primary", "Score", "Confirmatory"]


def canonical(value: str) -> str | None:
    mol = Chem.MolFromSmiles(value) if isinstance(value, str) else None
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None


def load_assay(directory: Path) -> tuple[pd.DataFrame, dict]:
    name = directory.name
    paths = [directory / f"{name}_train.csv", directory / f"{name}_val.csv"]
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    raw = pd.concat([pd.read_csv(path, usecols=COLUMNS, low_memory=False)
                     for path in paths], ignore_index=True)
    raw["Primary"] = pd.to_numeric(raw.Primary, errors="coerce")
    raw["Score"] = pd.to_numeric(raw.Score, errors="coerce")
    raw["Confirmatory"] = pd.to_numeric(raw.Confirmatory, errors="coerce")
    observed = raw.loc[raw.Primary.eq(1) & raw.Confirmatory.isin([0, 1])
                       & np.isfinite(raw.Score)].copy()
    observed["canonical_smiles"] = [canonical(s) for s in observed.SMILES]
    observed = observed.dropna(subset=["canonical_smiles"])
    conflict = observed.groupby("canonical_smiles").agg(
        labels=("Confirmatory", "nunique"), scores=("Score", "nunique"))
    bad = set(conflict.index[conflict.labels.gt(1) | conflict.scores.gt(1)])
    observed = observed.loc[~observed.canonical_smiles.isin(bad)]
    observed = observed.sort_values("canonical_smiles").drop_duplicates("canonical_smiles")
    cohort = observed[["canonical_smiles", "Score", "Confirmatory"]].rename(
        columns={"Score": "score", "Confirmatory": "label"}).reset_index(drop=True)
    cohort["label"] = cohort.label.astype(np.int8)
    return cohort, {"source_hashes": source_hashes, "all_source_rows": len(raw),
                    "observed_primary_actives": len(observed),
                    "conflicting_smiles_removed": len(bad)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != "604b8f0769ebe5902336fcf4113962f8f179e88f":
        raise AssertionError(f"Unexpected public source commit: {commit}")
    directories = sorted((source / "DataValuationPlatform/Datasets").iterdir())
    directories = [d for d in directories if d.is_dir()]
    if len(directories) != 25 or not EXPOSED.issubset({d.name for d in directories}):
        raise AssertionError("Public 25-task inventory changed")
    cohorts, source_info = {}, {}
    for directory in directories:
        if directory.name in EXPOSED:
            continue
        cohorts[directory.name], source_info[directory.name] = load_assay(directory)
        print("loaded", directory.name, len(cohorts[directory.name]), flush=True)
    if len(cohorts) != 20:
        raise AssertionError("Independent inventory is not 20 tasks")
    task_count = pd.concat([frame.canonical_smiles.rename("smiles")
                            for frame in cohorts.values()]).value_counts()
    shared = set(task_count.index[task_count.gt(1)])
    audit = []
    for name, cohort in cohorts.items():
        removed = int(cohort.canonical_smiles.isin(shared).sum())
        cohort = cohort.loc[~cohort.canonical_smiles.isin(shared)].reset_index(drop=True)
        scaffold = []
        for smiles in cohort.canonical_smiles:
            value = MurckoScaffold.MurckoScaffoldSmiles(
                mol=Chem.MolFromSmiles(smiles), includeChirality=False)
            scaffold.append(value if value else f"singleton:{smiles}")
        cohort["murcko_scaffold"] = scaffold
        cohort.to_parquet(output / f"{name}_cohort.parquet", index=False)
        for seed_index, seed in enumerate(SEEDS):
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            folds = list(splitter.split(cohort, cohort.label, cohort.murcko_scaffold))
            calibration_index, test_index = folds[seed_index]
            np.savez_compressed(output / f"{name}_seed{seed_index + 1}_indices.npz",
                                calibration=calibration_index, test=test_index)
            cal = cohort.iloc[calibration_index]
            test = cohort.iloc[test_index]
            iqr = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25)) if len(cal) else 0.0
            criteria = {
                "cal_scores_ge20": cal.score.nunique() >= 20,
                "cal_iqr_positive": iqr > 0,
                "cal_pos_ge20": int(cal.label.sum()) >= 20,
                "cal_neg_ge20": int(cal.label.eq(0).sum()) >= 20,
                "test_pos_ge10": int(test.label.sum()) >= 10,
                "test_neg_ge10": int(test.label.eq(0).sum()) >= 10,
                "test_molecules_ge50": len(test) >= 50,
                "test_scaffolds_ge50": test.murcko_scaffold.nunique() >= 50,
            }
            audit.append({"task": name, "seed": seed_index + 1,
                          "cross_task_smiles_removed": removed, "cohort_n": len(cohort),
                          "cohort_pos": int(cohort.label.sum()), "cal_n": len(cal),
                          "cal_pos": int(cal.label.sum()), "test_n": len(test),
                          "test_pos": int(test.label.sum()),
                          "cal_distinct_scores": cal.score.nunique(), "cal_iqr": iqr,
                          "test_scaffolds": test.murcko_scaffold.nunique(),
                          "eligible": all(criteria.values()),
                          "failed_checks": ";".join(k for k, passed in criteria.items() if not passed)})
        print("split", name, flush=True)
    ledger = pd.DataFrame(audit)
    ledger.to_csv(output / "cohort_audit.csv", index=False)
    eligible = [name for name in sorted(cohorts)
                if ledger.loc[ledger.task.eq(name), "eligible"].all()]
    (output / "cohort_manifest.json").write_text(json.dumps({
        "protocol_sha256": PROTOCOL_SHA256, "source_commit": commit,
        "all_tasks": [d.name for d in directories], "exposed_tasks": sorted(EXPOSED),
        "independent_tasks": sorted(cohorts), "eligible_tasks": eligible,
        "cross_task_shared_smiles_removed_globally": len(shared),
        "sources": source_info,
    }, indent=2) + "\n", encoding="utf-8")
    print("Eligible", len(eligible), eligible, flush=True)


if __name__ == "__main__":
    main()
