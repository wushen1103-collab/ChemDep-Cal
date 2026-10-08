#!/usr/bin/env python3
"""Build the precommitted twelve-chain explicit follow-up cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold

from download_mfpcba_twelve_new_20261004 import source
from prepare_pubchem_four_untouched_20261004 import canonical, verified_chunks


PROTOCOL_SHA256 = "f2542d1232d14437bea793780fbdf18fa2603ced5ca7fc04ca4642bd7780c2d8"
SOURCE_COMMIT = "776cb013a5ab9949adfc9c11a212cd7ad7dada7a"
CAMPAIGNS = source.CAMPAIGNS


def partition(frame: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    groups = frame.murcko_scaffold.where(
        frame.murcko_scaffold.ne(""), "singleton:" + frame.canonical_smiles)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=37000 + seed)
    calibration, test = list(splitter.split(frame, frame.label, groups))[seed - 1]
    if set(groups.iloc[calibration]) & set(groups.iloc[test]):
        raise AssertionError("Scaffold leakage")
    return {"calibration": calibration, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    output = root / "cohorts"
    output.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")
    source_file = root / "source_manifest.json"
    source_manifest = json.loads(source_file.read_text(encoding="utf-8"))
    if source_manifest["campaigns"] != {
            name: [aid, list(followups)] for name, (aid, followups) in CAMPAIGNS.items()}:
        raise AssertionError("Downloaded assay inventory differs from fixed list")
    for record in source_manifest["sources"]:
        file = root / record["file"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != record["sha256"]:
            raise AssertionError(f"Source checksum changed: {file}")

    cid_data = verified_chunks(root / "cid_isomeric_smiles_chunks")
    cid_data.CID = pd.to_numeric(cid_data.CID, errors="coerce")
    cid_data = cid_data.dropna(subset=["CID", "SMILES"]).copy()
    cid_data.CID = cid_data.CID.astype(int)
    cid_data["canonical_smiles"] = [canonical(s) for s in cid_data.SMILES]
    cid_data = cid_data.dropna(subset=["canonical_smiles"])
    if cid_data.CID.duplicated().any():
        raise AssertionError("Duplicate CID structure mapping")

    cohorts, counts = {}, {}
    for name, (primary_aid, followups) in CAMPAIGNS.items():
        primary = verified_chunks(root / f"AID{primary_aid}_active_full_chunks")
        primary["CID"] = pd.to_numeric(primary.PUBCHEM_CID, errors="coerce")
        primary["score"] = pd.to_numeric(primary.PUBCHEM_ACTIVITY_SCORE, errors="coerce")
        primary = primary.dropna(subset=["CID", "score"])
        primary = primary[np.isfinite(primary.score)].copy()
        primary.CID = primary.CID.astype(int)
        if not primary.PUBCHEM_ACTIVITY_OUTCOME.eq("Active").all():
            raise AssertionError(f"{name}: nonactive primary record")
        per_cid = primary.groupby("CID", as_index=False).score.median()
        per_cid = per_cid.merge(cid_data[["CID", "canonical_smiles"]], on="CID", how="inner")
        followup = []
        for followup_aid in followups:
            table = pd.read_csv(root / f"AID{followup_aid}_concise.csv", low_memory=False,
                                usecols=["CID", "Activity Outcome"])
            table.CID = pd.to_numeric(table.CID, errors="coerce")
            table = table.dropna(subset=["CID"]).copy()
            table.CID = table.CID.astype(int)
            followup.append(table)
        outcomes = pd.concat(followup).groupby("CID")["Activity Outcome"].agg(
            lambda values: frozenset(values.dropna()))
        per_cid["outcome_set"] = per_cid.CID.map(outcomes)
        status = per_cid.outcome_set.map(
            lambda x: 1 if x == frozenset({"Active"})
            else 0 if x == frozenset({"Inactive"}) else -1)
        labelled = per_cid.loc[status.ge(0), ["CID", "canonical_smiles", "score"]].copy()
        labelled["label"] = status.loc[labelled.index].astype(np.int8)
        collisions = labelled.groupby("canonical_smiles").agg(
            label_count=("label", "nunique"), score_count=("score", "nunique"))
        bad = set(collisions.index[collisions.label_count.gt(1) | collisions.score_count.gt(1)])
        labelled = labelled[~labelled.canonical_smiles.isin(bad)]
        labelled = labelled.sort_values("CID").drop_duplicates("canonical_smiles")
        cohorts[name] = labelled.reset_index(drop=True)
        counts[name] = {
            "primary_aid": primary_aid,
            "primary_active_sid_rows": len(primary),
            "primary_active_mapped_cids": len(per_cid),
            "explicit_positive_cids_before_dedup": int(status.eq(1).sum()),
            "explicit_negative_cids_before_dedup": int(status.eq(0).sum()),
            "untested_or_ambiguous_cids": int(status.eq(-1).sum()),
            "conflicting_duplicate_structures": len(bad),
            "after_within_task_structure_dedup": len(labelled),
        }
        print(name, counts[name], flush=True)

    assignments: dict[str, set[str]] = {}
    for name, frame in cohorts.items():
        for structure in frame.canonical_smiles:
            assignments.setdefault(structure, set()).add(name)
    shared = {structure for structure, names in assignments.items() if len(names) > 1}
    audit = []
    for name, frame in cohorts.items():
        removed = int(frame.canonical_smiles.isin(shared).sum())
        frame = frame.loc[~frame.canonical_smiles.isin(shared)].reset_index(drop=True)
        frame["murcko_scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(
            mol=Chem.MolFromSmiles(s), includeChirality=False) for s in frame.canonical_smiles]
        frame.to_parquet(output / f"{name}_cohort.parquet", index=False)
        details = {"task": name, **counts[name], "cross_task_structures_removed": removed,
                   "cohort_size": len(frame), "cohort_actives": int(frame.label.sum()),
                   "cohort_inactives": int(frame.label.eq(0).sum()),
                   "distinct_primary_scores": int(frame.score.nunique()),
                   "cohort_scaffolds": int(frame.murcko_scaffold.nunique())}
        for seed in range(1, 6):
            try:
                if len(frame) < 50 or frame.label.nunique() != 2:
                    raise ValueError("Too few rows or single-class cohort")
                indices = partition(frame, seed)
                np.savez_compressed(output / f"{name}_seed{seed}_indices.npz", **indices)
                cal, test = frame.iloc[indices["calibration"]], frame.iloc[indices["test"]]
                iqr = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
                test_blocks = test.murcko_scaffold.where(
                    test.murcko_scaffold.ne(""), "singleton:" + test.canonical_smiles)
                eligible = (cal.score.nunique() >= 20 and iqr > 0
                            and int(cal.label.sum()) >= 20 and int(cal.label.eq(0).sum()) >= 20
                            and int(test.label.sum()) >= 10 and int(test.label.eq(0).sum()) >= 10
                            and len(test) >= 50 and test_blocks.nunique() >= 50)
                row = {"n_calibration": len(cal), "n_test": len(test),
                       "active_calibration": int(cal.label.sum()),
                       "inactive_calibration": int(cal.label.eq(0).sum()),
                       "active_test": int(test.label.sum()),
                       "inactive_test": int(test.label.eq(0).sum()),
                       "test_selectable_scaffolds": int(test_blocks.nunique()),
                       "calibration_distinct_scores": int(cal.score.nunique()),
                       "calibration_score_iqr": iqr, "eligible": eligible,
                       "failure": "" if eligible else "fixed eligibility thresholds"}
            except ValueError as exc:
                row = {"n_calibration": 0, "n_test": 0, "active_calibration": 0,
                       "inactive_calibration": 0, "active_test": 0, "inactive_test": 0,
                       "test_selectable_scaffolds": 0, "calibration_distinct_scores": 0,
                       "calibration_score_iqr": 0., "eligible": False, "failure": str(exc)}
            audit.append({**details, "seed": seed, **row})
        print(name, "cohort", len(frame), "shared removed", removed, flush=True)
    ledger = pd.DataFrame(audit)
    ledger.to_csv(root / "cohort_audit.csv", index=False)
    eligible = [name for name in CAMPAIGNS if ledger.loc[ledger.task.eq(name), "eligible"].all()]
    (root / "cohort_manifest.json").write_text(json.dumps({
        "status": "internally precommitted twelve-chain independent source-transfer cohort",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": SOURCE_COMMIT,
        "source_manifest_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "globally_shared_structures_removed": len(shared),
        "independent_tasks": list(CAMPAIGNS), "eligible_tasks": eligible,
        "ineligible_tasks": [name for name in CAMPAIGNS if name not in eligible],
    }, indent=2) + "\n", encoding="utf-8")
    print(ledger[["task", "seed", "cohort_size", "cohort_actives", "n_calibration",
                  "active_calibration", "n_test", "active_test", "eligible"]].to_string(index=False),
          flush=True)


if __name__ == "__main__":
    main()
