#!/usr/bin/env python3
"""Prepare fixed primary-hit/follow-up cohorts without inspecting selector hits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from download_pubchem_four_untouched_20261004 import CAMPAIGNS


PROTOCOL_SHA256 = "9ce8b7dea2650a437293c7880e91eff2fada234e36de32de2d8fb5cfcc422eda"


def canonical(value: str) -> str | None:
    mol = Chem.MolFromSmiles(value) if isinstance(value, str) else None
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None


def verified_chunks(directory: Path) -> pd.DataFrame:
    manifest = json.loads((directory / "download_manifest.json").read_text(encoding="utf-8"))
    if not manifest["complete"]:
        raise AssertionError(f"Incomplete source: {directory}")
    frames = []
    for record in manifest["chunks"]:
        file = directory / record["part"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != record["sha256"]:
            raise AssertionError(f"Source checksum changed: {file}")
        frame = pd.read_csv(file, skiprows=[1, 2, 3] if "active_full" in directory.name else None,
                            low_memory=False)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def partition(frame: pd.DataFrame, name: str, seed: int) -> dict[str, np.ndarray]:
    groups = frame.groupby("murcko_scaffold", sort=True).label.max()
    positive = groups.index[groups.eq(1)].to_numpy(str)
    negative = groups.index[groups.eq(0)].to_numpy(str)
    digest = hashlib.sha256(f"four-untouched/{name}/{seed}".encode()).digest()
    rng = np.random.default_rng(20261004 + int.from_bytes(digest[:4], "big"))
    assignment = {"train": [], "calibration": [], "test": []}
    for scaffolds in (positive, negative):
        order = rng.permutation(scaffolds)
        n_test = int(round(.2 * len(order)))
        n_cal = int(round(.2 * len(order)))
        assignment["test"].extend(order[:n_test])
        assignment["calibration"].extend(order[n_test:n_test + n_cal])
        assignment["train"].extend(order[n_test + n_cal:])
    all_scaffolds = frame.murcko_scaffold.to_numpy(str)
    indices = {key: np.flatnonzero(np.isin(all_scaffolds, value))
               for key, value in assignment.items()}
    if sum(map(len, indices.values())) != len(frame):
        raise AssertionError("Split does not partition cohort")
    for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        if set(all_scaffolds[indices[left]]) & set(all_scaffolds[indices[right]]):
            raise AssertionError("Scaffold split leakage")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    output = root / "cohorts"
    output.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")
    source_manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
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
        raise AssertionError("Duplicate CID property")
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
        status = per_cid.outcome_set.map(lambda x: 1 if x == frozenset({"Active"})
                                         else 0 if x == frozenset({"Inactive"}) else -1)
        labelled = per_cid.loc[status.ge(0), ["CID", "canonical_smiles", "score"]].copy()
        labelled["label"] = status.loc[labelled.index].astype(np.int8)
        collisions = labelled.groupby("canonical_smiles").agg(
            label_count=("label", "nunique"), score_count=("score", "nunique"))
        bad = set(collisions.index[collisions.label_count.gt(1) | collisions.score_count.gt(1)])
        labelled = labelled[~labelled.canonical_smiles.isin(bad)]
        labelled = labelled.sort_values("CID").drop_duplicates("canonical_smiles")
        cohorts[name] = labelled.reset_index(drop=True)
        counts[name] = {"primary_aid": primary_aid, "primary_active_sid_rows": len(primary),
                        "primary_active_mapped_cids": len(per_cid),
                        "explicit_positive_cids_before_dedup": int(status.eq(1).sum()),
                        "explicit_negative_cids_before_dedup": int(status.eq(0).sum()),
                        "untested_or_ambiguous_cids": int(status.eq(-1).sum()),
                        "conflicting_duplicate_structures": len(bad),
                        "after_within_task_structure_dedup": len(labelled)}
    assignments = {}
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
        details = {"aid": name, **counts[name], "cross_task_structures_removed": removed,
                   "cohort_size": len(frame), "cohort_actives": int(frame.label.sum()),
                   "cohort_inactives": int(frame.label.eq(0).sum()),
                   "distinct_primary_scores": int(frame.score.nunique()),
                   "cohort_scaffolds": frame.murcko_scaffold.nunique()}
        for seed in range(1, 6):
            indices = partition(frame, name, seed)
            np.savez_compressed(output / f"{name}_seed{seed}_indices.npz", **indices)
            cal, test = frame.iloc[indices["calibration"]], frame.iloc[indices["test"]]
            iqr = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
            test_blocks = test.murcko_scaffold.where(
                test.murcko_scaffold.ne(""), "singleton:" + test.CID.astype(str))
            eligible = (len(frame) >= 50 and details["distinct_primary_scores"] >= 20
                        and int(cal.label.sum()) >= 10 and int(test.label.sum()) >= 10
                        and test_blocks.nunique() >= 50 and len(test) >= 50 and iqr > 0)
            audit.append({**details, "seed": seed, "n_train": len(indices["train"]),
                          "n_calibration": len(cal), "n_test": len(test),
                          "active_calibration": int(cal.label.sum()),
                          "active_test": int(test.label.sum()),
                          "test_selectable_scaffolds": test_blocks.nunique(),
                          "calibration_score_iqr": iqr, "eligible": eligible})
        print(name, details, flush=True)
    ledger = pd.DataFrame(audit)
    ledger.to_csv(output / "cohort_audit.csv", index=False)
    eligible = [name for name in CAMPAIGNS if ledger.loc[ledger.aid.eq(name), "eligible"].all()]
    (output / "cohort_manifest.json").write_text(json.dumps({
        "status": "internally frozen four-campaign source-transfer cohort; not official WelQrate split",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_manifest_sha256": hashlib.sha256((root / "source_manifest.json").read_bytes()).hexdigest(),
        "globally_shared_structures_removed": len(shared), "eligible_aids": eligible,
        "ineligible_aids": [name for name in CAMPAIGNS if name not in eligible],
    }, indent=2) + "\n", encoding="utf-8")
    print(ledger[["aid", "seed", "cohort_size", "cohort_actives", "n_calibration",
                  "active_calibration", "n_test", "active_test", "eligible"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
