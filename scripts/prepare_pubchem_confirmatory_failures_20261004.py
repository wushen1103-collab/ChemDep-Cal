#!/usr/bin/env python3
"""Build frozen scaffold splits of primary hits with explicit confirmation outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from audit_pubchem_primary_match_20261004 import PRIMARY, canonical, load_chunks


FOLLOWUP = {"AID1798": (1488,), "AID435034": (677, 859),
            "AID463087": (489005, 493021)}


def partition_scaffolds(frame: pd.DataFrame, aid: str, seed: int) -> dict[str, np.ndarray]:
    groups = frame.groupby("murcko_scaffold", sort=True).label.max()
    positive = groups.index[groups.eq(1)].to_numpy(str)
    negative = groups.index[groups.eq(0)].to_numpy(str)
    digest = hashlib.sha256(f"{aid}/{seed}/confirmatory".encode()).digest()
    rng = np.random.default_rng(20261004 + int.from_bytes(digest[:4], "big"))
    allocation = {"train": [], "calibration": [], "test": []}
    for cohort in (positive, negative):
        shuffled = rng.permutation(cohort)
        n_test = int(round(.2 * len(shuffled)))
        n_cal = int(round(.2 * len(shuffled)))
        allocation["test"].extend(shuffled[:n_test])
        allocation["calibration"].extend(shuffled[n_test:n_test + n_cal])
        allocation["train"].extend(shuffled[n_test + n_cal:])
    array = frame.murcko_scaffold.to_numpy(str)
    indices = {key: np.flatnonzero(np.isin(array, value))
               for key, value in allocation.items()}
    if sum(map(len, indices.values())) != len(frame):
        raise AssertionError("Split does not partition candidate rows")
    for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        if set(array[indices[left]]) & set(array[indices[right]]):
            raise AssertionError("Scaffold split leakage")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data = root / "data/pubchem_primary_welqrate_20261004"
    litmus = root / "data/pr_welqrate_litmus_20261004/prep"
    output = data / "confirmatory_failures"
    output.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")

    cid_dir = data / "cid_isomeric_smiles_chunks"
    cid_manifest_path = cid_dir / "download_manifest.json"
    cid_manifest = json.loads(cid_manifest_path.read_text(encoding="utf-8"))
    if not cid_manifest["complete"]:
        raise AssertionError("CID standardized property source incomplete")
    cid_frame = load_chunks(cid_dir, cid_manifest)
    cid_frame["canonical_smiles"] = [canonical(s) for s in cid_frame.SMILES]
    cid_frame = cid_frame.dropna(subset=["canonical_smiles"])
    cid_frame.CID = cid_frame.CID.astype(int)
    if cid_frame.CID.duplicated().any():
        raise AssertionError("Duplicate standardized CID")
    audit, source_hashes = [], {}
    for name, primary_aid in PRIMARY.items():
        primary_dir = data / f"AID{primary_aid}_full_chunks"
        primary_manifest_path = primary_dir / "download_manifest.json"
        primary_manifest = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
        if not primary_manifest["complete"]:
            raise AssertionError(f"Primary AID{primary_aid} source incomplete")
        primary = load_chunks(primary_dir, primary_manifest, skiprows=[1, 2, 3])
        primary["CID"] = pd.to_numeric(primary.PUBCHEM_CID, errors="coerce")
        primary["bscore"] = pd.to_numeric(primary.Bscore, errors="coerce")
        primary = primary.dropna(subset=["CID", "bscore"])
        primary = primary[np.isfinite(primary.bscore)].copy()
        primary.CID = primary.CID.astype(int)
        outcomes = primary.groupby("CID").PUBCHEM_ACTIVITY_OUTCOME.nunique()
        consistent_cids = set(outcomes.index[outcomes.eq(1)])
        consistent = primary[primary.CID.isin(consistent_cids)]
        primary_hits = consistent.loc[consistent.PUBCHEM_ACTIVITY_OUTCOME.eq("Active")]
        per_cid = primary_hits.groupby("CID", as_index=False).bscore.median()
        per_cid = per_cid.merge(cid_frame[["CID", "canonical_smiles"]], on="CID", how="inner")
        source_hashes[f"primary_{name}"] = hashlib.sha256(primary_manifest_path.read_bytes()).hexdigest()

        followup_rows = []
        for aid in FOLLOWUP[name]:
            file = data / f"AID{aid}_concise.csv"
            source_hashes[f"followup_{aid}"] = hashlib.sha256(file.read_bytes()).hexdigest()
            table = pd.read_csv(file, low_memory=False)
            table["CID"] = pd.to_numeric(table.CID, errors="coerce")
            table = table.dropna(subset=["CID"])
            table.CID = table.CID.astype(int)
            followup_rows.append(table[["CID", "Activity Outcome"]].rename(
                columns={"Activity Outcome": "outcome"}))
        followup = pd.concat(followup_rows, ignore_index=True)
        outcome_sets = followup.groupby("CID").outcome.agg(lambda x: frozenset(x.dropna()))
        explicit_failure = set(outcome_sets.index[outcome_sets.eq(frozenset({"Inactive"}))])
        any_inactive = set(outcome_sets.index[outcome_sets.map(lambda values: "Inactive" in values)])
        any_active = set(outcome_sets.index[outcome_sets.map(lambda values: "Active" in values)])

        curated = pd.read_parquet(litmus / f"{name}_molecules.parquet")
        final_actives = set(curated.loc[curated.label.eq(1), "canonical_smiles"])
        per_cid["curated_active"] = per_cid.canonical_smiles.isin(final_actives)
        per_cid["named_followup_active"] = per_cid.CID.isin(any_active)
        per_cid["explicit_failure"] = per_cid.CID.isin(explicit_failure)
        discordant = per_cid.curated_active & per_cid.CID.isin(any_inactive)
        confirmed_positive = per_cid.curated_active & per_cid.named_followup_active & ~discordant
        labelled = per_cid.loc[confirmed_positive ^ per_cid.explicit_failure].copy()
        labelled["label"] = confirmed_positive.loc[labelled.index].astype(np.int8)
        structure_conflict = labelled.groupby("canonical_smiles").agg(
            n_labels=("label", "nunique"), n_bscores=("bscore", "nunique"))
        bad_structures = set(structure_conflict.index[
            structure_conflict.n_labels.gt(1) | structure_conflict.n_bscores.gt(1)])
        labelled = labelled[~labelled.canonical_smiles.isin(bad_structures)]
        labelled = labelled.sort_values("CID").drop_duplicates("canonical_smiles").reset_index(drop=True)
        labelled["murcko_scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(
            mol=Chem.MolFromSmiles(smiles), includeChirality=False)
            for smiles in labelled.canonical_smiles]
        labelled = labelled[["CID", "canonical_smiles", "murcko_scaffold", "bscore", "label"]]
        labelled.to_parquet(output / f"{name}_cohort.parquet", index=False)
        details = {"aid": name, "primary_aid": primary_aid,
                   "primary_rows_with_cid_bscore": len(primary),
                   "primary_consistent_active_rows": len(primary_hits),
                   "primary_active_cids_mapped": len(per_cid),
                   "followup_explicit_failure_cids": len(explicit_failure),
                   "curated_positive_primary_cids": int(per_cid.curated_active.sum()),
                   "curated_positive_without_named_followup_active": int(
                       (per_cid.curated_active & ~per_cid.named_followup_active).sum()),
                   "discordant_positive_followup_cids": int(discordant.sum()),
                   "ambiguous_structures_excluded": len(bad_structures),
                   "cohort_size": len(labelled), "cohort_actives": int(labelled.label.sum()),
                   "cohort_inactives": int((labelled.label == 0).sum()),
                   "cohort_scaffolds": labelled.murcko_scaffold.nunique()}
        for seed in range(1, 6):
            indices = partition_scaffolds(labelled, name, seed)
            np.savez_compressed(output / f"{name}_seed{seed}_indices.npz", **indices)
            cal = labelled.iloc[indices["calibration"]]
            test = labelled.iloc[indices["test"]]
            sign = 1 if name == "AID1798" else -1
            oriented_cal = sign * cal.bscore.to_numpy(float)
            iqr = float(np.percentile(oriented_cal, 75) - np.percentile(oriented_cal, 25))
            eligible = (int(cal.label.sum()) >= 10 and int(test.label.sum()) >= 10
                        and test.murcko_scaffold.nunique() >= 50 and len(test) >= 50
                        and iqr > 0)
            audit.append({**details, "seed": seed,
                          "n_train": len(indices["train"]), "n_calibration": len(cal),
                          "n_test": len(test), "active_calibration": int(cal.label.sum()),
                          "active_test": int(test.label.sum()),
                          "test_scaffolds": test.murcko_scaffold.nunique(),
                          "calibration_iqr": iqr, "eligible": eligible})
        print(name, details, flush=True)
    ledger = pd.DataFrame(audit)
    ledger.to_csv(output / "cohort_audit.csv", index=False)
    (output / "cohort_manifest.json").write_text(json.dumps({
        "status": "retrospective explicit-confirmation cohort; frozen before selector outcomes",
        "positive": "curated WelQrate final active among primary Active compounds with Active in a named confirmatory assay (post-outcome source-audit amendment)",
        "negative": "primary Active plus only Inactive named confirmatory outcomes",
        "source_sha256": source_hashes,
        "cid_manifest_sha256": hashlib.sha256(cid_manifest_path.read_bytes()).hexdigest(),
        "eligible_aids": sorted(ledger.loc[ledger.eligible, "aid"].unique().tolist()),
    }, indent=2) + "\n", encoding="utf-8")
    print(ledger[["aid", "seed", "cohort_size", "cohort_actives", "n_calibration",
                  "active_calibration", "n_test", "active_test", "eligible"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
