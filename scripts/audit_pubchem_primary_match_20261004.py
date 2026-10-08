#!/usr/bin/env python3
"""Match pinned Litmus molecules to measured primary-screen scores by structure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


PRIMARY = {"AID1798": 626, "AID435034": 628, "AID463087": 449739}


def load_chunks(directory: Path, manifest: dict, skiprows: list[int] | None = None) -> pd.DataFrame:
    frames = []
    for record in manifest["chunks"]:
        file = directory / record["part"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != record["sha256"]:
            raise AssertionError(f"Source hash mismatch: {file}")
        frames.append(pd.read_csv(file, skiprows=skiprows, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def canonical(smiles: str) -> str | None:
    if not isinstance(smiles, str) or not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data = root / "data/pubchem_primary_welqrate_20261004"
    prep = root / "data/pr_welqrate_litmus_20261004/prep"
    output = data / "matched"
    output.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")

    cid_dir = data / "cid_isomeric_smiles_chunks"
    cid_manifest_path = cid_dir / "download_manifest.json"
    cid_manifest = json.loads(cid_manifest_path.read_text(encoding="utf-8"))
    if not cid_manifest["complete"]:
        raise AssertionError("CID-to-isomeric-SMILES download is incomplete")
    cid_frame = load_chunks(cid_dir, cid_manifest)
    if cid_frame.CID.duplicated().any():
        raise AssertionError("Repeated CID in standardized property source")
    cid_frame["canonical_smiles"] = [canonical(s) for s in cid_frame.SMILES]
    cid_frame = cid_frame.dropna(subset=["canonical_smiles"])
    cid_frame.CID = cid_frame.CID.astype(int)
    print("CID source", len(cid_frame), "standardized structures", flush=True)

    rows = []
    for name, aid in PRIMARY.items():
        source_dir = data / f"AID{aid}_full_chunks"
        source_manifest_path = source_dir / "download_manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if not source_manifest["complete"]:
            raise AssertionError(f"AID{aid} source incomplete")
        assay = load_chunks(source_dir, source_manifest, skiprows=[1, 2, 3])
        assay["CID"] = pd.to_numeric(assay.PUBCHEM_CID, errors="coerce")
        assay["score"] = pd.to_numeric(assay.PUBCHEM_ACTIVITY_SCORE, errors="coerce")
        assay = assay.dropna(subset=["CID", "score"]).copy()
        assay.CID = assay.CID.astype(int)
        source_score_unique = assay.score.nunique()
        source_score_min, source_score_max = assay.score.min(), assay.score.max()
        if not (0 <= source_score_min <= source_score_max <= 100):
            raise AssertionError(f"AID{aid}: activity score outside [0,100]")
        per_cid = assay.groupby("CID", as_index=False).score.median()
        mapped = per_cid.merge(cid_frame[["CID", "canonical_smiles"]], on="CID", how="inner")
        score_groups = mapped.groupby("canonical_smiles").score
        conflict = score_groups.nunique().gt(1)
        conflicting_structures = set(conflict.index[conflict])
        usable = mapped[~mapped.canonical_smiles.isin(conflicting_structures)]
        unique_scores = usable.groupby("canonical_smiles").score.first()

        litmus = pd.read_parquet(prep / f"{name}_molecules.parquet")
        duplicate_litmus_structures = int(litmus.canonical_smiles.duplicated().sum())
        matched = litmus.copy()
        matched["primary_score_raw"] = matched.canonical_smiles.map(unique_scores)
        matched["primary_score"] = matched.primary_score_raw / 100.
        matched.to_parquet(output / f"{name}_primary_matched.parquet", index=False)
        present = matched.primary_score.notna().to_numpy()
        distinct = matched.primary_score_raw.nunique(dropna=True)
        report = {"aid": name, "primary_aid": aid, "source_rows": len(assay),
                  "source_distinct_scores": int(source_score_unique),
                  "source_score_min": float(source_score_min),
                  "source_score_max": float(source_score_max),
                  "source_cids": len(per_cid), "mapped_cids": len(mapped),
                  "conflicting_structures": len(conflicting_structures),
                  "duplicate_litmus_structures": duplicate_litmus_structures,
                  "litmus_molecules": len(matched), "matched_molecules": int(present.sum()),
                  "coverage": float(present.mean()), "matched_distinct_scores": int(distinct),
                  "matched_actives": int(matched.loc[present, "label"].sum()),
                  "source_manifest_sha256": hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
                  "cid_manifest_sha256": hashlib.sha256(cid_manifest_path.read_bytes()).hexdigest()}
        eligible = report["coverage"] >= .9 and distinct >= 20
        for seed in range(1, 6):
            with np.load(prep / f"{name}_seed{seed}_indices.npz") as split:
                cal_idx, test_idx = split["calibration"], split["test"]
            cal = matched.iloc[cal_idx]
            test = matched.iloc[test_idx]
            cal = cal[cal.primary_score.notna()]
            test = test[test.primary_score.notna()]
            cal_active, test_active = int(cal.label.sum()), int(test.label.sum())
            scaffolds = test.murcko_scaffold.nunique()
            split_eligible = cal_active >= 5 and test_active >= 5 and scaffolds >= 50
            eligible &= split_eligible
            rows.append({**report, "seed": seed, "n_calibration": len(cal), "n_test": len(test),
                         "active_calibration": cal_active, "active_test": test_active,
                         "test_scaffolds": scaffolds, "split_eligible": split_eligible})
        print(name, report["matched_molecules"], "/", len(matched),
              "coverage", round(report["coverage"], 4), "distinct", distinct,
              "eligible", eligible, flush=True)
    audit = pd.DataFrame(rows)
    audit["eligible"] = audit.groupby("aid").split_eligible.transform("all") & audit.coverage.ge(.9) & audit.matched_distinct_scores.ge(20)
    audit.to_csv(output / "match_audit.csv", index=False)
    (output / "match_manifest.json").write_text(json.dumps({
        "status": "post hoc source-feasibility audit before selector outcomes",
        "identity": "exact RDKit canonical isomeric SMILES from PubChem CID properties",
        "conflicts": "multiple CIDs with discordant median primary scores excluded",
        "score": "PUBCHEM_ACTIVITY_SCORE / 100; no readout substitution",
        "eligible_aids": sorted(audit.loc[audit.eligible, "aid"].unique().tolist()),
    }, indent=2) + "\n", encoding="utf-8")
    print(audit[["aid", "seed", "coverage", "matched_distinct_scores",
                 "active_calibration", "active_test", "eligible"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
