#!/usr/bin/env python3
"""Feasibility audit for the separately frozen primary-screen Bscore experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from audit_pubchem_primary_match_20261004 import PRIMARY, canonical, load_chunks


SIGNS = {"AID1798": 1, "AID435034": -1, "AID463087": -1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data = root / "data/pubchem_primary_welqrate_20261004"
    prep = root / "data/pr_welqrate_litmus_20261004/prep"
    output = data / "matched_bscore"
    output.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")

    cid_dir = data / "cid_isomeric_smiles_chunks"
    cid_manifest_path = cid_dir / "download_manifest.json"
    cid_manifest = json.loads(cid_manifest_path.read_text(encoding="utf-8"))
    if not cid_manifest["complete"]:
        raise AssertionError("CID property download incomplete")
    cid_frame = load_chunks(cid_dir, cid_manifest)
    cid_frame["canonical_smiles"] = [canonical(s) for s in cid_frame.SMILES]
    cid_frame = cid_frame.dropna(subset=["canonical_smiles"])
    cid_frame.CID = cid_frame.CID.astype(int)
    if cid_frame.CID.duplicated().any():
        raise AssertionError("Duplicate CID property rows")

    rows = []
    for name, aid in PRIMARY.items():
        source_dir = data / f"AID{aid}_full_chunks"
        source_manifest_path = source_dir / "download_manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if not source_manifest["complete"]:
            raise AssertionError(f"AID{aid} full assay incomplete")
        assay = load_chunks(source_dir, source_manifest, skiprows=[1, 2, 3])
        assay["CID"] = pd.to_numeric(assay.PUBCHEM_CID, errors="coerce")
        assay["bscore"] = pd.to_numeric(assay.Bscore, errors="coerce")
        assay = assay.dropna(subset=["CID", "bscore"])
        assay = assay[np.isfinite(assay.bscore)].copy()
        assay.CID = assay.CID.astype(int)
        per_cid = assay.groupby("CID", as_index=False).bscore.median()
        mapped = per_cid.merge(cid_frame[["CID", "canonical_smiles"]], on="CID", how="inner")
        n_unique = mapped.groupby("canonical_smiles").bscore.nunique()
        conflicts = set(n_unique.index[n_unique.gt(1)])
        unique_scores = mapped[~mapped.canonical_smiles.isin(conflicts)].groupby(
            "canonical_smiles").bscore.first()
        litmus = pd.read_parquet(prep / f"{name}_molecules.parquet")
        matched = litmus.copy()
        matched["primary_bscore_raw"] = matched.canonical_smiles.map(unique_scores)
        matched.to_parquet(output / f"{name}_bscore_matched.parquet", index=False)
        present = matched.primary_bscore_raw.notna().to_numpy()
        coverage = float(present.mean())
        distinct = int(matched.primary_bscore_raw.nunique(dropna=True))
        common = {"aid": name, "primary_aid": aid, "sign": SIGNS[name],
                  "litmus_molecules": len(litmus), "source_rows_with_bscore": len(assay),
                  "source_cids_with_bscore": len(per_cid),
                  "matched_molecules": int(present.sum()), "coverage": coverage,
                  "matched_distinct_bscores": distinct,
                  "matched_actives": int(matched.loc[present, "label"].sum()),
                  "conflicting_structures": len(conflicts),
                  "duplicate_litmus_structures": int(litmus.canonical_smiles.duplicated().sum()),
                  "source_manifest_sha256": hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
                  "description_sha256": hashlib.sha256((data / f"AID{aid}_description.json").read_bytes()).hexdigest(),
                  "cid_manifest_sha256": hashlib.sha256(cid_manifest_path.read_bytes()).hexdigest()}
        for seed in range(1, 6):
            with np.load(prep / f"{name}_seed{seed}_indices.npz") as split:
                cal_idx, test_idx = split["calibration"], split["test"]
            cal = matched.iloc[cal_idx]
            test = matched.iloc[test_idx]
            cal = cal[cal.primary_bscore_raw.notna()]
            test = test[test.primary_bscore_raw.notna()]
            oriented = SIGNS[name] * cal.primary_bscore_raw.to_numpy(float)
            iqr = float(np.percentile(oriented, 75) - np.percentile(oriented, 25))
            split_eligible = (int(cal.label.sum()) >= 5 and int(test.label.sum()) >= 5
                              and test.murcko_scaffold.nunique() >= 50 and iqr > 0)
            rows.append({**common, "seed": seed,
                         "n_calibration": len(cal), "n_test": len(test),
                         "active_calibration": int(cal.label.sum()),
                         "active_test": int(test.label.sum()),
                         "test_scaffolds": test.murcko_scaffold.nunique(),
                         "calibration_iqr": iqr, "split_eligible": split_eligible})
        print(name, "coverage", round(coverage, 4), "distinct", distinct,
              "conflicts", len(conflicts), flush=True)
    audit = pd.DataFrame(rows)
    audit["eligible"] = (audit.groupby("aid").split_eligible.transform("all")
                         & audit.coverage.ge(.9) & audit.matched_distinct_bscores.ge(20))
    audit.to_csv(output / "match_audit.csv", index=False)
    (output / "match_manifest.json").write_text(json.dumps({
        "status": "Bscore feasibility audit before selector outcomes",
        "source": "full primary-screen PubChem Bscore joined by standardized CID isomeric SMILES",
        "signs": SIGNS, "eligible_aids": sorted(audit.loc[audit.eligible, "aid"].unique().tolist()),
    }, indent=2) + "\n", encoding="utf-8")
    print(audit[["aid", "seed", "coverage", "matched_distinct_bscores",
                 "active_calibration", "active_test", "calibration_iqr", "eligible"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
