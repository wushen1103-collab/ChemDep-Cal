#!/usr/bin/env python3
"""Audit source-level confirmation and cross-task overlap of frozen cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FOLLOWUP = {"AID1798": (1488,), "AID435034": (677, 859),
            "AID463087": (489005, 493021)}
AIDS = tuple(FOLLOWUP)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    base = args.root.resolve() / "data/pubchem_primary_welqrate_20261004"
    cohort_dir = base / "confirmatory_failures"
    cohorts = {aid: pd.read_parquet(cohort_dir / f"{aid}_cohort.parquet") for aid in AIDS}
    records = []
    for aid, cohort in cohorts.items():
        frames = []
        for followup_aid in FOLLOWUP[aid]:
            frame = pd.read_csv(base / f"AID{followup_aid}_concise.csv", low_memory=False,
                                usecols=["CID", "Activity Outcome"])
            frame.CID = pd.to_numeric(frame.CID, errors="coerce")
            frames.append(frame.dropna(subset=["CID"]).assign(CID=lambda x: x.CID.astype(int)))
        outcomes = pd.concat(frames).groupby("CID")["Activity Outcome"].agg(
            lambda values: frozenset(values.dropna()))
        for label in (0, 1):
            subset = cohort[cohort.label.eq(label)]
            mapped = subset.CID.map(outcomes)
            records.append({"aid": aid, "label": label, "n": len(subset),
                            "followup_any_active": int(mapped.map(
                                lambda x: isinstance(x, frozenset) and "Active" in x).sum()),
                            "followup_any_inactive": int(mapped.map(
                                lambda x: isinstance(x, frozenset) and "Inactive" in x).sum()),
                            "followup_untested": int(mapped.isna().sum()),
                            "followup_mixed": int(mapped.map(lambda x: isinstance(x, frozenset)
                                and "Active" in x and "Inactive" in x).sum())})
    overlap = []
    for i, left in enumerate(AIDS):
        for right in AIDS[i + 1:]:
            left_frame = cohorts[left].set_index("canonical_smiles")
            right_frame = cohorts[right].set_index("canonical_smiles")
            shared = left_frame.index.intersection(right_frame.index)
            overlap.append({"left": left, "right": right, "shared_structures": len(shared),
                            "same_label": int((left_frame.loc[shared].label.to_numpy()
                                               == right_frame.loc[shared].label.to_numpy()).sum()),
                            "opposite_label": int((left_frame.loc[shared].label.to_numpy()
                                                   != right_frame.loc[shared].label.to_numpy()).sum())})
    split_overlap = []
    for seed in range(1, 6):
        for target in AIDS:
            target_split = np.load(cohort_dir / f"{target}_seed{seed}_indices.npz")
            test = set(cohorts[target].iloc[target_split["test"]].canonical_smiles)
            for other in AIDS:
                if other == target:
                    continue
                other_split = np.load(cohort_dir / f"{other}_seed{seed}_indices.npz")
                cal = set(cohorts[other].iloc[other_split["calibration"]].canonical_smiles)
                split_overlap.append({"seed": seed, "test_aid": target, "calibration_aid": other,
                                      "overlap_structures": len(test & cal)})
    output = {"confirmation": records, "cohort_overlap": overlap,
              "test_vs_other_calibration_overlap": split_overlap}
    (cohort_dir / "provenance_overlap_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
