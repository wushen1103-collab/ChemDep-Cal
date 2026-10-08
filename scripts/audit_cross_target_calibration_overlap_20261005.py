#!/usr/bin/env python3
"""Audit exact structure overlap between pooled calibration and other test targets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def cohort_paths(root: Path, cohort: str, seed: int) -> list[Path]:
    if cohort == "fifth":
        return sorted((root / "data/chembl_meta_ungated_fifth_xgb_20261003"
                       / f"seed{seed}" / "scores").glob("*_scores.csv"))
    directory = root / f"data/chembl_modern_{cohort}_20261005"
    eligible = pd.read_csv(directory / "technical_eligibility.csv").query(
        "eligible").target_chembl_id.tolist()
    return [directory / f"eligible_seed{seed}" / "scores"
            / f"{target}_scores.csv" for target in eligible]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from train_artifact_meta_ranker_20261003 import load_panel

    rows = []
    seeds_by_cohort = {"fifth": range(35501, 35506),
                       "seventh": range(35701, 35706),
                       "eighth": range(35801, 35806)}
    for cohort, seeds in seeds_by_cohort.items():
        for seed in seeds:
            panels = [load_panel(path) for path in cohort_paths(root, cohort, seed)]
            calibration = defaultdict(list)
            for panel in panels:
                for record in panel["cal"].itertuples(index=False):
                    calibration[record.canonical_smiles].append((panel["target"], int(record.label)))
            for panel in panels:
                target = panel["target"]
                test = panel["test"]
                overlap = 0
                concordant = 0
                discordant = 0
                opposite_labels = 0
                for record in test.itertuples(index=False):
                    others = [(t, y) for t, y in calibration.get(record.canonical_smiles, [])
                              if t != target]
                    if not others:
                        continue
                    overlap += 1
                    agree = any(y == int(record.label) for _, y in others)
                    disagree = any(y != int(record.label) for _, y in others)
                    concordant += int(agree)
                    discordant += int(disagree)
                    opposite_labels += int(not agree and disagree)
                rows.append({"cohort": cohort, "seed": seed, "target": target,
                             "test_n": len(test), "overlap_n": overlap,
                             "overlap_fraction": overlap / len(test),
                             "any_concordant_n": concordant,
                             "any_discordant_n": discordant,
                             "only_discordant_n": opposite_labels})
    detail = pd.DataFrame(rows)
    out = root / "data/pr_cross_target_calibration_overlap_20261005"
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "target_seed_overlap.csv", index=False)
    summary = detail.groupby("cohort", as_index=False).agg(
        target_seed_panels=("target", "size"), test_n=("test_n", "sum"),
        overlap_n=("overlap_n", "sum"), concordant_n=("any_concordant_n", "sum"),
        discordant_n=("any_discordant_n", "sum"),
        only_discordant_n=("only_discordant_n", "sum"))
    summary["overlap_fraction"] = summary.overlap_n / summary.test_n
    summary.to_csv(out / "cohort_summary.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "question": "exact test SMILES seen in calibration of another target",
        "scope": "eligible panels within each cohort and upstream scorer seed",
        "note": "label comparisons use held-out labels for retrospective audit only",
    }, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(detail.sort_values("overlap_fraction", ascending=False).head(12).to_string(index=False),
          flush=True)


if __name__ == "__main__":
    main()
