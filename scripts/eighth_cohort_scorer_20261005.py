#!/usr/bin/env python3
"""Freeze eligibility and fit five independent upstream scorers for eighth cohort."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
from pathlib import Path

import pandas as pd

from train_chembl_ecfp_xgb import train_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.seed not in range(35801, 35806):
        raise ValueError("Scorer seed not in frozen range")
    root = args.root.resolve()
    cohort = root / "data/chembl_modern_eighth_20261005"
    catalog = json.loads((cohort / "catalog_metadata.json").read_text(encoding="utf-8"))
    inventory = [r["target_chembl_id"] for r in catalog["human_single_protein"]
                 if int(r["target_chembl_id"][6:]) > 1821][:100]
    if len(inventory) != 100 or inventory[0] != "CHEMBL1822" or inventory[-1] != "CHEMBL1958":
        raise AssertionError("Frozen target inventory changed")
    manifest = pd.read_csv(cohort / "screening_manifest.csv",
                           dtype={"passes_smoke_threshold": "string"})
    if manifest.target_chembl_id.duplicated().any() or not set(manifest.target_chembl_id).issubset(inventory):
        raise AssertionError("Curated manifest differs from frozen inventory")
    by_id = manifest.set_index("target_chembl_id")
    rows = []
    for target in inventory:
        if target not in by_id.index:
            rows.append({"target_chembl_id": target,
                         "passes_global_quality": False,
                         "eligible": False, "reason": "no nonempty curated panel"})
            continue
        row = by_id.loc[target]
        reason = ""
        if str(row.passes_smoke_threshold).lower() != "true":
            reason = "global quality threshold"
        else:
            panel = pd.read_csv(root / row.panel_path,
                                usecols=["split", "label", "murcko_scaffold"])
            train, cal, test = (panel[panel.split.eq(s)]
                                for s in ("train", "calibration", "testpool"))
            if train.label.nunique() != 2 or cal.label.nunique() != 2:
                reason = "single-class train or calibration"
            elif test.murcko_scaffold.nunique() < 50:
                reason = "test scaffold capacity below 50"
        rows.append({"target_chembl_id": target,
                     "passes_global_quality": str(row.passes_smoke_threshold).lower() == "true",
                     "eligible": not reason, "reason": reason})
    ledger = pd.DataFrame(rows)
    eligible = ledger.loc[ledger.eligible, "target_chembl_id"].tolist()
    if not eligible:
        raise RuntimeError("No technically eligible eighth-cohort panels")
    ledger_path = cohort / "technical_eligibility.csv"
    if args.seed == 35801:
        ledger.to_csv(ledger_path, index=False)
    else:
        frozen = pd.read_csv(ledger_path, keep_default_na=False)
        if not frozen.equals(ledger):
            raise AssertionError("Technical eligibility changed across scorer seeds")
    out = cohort / f"eligible_seed{args.seed}"
    tasks = [(root / manifest.loc[manifest.target_chembl_id.eq(target), "panel_path"].iloc[0],
              out, args.seed + index, 4, 350, 4, .04)
             for index, target in enumerate(eligible)]
    with futures.ProcessPoolExecutor(max_workers=min(8, len(tasks))) as pool:
        results = list(pool.map(train_one, tasks))
    pd.DataFrame(results).sort_values("target_chembl_id").to_csv(
        out / "scorer_metrics.csv", index=False)
    (out / "metadata.json").write_text(json.dumps({
        "status": "frozen eighth-cohort upstream scorers; no outcome-based substitutions",
        "base_seed": args.seed, "eligible_targets": eligible,
        "threads_per_model": 4, "trees": 350, "max_depth": 4,
        "learning_rate": .04,
    }, indent=2) + "\n")
    print(args.seed, eligible, flush=True)


if __name__ == "__main__":
    main()
