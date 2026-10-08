#!/usr/bin/env python3
"""Score every feasible seventh-cohort panel with frozen fifth/sixth XGB settings."""

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
    root = args.root.resolve()
    cohort = root / "data/chembl_modern_seventh_20261005"
    manifest = pd.read_csv(cohort / "screening_manifest.csv",
                           dtype={"passes_smoke_threshold": "string"})
    rows = []
    for row in manifest.itertuples(index=False):
        reason = ""
        if str(row.passes_smoke_threshold).lower() != "true":
            reason = "global quality threshold"
        else:
            panel = pd.read_csv(root / row.panel_path,
                                usecols=["split", "label", "murcko_scaffold"])
            train = panel[panel.split.eq("train")]
            cal = panel[panel.split.eq("calibration")]
            test = panel[panel.split.eq("testpool")]
            if train.label.nunique() != 2 or cal.label.nunique() != 2:
                reason = "single-class train or calibration"
            elif test.murcko_scaffold.nunique() < 50:
                reason = "test scaffold capacity below 50"
        rows.append({"target_chembl_id": row.target_chembl_id,
                     "passes_global_quality": str(row.passes_smoke_threshold).lower() == "true",
                     "eligible": not reason, "reason": reason})
    ledger = pd.DataFrame(rows)
    eligible = ledger.loc[ledger.eligible, "target_chembl_id"].tolist()
    if not eligible:
        raise RuntimeError("No technically evaluable seventh-cohort panels")
    if args.seed == 35701:
        ledger.to_csv(cohort / "technical_eligibility.csv", index=False)
    else:
        frozen = pd.read_csv(cohort / "technical_eligibility.csv", keep_default_na=False)
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
        "base_seed": args.seed, "eligible_targets": eligible,
        "target_count": len(eligible), "workers": min(8, len(tasks)),
        "threads_per_model": 4, "technical_eligibility": str(cohort / "technical_eligibility.csv"),
        "status": "fixed seventh-cohort scorer; no outcome-based substitution",
    }, indent=2) + "\n", encoding="utf-8")
    print(args.seed, eligible, flush=True)


if __name__ == "__main__":
    main()
