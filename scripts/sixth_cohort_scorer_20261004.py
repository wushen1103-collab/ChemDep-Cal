#!/usr/bin/env python3
"""Score only manifest-eligible sixth-cohort targets with frozen XGB settings."""

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
    cohort = root / "data/chembl_modern_sixth_20261004"
    manifest = pd.read_csv(cohort / "screening_manifest.csv", dtype={"passes_smoke_threshold": "string"})
    eligible = manifest[manifest.passes_smoke_threshold.str.lower().eq("true")].copy()
    if len(eligible) != 8 or not eligible.target_chembl_id.is_unique:
        raise AssertionError("Frozen sixth-cohort eligibility changed")
    out = cohort / f"eligible_seed{args.seed}"
    tasks = [
        (root / row.panel_path, out, args.seed + index, 4, 350, 4, .04)
        for index, row in enumerate(eligible.itertuples(index=False))
    ]
    with futures.ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(train_one, tasks))
    result = pd.DataFrame(rows).sort_values("target_chembl_id")
    if set(result.target_chembl_id) != set(eligible.target_chembl_id):
        raise AssertionError("Scored target list differs from eligibility manifest")
    result.to_csv(out / "scorer_metrics.csv", index=False)
    (out / "metadata.json").write_text(json.dumps({
        "base_seed": args.seed,
        "eligible_targets": eligible.target_chembl_id.tolist(),
        "target_count": 8,
        "workers": 8,
        "threads_per_model": 4,
        "note": "Correct explicit string-boolean eligibility; earlier unfiltered scorer output unused",
    }, indent=2) + "\n", encoding="utf-8")
    print(args.seed, eligible.target_chembl_id.tolist(), flush=True)


if __name__ == "__main__":
    main()
