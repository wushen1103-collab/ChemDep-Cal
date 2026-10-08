#!/usr/bin/env python3
"""Matched multi-FP RF with ChemDep evidence features on exposed assays."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from develop_mfpcba_multifp_rf_20261004 import fingerprints, fit_rf
from develop_mfpcba_readout_context_20261004 import quality_frame
from evaluate_mfpcba_twenty_20261004 import choose


def run_task(root_s: str, campaign: str, name: str) -> list[dict]:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from probe_meta_band_v2_20261003 import evidence_features

    directory = root / f"data/mfpcba_{campaign}_20261004"
    cohort_file = (directory / f"{name}_cohort.parquet" if campaign == "twenty"
                   else directory / "cohorts" / f"{name}_cohort.parquet")
    cohort = pd.read_parquet(cohort_file)
    fp = fingerprints(cohort.canonical_smiles)
    quality, _ = ((np.empty((len(cohort), 0)), []) if campaign == "twenty"
                  else quality_frame(directory, name, cohort))
    reference_file = (root / "data" /
                      ("mfpcba_multifp_rf_twenty_development_20261004" if campaign == "twenty"
                       else "mfpcba_multifp_rf_development_20261004") / "cells.csv")
    reference = pd.read_csv(reference_file)
    rf_method = "multifp_score_rf" if campaign == "twenty" else "multifp_score_readout_rf"
    rows = []
    for seed in range(1, 6):
        split_file = (directory / f"{name}_seed{seed}_indices.npz" if campaign == "twenty"
                      else directory / "cohorts" / f"{name}_seed{seed}_indices.npz")
        with np.load(split_file) as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        cal, test = cohort.iloc[cal_idx], cohort.iloc[test_idx]
        center = float(np.median(cal.score))
        scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        score = expit((cohort.score.to_numpy(float) - center) / scale)
        cal_score, test_score = score[cal_idx], score[test_idx]
        cal_blocks = cal.murcko_scaffold.where(
            cal.murcko_scaffold.ne(""), "singleton:" + cal.canonical_smiles).to_numpy(str)
        test_blocks = test.murcko_scaffold.where(
            test.murcko_scaffold.ne(""), "singleton:" + test.canonical_smiles).to_numpy(str)
        null = cal_score[cal.label.to_numpy(int) == 0]
        context_cal = evidence_features(cal_score, cal_blocks, null)
        context_test = evidence_features(test_score, test_blocks, null)
        y_cal, y_test = cal.label.to_numpy(int), test.label.to_numpy(int)
        base_cal = np.column_stack([fp[cal_idx], cal_score, quality[cal_idx]])
        base_test = np.column_stack([fp[test_idx], test_score, quality[test_idx]])
        predictions = {
            "multi_fp_rf": fit_rf(base_cal, y_cal, seed).predict_proba(base_test)[:, 1],
            "multi_fp_chemdep_rf": fit_rf(
                np.column_stack([base_cal, context_cal]), y_cal, seed).predict_proba(
                    np.column_stack([base_test, context_test]))[:, 1],
        }
        ties = np.asarray([int.from_bytes(hashlib.sha256(
            f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        for method, values in predictions.items():
            selected = choose(values, test_blocks, ties)
            if len(selected) != 50 or len(set(test_blocks[selected])) != 50:
                raise AssertionError("Budget or scaffold-cap violation")
            hits = int(y_test[selected].sum())
            if method == "multi_fp_rf":
                expected = reference.loc[reference.task.eq(name)
                                         & reference.seed.eq(seed)
                                         & reference.method.eq(rf_method), "hits"]
                if len(expected) != 1 or hits != int(expected.iloc[0]):
                    raise AssertionError(f"RF replay mismatch: {name}/{seed}")
            rows.append({"task": name, "seed": seed, "method": method, "hits": hits})
        print(campaign, name, seed, "completed", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", choices=("twelve_new", "twenty"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / f"data/mfpcba_{args.campaign}_20261004"
                           / "cohort_manifest.json").read_text())
    names = manifest["eligible_tasks"]
    with futures.ProcessPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(run_task, [str(root)] * len(names),
                               [args.campaign] * len(names), names))
    output = root / f"data/mfpcba_structured_rf_{args.campaign}_development_20261005"
    output.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame([row for group in groups for row in group])
    cells.to_csv(output / "cells.csv", index=False)
    assay = cells.groupby(["task", "method"]).hits.agg(["mean", "std"]).reset_index()
    assay.to_csv(output / "assay_mean_sd.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "status": "post hoc development on exposed assays, not independent SOTA evidence",
        "protocol_sha256": manifest["protocol_sha256"],
        "input_difference": "16 ChemDep score/scaffold-context features added to same RF",
        "reference_replayed_exactly": True,
    }, indent=2) + "\n")
    print(assay.pivot(index="task", columns="method", values="mean").to_string(), flush=True)
    print(assay.groupby("method")["mean"].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
