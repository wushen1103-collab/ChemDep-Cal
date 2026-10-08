#!/usr/bin/env python3
"""Matched frozen-MoLFormer comparison on exposed confirmatory chains."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from xgboost import XGBClassifier

from develop_mfpcba_readout_context_20261004 import quality_frame
from evaluate_mfpcba_twenty_20261004 import choose


def run_task(root_s: str, campaign: str, name: str) -> list[dict]:
    root = Path(root_s)
    directory = root / f"data/mfpcba_{campaign}_20261004"
    cohort_file = (directory / f"{name}_cohort.parquet" if campaign == "twenty"
                   else directory / "cohorts" / f"{name}_cohort.parquet")
    cohort = pd.read_parquet(cohort_file)
    with np.load(root / "data/mfpcba_molformer_exposed_development_20261005/embeddings.npz") as source:
        smiles = source["smiles"]
        latent = source["embedding"]
    lookup = pd.Series(np.arange(len(smiles)), index=smiles)
    positions = lookup.reindex(cohort.canonical_smiles).to_numpy()
    if not np.isfinite(positions).all():
        raise AssertionError("Missing MoLFormer embedding")
    embedded = latent[positions.astype(int)]
    quality, quality_columns = ((np.empty((len(cohort), 0)), []) if campaign == "twenty"
                                else quality_frame(directory, name, cohort))
    reference_file = (root / "data" /
                      ("mfpcba_multifp_rf_twenty_development_20261004" if campaign == "twenty"
                       else "mfpcba_multifp_rf_development_20261004") / "cells.csv")
    reference = pd.read_csv(reference_file)
    reference_method = ("multifp_score_rf" if campaign == "twenty"
                        else "multifp_score_readout_rf")
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
        features = np.column_stack([embedded, score, quality])
        y_cal, y_test = cal.label.to_numpy(int), test.label.to_numpy(int)
        positives = int(y_cal.sum())
        model = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=.035, subsample=.9,
            colsample_bytree=.9, min_child_weight=8, reg_lambda=4.,
            tree_method="hist", eval_metric="logloss", n_jobs=8,
            random_state=862026 + seed,
            scale_pos_weight=min(20., max(1., (len(y_cal) - positives) / positives)),
        ).fit(features[cal_idx], y_cal)
        tie = np.asarray([int.from_bytes(hashlib.sha256(
            f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        blocks = test.murcko_scaffold.where(
            test.murcko_scaffold.ne(""), "singleton:" + test.canonical_smiles).to_numpy(str)
        selected = choose(model.predict_proba(features[test_idx])[:, 1], blocks, tie)
        if len(selected) != 50 or len(set(blocks[selected])) != 50:
            raise AssertionError("Budget or scaffold-cap violation")
        baseline = reference.loc[reference.task.eq(name) & reference.seed.eq(seed)
                                 & reference.method.eq(reference_method), "hits"]
        if len(baseline) != 1:
            raise AssertionError("Missing matched RF reference")
        rows.append({"task": name, "seed": seed, "method": "frozen_molformer_score_xgb",
                     "hits": int(y_test[selected].sum()), "rf_hits": int(baseline.iloc[0]),
                     "test_n": len(test), "test_pos": int(y_test.sum()),
                     "quality_dimensions": len(quality_columns)})
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
    output = root / f"data/mfpcba_molformer_{args.campaign}_development_20261005"
    output.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame([row for group in groups for row in group])
    cells.to_csv(output / "cells.csv", index=False)
    assay = cells.groupby(["task", "method"]).agg(
        mean=("hits", "mean"), std=("hits", "std"), rf_mean=("rf_hits", "mean")
    ).reset_index()
    assay.to_csv(output / "assay_mean_sd.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "status": "post hoc exposed-assay development; not independent SOTA evidence",
        "source_protocol_sha256": manifest["protocol_sha256"],
        "embedding_manifest": "data/mfpcba_molformer_exposed_development_20261005/manifest.json",
        "model": "frozen_molformer_score_xgb",
        "training_labels": "same per-assay calibration fold as multi-FP RF",
    }, indent=2) + "\n")
    print(assay.to_string(index=False), flush=True)
    print("macro", assay["mean"].mean(), "RF reference", assay["rf_mean"].mean(), flush=True)


if __name__ == "__main__":
    main()
