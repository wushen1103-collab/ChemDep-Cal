#!/usr/bin/env python3
"""Post hoc cross-assay confirmation transfer with global test-molecule embargo."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rdkit import RDLogger
from scipy.special import expit
from scipy.stats import rankdata

from develop_mfpcba_multifp_rf_20261004 import fingerprints
from evaluate_mfpcba_twenty_20261004 import choose


def run_seed(root_s: str, seed: int, include_twelve: bool) -> tuple[list[dict], list[dict], dict]:
    root = Path(root_s)
    campaigns = ["twenty"] + (["twelve_new"] if include_twelve else [])
    tasks = []
    for campaign in campaigns:
        directory = root / f"data/mfpcba_{campaign}_20261004"
        manifest = json.loads((directory / "cohort_manifest.json").read_text())
        tasks.extend((campaign, name) for name in manifest["eligible_tasks"])
    names = [name for _, name in tasks]
    RDLogger.DisableLog("rdApp.*")
    panels = []
    all_test_smiles = set()
    for campaign, name in tasks:
        directory = root / f"data/mfpcba_{campaign}_20261004"
        cohort_file = (directory / f"{name}_cohort.parquet" if campaign == "twenty"
                       else directory / "cohorts" / f"{name}_cohort.parquet")
        split_file = (directory / f"{name}_seed{seed}_indices.npz" if campaign == "twenty"
                      else directory / "cohorts" / f"{name}_seed{seed}_indices.npz")
        cohort = pd.read_parquet(cohort_file)
        with np.load(split_file) as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        cal = cohort.iloc[cal_idx]
        test = cohort.iloc[test_idx]
        center = float(np.median(cal.score))
        scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        features = fingerprints(cohort.canonical_smiles)
        scores = expit((cohort.score.to_numpy(float) - center) / scale)
        panels.append({"name": name, "campaign": campaign,
                       "cohort": cohort, "cal_idx": cal_idx,
                       "test_idx": test_idx, "fp": features, "score": scores})
        all_test_smiles.update(test.canonical_smiles)
    x_parts, y_parts, w_parts, excluded = [], [], [], {}
    for i, panel in enumerate(panels):
        cohort, cal_idx = panel["cohort"], panel["cal_idx"]
        keep = ~cohort.iloc[cal_idx].canonical_smiles.isin(all_test_smiles).to_numpy()
        kept_idx = cal_idx[keep]
        excluded[panel["name"]] = int(len(cal_idx) - len(kept_idx))
        labels = cohort.iloc[kept_idx].label.to_numpy(int)
        if len(kept_idx) < 50 or labels.sum() < 5 or (1 - labels).sum() < 5:
            raise AssertionError("Embargo removed too much of the calibration data")
        task_column = np.zeros((len(kept_idx), len(names)), dtype=np.float32)
        task_column[:, i] = 1.
        x_parts.append(np.column_stack([panel["fp"][kept_idx],
                                        panel["score"][kept_idx], task_column]).astype(np.float32))
        y_parts.append(labels)
        per_class = np.where(labels == 1, .5 / labels.sum(), .5 / (len(labels) - labels.sum()))
        w_parts.append(per_class * len(labels))
    x_train, y_train = np.concatenate(x_parts), np.concatenate(y_parts)
    weights = np.concatenate(w_parts)
    model = LGBMClassifier(
        n_estimators=400, learning_rate=.035, num_leaves=31,
        min_child_samples=30, colsample_bytree=.8, subsample=.9,
        subsample_freq=1, reg_lambda=5., verbosity=-1,
        n_jobs=16, random_state=862026 + seed).fit(
            x_train, y_train, sample_weight=weights)
    reference = pd.concat([
        pd.read_parquet(root / f"data/mfpcba_catboost_{campaign}_development_20261004"
                        / "test_predictions.parquet") for campaign in campaigns], ignore_index=True)
    old = pd.concat([
        pd.read_csv(root / "data" / ("mfpcba_multifp_rf_twenty_development_20261004"
                                     if campaign == "twenty" else
                                     "mfpcba_multifp_rf_development_20261004") / "cells.csv")
        for campaign in campaigns], ignore_index=True)
    rows, predictions = [], []
    for i, panel in enumerate(panels):
        name, cohort, test_idx = panel["name"], panel["cohort"], panel["test_idx"]
        test = cohort.iloc[test_idx]
        task_column = np.zeros((len(test_idx), len(names)), dtype=np.float32)
        task_column[:, i] = 1.
        x_test = np.column_stack([panel["fp"][test_idx],
                                  panel["score"][test_idx], task_column]).astype(np.float32)
        global_score = model.predict_proba(x_test)[:, 1]
        rf_frame = reference.loc[reference.task.eq(name) & reference.seed.eq(seed)
                                 & reference.method.eq("multifp_rf_replay")]
        rf_score = rf_frame.set_index("canonical_smiles").score.reindex(
            test.canonical_smiles.to_numpy()).to_numpy(float)
        if not np.isfinite(rf_score).all():
            raise AssertionError("RF prediction coverage mismatch")
        blocks = test.murcko_scaffold.where(
            test.murcko_scaffold.ne(""), "singleton:" + test.canonical_smiles).to_numpy(str)
        keys = np.asarray([int.from_bytes(hashlib.sha256(
            f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        rf_rank = rankdata(rf_score) / len(rf_score)
        global_rank = rankdata(global_score) / len(global_score)
        candidates = {
            "multifp_rf_replay": rf_score,
            "global_transfer": global_score,
            "rf_global_rank_8020": .8 * rf_rank + .2 * global_rank,
            "rf_global_rank_5050": .5 * rf_rank + .5 * global_rank,
        }
        for method, score in candidates.items():
            selected = choose(score, blocks, keys)
            if len(selected) != 50 or len(set(blocks[selected])) != 50:
                raise AssertionError("Budget or scaffold-cap violation")
            hits = int(test.label.to_numpy(int)[selected].sum())
            rows.append({"task": name, "campaign": panel["campaign"],
                         "seed": seed, "method": method,
                         "hits": hits, "test_n": len(test),
                         "test_pos": int(test.label.sum()),
                         "other_assay_train_embargoed": excluded[name]})
            if method == "multifp_rf_replay":
                rf_method = ("multifp_score_rf" if panel["campaign"] == "twenty"
                             else "multifp_score_readout_rf")
                expected = int(old.loc[old.task.eq(name) & old.seed.eq(seed)
                                       & old.method.eq(rf_method), "hits"].iloc[0])
                if hits != expected:
                    raise AssertionError(f"RF baseline mismatch {name}/{seed}")
        predictions.extend({"task": name, "campaign": panel["campaign"], "seed": seed,
                            "canonical_smiles": smiles, "global_score": float(g),
                            "rf_score": float(r)}
                           for smiles, g, r in zip(test.canonical_smiles,
                                                   global_score, rf_score))
        print(name, seed, "completed", flush=True)
    provenance = {"seed": seed, "train_n": len(y_train),
                  "train_pos": int(y_train.sum()), "embargoed_by_task": excluded}
    return rows, predictions, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--include-twelve", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        groups = list(pool.map(run_seed, [str(root)] * 5, range(1, 6),
                               [args.include_twelve] * 5))
    out_name = ("mfpcba_cross_assay_transfer_21task_development_20261004"
                if args.include_twelve else "mfpcba_cross_assay_transfer_development_20261004")
    out = root / "data" / out_name
    out.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame([row for group in groups for row in group[0]])
    predictions = pd.DataFrame([row for group in groups for row in group[1]])
    cells.to_csv(out / "cells.csv", index=False)
    predictions.to_parquet(out / "test_predictions.parquet", index=False)
    (out / "provenance.json").write_text(json.dumps([group[2] for group in groups],
                                               indent=2) + "\n")
    assay = cells.groupby(["task", "method"]).hits.agg(["mean", "std"]).reset_index()
    assay.to_csv(out / "assay_mean_sd.csv", index=False)
    print(assay.groupby("method")["mean"].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
