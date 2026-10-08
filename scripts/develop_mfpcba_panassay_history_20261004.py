#!/usr/bin/env python3
"""Post hoc cross-primary-screen history under fixed confirmation splits."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier

from develop_mfpcba_multifp_rf_20261004 import fingerprints
from develop_mfpcba_readout_context_20261004 import quality_frame
from evaluate_mfpcba_twenty_20261004 import choose

OUTPUT_DIR = "mfpcba_panassay_history_v2_development_20261004"


def sources(root: Path) -> list[Path]:
    return sorted((root / "data/external_benchmark_sources/AIC_Finder/Datasets").glob("*.csv"))


def build_profile(root: Path) -> None:
    tasks = []
    for campaign in ("twenty", "twelve_new"):
        directory = root / f"data/mfpcba_{campaign}_20261004"
        manifest = json.loads((directory / "cohort_manifest.json").read_text())
        for name in manifest["eligible_tasks"]:
            file = (directory / f"{name}_cohort.parquet" if campaign == "twenty"
                    else directory / "cohorts" / f"{name}_cohort.parquet")
            cohort = pd.read_parquet(file, columns=["canonical_smiles"])
            tasks.append((campaign, name, cohort.canonical_smiles.tolist()))
    all_smiles = set(smiles for _, _, members in tasks for smiles in members)
    assay_maps = {}
    for file in sources(root):
        values = {}
        for chunk in pd.read_csv(file, usecols=["SMILES", "Primary"], chunksize=100000):
            matched = chunk.loc[chunk.SMILES.isin(all_smiles) & chunk.Primary.isin([0, 1])]
            for smiles, primary in zip(matched.SMILES, matched.Primary):
                values[smiles] = max(values.get(smiles, 0), int(primary))
        assay_maps[file.stem] = values
        print(file.stem, "matched", len(values), flush=True)
    rows = []
    for campaign, name, members in tasks:
        own = {"transcription2": "transcription_2",
               "transcription3": "transcription_3"}.get(name, name)
        for smiles in members:
            observed = [values[smiles] for assay, values in assay_maps.items()
                        if assay != own and smiles in values]
            rows.append({"campaign": campaign, "task": name,
                         "canonical_smiles": smiles, "other_tested": len(observed),
                         "other_active": sum(observed),
                         "other_rate": (sum(observed) + 1) / (len(observed) + 2)})
    out = root / "data" / OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    profile = pd.DataFrame(rows)
    profile.to_parquet(out / "primary_history.parquet", index=False)
    coverage = profile.groupby(["campaign", "task"]).agg(
        n=("canonical_smiles", "size"),
        covered=("other_tested", lambda s: int(s.gt(0).sum())),
        median_tested=("other_tested", "median"),
        mean_active=("other_active", "mean")).reset_index()
    coverage["coverage"] = coverage.covered / coverage.n
    coverage.to_csv(out / "coverage.csv", index=False)
    print(coverage.to_string(index=False), flush=True)


def run_task(root_s: str, campaign: str, name: str) -> list[dict]:
    root = Path(root_s)
    directory = root / f"data/mfpcba_{campaign}_20261004"
    cohort_file = (directory / f"{name}_cohort.parquet" if campaign == "twenty"
                   else directory / "cohorts" / f"{name}_cohort.parquet")
    RDLogger.DisableLog("rdApp.*")
    cohort = pd.read_parquet(cohort_file)
    profile = pd.read_parquet(root / "data" / OUTPUT_DIR
                              / "primary_history.parquet")
    profile = profile.loc[profile.campaign.eq(campaign) & profile.task.eq(name)]
    aligned = profile.set_index("canonical_smiles").reindex(cohort.canonical_smiles)
    if aligned.other_tested.isna().any():
        raise AssertionError("Primary history row missing")
    history = aligned[["other_tested", "other_active", "other_rate"]].to_numpy(float)
    fp = fingerprints(cohort.canonical_smiles)
    quality, columns = ((np.empty((len(cohort), 0)), []) if campaign == "twenty"
                        else quality_frame(directory, name, cohort))
    rf_dir = ("mfpcba_multifp_rf_twenty_development_20261004" if campaign == "twenty"
              else "mfpcba_multifp_rf_development_20261004")
    reference = pd.read_csv(root / "data" / rf_dir / "cells.csv")
    ref_method = "multifp_score_rf" if campaign == "twenty" else "multifp_score_readout_rf"
    rows = []
    for seed in range(1, 6):
        split_file = (directory / f"{name}_seed{seed}_indices.npz" if campaign == "twenty"
                      else directory / "cohorts" / f"{name}_seed{seed}_indices.npz")
        with np.load(split_file) as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        cal, test = cohort.iloc[cal_idx], cohort.iloc[test_idx]
        center = float(np.median(cal.score))
        scale = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        scores = expit((cohort.score.to_numpy(float) - center) / scale)
        base_x = np.column_stack([fp, scores, quality])
        informed_x = np.column_stack([base_x, history])
        y_cal, y_test = cal.label.to_numpy(int), test.label.to_numpy(int)
        keys = np.asarray([int.from_bytes(hashlib.sha256(
            f"{name}/{seed}/{smiles}".encode()).digest()[:8], "big")
            for smiles in test.canonical_smiles], dtype=np.uint64)
        blocks = test.murcko_scaffold.where(
            test.murcko_scaffold.ne(""), "singleton:" + test.canonical_smiles).to_numpy(str)
        for method, matrix in (("multifp_rf_replay", base_x),
                               ("multifp_panassay_history_rf", informed_x)):
            model = RandomForestClassifier(
                n_estimators=500, min_samples_leaf=3, max_features=.2,
                class_weight="balanced_subsample", n_jobs=8,
                random_state=862026 + seed).fit(matrix[cal_idx], y_cal)
            prediction = model.predict_proba(matrix[test_idx])[:, 1]
            selected = choose(prediction, blocks, keys)
            if len(selected) != 50 or len(set(blocks[selected])) != 50:
                raise AssertionError("Budget or scaffold-cap violation")
            hits = int(y_test[selected].sum())
            rows.append({"task": name, "campaign": campaign, "seed": seed,
                         "method": method, "hits": hits, "test_n": len(test),
                         "test_pos": int(y_test.sum()),
                         "history_coverage": float((history[:, 0] > 0).mean()),
                         "quality_dimensions": len(columns)})
            if method == "multifp_rf_replay":
                expected = int(reference.loc[reference.task.eq(name)
                                             & reference.seed.eq(seed)
                                             & reference.method.eq(ref_method), "hits"].iloc[0])
                if hits != expected:
                    raise AssertionError(f"RF replay mismatch {name}/{seed}")
        print(name, seed, "completed", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "data" / OUTPUT_DIR
            / "primary_history.parquet").exists():
        build_profile(root)
    if args.profile_only:
        return
    tasks = []
    for campaign in ("twenty", "twelve_new"):
        directory = root / f"data/mfpcba_{campaign}_20261004"
        manifest = json.loads((directory / "cohort_manifest.json").read_text())
        tasks.extend((campaign, name) for name in manifest["eligible_tasks"])
    with futures.ProcessPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(run_task, [str(root)] * len(tasks),
                               [c for c, _ in tasks], [n for _, n in tasks]))
    out = root / "data" / OUTPUT_DIR
    cells = pd.DataFrame([row for group in groups for row in group])
    cells.to_csv(out / "cells.csv", index=False)
    assay = cells.groupby(["campaign", "task", "method"]).hits.agg(
        ["mean", "std"]).reset_index()
    assay.to_csv(out / "assay_mean_sd.csv", index=False)
    print(assay.groupby(["campaign", "method"])["mean"].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
