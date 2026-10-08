#!/usr/bin/env python3
"""Fixed equal-rank ensemble of clean and artifact-augmented RF on exposed cohorts."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from scipy.stats import rankdata

from develop_fifth_augmented_multifp_20261005 import augmented_views
from develop_mfpcba_multifp_rf_20261004 import fingerprints, fit_rf


def run_seed(root_s: str, cohort: str, seed: int) -> int:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import cap_one, load_panel

    RDLogger.DisableLog("rdApp.*")
    if cohort == "fifth":
        paths = sorted((root / "data/chembl_meta_ungated_fifth_xgb_20261003"
                        / f"seed{seed}" / "scores").glob("*_scores.csv"))
        ref = pd.read_csv(root / "data/pr_fifth_augmented_multifp_development_20261005"
                          / f"seed{seed}_cells.csv")
        seed_offset = 355700
    else:
        directory = root / "data/chembl_modern_seventh_20261005"
        eligible = pd.read_csv(directory / "technical_eligibility.csv").query(
            "eligible").target_chembl_id.tolist()
        paths = [directory / f"eligible_seed{seed}" / "scores"
                 / f"{target}_scores.csv" for target in eligible]
        ref = pd.read_csv(root / "data/pr_seventh_augmented_multifp_development_20261005"
                          / f"seed{seed}_cells.csv")
        seed_offset = 357700
    if len(paths) != (10 if cohort == "fifth" else 3):
        raise AssertionError("Unexpected target count")
    panels = [load_panel(path) for path in paths]
    expected = {(r.target, int(r.rep), float(r.fraction), float(r.boost)):
                (int(r.score_cap1_hits), int(r.multifp_rf_hits),
                 int(r.augmented_score_rf_hits))
                for r in ref.itertuples(index=False)}
    out = root / f"data/pr_{cohort}_dual_rf_equal_fusion_development_20261005"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, panel in enumerate(panels):
        target, cal, test = panel["target"], panel["cal"], panel["test"]
        fp_cal, fp_test = fingerprints(cal.canonical_smiles), fingerprints(test.canonical_smiles)
        clean_rf = fit_rf(np.column_stack([fp_cal, cal.score.to_numpy(float)]),
                          cal.label.to_numpy(int), seed)
        augmented, labels = augmented_views(
            panel, evidence_features, choose_artifact_blocks, inject_artifact)
        robust_rf = fit_rf(np.column_stack([np.tile(fp_cal, (13, 1)), augmented[:, 0]]),
                           labels, seed)
        base, truth = test.score.to_numpy(float), test.label.to_numpy(int)
        blocks = panel["test_blocks"]
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = seed_offset + index * 100_000 + target_hash
        for rep in range(10):
            for fraction in (.2, .3):
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
                artifacts = choose_artifact_blocks(
                    test, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2)
                for boost in (4., 6.):
                    scores = inject_artifact(
                        test, base, artifacts, artifact_block_col="murcko_scaffold",
                        logit_boost=boost)[0]
                    key = (target, rep, fraction, boost)
                    x = np.column_stack([fp_test, scores])
                    clean = clean_rf.predict_proba(x)[:, 1]
                    robust = robust_rf.predict_proba(x)[:, 1]
                    replay = (int(truth[cap_one(scores, blocks, 50)].sum()),
                              int(truth[cap_one(clean, blocks, 50)].sum()),
                              int(truth[cap_one(robust, blocks, 50)].sum()))
                    if replay != expected[key]:
                        raise AssertionError(f"Matched reference mismatch: {key}: {replay}")
                    fusion = (rankdata(clean, method="average")
                              + rankdata(robust, method="average")) / (2 * len(clean))
                    chosen = cap_one(fusion, blocks, 50)
                    if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                        raise AssertionError("Budget/scaffold violation")
                    rows.append({"target": target, "seed": seed, "rep": rep,
                                 "fraction": fraction, "boost": boost,
                                 "score_cap1_hits": replay[0], "multifp_rf_hits": replay[1],
                                 "augmented_score_rf_hits": replay[2],
                                 "equal_rank_fusion_hits": int(truth[chosen].sum())})
        print(cohort, "seed", seed, "target", target, "completed", flush=True)
    cells = pd.DataFrame(rows)
    if len(cells) != len(panels) * 40:
        raise AssertionError("Incomplete fusion run")
    cells.to_csv(out / f"seed{seed}_cells.csv", index=False)
    (out / f"seed{seed}_manifest.json").write_text(json.dumps({
        "status": "exposed-cohort development only; not independent SOTA evidence",
        "cohort": cohort, "seed": seed, "targets": [p["target"] for p in panels],
        "combination": "equal mean of percentile ranks, no tuned weight",
        "matched_baselines_replayed_per_cell": True,
    }, indent=2) + "\n")
    return len(cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cohort", choices=("fifth", "seventh"), required=True)
    args = parser.parse_args()
    seeds = list(range(35501, 35506) if args.cohort == "fifth" else range(35701, 35706))
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        print(list(pool.map(run_seed, [str(args.root.resolve())] * 5,
                            [args.cohort] * 5, seeds)), flush=True)


if __name__ == "__main__":
    main()
