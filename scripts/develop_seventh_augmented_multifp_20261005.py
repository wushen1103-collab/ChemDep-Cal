#!/usr/bin/env python3
"""Development-only transfer of fixed augmentation-only RF to exposed seventh cohort."""

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

from develop_fifth_augmented_multifp_20261005 import augmented_views
from develop_mfpcba_multifp_rf_20261004 import fingerprints, fit_rf


def run_seed(root_s: str, seed: int) -> int:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import cap_one, load_panel

    RDLogger.DisableLog("rdApp.*")
    cohort = root / "data/chembl_modern_seventh_20261005"
    eligible = pd.read_csv(cohort / "technical_eligibility.csv").query("eligible").target_chembl_id.tolist()
    paths = [cohort / f"eligible_seed{seed}" / "scores" / f"{target}_scores.csv"
             for target in eligible]
    if len(paths) != 3 or not all(path.is_file() for path in paths):
        raise AssertionError("Expected all three seventh-cohort eligible panels")
    panels = [load_panel(path) for path in paths]
    reference = pd.read_csv(cohort / "fixed_selector_results" / f"seed{seed}.csv")
    expected = {(r.target, int(r.rep), float(r.fraction), float(r.boost), r.method): int(r.hits)
                for r in reference.itertuples(index=False)}
    out = root / "data/pr_seventh_augmented_multifp_development_20261005"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, panel in enumerate(panels):
        target, cal, test = panel["target"], panel["cal"], panel["test"]
        fp_cal, fp_test = fingerprints(cal.canonical_smiles), fingerprints(test.canonical_smiles)
        context, y = augmented_views(
            panel, evidence_features, choose_artifact_blocks, inject_artifact)
        rf = fit_rf(np.column_stack([np.tile(fp_cal, (13, 1)), context[:, 0]]), y, seed)
        base, labels = test.score.to_numpy(float), test.label.to_numpy(int)
        blocks = panel["test_blocks"]
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = 357700 + index * 100_000 + target_hash
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
                    cap_hits = int(labels[cap_one(scores, blocks, 50)].sum())
                    if cap_hits != expected[(*key, "score_cap1")]:
                        raise AssertionError(f"Seventh score-cap replay mismatch: {key}")
                    pred = rf.predict_proba(np.column_stack([fp_test, scores]))[:, 1]
                    chosen = cap_one(pred, blocks, 50)
                    if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                        raise AssertionError(f"Selection feasibility mismatch: {key}")
                    rows.append({
                        "target": target, "seed": seed, "rep": rep, "fraction": fraction,
                        "boost": boost, "score_cap1_hits": cap_hits,
                        "multifp_rf_hits": expected[(*key, "multi_fp_rf")],
                        "chemdep_hits": expected[(*key, "chemdep_cal")],
                        "augmented_score_rf_hits": int(labels[chosen].sum()),
                    })
        print("seed", seed, "target", target, "completed", flush=True)
    cells = pd.DataFrame(rows)
    if len(cells) != 120:
        raise AssertionError("Incomplete seventh transfer")
    cells.to_csv(out / f"seed{seed}_cells.csv", index=False)
    (out / f"seed{seed}_manifest.json").write_text(json.dumps({
        "status": "exposed seventh-cohort development, not independent confirmation",
        "seed": seed, "eligible_targets": eligible,
        "calibration_views_per_molecule": 13,
        "test_labels": "artifact generation and final hit count only",
    }, indent=2) + "\n")
    return len(cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    seeds = list(range(35701, 35706))
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        print(list(pool.map(run_seed, [str(args.root.resolve())] * 5, seeds)), flush=True)


if __name__ == "__main__":
    main()
