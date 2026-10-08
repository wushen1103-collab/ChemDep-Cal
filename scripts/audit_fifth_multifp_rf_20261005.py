#!/usr/bin/env python3
"""Post hoc multi-fingerprint RF control on all frozen fifth-cohort cells."""

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

from develop_mfpcba_multifp_rf_20261004 import fingerprints, fit_rf


def run_seed(root_s: str, seed: int) -> int:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import cap_one, load_panel

    RDLogger.DisableLog("rdApp.*")
    paths = sorted((root / "data/chembl_meta_ungated_fifth_xgb_20261003"
                    / f"seed{seed}" / "scores").glob("*_scores.csv"))
    if len(paths) != 10:
        raise AssertionError("Expected original ten frozen targets")
    panels = [load_panel(path) for path in paths]
    reference = pd.read_csv(root / "data/conditional_sota_meta_ungated_fifth_20261003"
                            / f"meta_seed{seed}.csv")
    reference = reference[reference.budget.eq(50) & reference.method.isin(
        ("meta_calibration", "score_cap1"))]
    expected = {(r.target, int(r.rep), float(r.fraction), float(r.boost), r.method): int(r.hits)
                for r in reference.itertuples(index=False)}
    if len(expected) != 800:
        raise AssertionError("Frozen reference incomplete")
    output = root / "data/pr_fifth_multifp_rf_posthoc_20261005"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for panel_index, panel in enumerate(panels):
        target = panel["target"]
        cal, test = panel["cal"], panel["test"]
        fp_cal = fingerprints(cal.canonical_smiles)
        fp_test = fingerprints(test.canonical_smiles)
        labels_cal = cal.label.to_numpy(int)
        if len(np.unique(labels_cal)) != 2:
            raise AssertionError(f"Unfit target calibration: {target}")
        train_x = np.column_stack([fp_cal, cal.score.to_numpy(float)])
        model = fit_rf(train_x, labels_cal, seed)
        base = test.score.to_numpy(float)
        labels = test.label.to_numpy(int)
        blocks = panel["test_blocks"]
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = 355700 + panel_index * 100_000 + target_hash
        for rep in range(10):
            for fraction in (.2, .3):
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
                artifacts = choose_artifact_blocks(
                    test, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2)
                for boost in (4., 6.):
                    scores = inject_artifact(
                        test, base, artifacts,
                        artifact_block_col="murcko_scaffold", logit_boost=boost)[0]
                    key = (target, rep, fraction, boost)
                    score_hits = int(labels[cap_one(scores, blocks, 50)].sum())
                    if score_hits != expected[(*key, "score_cap1")]:
                        raise AssertionError(f"Frozen score-cap replay mismatch: {key}")
                    predicted = model.predict_proba(np.column_stack([fp_test, scores]))[:, 1]
                    chosen = cap_one(predicted, blocks, 50)
                    if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                        raise AssertionError("Budget or scaffold-cap violation")
                    rows.append({"target": target, "seed": seed, "rep": rep,
                                 "fraction": fraction, "boost": boost,
                                 "chemdep_hits": expected[(*key, "meta_calibration")],
                                 "scorecap_hits": score_hits,
                                 "multifp_rf_hits": int(labels[chosen].sum()),
                                 "selected": len(chosen), "scaffolds": len(set(blocks[chosen]))})
        print("seed", seed, "target", target, "completed", flush=True)
    result = pd.DataFrame(rows)
    if len(result) != 400:
        raise AssertionError("Incomplete fifth-cohort RF grid")
    result.to_csv(output / f"seed{seed}_cells.csv", index=False)
    (output / f"seed{seed}_manifest.json").write_text(json.dumps({
        "status": "post hoc development on inspected fifth cohort, not independent confirmation",
        "seed": seed,
        "score_file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "frozen_scorecap_cells_replayed": 400,
        "frozen_chemdep_hits_loaded": 400,
        "test_label_use": "original label-aware artifact construction and final hit evaluation",
        "classifier": "per-target 4263-bit multi-FP plus perturbed score RF; same calibration labels",
    }, indent=2) + "\n")
    return len(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        print(list(pool.map(run_seed, [str(args.root.resolve())] * 5,
                            range(35501, 35506))), flush=True)


if __name__ == "__main__":
    main()
