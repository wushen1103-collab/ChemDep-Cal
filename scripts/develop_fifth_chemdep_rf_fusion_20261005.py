#!/usr/bin/env python3
"""Post hoc fixed-weight ChemDep-Cal/multi-FP RF rank-fusion diagnostic."""

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
from xgboost import XGBClassifier

from develop_mfpcba_multifp_rf_20261004 import fingerprints, fit_rf


def run_seed(root_s: str, seed: int) -> int:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import cap_one, load_panel

    RDLogger.DisableLog("rdApp.*")
    paths = sorted((root / "data/chembl_meta_ungated_fifth_xgb_20261003"
                    / f"seed{seed}" / "scores").glob("*_scores.csv"))
    if len(paths) != 10:
        raise AssertionError("Expected ten targets")
    panels = [load_panel(path) for path in paths]
    reference = pd.read_csv(root / "data/conditional_sota_meta_ungated_fifth_20261003"
                            / f"meta_seed{seed}.csv")
    reference = reference[reference.budget.eq(50) & reference.method.isin(
        ("meta_calibration", "score_cap1"))]
    expected = {(r.target, int(r.rep), float(r.fraction), float(r.boost), r.method): int(r.hits)
                for r in reference.itertuples(index=False)}
    if len(expected) != 800:
        raise AssertionError("Frozen reference incomplete")
    rf_reference = pd.read_csv(root / "data/pr_fifth_multifp_rf_posthoc_20261005"
                               / f"seed{seed}_cells.csv")
    rf_expected = {(r.target, int(r.rep), float(r.fraction), float(r.boost)):
                   int(r.multifp_rf_hits) for r in rf_reference.itertuples(index=False)}
    if len(rf_expected) != 400:
        raise AssertionError("RF reference incomplete")
    model_path = (root / "data/conditional_sota_meta_ungated_fifth_20261003"
                  / f"meta_seed{seed}.model.json")
    chemdep = XGBClassifier()
    chemdep.load_model(model_path)
    output = root / "data/pr_fifth_chemdep_rf_fusion_development_20261005"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for panel_index, panel in enumerate(panels):
        target, cal, test = panel["target"], panel["cal"], panel["test"]
        fp_cal, fp_test = fingerprints(cal.canonical_smiles), fingerprints(test.canonical_smiles)
        rf = fit_rf(np.column_stack([fp_cal, cal.score.to_numpy(float)]),
                    cal.label.to_numpy(int), seed)
        base, labels = test.score.to_numpy(float), test.label.to_numpy(int)
        blocks, null = panel["test_blocks"], panel["null"]
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
                    if int(labels[cap_one(scores, blocks, 50)].sum()) != expected[(*key, "score_cap1")]:
                        raise AssertionError(f"Score-cap replay mismatch: {key}")
                    cp = chemdep.predict_proba(evidence_features(scores, blocks, null))[:, 1]
                    rp = rf.predict_proba(np.column_stack([fp_test, scores]))[:, 1]
                    if int(labels[cap_one(cp, blocks, 50)].sum()) != expected[(*key, "meta_calibration")]:
                        raise AssertionError(f"ChemDep replay mismatch: {key}")
                    if int(labels[cap_one(rp, blocks, 50)].sum()) != rf_expected[key]:
                        raise AssertionError(f"RF replay mismatch: {key}")
                    cr = rankdata(cp, method="average") / len(cp)
                    rr = rankdata(rp, method="average") / len(rp)
                    for weight in (.25, .5, .75):
                        chosen = cap_one(weight * rr + (1. - weight) * cr, blocks, 50)
                        if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                            raise AssertionError("Budget or scaffold-cap violation")
                        rows.append({"target": target, "seed": seed, "rep": rep,
                                     "fraction": fraction, "boost": boost,
                                     "rf_weight": weight, "chemdep_hits": expected[(*key, "meta_calibration")],
                                     "rf_hits": rf_expected[key], "fusion_hits": int(labels[chosen].sum())})
        print("seed", seed, "target", target, "completed", flush=True)
    cells = pd.DataFrame(rows)
    if len(cells) != 1200:
        raise AssertionError("Incomplete fusion grid")
    cells.to_csv(output / f"seed{seed}_cells.csv", index=False)
    (output / f"seed{seed}_manifest.json").write_text(json.dumps({
        "status": "post hoc inspected-development fusion; not independent SOTA evidence",
        "seed": seed, "chemdep_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "original_chemdep_cells_replayed": 400,
        "original_rf_cells_replayed": 400,
        "rf_weights": [.25, .5, .75],
    }, indent=2) + "\n")
    return len(cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        print(list(pool.map(run_seed, [str(args.root.resolve())] * 5,
                            range(35501, 35506))), flush=True)


if __name__ == "__main__":
    main()
