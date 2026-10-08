#!/usr/bin/env python3
"""Replay the previously fixed 75:25 RF/ChemDep rank fusion on cohort eight."""

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


METHODS = ("score_cap1", "chemdep_cal", "multi_fp_rf")


def run_seed(root_s: str, seed: int) -> int:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import cap_one, load_panel

    RDLogger.DisableLog("rdApp.*")
    cohort = root / "data/chembl_modern_eighth_20261005"
    ledger = pd.read_csv(cohort / "technical_eligibility.csv")
    if len(ledger) != 100:
        raise AssertionError("Eighth metadata-fixed inventory changed")
    eligible = ledger.loc[ledger.eligible, "target_chembl_id"].tolist()
    score_dir = cohort / f"eligible_seed{seed}" / "scores"
    paths = [score_dir / f"{target}_scores.csv" for target in eligible]
    if len(paths) != 30 or not all(path.is_file() for path in paths):
        raise AssertionError("Eighth eligible scorer panels changed")
    panels = [load_panel(path) for path in paths]
    reference_path = cohort / "fixed_selector_results" / f"seed{seed}.csv"
    reference = pd.read_csv(reference_path)
    expected = {
        (r.target, int(r.rep), float(r.fraction), float(r.boost), r.method): int(r.hits)
        for r in reference.itertuples(index=False)
        if r.method in METHODS
    }
    if len(expected) != 30 * 40 * len(METHODS):
        raise AssertionError("Missing or duplicate original eighth cells")
    output = root / "data/pr_eighth_fixed_75_25_fusion_20261005"
    output.mkdir(parents=True, exist_ok=True)
    model_path = cohort / "fixed_selector_results" / f"seed{seed}.chemdep.json"
    if not model_path.is_file():
        raise AssertionError("Original ChemDep model missing")
    chemdep = XGBClassifier()
    chemdep.load_model(model_path)
    rows = []

    for index, panel in enumerate(panels):
        target, cal, test = panel["target"], panel["cal"], panel["test"]
        fp_cal, fp_test = fingerprints(cal.canonical_smiles), fingerprints(test.canonical_smiles)
        rf = fit_rf(
            np.column_stack([fp_cal, cal.score.to_numpy(float)]),
            cal.label.to_numpy(int), seed,
        )
        base, labels = test.score.to_numpy(float), test.label.to_numpy(int)
        blocks, null = panel["test_blocks"], panel["null"]
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = 358700 + index * 100_000 + target_hash
        for rep in range(10):
            for fraction in (.2, .3):
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
                artifacts = choose_artifact_blocks(
                    test, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2,
                )
                for boost in (4., 6.):
                    scores = inject_artifact(
                        test, base, artifacts,
                        artifact_block_col="murcko_scaffold", logit_boost=boost,
                    )[0]
                    key = (target, rep, fraction, boost)
                    cp = chemdep.predict_proba(evidence_features(scores, blocks, null))[:, 1]
                    rp = rf.predict_proba(np.column_stack([fp_test, scores]))[:, 1]
                    predictions = {
                        "score_cap1": scores,
                        "chemdep_cal": cp,
                        "multi_fp_rf": rp,
                    }
                    for method, values in predictions.items():
                        chosen = cap_one(values, blocks, 50)
                        if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                            raise AssertionError(f"Original cap violation: {key}/{method}")
                        if int(labels[chosen].sum()) != expected[(*key, method)]:
                            raise AssertionError(f"Original replay mismatch: {key}/{method}")
                    cr = rankdata(cp, method="average") / len(cp)
                    rr = rankdata(rp, method="average") / len(rp)
                    chosen = cap_one(.75 * rr + .25 * cr, blocks, 50)
                    if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                        raise AssertionError(f"Fusion cap violation: {key}")
                    rows.append({
                        "target": target, "scorer_seed": seed, "rep": rep,
                        "fraction": fraction, "boost": boost,
                        "fusion_hits": int(labels[chosen].sum()),
                        "rf_hits": expected[(*key, "multi_fp_rf")],
                        "chemdep_hits": expected[(*key, "chemdep_cal")],
                        "scorecap_hits": expected[(*key, "score_cap1")],
                        "selected": len(chosen), "scaffolds": len(set(blocks[chosen])),
                    })
        print("seed", seed, "target", target, "completed", flush=True)

    cells = pd.DataFrame(rows)
    if len(cells) != 30 * 40:
        raise AssertionError("Incomplete eighth fusion grid")
    cells.to_csv(output / f"seed{seed}_cells.csv", index=False)
    (output / f"seed{seed}_manifest.json").write_text(json.dumps({
        "status": "fixed-weight replay on previously inspected eighth cohort",
        "scorer_seed": seed, "eligible_targets": eligible,
        "rf_weight": .75, "chemdep_weight": .25,
        "score_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in paths},
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "chemdep_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "test_label_use": "synthetic artifact generation, replay QA, final hit count only",
    }, indent=2) + "\n")
    print("seed", seed, cells[["fusion_hits", "rf_hits", "chemdep_hits", "scorecap_hits"]]
          .mean().to_dict(), flush=True)
    return len(cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        counts = list(pool.map(run_seed, [str(root)] * 5, range(35801, 35806)))
    if counts != [1200] * 5:
        raise AssertionError("Incomplete seeds")
    print("complete", counts, flush=True)


if __name__ == "__main__":
    main()
