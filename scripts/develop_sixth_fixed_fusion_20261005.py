#!/usr/bin/env python3
"""Fixed 75:25 RF/ChemDep transfer diagnostic on exposed sixth targets."""

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
    cohort = root / "data/chembl_modern_sixth_20261004"
    manifest = pd.read_csv(cohort / "screening_manifest.csv",
                           dtype={"passes_smoke_threshold": "string"})
    eligible = manifest[manifest.passes_smoke_threshold.str.lower().eq("true")]
    score_dir = cohort / f"eligible_seed{seed}" / "scores"
    paths = [score_dir / f"{target}_scores.csv" for target in eligible.target_chembl_id]
    if len(paths) != 8 or not all(path.is_file() for path in paths):
        raise AssertionError("Sixth target inventory changed")
    panels = [load_panel(path) for path in paths]
    ref = pd.read_csv(cohort / "selector_results" / f"seed{seed}.csv")
    ref = ref[ref.method.isin(("chemdep_cal", "score_cap1"))]
    expected = {(r.target, int(r.rep), float(r.fraction), float(r.boost), r.method): int(r.hits)
                for r in ref.itertuples(index=False)}
    if len(expected) != 640:
        raise AssertionError("Original sixth selector cells incomplete")
    model_path = cohort / "selector_results" / f"seed{seed}.chemdep.json"
    chemdep = XGBClassifier()
    chemdep.load_model(model_path)
    output = root / "data/pr_sixth_fixed_fusion_development_20261005"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, panel in enumerate(panels):
        target, cal, test = panel["target"], panel["cal"], panel["test"]
        fp_cal, fp_test = fingerprints(cal.canonical_smiles), fingerprints(test.canonical_smiles)
        rf = fit_rf(np.column_stack([fp_cal, cal.score.to_numpy(float)]),
                    cal.label.to_numpy(int), seed)
        base, labels = test.score.to_numpy(float), test.label.to_numpy(int)
        blocks, null = panel["test_blocks"], panel["null"]
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = 356700 + index * 100_000 + target_hash
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
                    if int(labels[cap_one(cp, blocks, 50)].sum()) != expected[(*key, "chemdep_cal")]:
                        raise AssertionError(f"ChemDep replay mismatch: {key}")
                    rp = rf.predict_proba(np.column_stack([fp_test, scores]))[:, 1]
                    cr = rankdata(cp, method="average") / len(cp)
                    rr = rankdata(rp, method="average") / len(rp)
                    chosen_rf = cap_one(rp, blocks, 50)
                    chosen_fusion = cap_one(.75 * rr + .25 * cr, blocks, 50)
                    if len(chosen_rf) != 50 or len(chosen_fusion) != 50:
                        raise AssertionError("Budget violation")
                    if len(set(blocks[chosen_rf])) != 50 or len(set(blocks[chosen_fusion])) != 50:
                        raise AssertionError("Scaffold-cap violation")
                    rows.append({"target": target, "seed": seed, "rep": rep,
                                 "fraction": fraction, "boost": boost,
                                 "chemdep_hits": expected[(*key, "chemdep_cal")],
                                 "scorecap_hits": expected[(*key, "score_cap1")],
                                 "rf_hits": int(labels[chosen_rf].sum()),
                                 "fusion_hits": int(labels[chosen_fusion].sum())})
        print("seed", seed, "target", target, "completed", flush=True)
    cells = pd.DataFrame(rows)
    if len(cells) != 320:
        raise AssertionError("Incomplete sixth fusion grid")
    cells.to_csv(output / f"seed{seed}_cells.csv", index=False)
    (output / f"seed{seed}_manifest.json").write_text(json.dumps({
        "status": "exposed-target transfer development, not independent SOTA confirmation",
        "seed": seed, "rf_weight": .75, "chemdep_weight": .25,
        "chemdep_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "original_chemdep_cells_replayed": 320, "original_scorecap_cells_replayed": 320,
    }, indent=2) + "\n")
    return len(cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        print(list(pool.map(run_seed, [str(args.root.resolve())] * 5,
                            range(35601, 35606))), flush=True)


if __name__ == "__main__":
    main()
