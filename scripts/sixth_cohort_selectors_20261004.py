#!/usr/bin/env python3
"""Exploratory execution of the frozen selector rule on eight eligible targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("chemdep_cal", "xgb_pairwise_cap1", "score_cap1", "mmr_cap1_l095")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]

    import probe_meta_band_v2_20261003  # Registers the frozen 16-feature map.
    from modern_selector_benchmark_20261004 import fit_pairwise, mmr_cap_one, similarity_matrix
    from probe_meta_band_20261003 import training_matrix
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import cap_one, fit_ranker, load_panel

    cohort = root / "data/chembl_modern_sixth_20261004"
    manifest = pd.read_csv(cohort / "screening_manifest.csv", dtype={"passes_smoke_threshold": "string"})
    eligible = manifest[manifest.passes_smoke_threshold.str.lower().eq("true")]
    if len(eligible) != 8:
        raise AssertionError("Expected eight eligible targets after frozen-list shortfall")
    scores_dir = cohort / f"eligible_seed{args.seed}" / "scores"
    paths = [scores_dir / f"{target}_scores.csv" for target in eligible.target_chembl_id]
    if not all(path.is_file() for path in paths):
        raise AssertionError("Eligible score panel missing")
    if set(scores_dir.glob("*_scores.csv")) != set(paths):
        raise AssertionError("Unexpected scored target in eligible-only directory")
    panels = [load_panel(path) for path in paths]
    x, y = training_matrix(panels, repeats=3)
    pooled_model = fit_ranker(x, y, 862026)
    pairwise_model = fit_pairwise(panels)

    out = cohort / "selector_results"
    out.mkdir(parents=True, exist_ok=True)
    pooled_model.save_model(out / f"seed{args.seed}.chemdep.json")
    pairwise_model.save_model(out / f"seed{args.seed}.pairwise.json")
    rows = []
    for panel_index, panel in enumerate(panels):
        target = panel["target"]
        test = panel["test"]
        blocks = panel["test_blocks"]
        base = test.score.to_numpy(float)
        labels = test.label.to_numpy(int)
        similarities = similarity_matrix(test.canonical_smiles.astype(str).tolist())
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = 356700 + panel_index * 100_000 + target_hash
        for rep in range(10):
            for fraction in (.2, .3):
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
                artifact_blocks = choose_artifact_blocks(
                    test, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2,
                )
                for boost in (4., 6.):
                    scores = inject_artifact(
                        test, base, artifact_blocks, artifact_block_col="murcko_scaffold",
                        logit_boost=boost,
                    )[0]
                    features = evidence_features(scores, blocks, panel["null"])
                    selections = {
                        "chemdep_cal": cap_one(pooled_model.predict_proba(features)[:, 1], blocks, 50),
                        "xgb_pairwise_cap1": cap_one(pairwise_model.predict(features), blocks, 50),
                        "score_cap1": cap_one(scores, blocks, 50),
                        "mmr_cap1_l095": mmr_cap_one(scores, blocks, similarities, 50, .95),
                    }
                    for method, chosen in selections.items():
                        if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                            raise AssertionError(f"Invalid capacity: {target} {method}")
                        hits = int(labels[chosen].sum())
                        rows.append({"target": target, "scorer_seed": args.seed,
                                     "rep": rep, "fraction": fraction, "boost": boost,
                                     "method": method, "hits": hits, "selected": 50,
                                     "scaffolds": 50, "fdp": (50 - hits) / 50})
        print(f"completed {target}", flush=True)
    result = pd.DataFrame(rows)
    if len(result) != 8 * 10 * 4 * len(METHODS):
        raise AssertionError("Incomplete sixth-cohort evaluation grid")
    result.to_csv(out / f"seed{args.seed}.csv", index=False)
    (out / f"seed{args.seed}.manifest.json").write_text(json.dumps({
        "status": "exploratory: only eight of ten frozen-list targets qualified",
        "eligible_targets": eligible.target_chembl_id.tolist(),
        "methods": METHODS,
        "scorer_seed": args.seed,
        "score_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }, indent=2) + "\n", encoding="utf-8")
    print(result.groupby("method").hits.mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
