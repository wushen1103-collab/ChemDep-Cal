#!/usr/bin/env python3
"""Same-score cap-1 selector benchmark on fixed WelQrate/Litmus panels."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


AIDS = ("AID1798", "AID435034", "AID463087")
METHODS = ("chemdep_cal", "score_only_ranker", "xgb_pairwise_cap1", "score_cap1")


def evaluate_seed(root_s: str, seed: int) -> tuple[int, int]:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    import probe_meta_band_v2_20261003  # Registers the frozen 16-feature map.
    from modern_selector_benchmark_20261004 import fit_pairwise
    from probe_meta_band_20261003 import training_matrix
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from run_chembl_selection import block_array
    from train_artifact_meta_ranker_20261003 import cap_one, fit_ranker

    base = root / "data/pr_welqrate_litmus_20261004"
    prep, scored = base / "prep", base / "scores"
    output = base / "selectors"
    output.mkdir(parents=True, exist_ok=True)
    panels, provenance = [], {}
    for aid in AIDS:
        meta_path = prep / f"{aid}_molecules.parquet"
        index_path = prep / f"{aid}_seed{seed}_indices.npz"
        score_path = scored / f"{aid}_seed{seed}_scores.npz"
        frame = pd.read_parquet(meta_path)
        with np.load(index_path) as indices:
            cal_idx = indices["calibration"].copy()
            test_idx = indices["test"].copy()
        with np.load(score_path) as scores:
            cal_scores = scores["calibration"].copy()
            test_scores = scores["test"].copy()
        cal = frame.iloc[cal_idx].reset_index(drop=True).copy()
        test = frame.iloc[test_idx].reset_index(drop=True).copy()
        cal["molecule_chembl_id"] = [f"{aid}:{row}" for row in cal_idx]
        test["molecule_chembl_id"] = [f"{aid}:{row}" for row in test_idx]
        cal["score"], test["score"] = cal_scores, test_scores
        if len(cal) != len(cal_scores) or len(test) != len(test_scores):
            raise AssertionError("Score-panel length mismatch")
        cal["target_chembl_id"], test["target_chembl_id"] = aid, aid
        null = cal.loc[cal.label.eq(0), "score"].to_numpy(float)
        if len(null) < 10 or int(cal.label.sum()) < 5 or int(test.label.sum()) < 5:
            raise AssertionError("Frozen calibration/test eligibility failed")
        panels.append({"target": aid, "cal": cal, "test": test, "null": null,
                       "cal_blocks": block_array(cal, "murcko_scaffold"),
                       "test_blocks": block_array(test, "murcko_scaffold")})
        provenance[aid] = {"metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                           "indices_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                           "score_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest()}
    train_x, train_y = training_matrix(panels, repeats=3)
    if train_x.shape[1] != 16 or train_y.sum() < 2:
        raise AssertionError("Unexpected ranker training geometry")
    full_ranker = fit_ranker(train_x, train_y, 862026)
    score_only_ranker = fit_ranker(train_x[:, :4], train_y, 862026)
    pairwise_ranker = fit_pairwise(panels)
    full_ranker.save_model(output / f"seed{seed}.chemdep.json")
    score_only_ranker.save_model(output / f"seed{seed}.score_only.json")
    pairwise_ranker.save_model(output / f"seed{seed}.pairwise.json")

    rows = []
    for index, panel in enumerate(panels):
        test = panel["test"]
        aid = panel["target"]
        labels = test.label.to_numpy(int)
        blocks = panel["test_blocks"]
        base_scores = test.score.to_numpy(float)

        def record(scenario: str, rep: int, fraction: float, boost: float,
                   scores: np.ndarray) -> None:
            features = evidence_features(scores, blocks, panel["null"])
            selections = {
                "chemdep_cal": cap_one(full_ranker.predict_proba(features)[:, 1], blocks, 50),
                "score_only_ranker": cap_one(score_only_ranker.predict_proba(features[:, :4])[:, 1], blocks, 50),
                "xgb_pairwise_cap1": cap_one(pairwise_ranker.predict(features), blocks, 50),
                "score_cap1": cap_one(scores, blocks, 50),
            }
            for method, chosen in selections.items():
                if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                    raise AssertionError(f"{aid}/seed{seed}: budget or scaffold capacity mismatch")
                rows.append({"aid": aid, "split_seed": seed, "scenario": scenario,
                             "rep": rep, "fraction": fraction, "boost": boost,
                             "method": method, "hits": int(labels[chosen].sum()),
                             "selected": 50, "scaffolds": 50,
                             "test_active": int(labels.sum()), "testpool_size": len(test)})

        record("natural", -1, 0., 0., base_scores)
        target_hash = int(hashlib.sha256(aid.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = 355700 + index * 100_000 + target_hash
        for rep in range(10):
            for fraction in (.2, .3):
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
                artifacts = choose_artifact_blocks(
                    test, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2)
                for boost in (4., 6.):
                    changed = inject_artifact(
                        test, base_scores, artifacts,
                        artifact_block_col="murcko_scaffold", logit_boost=boost)[0]
                    record("synthetic_high_artifact", rep, fraction, boost, changed)
        print(f"seed{seed} {aid} completed", flush=True)
    result = pd.DataFrame(rows)
    if len(result) != len(AIDS) * 41 * len(METHODS) or result.duplicated(
            ["aid", "split_seed", "scenario", "rep", "fraction", "boost", "method"]).any():
        raise AssertionError("Incomplete selector grid")
    result.to_csv(output / f"seed{seed}_cells.csv", index=False)
    (output / f"seed{seed}_manifest.json").write_text(json.dumps({
        "status": "post hoc third-party Litmus-derived WelQrate split; all outcomes retained",
        "not_official_welqrate_split": True,
        "seed": seed, "aids": AIDS, "methods": METHODS,
        "ranker_training": "three AID calibration panels pooled, 13 views each, 16 frozen features",
        "molecule_chembl_id": "internal AID:source-row ID for compatibility; not a ChEMBL identifier",
        "test_label_use": "outcome and explicitly named synthetic_high_artifact simulator only",
        "provenance": provenance,
    }, indent=2) + "\n", encoding="utf-8")
    return seed, len(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        rows = list(pool.map(evaluate_seed, [str(root)] * 5, range(1, 6)))
    print(rows, flush=True)


if __name__ == "__main__":
    main()
