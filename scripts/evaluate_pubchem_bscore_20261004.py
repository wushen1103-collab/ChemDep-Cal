#!/usr/bin/env python3
"""Matched selector test on real PubChem primary-screen Bscore."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


AIDS = ("AID1798", "AID435034", "AID463087")
SIGNS = {"AID1798": 1, "AID435034": -1, "AID463087": -1}
METHODS = ("chemdep_cal", "score_only_ranker", "xgb_pairwise_cap1", "score_cap1")


def evaluate_seed(root_s: str, seed: int) -> tuple[int, int]:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    import probe_meta_band_v2_20261003  # Registers the frozen feature map.
    from modern_selector_benchmark_20261004 import fit_pairwise
    from probe_meta_band_20261003 import training_matrix
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_selection import block_array
    from train_artifact_meta_ranker_20261003 import cap_one, fit_ranker

    data = root / "data/pubchem_primary_welqrate_20261004/matched_bscore"
    prep = root / "data/pr_welqrate_litmus_20261004/prep"
    audit = pd.read_csv(data / "match_audit.csv")
    if len(audit) != 15 or not audit.eligible.all():
        raise AssertionError("Frozen Bscore eligibility failed")
    output = data / "selectors"
    output.mkdir(parents=True, exist_ok=True)
    panels, provenance = [], {}
    for aid in AIDS:
        match_file = data / f"{aid}_bscore_matched.parquet"
        split_file = prep / f"{aid}_seed{seed}_indices.npz"
        frame = pd.read_parquet(match_file)
        with np.load(split_file) as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        cal = frame.iloc[cal_idx].copy()
        test = frame.iloc[test_idx].copy()
        cal_idx = cal.index[cal.primary_bscore_raw.notna()].to_numpy(int)
        test_idx = test.index[test.primary_bscore_raw.notna()].to_numpy(int)
        cal = frame.iloc[cal_idx].reset_index(drop=True).copy()
        test = frame.iloc[test_idx].reset_index(drop=True).copy()
        sign = SIGNS[aid]
        oriented_cal = sign * cal.primary_bscore_raw.to_numpy(float)
        oriented_test = sign * test.primary_bscore_raw.to_numpy(float)
        center = float(np.median(oriented_cal))
        iqr = float(np.percentile(oriented_cal, 75) - np.percentile(oriented_cal, 25))
        if iqr <= 0:
            raise AssertionError(f"{aid}/seed{seed}: zero calibration IQR")
        cal["score"] = expit((oriented_cal - center) / iqr)
        test["score"] = expit((oriented_test - center) / iqr)
        cal["molecule_chembl_id"] = [f"{aid}:{row}" for row in cal_idx]
        test["molecule_chembl_id"] = [f"{aid}:{row}" for row in test_idx]
        cal["target_chembl_id"], test["target_chembl_id"] = aid, aid
        if min(int(cal.label.sum()), int(test.label.sum())) < 5:
            raise AssertionError(f"{aid}/seed{seed}: active-count eligibility failed")
        panels.append({"target": aid, "cal": cal, "test": test,
                       "null": cal.loc[cal.label.eq(0), "score"].to_numpy(float),
                       "cal_blocks": block_array(cal, "murcko_scaffold"),
                       "test_blocks": block_array(test, "murcko_scaffold")})
        provenance[aid] = {"match_sha256": hashlib.sha256(match_file.read_bytes()).hexdigest(),
                           "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
                           "sign": sign, "calibration_median": center,
                           "calibration_iqr": iqr, "n_cal": len(cal), "n_test": len(test)}
    x, y = training_matrix(panels, repeats=3)
    if x.shape[1] != 16:
        raise AssertionError("Unexpected feature geometry")
    full = fit_ranker(x, y, 862026)
    score_only = fit_ranker(x[:, :4], y, 862026)
    pairwise = fit_pairwise(panels)
    full.save_model(output / f"seed{seed}.chemdep.json")
    score_only.save_model(output / f"seed{seed}.score_only.json")
    pairwise.save_model(output / f"seed{seed}.pairwise.json")

    rows = []
    for panel in panels:
        aid = panel["target"]
        test = panel["test"]
        scores = test.score.to_numpy(float)
        labels = test.label.to_numpy(int)
        blocks = panel["test_blocks"]
        features = evidence_features(scores, blocks, panel["null"])
        selections = {
            "chemdep_cal": cap_one(full.predict_proba(features)[:, 1], blocks, 50),
            "score_only_ranker": cap_one(score_only.predict_proba(features[:, :4])[:, 1], blocks, 50),
            "xgb_pairwise_cap1": cap_one(pairwise.predict(features), blocks, 50),
            "score_cap1": cap_one(scores, blocks, 50),
        }
        leaders = pd.DataFrame({"block": blocks, "score": scores}).groupby("block").score.max()
        boundary = float(np.sort(leaders.to_numpy())[-50])
        tied_leaders = int(leaders.eq(boundary).sum())
        tie_hits = []
        if tied_leaders > 5:
            rng = np.random.default_rng(20261004 + seed + int(aid[3:]))
            for _ in range(1000):
                order = np.lexsort((rng.random(len(scores)), -scores))
                chosen_tie, seen_tie = [], set()
                for row in order:
                    if blocks[row] not in seen_tie:
                        chosen_tie.append(row)
                        seen_tie.add(blocks[row])
                        if len(chosen_tie) == 50:
                            break
                tie_hits.append(int(labels[chosen_tie].sum()))
        for method, chosen in selections.items():
            if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                raise AssertionError(f"{aid}/seed{seed}: selection capacity error")
            rows.append({"aid": aid, "split_seed": seed, "method": method,
                         "hits": int(labels[chosen].sum()),
                         "selected": len(chosen), "scaffolds": len(set(blocks[chosen])),
                         "test_active": int(labels.sum()), "testpool_size": len(test),
                         "score_boundary_tied_leaders": tied_leaders,
                         "scorecap_tie_random_mean": float(np.mean(tie_hits)) if tie_hits else np.nan,
                         "scorecap_tie_random_sd": float(np.std(tie_hits, ddof=1)) if tie_hits else np.nan})
        print(seed, aid, "completed; boundary ties", tied_leaders, flush=True)
    result = pd.DataFrame(rows)
    if len(result) != len(AIDS) * len(METHODS) or result.duplicated(
            ["aid", "split_seed", "method"]).any():
        raise AssertionError("Incomplete selector grid")
    result.to_csv(output / f"seed{seed}_cells.csv", index=False)
    (output / f"seed{seed}_manifest.json").write_text(json.dumps({
        "status": "post hoc real PubChem primary-screen Bscore test on third-party Litmus splits",
        "not_official_welqrate_split": True, "seed": seed,
        "methods": METHODS, "test_label_use": "endpoint evaluation only",
        "primary_test": "unperturbed measured Bscore, no synthetic injection",
        "score_transform": "expit((signed_Bscore - calibration_median) / calibration_IQR)",
        "provenance": provenance,
    }, indent=2) + "\n", encoding="utf-8")
    return seed, len(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        print(list(pool.map(evaluate_seed, [str(args.root.resolve())] * 5, range(1, 6))), flush=True)


if __name__ == "__main__":
    main()
