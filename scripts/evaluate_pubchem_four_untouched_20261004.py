#!/usr/bin/env python3
"""Matched selectors on all eligible fixed new PubChem follow-up campaigns."""

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


def cap_one_tiebreak(values: np.ndarray, blocks: np.ndarray, budget: int,
                     keys: np.ndarray) -> np.ndarray:
    order = np.lexsort((keys, -values))
    selected, seen = [], set()
    for index in order:
        block = blocks[index]
        if block not in seen:
            selected.append(int(index))
            seen.add(block)
            if len(selected) == budget:
                break
    return np.asarray(selected, dtype=int)


def evaluate_seed(root_s: str, seed: int) -> int:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    import probe_meta_band_v2_20261003  # noqa: F401; register original 16 features
    from modern_selector_benchmark_20261004 import fit_pairwise
    from probe_meta_band_20261003 import training_matrix
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_selection import block_array
    from train_artifact_meta_ranker_20261003 import fit_ranker

    base = root / "data/pubchem_four_untouched_20261004/cohorts"
    ledger = pd.read_csv(base / "cohort_audit.csv")
    if len(ledger) != 20 or not ledger.groupby("aid").size().eq(5).all():
        raise AssertionError("Incomplete frozen eligibility ledger")
    eligible = sorted(name for name, group in ledger.groupby("aid") if group.eligible.all())
    output = base / "selectors"
    output.mkdir(parents=True, exist_ok=True)
    panels, provenance = [], {}
    for aid in eligible:
        cohort_file = base / f"{aid}_cohort.parquet"
        split_file = base / f"{aid}_seed{seed}_indices.npz"
        cohort = pd.read_parquet(cohort_file)
        with np.load(split_file) as split:
            cal_idx, test_idx = split["calibration"], split["test"]
        cal = cohort.iloc[cal_idx].reset_index(drop=True).copy()
        test = cohort.iloc[test_idx].reset_index(drop=True).copy()
        center = float(np.median(cal.score))
        iqr = float(np.percentile(cal.score, 75) - np.percentile(cal.score, 25))
        if iqr <= 0:
            raise AssertionError("Eligibility score IQR changed")
        for frame in (cal, test):
            frame["score"] = expit((frame.score.to_numpy(float) - center) / iqr)
            frame["molecule_chembl_id"] = frame.CID.astype(str)
            frame["target_chembl_id"] = aid
        panels.append({"target": aid, "cal": cal, "test": test,
                       "null": cal.loc[cal.label.eq(0), "score"].to_numpy(float),
                       "cal_blocks": block_array(cal, "murcko_scaffold"),
                       "test_blocks": block_array(test, "murcko_scaffold")})
        provenance[aid] = {"cohort_sha256": hashlib.sha256(cohort_file.read_bytes()).hexdigest(),
                           "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
                           "calibration_median": center, "calibration_iqr": iqr,
                           "n_cal": len(cal), "n_test": len(test),
                           "test_actives": int(test.label.sum())}
    rows = []
    if eligible:
        x, y = training_matrix(panels, repeats=3)
        if x.shape[1] != 16 or y.sum() < 2:
            raise AssertionError("Unexpected frozen training matrix")
        full = fit_ranker(x, y, 862026)
        score_only = fit_ranker(x[:, :4], y, 862026)
        pairwise = fit_pairwise(panels)
        full.save_model(output / f"seed{seed}.chemdep.json")
        score_only.save_model(output / f"seed{seed}.score_only.json")
        pairwise.save_model(output / f"seed{seed}.pairwise.json")
        for panel in panels:
            aid = panel["target"]
            test = panel["test"]
            scores = test.score.to_numpy(float)
            labels = test.label.to_numpy(int)
            blocks = panel["test_blocks"]
            features = evidence_features(scores, blocks, panel["null"])
            keys = np.asarray([int.from_bytes(hashlib.sha256(
                f"{aid}/{seed}/{cid}".encode()).digest()[:8], "big")
                for cid in test.CID], dtype=np.uint64)
            value_map = {"chemdep_cal": full.predict_proba(features)[:, 1],
                         "score_only_ranker": score_only.predict_proba(features[:, :4])[:, 1],
                         "xgb_pairwise_cap1": pairwise.predict(features),
                         "score_cap1": scores}
            leaders = pd.DataFrame({"block": blocks, "score": scores}).groupby("block").score.max()
            boundary = float(np.sort(leaders.to_numpy())[-50])
            n_boundary = int(leaders.eq(boundary).sum())
            tie_rng = np.random.default_rng(20261004 + seed + int(aid[3:]))
            tie_hits = [int(labels[cap_one_tiebreak(scores, blocks, 50,
                                                    tie_rng.random(len(scores)))].sum())
                        for _ in range(1000)]
            for method, values in value_map.items():
                selected = cap_one_tiebreak(values, blocks, 50, keys)
                if len(selected) != 50 or len(set(blocks[selected])) != 50:
                    raise AssertionError("Capacity mismatch")
                rows.append({"aid": aid, "seed": seed, "method": method,
                             "hits": int(labels[selected].sum()), "selected": len(selected),
                             "scaffolds": len(set(blocks[selected])),
                             "test_active": int(labels.sum()), "testpool_size": len(test),
                             "score_boundary_tied_leaders": n_boundary,
                             "scorecap_random_tie_mean": float(np.mean(tie_hits)),
                             "scorecap_random_tie_sd": float(np.std(tie_hits, ddof=1))})
            print(seed, aid, "completed; boundary ties", n_boundary, flush=True)
    result = pd.DataFrame(rows)
    if len(result) != 4 * len(eligible) or (not result.empty and
                                           result.duplicated(["aid", "seed", "method"]).any()):
        raise AssertionError("Incomplete selector grid")
    result.to_csv(output / f"seed{seed}_cells.csv", index=False)
    (output / f"seed{seed}_manifest.json").write_text(json.dumps({
        "status": "new source matched selector audit; transfer gate separately requires >=3 eligible AIDs",
        "seed": seed, "eligible_aids": eligible, "methods": list(value_map) if eligible else [],
        "tie_break": "shared label-blind SHA256(AID/seed/CID) ascending; score-cap random tie sensitivity 1000",
        "test_label_use": "endpoint evaluation and random-tie audit only",
        "provenance": provenance,
    }, indent=2) + "\n", encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        print(list(pool.map(evaluate_seed, [str(args.root.resolve())] * 5, range(1, 6))), flush=True)


if __name__ == "__main__":
    main()
