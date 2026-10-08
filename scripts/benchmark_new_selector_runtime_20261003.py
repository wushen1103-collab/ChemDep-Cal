#!/usr/bin/env python3
"""Benchmark frozen fifth-cohort selectors, excluding CSV I/O and model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


SEEDS = range(35501, 35506)
REPETITIONS = 9


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from train_artifact_meta_ranker_20261003 import cap_one, load_panel
    from probe_meta_band_v2_20261003 import evidence_features
    from chemdeprc.selection import select
    from run_chembl_selection import conformal_right_tail_pvalues

    rows = []
    raw = []
    files = {}
    methods = ("meta_calibration", "score_cap1", "chemdeprc_cap1")
    for seed in SEEDS:
        model_path = root / "data/conditional_sota_meta_ungated_fifth_20261003" / f"meta_seed{seed}.model.json"
        model = XGBClassifier(n_jobs=4)
        model.load_model(model_path)
        files[str(model_path.relative_to(root))] = digest(model_path)
        scores_dir = root / "data/chembl_meta_ungated_fifth_xgb_20261003" / f"seed{seed}/scores"
        for panel_path in sorted(scores_dir.glob("*_scores.csv")):
            panel = load_panel(panel_path)
            files[str(panel_path.relative_to(root))] = digest(panel_path)
            scores = panel["test"]["score"].to_numpy(float)
            blocks = panel["test_blocks"]
            null = panel["null"]
            dummy_p = np.ones(len(scores), dtype=float)

            def meta():
                prob = model.predict_proba(evidence_features(scores, blocks, null))[:, 1]
                return cap_one(prob, blocks, 50)

            def scorecap():
                return select("score_cap1", scores=scores, pvalues=dummy_p,
                              blocks=blocks, q=.7, budget=50).selected

            def chemdep():
                p = conformal_right_tail_pvalues(scores, null)
                return select("chemdeprc_cap1", scores=scores, pvalues=p,
                              blocks=blocks, q=.7, budget=50).selected

            funcs = {"meta_calibration": meta, "score_cap1": scorecap,
                     "chemdeprc_cap1": chemdep}
            for fn in funcs.values():
                if len(fn()) > 50:
                    raise AssertionError("Selector exceeded budget")
            times = {m: [] for m in methods}
            for rep in range(REPETITIONS):
                for method in methods[rep % len(methods):] + methods[:rep % len(methods)]:
                    start = perf_counter_ns()
                    funcs[method]()
                    ms = (perf_counter_ns() - start) / 1e6
                    times[method].append(ms)
                    raw.append({"scorer_seed": seed, "target": panel["target"],
                                "testpool_size": len(scores), "rep": rep,
                                "method": method, "latency_ms": ms})
            for method, values in times.items():
                rows.append({"scorer_seed": seed, "target": panel["target"],
                             "testpool_size": len(scores), "method": method,
                             "median_ms": float(np.median(values)),
                             "q25_ms": float(np.quantile(values, .25)),
                             "q75_ms": float(np.quantile(values, .75)),
                             "repetitions": REPETITIONS})
        print(f"seed {seed}: ten panels benchmarked", flush=True)

    result = pd.DataFrame(rows)
    if len(result) != 150 or result.groupby("method").size().ne(50).any():
        raise AssertionError("Expected three methods x five seeds x ten targets")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    pd.DataFrame(raw).to_csv(args.out.with_name(args.out.stem + "_raw.csv"), index=False)
    manifest = {
        "source": "frozen fifth-cohort, unperturbed test scores",
        "included": "selection with feature extraction, conformal p-values where required, prediction and cap allocation",
        "excluded": "CSV read, scorer training/inference, ranker training, model loading, artifact injection",
        "repetitions": REPETITIONS,
        "warmup": "one call per method per target panel",
        "timing": "perf_counter_ns; method order rotated by repetition",
        "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "input_sha256": files,
        "output_rows": len(result),
    }
    args.out.with_name(args.out.stem + "_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(result.groupby("method").median_ms.agg(["median", "min", "max"]), flush=True)


if __name__ == "__main__":
    main()
