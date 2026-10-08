#!/usr/bin/env python3
"""Same-supervision score-only control and natural external transfer audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def cap_one(values: np.ndarray, blocks: np.ndarray, budget: int) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    selected = []
    seen = set()
    for i in order:
        if blocks[i] not in seen:
            selected.append(int(i))
            seen.add(blocks[i])
            if len(selected) == budget:
                break
    return np.asarray(selected, dtype=int)


def filled_minp(scores: np.ndarray, blocks: np.ndarray, selected: np.ndarray, budget: int) -> np.ndarray:
    chosen = list(map(int, selected))
    seen = set(blocks[selected])
    for i in np.argsort(-scores, kind="mergesort"):
        if len(chosen) >= budget:
            break
        if blocks[i] not in seen:
            chosen.append(int(i))
            seen.add(blocks[i])
    return np.asarray(chosen, dtype=int)


def one_panel(path: Path, panel: dict, full, score_only, index: int, mode: str,
              seed: int, reps: int, reference: dict) -> list[dict]:
    from chemdeprc.selection import select
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from run_chembl_selection import conformal_right_tail_pvalues

    target = panel["target"]
    frame = pd.read_csv(path, nrows=1)
    dataset = str(frame["dataset"].iloc[0]) if "dataset" in frame else "ChEMBL"
    scenario = str(frame["scenario"].iloc[0]) if "scenario" in frame else mode
    test = panel["test"]
    base = test["score"].to_numpy(float)
    labels = test["label"].to_numpy(int)
    blocks = panel["test_blocks"]
    null = panel["null"]
    target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
    target_seed = seed + index * 100_000 + target_hash
    rows = []
    settings = [(0, 0., 0.)] if mode == "natural" else [
        (rep, frac, boost)
        for rep in range(reps) for frac in (.2, .3) for boost in (4., 6.)
    ]
    for rep, fraction, boost in settings:
        if mode == "natural":
            scores = base.copy()
        else:
            rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
            artifacts = choose_artifact_blocks(
                test, artifact_block_col="murcko_scaffold", rng=rng,
                artifact_fraction=fraction, max_artifact_blocks=25,
                min_inactive_per_block=2,
            )
            scores, _, _ = inject_artifact(
                test, base, artifacts, artifact_block_col="murcko_scaffold", logit_boost=boost,
            )
        pv = conformal_right_tail_pvalues(scores, null)
        features = evidence_features(scores, blocks, null)
        full_prob = full.predict_proba(features)[:, 1]
        score_prob = score_only.predict_proba(features[:, :4])[:, 1]
        minp = select("minp_block_bh_cap1", scores=scores, pvalues=pv,
                      blocks=blocks, q=.7, budget=50).selected
        picks = {
            "meta_calibration": cap_one(full_prob, blocks, 50),
            "scoreonly_calibration": cap_one(score_prob, blocks, 50),
            "score_cap1": select("score_cap1", scores=scores, pvalues=pv,
                                  blocks=blocks, q=.7, budget=50).selected,
            "chemdeprc_cap1": select("chemdeprc_cap1", scores=scores, pvalues=pv,
                                      blocks=blocks, q=.7, budget=50).selected,
            "minp_score_fill": filled_minp(scores, blocks, minp, 50),
        }
        for method, chosen in picks.items():
            hits = int(labels[chosen].sum())
            key = (target, rep, fraction, boost, method)
            if key in reference and reference[key] != hits:
                raise AssertionError(f"Frozen fifth-cohort replay mismatch: {key}")
            rows.append({
                "target": target, "dataset": dataset, "scenario": scenario,
                "rep": rep, "fraction": fraction, "boost": boost,
                "method": method, "hits": hits,
                "fdp": (len(chosen) - hits) / len(chosen) if len(chosen) else 0.,
                "selected": len(chosen), "blocks": len(np.unique(blocks[chosen])),
                "calibration_active": int(panel["cal"]["label"].sum()),
                "calibration_inactive": int(len(panel["cal"]) - panel["cal"]["label"].sum()),
                "testpool_active": int(labels.sum()), "testpool_size": len(labels),
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--mode", choices=["natural", "stress"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--reference", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    paths = sorted((root / args.scores_dir).glob("*_scores.csv"))
    if args.mode == "natural":
        paths = [path for path in paths if "scaffold_ood" in path.name]
    if not paths:
        raise FileNotFoundError(args.scores_dir)
    import probe_meta_band_v2_20261003  # installs the frozen feature function
    from probe_meta_band_20261003 import training_matrix
    from train_artifact_meta_ranker_20261003 import fit_ranker, load_panel

    panels = [load_panel(path) for path in paths]
    x, y = training_matrix(panels, repeats=3)
    if np.unique(y).size != 2:
        raise ValueError("Pooled calibration labels contain only one class")
    full = fit_ranker(x, y, 862026)
    score_only = fit_ranker(x[:, :4], y, 862026)
    reference = {}
    if args.reference:
        frozen = pd.read_csv(args.reference)
        frozen = frozen[frozen["budget"].eq(50) & frozen["method"].isin(
            ["meta_calibration", "score_cap1", "chemdeprc_cap1"])]
        reference = {
            (r.target, int(r.rep), float(r.fraction), float(r.boost), r.method): int(r.hits)
            for r in frozen.itertuples(index=False)
        }
    records = []
    for idx, (path, panel) in enumerate(zip(paths, panels)):
        records.extend(one_panel(path, panel, full, score_only, idx, args.mode,
                                 args.seed, args.reps, reference))
        print(f"Evaluated {path.name}", flush=True)
    raw = pd.DataFrame(records)
    if reference and len(reference) != len(paths) * args.reps * 4 * 3:
        raise AssertionError("Incomplete frozen reference")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out, index=False)
    summary = raw.groupby(["dataset", "scenario", "method"], as_index=False).agg(
        hits=("hits", "mean"), fdp=("fdp", "mean"),
        selected=("selected", "mean"), blocks=("blocks", "mean"),
        n_panels=("target", "nunique"), calibration_active_min=("calibration_active", "min"),
    )
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    args.out.with_name(args.out.stem + "_manifest.json").write_text(json.dumps({
        "scores_dir": args.scores_dir, "mode": args.mode, "seed": args.seed,
        "reps": args.reps, "n_panels": len(paths),
        "score_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "training": "pooled calibration labels only, same augmentation/fit; first four score-only features are the control",
        "scope": "post-confirmation mechanism and external transfer audit",
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
