#!/usr/bin/env python3
"""Exploratory target-specific ChemDep-Cal sensitivity on frozen fifth panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]

    import probe_meta_band_v2_20261003  # Registers the frozen 16-feature map.
    from probe_meta_band_20261003 import training_matrix
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import cap_one, fit_ranker, load_panel

    score_dir = root / "data/chembl_meta_ungated_fifth_xgb_20261003" / f"seed{args.seed}" / "scores"
    paths = sorted(score_dir.glob("*_scores.csv"))
    if len(paths) != 10:
        raise AssertionError(f"Expected ten targets, got {len(paths)}")
    panels = [load_panel(path) for path in paths]
    frozen_dir = root / "data/conditional_sota_meta_ungated_fifth_20261003"
    reference = pd.read_csv(frozen_dir / f"meta_seed{args.seed}.csv")
    reference = reference[reference.budget.eq(50) & reference.method.eq("score_cap1")]
    expected = {(r.target, int(r.rep), float(r.fraction), float(r.boost)): int(r.hits)
                for r in reference.itertuples(index=False)}
    if len(expected) != 400:
        raise AssertionError("Incomplete score-cap replay reference")

    rows = []
    for index, panel in enumerate(panels):
        x, y = training_matrix([panel], repeats=3)
        if x.shape[1] != 16:
            raise AssertionError(x.shape)
        model = fit_ranker(x, y, 862026)
        test = panel["test"]
        target = panel["target"]
        base = test.score.to_numpy(float)
        labels = test.label.to_numpy(int)
        blocks = panel["test_blocks"]
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = 355700 + index * 100_000 + target_hash
        for rep in range(10):
            for fraction in (.2, .3):
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
                chosen = choose_artifact_blocks(
                    test, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2,
                )
                for boost in (4., 6.):
                    scores = inject_artifact(
                        test, base, chosen, artifact_block_col="murcko_scaffold",
                        logit_boost=boost,
                    )[0]
                    rank = model.predict_proba(evidence_features(scores, blocks, panel["null"]))[:, 1]
                    model_picks = cap_one(rank, blocks, 50)
                    score_picks = cap_one(scores, blocks, 50)
                    score_hits = int(labels[score_picks].sum())
                    key = (target, rep, fraction, boost)
                    if score_hits != expected[key]:
                        raise AssertionError(f"Score-cap replay mismatch: {key}")
                    rows.append(dict(target=target, scorer_seed=args.seed, rep=rep,
                                     fraction=fraction, boost=boost,
                                     target_specific_hits=int(labels[model_picks].sum()),
                                     score_cap1_hits=score_hits,
                                     selected=len(model_picks),
                                     scaffolds=len(np.unique(blocks[model_picks]))))
        print(f"Completed {target}", flush=True)

    output = pd.DataFrame(rows)
    if len(output) != 400 or not output.selected.eq(50).all() or not output.scaffolds.eq(50).all():
        raise AssertionError("Incomplete target-specific sensitivity")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    args.out.with_suffix(".manifest.json").write_text(json.dumps({
        "status": "post hoc on inspected fifth cohort",
        "scorer_seed": args.seed,
        "training": "one 16-feature ranker per target, labelled calibration rows only",
        "seed": 862026,
        "score_cap1_replay_rows_checked": len(output),
        "score_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in paths},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
