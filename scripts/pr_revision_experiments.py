#!/usr/bin/env python3
"""Post-confirmation budget, feature-family, and unseen-artifact diagnostics.

These analyses reuse fifth-cohort score files already inspected. They are
exploratory and must not be merged into the frozen three-test family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


FAMILIES = {
    "score_only": [0, 1, 2, 3],
    "plus_group_stats": list(range(9)),
    "plus_group_evidence": list(range(13)),
}
BUDGETS = (20, 40, 50, 75)
FROZEN_SEED = 355700


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_picks(scores, labels, blocks, ranking, budget):
    from train_artifact_meta_ranker_20261003 import cap_one

    chosen = cap_one(ranking, blocks, budget)
    return int(labels[chosen].sum()), len(chosen), len(np.unique(blocks[chosen]))


def make_unseen(test, base, target_seed, rep, mechanism):
    from run_chembl_block_artifact_stress import (
        choose_artifact_blocks, expit, inject_artifact, logit,
    )

    fraction = {"logit_b5": .2, "logit_b7": .3}.get(mechanism, .25)
    rng = np.random.default_rng(target_seed + 5_000_000 + rep * 10_003 +
                                int(fraction * 10_000))
    chosen = choose_artifact_blocks(
        test, artifact_block_col="murcko_scaffold", rng=rng,
        artifact_fraction=fraction, max_artifact_blocks=25,
        min_inactive_per_block=2,
    )
    if mechanism.startswith("logit_b"):
        return inject_artifact(
            test, base, chosen, artifact_block_col="murcko_scaffold",
            logit_boost=float(mechanism.removeprefix("logit_b")),
        )[0]
    block_mask = test["murcko_scaffold"].astype(str).isin(chosen).to_numpy()
    inactive = test["label"].to_numpy(int) == 0
    changed = base.copy()
    if mechanism == "additive_inactive":
        mask = block_mask & inactive
        changed[mask] = np.clip(base[mask] + .15, 1e-6, 1 - 1e-6)
    elif mechanism == "blockwide_logit":
        changed[block_mask] = expit(logit(base[block_mask]) + 4.)
    elif mechanism == "heterogeneous_inactive":
        mask = block_mask & inactive
        boosts = rng.uniform(2., 8., size=int(mask.sum()))
        changed[mask] = expit(logit(base[mask]) + boosts)
    else:
        raise ValueError(mechanism)
    return changed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reps", type=int, default=10)
    args = p.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]

    import probe_meta_band_v2_20261003  # installs frozen 16-feature function
    from probe_meta_band_20261003 import training_matrix
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import fit_ranker, load_panel

    score_dir = root / "data/chembl_meta_ungated_fifth_xgb_20261003" / f"seed{args.seed}" / "scores"
    paths = sorted(score_dir.glob("*_scores.csv"))
    if len(paths) != 10:
        raise AssertionError(f"Expected ten fifth-cohort targets; found {len(paths)}")
    panels = [load_panel(path) for path in paths]
    x, y = training_matrix(panels, repeats=3)
    if x.shape[1] != 16:
        raise AssertionError(x.shape)
    models = {}
    for name, columns in FAMILIES.items():
        models[name] = fit_ranker(x[:, columns], y, 862026)
    frozen_dir = root / "data/conditional_sota_meta_ungated_fifth_20261003"
    model_path = frozen_dir / f"meta_seed{args.seed}.model.json"
    models["full"] = XGBClassifier()
    models["full"].load_model(model_path)

    reference = pd.read_csv(frozen_dir / f"meta_seed{args.seed}.csv")
    reference = reference[reference["budget"].eq(50) &
                          reference["method"].isin(["meta_calibration", "score_cap1"])]
    expected = {(r.target, int(r.rep), float(r.fraction), float(r.boost), r.method):
                int(r.hits) for r in reference.itertuples(index=False)}
    if len(expected) != len(paths) * args.reps * 4 * 2:
        raise AssertionError("Frozen reference is incomplete")

    rows = []
    checked = 0
    unseen = ("logit_b5", "logit_b7", "additive_inactive",
              "blockwide_logit", "heterogeneous_inactive")
    for idx, panel in enumerate(panels):
        test = panel["test"]
        base = test["score"].to_numpy(float)
        labels = test["label"].to_numpy(int)
        blocks = panel["test_blocks"]
        target = panel["target"]
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = FROZEN_SEED + idx * 100_000 + target_hash
        for rep in range(args.reps):
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
                    feature_matrix = evidence_features(scores, blocks, panel["null"])
                    rankings = {name: model.predict_proba(feature_matrix[:, columns])[:, 1]
                                for name, (columns, model) in
                                ((name, (FAMILIES[name], models[name])) for name in FAMILIES)}
                    rankings["full"] = models["full"].predict_proba(feature_matrix)[:, 1]
                    rankings["score_cap1"] = scores
                    for budget in BUDGETS:
                        methods = ("full", "score_cap1") if budget != 50 else tuple(rankings)
                        for method in methods:
                            hits, selected, scaffolds = evaluate_picks(
                                scores, labels, blocks, rankings[method], budget)
                            if budget == 50 and method in ("full", "score_cap1"):
                                frozen_name = "meta_calibration" if method == "full" else method
                                key = (target, rep, fraction, boost, frozen_name)
                                if hits != expected[key]:
                                    raise AssertionError(f"Frozen replay mismatch: {key}, {hits} != {expected[key]}")
                                checked += 1
                            rows.append(dict(target=target, scorer_seed=args.seed,
                                             regime="original", rep=rep, fraction=fraction,
                                             boost=boost, mechanism="inactive_logit",
                                             budget=budget, method=method, hits=hits,
                                             selected=selected, scaffolds=scaffolds))
            for mechanism in unseen:
                scores = make_unseen(test, base, target_seed, rep, mechanism)
                feature_matrix = evidence_features(scores, blocks, panel["null"])
                rankings = {
                    "full": models["full"].predict_proba(feature_matrix)[:, 1],
                    "score_only": models["score_only"].predict_proba(feature_matrix[:, :4])[:, 1],
                    "score_cap1": scores,
                }
                for method, ranking in rankings.items():
                    hits, selected, scaffolds = evaluate_picks(
                        scores, labels, blocks, ranking, 50)
                    rows.append(dict(target=target, scorer_seed=args.seed,
                                     regime="unseen", rep=rep, fraction=np.nan,
                                     boost=np.nan, mechanism=mechanism, budget=50,
                                     method=method, hits=hits, selected=selected,
                                     scaffolds=scaffolds))
        print(f"Completed {target}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    args.out.with_suffix(".manifest.json").write_text(json.dumps({
        "status": "post-hoc diagnostic on previously inspected fifth cohort",
        "scorer_seed": args.seed,
        "frozen_replay_rows_checked": checked,
        "feature_families": FAMILIES,
        "budgets": BUDGETS,
        "unseen_mechanisms": unseen,
        "calibration_training": "unchanged logit-shift augmentation; no test labels in ranker",
        "test_simulation": "test labels choose inactive-enriched blocks; only inside evaluator",
        "source_sha256": {path.name: sha256(path) for path in paths},
        "frozen_model_sha256": sha256(model_path),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Rows: {len(rows)}, frozen replay checks: {checked}", flush=True)


if __name__ == "__main__":
    main()
