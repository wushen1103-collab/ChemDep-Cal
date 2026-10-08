#!/usr/bin/env python3
"""Out-of-target calibration gate between meta ranking and min-p cap one."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import probe_meta_band_20261003 as base
from probe_meta_band_v2_20261003 import evidence_features
from train_artifact_meta_ranker_20261003 import cap_one, fit_ranker, load_panel


base.features = evidence_features


def gate_panel(panel: dict, model, reps: int) -> dict[int, dict]:
    from chemdeprc.selection import select
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from run_chembl_selection import conformal_right_tail_pvalues

    cal = panel["cal"]
    base_scores = cal["score"].to_numpy(float)
    labels = cal["label"].to_numpy(int)
    blocks = panel["cal_blocks"]
    null = panel["null"]
    target_hash = int(hashlib.sha256(panel["target"].encode()).hexdigest()[:12], 16) % 1_000_000
    differences = {budget: [] for budget in base.BUDGETS}
    for rep in range(reps):
        for fraction in base.FRACTIONS:
            rng = np.random.default_rng(882026 + target_hash + rep * 1009 + int(fraction * 10000))
            artifacts = choose_artifact_blocks(
                cal, artifact_block_col="murcko_scaffold", rng=rng,
                artifact_fraction=fraction, max_artifact_blocks=25,
                min_inactive_per_block=2,
            )
            for boost in base.BOOSTS:
                scores, _, _ = inject_artifact(
                    cal, base_scores, artifacts, artifact_block_col="murcko_scaffold",
                    logit_boost=boost,
                )
                pvalues = conformal_right_tail_pvalues(scores, null)
                prob = model.predict_proba(evidence_features(scores, blocks, null))[:, 1]
                for budget in base.BUDGETS:
                    meta = cap_one(prob, blocks, budget)
                    minp = select(
                        "minp_block_bh_cap1", scores=scores, pvalues=pvalues,
                        blocks=blocks, q=.7, budget=budget,
                    ).selected
                    differences[budget].append(int(labels[meta].sum()) - int(labels[minp].sum()))
    return {
        budget: {
            "calibration_delta_hits": float(np.mean(values)),
            "choice": "meta_calibration" if np.mean(values) > 0 else "minp_block_bh_cap1",
            "n_calibration_stresses": len(values),
        }
        for budget, values in differences.items()
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--train-repeats", type=int, default=3)
    p.add_argument("--gate-repeats", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    paths = sorted((root / args.scores_dir).glob("*_scores.csv"))
    panels = [load_panel(path) for path in paths]
    matrices = [base.training_matrix([panel], args.train_repeats) for panel in panels]
    all_x = np.concatenate([part[0] for part in matrices])
    all_y = np.concatenate([part[1] for part in matrices])
    global_model = fit_ranker(all_x, all_y, 862026)
    gate_rows = []
    for idx, panel in enumerate(panels):
        x = np.concatenate([part[0] for j, part in enumerate(matrices) if j != idx])
        y = np.concatenate([part[1] for j, part in enumerate(matrices) if j != idx])
        loo = fit_ranker(x, y, 862026)
        decisions = gate_panel(panel, loo, args.gate_repeats)
        for budget, decision in decisions.items():
            gate_rows.append({"target": panel["target"], "budget": budget, **decision})
        print(f"Gate {panel['target']}: {decisions}", flush=True)
    gate = pd.DataFrame(gate_rows)
    raw = base.evaluate(panels, global_model, args.seed, args.reps)
    gated = raw.merge(gate[["target", "budget", "choice"]], on=["target", "budget"], validate="many_to_one")
    gated = gated[gated["method"].eq(gated["choice"])].drop(columns="choice")
    if len(gated) * 5 != len(raw):
        raise AssertionError("Gate omitted or duplicated test cells")
    gated["method"] = "meta_gate"
    raw = pd.concat([raw, gated], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out, index=False)
    gate.to_csv(args.out.with_name(args.out.stem + "_gate.csv"), index=False)
    target_means = raw.groupby(["target", "budget", "method"], as_index=False).agg(
        hits=("hits", "mean"), fdp=("fdp", "mean"),
        selected=("selected", "mean"), blocks=("blocks", "mean"),
    )
    summary = target_means.groupby(["budget", "method"], as_index=False).agg(
        hits=("hits", "mean"), hits_target_sd=("hits", "std"),
        fdp=("fdp", "mean"), selected=("selected", "mean"), blocks=("blocks", "mean"),
    )
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    args.out.with_name(args.out.stem + "_manifest.json").write_text(json.dumps({
        "scores_dir": args.scores_dir, "seed": args.seed, "reps": args.reps,
        "train_repeats": args.train_repeats, "gate_repeats": args.gate_repeats,
        "gate": "leave-one-target-out calibration model; calibration labels only for gate",
        "global_model": "fit on all calibration panels, never test labels",
        "status": "exploratory new method, not original ChemDep-RC",
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
