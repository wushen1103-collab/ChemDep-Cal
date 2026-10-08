#!/usr/bin/env python3
"""Budget-matched min-p plus score-only scaffold fill control."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run_panel(path: Path, index: int, root: Path, seed: int, reps: int, expected: dict) -> list[dict]:
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from chemdeprc.selection import select
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from run_chembl_selection import block_array, conformal_right_tail_pvalues

    frame = pd.read_csv(path)
    target = str(frame["target_chembl_id"].iloc[0])
    cal = frame[frame["split"] == "calibration"]
    test = frame[frame["split"] == "testpool"].reset_index(drop=True)
    null = cal.loc[cal["label"] == 0, "score"].to_numpy(float)
    base = test["score"].to_numpy(float)
    labels = test["label"].to_numpy(int)
    blocks = block_array(test, "murcko_scaffold")
    target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
    target_seed = seed + index * 100_000 + target_hash
    rows = []
    for rep in range(reps):
        for fraction in (.2, .3):
            rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
            artifacts = choose_artifact_blocks(
                test, artifact_block_col="murcko_scaffold", rng=rng,
                artifact_fraction=fraction, max_artifact_blocks=25, min_inactive_per_block=2,
            )
            for boost in (4., 6.):
                scores, _, _ = inject_artifact(
                    test, base, artifacts, artifact_block_col="murcko_scaffold", logit_boost=boost,
                )
                pvalues = conformal_right_tail_pvalues(scores, null)
                minp = select(
                    "minp_block_bh_cap1", scores=scores, pvalues=pvalues,
                    blocks=blocks, q=.7, budget=50,
                ).selected
                scorecap = select(
                    "score_cap1", scores=scores, pvalues=pvalues,
                    blocks=blocks, q=.7, budget=50,
                ).selected
                key = (target, rep, fraction, boost)
                for method, selected in (("minp_block_bh_cap1", minp), ("score_cap1", scorecap)):
                    if expected.get((*key, method)) != int(labels[selected].sum()):
                        raise AssertionError(f"Base grid mismatch: {key} {method}")
                order = np.argsort(-scores, kind="mergesort")
                selected = list(map(int, minp))
                seen = set(blocks[minp])
                if len(selected) < 50:
                    for candidate in order:
                        block = blocks[candidate]
                        if block not in seen:
                            selected.append(int(candidate))
                            seen.add(block)
                            if len(selected) == 50:
                                break
                if len(selected) != 50 or len(set(blocks[selected])) != 50:
                    raise AssertionError("Budget-filling or cap-one failure")
                hits = int(labels[selected].sum())
                rows.append({
                    "target": target, "rep": rep, "fraction": fraction,
                    "boost": boost, "budget": 50, "method": "minp_score_fill",
                    "selected": 50, "hits": hits, "fdp": (50 - hits) / 50,
                    "blocks": 50, "minp_selected_before_fill": len(minp),
                })
    print(f"Audited {target}", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=353700)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--candidate-method", default="meta_gate")
    args = p.parse_args()
    root = args.root.resolve()
    paths = sorted((root / args.scores_dir).glob("*_scores.csv"))
    gate = pd.read_csv(args.gate)
    gate = gate[gate["budget"].eq(50)]
    reference = gate[gate["method"].isin(["minp_block_bh_cap1", "score_cap1"])]
    expected = {
        (str(r.target), int(r.rep), float(r.fraction), float(r.boost), str(r.method)): int(r.hits)
        for r in reference.itertuples(index=False)
    }
    if len(expected) != len(paths) * args.reps * 4 * 2:
        raise AssertionError("Missing reference cells")
    raw = pd.DataFrame([
        row for idx, path in enumerate(paths)
        for row in run_panel(path, idx, root, args.seed, args.reps, expected)
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out, index=False)
    left = gate[gate["method"].eq(args.candidate_method)].groupby("target")["hits"].mean().sort_index()
    right = raw.groupby("target")["hits"].mean().sort_index()
    if not left.index.equals(right.index):
        raise AssertionError("Target mismatch")
    diff = (left - right).to_numpy(float)
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(diff), size=(100000, len(diff)))
    ci = np.quantile(diff[draws].mean(axis=1), [.025, .975])
    p_exact = np.mean([
        abs(np.mean(np.asarray(signs) * diff)) >= abs(diff.mean()) - 1e-12
        for signs in itertools.product((-1., 1.), repeat=len(diff))
    ])
    result = {
        "n_targets": len(paths), "minp_score_fill_hits_mean": float(right.mean()),
        "minp_score_fill_hits_target_sd": float(right.std()),
        "mean_minp_selected_before_fill": float(raw["minp_selected_before_fill"].mean()),
        "candidate_method": args.candidate_method,
        "candidate_minus_filled_hits": float(diff.mean()),
        "ci_low": float(ci[0]), "ci_high": float(ci[1]),
        "positive_targets": int((diff > 0).sum()),
        "negative_targets": int((diff < 0).sum()),
        "exact_signflip_p_two_sided": float(p_exact),
        "caveat": "Budget fill voids the original min-p risk certificate; this is a fair-capacity selection control, not a controlled-FDR procedure.",
    }
    args.out.with_name(args.out.stem + "_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
