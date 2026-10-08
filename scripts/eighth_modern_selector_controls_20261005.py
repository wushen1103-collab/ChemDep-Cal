#!/usr/bin/env python3
"""Fixed pairwise and MMR-module controls on the eighth matched selector cells."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from modern_selector_benchmark_20261004 import fit_pairwise, mmr_cap_one, similarity_matrix


def run_seed(root_s: str, seed: int) -> int:
    root = Path(root_s)
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    import probe_meta_band_v2_20261003  # noqa: F401; activate fixed 16-feature map
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import cap_one, load_panel

    RDLogger.DisableLog("rdApp.*")
    cohort = root / "data/chembl_modern_eighth_20261005"
    ledger = pd.read_csv(cohort / "technical_eligibility.csv")
    if len(ledger) != 100:
        raise AssertionError("Eighth inventory changed")
    eligible = ledger.loc[ledger.eligible, "target_chembl_id"].tolist()
    score_dir = cohort / f"eligible_seed{seed}" / "scores"
    paths = [score_dir / f"{target}_scores.csv" for target in eligible]
    if len(paths) != 30 or not all(path.is_file() for path in paths):
        raise AssertionError("Eighth eligible panels changed")
    panels = [load_panel(path) for path in paths]
    pairwise = fit_pairwise(panels)
    output = root / "data/pr_eighth_modern_selector_controls_20261005"
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / f"seed{seed}.pairwise.json"
    pairwise.save_model(model_path)
    ref = pd.read_csv(cohort / "fixed_selector_results" / f"seed{seed}.csv")
    ref = ref[ref.method.isin(("score_cap1", "chemdep_cal"))]
    expected = {(r.target, int(r.rep), float(r.fraction), float(r.boost), r.method):
                int(r.hits) for r in ref.itertuples(index=False)}
    if len(expected) != 30 * 40 * 2:
        raise AssertionError("Original controls incomplete")
    rows = []
    for index, panel in enumerate(panels):
        target, test = panel["target"], panel["test"]
        labels = test.label.to_numpy(int)
        base = test.score.to_numpy(float)
        blocks = panel["test_blocks"]
        similarity = similarity_matrix(test.canonical_smiles.astype(str).tolist())
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = 358700 + index * 100_000 + target_hash
        for rep in range(10):
            for fraction in (.2, .3):
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
                artifacts = choose_artifact_blocks(
                    test, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2,
                )
                for boost in (4., 6.):
                    scores = inject_artifact(
                        test, base, artifacts,
                        artifact_block_col="murcko_scaffold", logit_boost=boost,
                    )[0]
                    key = (target, rep, fraction, boost)
                    selected_scorecap = cap_one(scores, blocks, 50)
                    if int(labels[selected_scorecap].sum()) != expected[(*key, "score_cap1")]:
                        raise AssertionError(f"Score-cap replay mismatch: {key}")
                    predictions = pairwise.predict(evidence_features(scores, blocks, panel["null"]))
                    selections = {
                        "xgb_pairwise_cap1": cap_one(predictions, blocks, 50),
                        "mmr_module_l095_cap1": mmr_cap_one(scores, blocks, similarity, 50, .95),
                    }
                    for method, chosen in selections.items():
                        if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                            raise AssertionError(f"Budget/cap violation: {key}/{method}")
                        rows.append({
                            "target": target, "scorer_seed": seed, "rep": rep,
                            "fraction": fraction, "boost": boost, "method": method,
                            "hits": int(labels[chosen].sum()),
                            "selected": 50, "scaffolds": 50,
                        })
        print("seed", seed, "target", target, "completed", flush=True)
    cells = pd.DataFrame(rows)
    if len(cells) != 30 * 40 * 2:
        raise AssertionError("Incomplete eighth modern-control grid")
    cells.to_csv(output / f"seed{seed}_cells.csv", index=False)
    (output / f"seed{seed}_manifest.json").write_text(json.dumps({
        "status": "extra controls added after eighth fusion result was seen",
        "scorer_seed": seed, "eligible_targets": eligible,
        "pairwise": "original fifth settings: 16 features, 13 calibration views, rank:pairwise",
        "mmr": "ScaffAug-style reranking module only; fixed lambda=0.95, ECFP4 Tanimoto",
        "test_label_use": "synthetic artifact generation, score-cap replay QA, hit count only",
        "score_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in paths},
        "pairwise_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    print("seed", seed, cells.groupby("method").hits.mean().to_dict(), flush=True)
    return len(cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with futures.ProcessPoolExecutor(max_workers=5) as pool:
        counts = list(pool.map(run_seed, [str(root)] * 5, range(35801, 35806)))
    if counts != [2400] * 5:
        raise AssertionError("Incomplete seed results")
    print("complete", counts, flush=True)


if __name__ == "__main__":
    main()
