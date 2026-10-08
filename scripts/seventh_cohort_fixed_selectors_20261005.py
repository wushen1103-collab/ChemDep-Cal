#!/usr/bin/env python3
"""Fixed seventh-cohort 75:25 fusion and matched controls on five scorer seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from scipy.stats import rankdata

from develop_mfpcba_multifp_rf_20261004 import fingerprints, fit_rf


METHODS = ("score_cap1", "chemdep_cal", "multi_fp_rf", "rf75_chemdep25")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    import probe_meta_band_v2_20261003  # noqa: F401; activate the frozen 16-feature map
    from probe_meta_band_20261003 import training_matrix
    from probe_meta_band_v2_20261003 import evidence_features
    from run_chembl_block_artifact_stress import choose_artifact_blocks, inject_artifact
    from train_artifact_meta_ranker_20261003 import cap_one, fit_ranker, load_panel

    RDLogger.DisableLog("rdApp.*")
    cohort = root / "data/chembl_modern_seventh_20261005"
    ledger = pd.read_csv(cohort / "technical_eligibility.csv")
    eligible = ledger.loc[ledger.eligible, "target_chembl_id"].tolist()
    score_dir = cohort / f"eligible_seed{args.seed}" / "scores"
    paths = [score_dir / f"{target}_scores.csv" for target in eligible]
    if not eligible or not all(path.is_file() for path in paths):
        raise AssertionError("Missing fixed eligible score panel")
    if set(score_dir.glob("*_scores.csv")) != set(paths):
        raise AssertionError("Unexpected target in scorer output")
    panels = [load_panel(path) for path in paths]
    x, y = training_matrix(panels, repeats=3)
    chemdep = fit_ranker(x, y, 862026)
    out = cohort / "fixed_selector_results"
    out.mkdir(parents=True, exist_ok=True)
    chemdep_path = out / f"seed{args.seed}.chemdep.json"
    chemdep.save_model(chemdep_path)
    rows = []
    for index, panel in enumerate(panels):
        target, cal, test = panel["target"], panel["cal"], panel["test"]
        fp_cal, fp_test = fingerprints(cal.canonical_smiles), fingerprints(test.canonical_smiles)
        rf = fit_rf(np.column_stack([fp_cal, cal.score.to_numpy(float)]),
                    cal.label.to_numpy(int), args.seed)
        base, labels = test.score.to_numpy(float), test.label.to_numpy(int)
        blocks, null = panel["test_blocks"], panel["null"]
        target_hash = int(hashlib.sha256(target.encode()).hexdigest()[:12], 16) % 1_000_000
        target_seed = 357700 + index * 100_000 + target_hash
        for rep in range(10):
            for fraction in (.2, .3):
                rng = np.random.default_rng(target_seed + rep * 10_003 + int(fraction * 10_000))
                artifacts = choose_artifact_blocks(
                    test, artifact_block_col="murcko_scaffold", rng=rng,
                    artifact_fraction=fraction, max_artifact_blocks=25,
                    min_inactive_per_block=2)
                for boost in (4., 6.):
                    scores = inject_artifact(
                        test, base, artifacts,
                        artifact_block_col="murcko_scaffold", logit_boost=boost)[0]
                    cp = chemdep.predict_proba(evidence_features(scores, blocks, null))[:, 1]
                    rp = rf.predict_proba(np.column_stack([fp_test, scores]))[:, 1]
                    cr = rankdata(cp, method="average") / len(cp)
                    rr = rankdata(rp, method="average") / len(rp)
                    predictions = {"score_cap1": scores, "chemdep_cal": cp,
                                   "multi_fp_rf": rp,
                                   "rf75_chemdep25": .75 * rr + .25 * cr}
                    for method, values in predictions.items():
                        chosen = cap_one(values, blocks, 50)
                        if len(chosen) != 50 or len(set(blocks[chosen])) != 50:
                            raise AssertionError(f"Budget/scaffold violation: {target}/{method}")
                        rows.append({"target": target, "scorer_seed": args.seed,
                                     "rep": rep, "fraction": fraction, "boost": boost,
                                     "method": method, "hits": int(labels[chosen].sum()),
                                     "selected": len(chosen), "scaffolds": len(set(blocks[chosen]))})
        print("seed", args.seed, "target", target, "completed", flush=True)
    cells = pd.DataFrame(rows)
    if len(cells) != len(eligible) * 40 * len(METHODS):
        raise AssertionError("Incomplete seventh selector grid")
    cells.to_csv(out / f"seed{args.seed}.csv", index=False)
    (out / f"seed{args.seed}.manifest.json").write_text(json.dumps({
        "status": "fixed before seventh activity retrieval; label-aware synthetic stress",
        "scorer_seed": args.seed, "eligible_targets": eligible, "methods": METHODS,
        "rf_weight": .75, "chemdep_weight": .25,
        "score_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in paths},
        "chemdep_model_sha256": hashlib.sha256(chemdep_path.read_bytes()).hexdigest(),
        "test_label_use": "artifact construction and final hit counting only",
    }, indent=2) + "\n", encoding="utf-8")
    print(cells.groupby("method").hits.mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
