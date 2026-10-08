"""Recompute the primary fifth-cohort hit means from five archived scorer seeds."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "source_tables/fifth_primary"
METHODS = ("meta_calibration", "score_cap1")


def main() -> None:
    cells = pd.concat(
        [pd.read_csv(ROOT / "fifth_raw" / f"meta_seed{seed}.csv").assign(scorer_seed=seed)
         for seed in range(35501, 35506)], ignore_index=True
    )
    cells = cells.loc[cells.budget.eq(50) & cells.method.isin(METHODS)]
    assert cells.target.nunique() == 10 and cells.scorer_seed.nunique() == 5
    assert cells.groupby("method").size().eq(2000).all()
    assert cells.selected.eq(50).all() and cells.blocks.eq(50).all()
    per_target = cells.groupby(["target", "method"]).hits.mean().unstack()
    saved = pd.read_csv(ROOT / "fifth_method_mean_target_sd.csv").set_index("method")
    for method in METHODS:
        assert np.isclose(per_target[method].mean(), saved.loc[method, "hits"])
        assert np.isclose(per_target[method].std(ddof=1), saved.loc[method, "hits_target_sd"])
    gain = (per_target.meta_calibration - per_target.score_cap1).mean()
    assert np.isclose(gain, 2.322)
    print("PASS: five seeds, ten paired targets, 2,000 matched cells per method")
    print(f"ChemDep-Cal {per_target.meta_calibration.mean():.4f}; "
          f"score cap-1 {per_target.score_cap1.mean():.4f}; gain {gain:.4f} hits")


if __name__ == "__main__":
    main()
