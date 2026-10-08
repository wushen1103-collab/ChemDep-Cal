#!/usr/bin/env python3
"""Run frozen eighth-cohort scorer/selector scripts with bounded CPU concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
import subprocess
import sys
from pathlib import Path


def run(script: Path, root: Path, seed: int, env: dict[str, str]) -> int:
    subprocess.run([sys.executable, str(script), "--root", str(root), "--seed", str(seed)],
                   cwd=root, env=env, check=True)
    return seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    scripts = root / "scripts"
    env = os.environ.copy()
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1"})
    scorer = scripts / "eighth_cohort_scorer_20261005.py"
    selector = scripts / "eighth_cohort_fixed_selectors_20261005.py"
    run(scorer, root, 35801, env)
    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        print("scorers complete", list(pool.map(
            lambda seed: run(scorer, root, seed, env), range(35802, 35806))), flush=True)
    with futures.ThreadPoolExecutor(max_workers=5) as pool:
        print("selectors complete", list(pool.map(
            lambda seed: run(selector, root, seed, env), range(35801, 35806))), flush=True)
    subprocess.run([sys.executable, str(scripts / "audit_eighth_fixed_results_20261005.py"),
                    "--source", str(root / "data/chembl_modern_eighth_20261005")],
                   cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
