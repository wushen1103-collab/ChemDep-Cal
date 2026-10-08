#!/usr/bin/env python3
"""Fetch a pinned, third-party WelQrate mirror through the China HF mirror."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path


COMMIT = "10c6bd4d8d6bd0116929239864d6bbddb4e46ab0"
AIDS = ("AID1798", "AID435034", "AID463087")
BASE = f"https://hf-mirror.com/datasets/scikit-fingerprints/litmus/resolve/{COMMIT}/welqrate"


def fetch(task: tuple[Path, str]) -> dict:
    directory, relative = task
    destination = directory / relative.split("/")[-1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        error = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(f"{BASE}/{relative}", headers={"User-Agent": "ChemDep-Cal-repro/1.0"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
                if relative.endswith(".parquet") and data[:4] != b"PAR1":
                    raise ValueError("Not a Parquet payload")
                if relative.endswith(".csv") and not data.strip().splitlines()[0].isdigit():
                    raise ValueError("Unexpected split CSV payload")
                temporary = destination.with_suffix(destination.suffix + ".part")
                temporary.write_bytes(data)
                temporary.replace(destination)
                error = None
                break
            except Exception as exc:
                error = exc
                time.sleep(attempt + 1)
        if error is not None:
            raise RuntimeError(f"Could not fetch {relative}") from error
    data = destination.read_bytes()
    return {"source": relative, "file": destination.name, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    paths = [f"data/WELQRATE_{aid}.parquet" for aid in AIDS]
    paths.extend(f"splits/seed{seed}_WELQRATE_{aid}.{part}_idx.csv"
                 for aid in AIDS for seed in range(1, 6) for part in ("train", "test"))
    with futures.ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(fetch, [(directory, path) for path in paths]))
    manifest = {"status": "third-party Litmus mirror; not official WelQrate split",
                "mirror_commit": COMMIT, "files": rows}
    (directory / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"verified {len(rows)} pinned files; {sum(row['bytes'] for row in rows):,} bytes", flush=True)


if __name__ == "__main__":
    main()
