#!/usr/bin/env python3
"""Safely unpack the public ChEMBL scorer-input archive into the repository."""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path, PurePosixPath


EXPECTED_SHA256 = "d1f52e4942ea668d272392865ecda81ffd4fef45a1415249af3874dd6bb11d69"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    archive = root / "data/archives/chemdep-cal-chembl-scores-clean.tar.gz"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Archive SHA-256 mismatch")

    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or not path.parts
                    or path.parts[0] != "data" or not (member.isfile() or member.isdir())):
                raise ValueError(f"Unexpected archive entry: {member.name}")
        if sys.version_info >= (3, 12):
            bundle.extractall(root, members=members, filter="data")
        else:
            bundle.extractall(root, members=members)
    print(f"Extracted {len(members)} entries; SHA-256 verified")


if __name__ == "__main__":
    main()
