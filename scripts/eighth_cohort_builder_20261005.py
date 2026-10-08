#!/usr/bin/env python3
"""Build a metadata-precommitted 100-target ChEMBL eighth cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import build_chembl_smoke_panels as builder


CATALOG_SHA256 = "bfd5faef9f0c17df900ef40310964b3e4acd4ece6094b735908390cd14e87fda"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    catalog_file = root / "data/chembl_modern_eighth_20261005/catalog_metadata.json"
    if hashlib.sha256(catalog_file.read_bytes()).hexdigest() != CATALOG_SHA256:
        raise AssertionError("Metadata catalog hash differs from frozen source")
    catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    selected = [record for record in catalog["human_single_protein"]
                if int(record["target_chembl_id"][6:]) > 1821][:100]
    ids = [row["target_chembl_id"] for row in selected]
    if len(ids) != 100 or ids[0] != "CHEMBL1822" or ids[-1] != "CHEMBL1958":
        raise AssertionError("Frozen metadata inventory changed")
    for item in selected:
        if item["organism"] != "Homo sapiens" or item["target_type"] != "SINGLE PROTEIN":
            raise AssertionError("Incorrect target metadata")
    fixed = {row["target_chembl_id"]: builder.TargetInfo(
        row["target_chembl_id"], row["pref_name"] or row["target_chembl_id"],
        row["organism"], row["target_type"]) for row in selected}

    def resolve_targets(requested: list[str]) -> list[builder.TargetInfo]:
        if requested != ids:
            raise AssertionError("Target list differs from frozen metadata inventory")
        return [fixed[target] for target in requested]

    builder.resolve_targets = resolve_targets
    sys.argv = [sys.argv[0], "--outdir", str(root / "data/chembl_modern_eighth_20261005"),
                "--target-ids", *ids, "--max-targets", "100",
                "--max-activities-per-target", "4000", "--page-size", "1000",
                "--active-threshold", "7", "--inactive-threshold", "6",
                "--min-active", "50", "--min-inactive", "200",
                "--min-scaffolds", "50", "--seed", "35", "--sleep", "0.2"]
    builder.main()


if __name__ == "__main__":
    main()
