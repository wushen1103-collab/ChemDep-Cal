#!/usr/bin/env python3
"""Metadata-only ChEMBL target inventory; never queries activities or labels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


API = "https://www.ebi.ac.uk/chembl/api/data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.first > args.last:
        raise ValueError("Empty numeric range")
    session = requests.Session()
    session.headers.update({"User-Agent": "ChemDep-Cal-metadata-only-audit/1.0"})
    rows = []
    for number in range(args.first, args.last + 1):
        identifier = f"CHEMBL{number}"
        response = session.get(f"{API}/target/{identifier}.json", timeout=60)
        if response.status_code == 404:
            rows.append({"target_chembl_id": identifier, "exists": False,
                         "organism": "", "target_type": "", "pref_name": ""})
        else:
            response.raise_for_status()
            target = response.json()
            rows.append({"target_chembl_id": identifier, "exists": True,
                         "organism": target.get("organism") or "",
                         "target_type": target.get("target_type") or "",
                         "pref_name": target.get("pref_name") or ""})
        if number % 10 == 0:
            print("scanned through", identifier, flush=True)
        time.sleep(.05)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "status": "metadata only; no activity or label endpoint accessed",
        "range": [args.first, args.last], "targets": rows,
    }, indent=2) + "\n", encoding="utf-8")
    eligible = [row["target_chembl_id"] for row in rows
                if row["organism"] == "Homo sapiens"
                and row["target_type"] == "SINGLE PROTEIN"]
    print("human_single_protein", len(eligible), eligible, flush=True)


if __name__ == "__main__":
    main()
