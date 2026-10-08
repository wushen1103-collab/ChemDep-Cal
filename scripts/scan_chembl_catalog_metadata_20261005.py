#!/usr/bin/env python3
"""Catalog human single-protein target IDs using metadata only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


API = "https://www.ebi.ac.uk/chembl/api/data/target.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=5)
    args = parser.parse_args()
    session = requests.Session()
    session.headers.update({"User-Agent": "ChemDep-Cal-metadata-only-catalog/1.0"})
    records = []
    for page in range(args.pages):
        response = session.get(API, params={"organism": "Homo sapiens",
                                             "target_type": "SINGLE PROTEIN",
                                             "limit": 1000, "offset": page * 1000}, timeout=120)
        response.raise_for_status()
        payload = response.json()
        targets = payload.get("targets", [])
        if not targets:
            break
        records.extend({"target_chembl_id": item.get("target_chembl_id"),
                        "organism": item.get("organism"),
                        "target_type": item.get("target_type"),
                        "pref_name": item.get("pref_name")}
                       for item in targets)
        print("metadata page", page, "rows", len(targets), flush=True)
        if not payload.get("page_meta", {}).get("next"):
            break
    candidates = [row for row in records
                  if row["organism"] == "Homo sapiens"
                  and row["target_type"] == "SINGLE PROTEIN"
                  and isinstance(row["target_chembl_id"], str)
                  and row["target_chembl_id"].startswith("CHEMBL")
                  and row["target_chembl_id"][6:].isdigit()]
    candidates.sort(key=lambda row: int(row["target_chembl_id"][6:]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "status": "metadata only; no activity or label endpoint accessed",
        "endpoint": API, "params": {"organism": "Homo sapiens",
                                       "target_type": "SINGLE PROTEIN", "limit": 1000},
        "pages_requested": args.pages, "records_returned": len(records),
        "human_single_protein": candidates,
    }, indent=2) + "\n", encoding="utf-8")
    print("eligible metadata records", len(candidates), flush=True)
    print([row["target_chembl_id"] for row in candidates[:80]], flush=True)


if __name__ == "__main__":
    main()
