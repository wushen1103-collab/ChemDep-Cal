#!/usr/bin/env python3
"""Run the frozen numeric target list while skipping nonexistent ChEMBL IDs."""

from __future__ import annotations

import time

import requests

import build_chembl_smoke_panels as builder


def resolve_existing_human_targets(target_ids: list[str]) -> list[builder.TargetInfo]:
    result = []
    session = requests.Session()
    session.headers.update({"User-Agent": "ChemDep-Cal sixth-cohort research benchmark"})
    for target_id in dict.fromkeys(target_ids):
        url = f"{builder.CHEMBL_API}/target/{target_id}.json"
        for attempt in range(3):
            try:
                response = session.get(url, timeout=60)
                if response.status_code == 404:
                    print(f"Skipping {target_id}: target ID does not exist", flush=True)
                    break
                response.raise_for_status()
                item = response.json()
                organism = item.get("organism") or ""
                target_type = item.get("target_type") or ""
                if organism != "Homo sapiens" or target_type != "SINGLE PROTEIN":
                    print(f"Skipping {target_id}: {organism}, {target_type}", flush=True)
                    break
                result.append(builder.TargetInfo(target_id, item.get("pref_name") or target_id,
                                                 organism, target_type))
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2.0 * (attempt + 1))
    return result


builder.resolve_targets = resolve_existing_human_targets

if __name__ == "__main__":
    builder.main()
