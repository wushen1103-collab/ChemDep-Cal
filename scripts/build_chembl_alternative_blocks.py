#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from run_chembl_selection import FP_COL, fingerprint_matrix


@dataclass
class UnionFind:
    parent: list[int]

    @classmethod
    def create(cls, n: int) -> "UnionFind":
        return cls(list(range(n)))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def labels(self) -> np.ndarray:
        seen: dict[int, int] = {}
        labels = np.empty(len(self.parent), dtype=int)
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in seen:
                seen[root] = len(seen)
            labels[i] = seen[root]
        return labels


def add_tanimoto_edges(uf: UnionFind, bits: np.ndarray, threshold: float) -> int:
    n = len(bits)
    if n < 2:
        return 0
    x = bits.astype(np.uint16)
    intersections = x @ x.T
    bit_counts = x.sum(axis=1, dtype=np.uint16)
    unions = bit_counts[:, None] + bit_counts[None, :] - intersections
    tri = np.triu_indices(n, k=1)
    similarities = intersections[tri] / np.maximum(unions[tri], 1)
    hits = np.flatnonzero(similarities >= threshold)
    for pos in hits:
        uf.union(int(tri[0][pos]), int(tri[1][pos]))
    return int(len(hits))


def add_murcko_edges(uf: UnionFind, frame: pd.DataFrame) -> int:
    edge_count = 0
    scaffold_values = frame["murcko_scaffold"].fillna(frame["molecule_chembl_id"]).astype(str)
    for indices in frame.groupby(scaffold_values, sort=False).groups.values():
        idx = list(indices)
        if len(idx) < 2:
            continue
        first = int(idx[0])
        for other in idx[1:]:
            uf.union(first, int(other))
            edge_count += 1
    return edge_count


def component_labels(frame: pd.DataFrame, mode: str, threshold: float) -> tuple[np.ndarray, int, int]:
    uf = UnionFind.create(len(frame))
    tanimoto_edges = 0
    murcko_edges = 0
    if mode in {"tanimoto", "hybrid"} and len(frame):
        bits = fingerprint_matrix(frame[FP_COL])
        tanimoto_edges = add_tanimoto_edges(uf, bits, threshold)
    if mode == "hybrid":
        murcko_edges = add_murcko_edges(uf, frame)
    return uf.labels(), tanimoto_edges, murcko_edges


def add_block_column(df: pd.DataFrame, *, mode: str, threshold: float, block_col: str) -> tuple[pd.DataFrame, list[dict]]:
    out = df.copy()
    out[block_col] = ""
    stats: list[dict] = []
    for split, split_df in out.groupby("split", sort=False):
        labels, tanimoto_edges, murcko_edges = component_labels(split_df.reset_index(drop=True), mode, threshold)
        values = [f"{mode}{int(round(threshold * 100)):02d}_{split}_{label:04d}" for label in labels]
        out.loc[split_df.index, block_col] = values
        counts = pd.Series(values).value_counts()
        stats.append(
            {
                "split": split,
                "n_molecules": int(len(split_df)),
                "block_col": block_col,
                "mode": mode,
                "threshold": threshold,
                "n_blocks": int(len(counts)),
                "max_block_size": int(counts.max()) if len(counts) else 0,
                "median_block_size": float(counts.median()) if len(counts) else 0.0,
                "singleton_rate": float(np.mean(counts.to_numpy() == 1)) if len(counts) else 0.0,
                "tanimoto_edges": tanimoto_edges,
                "murcko_edges": murcko_edges,
            }
        )
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--mode", choices=["tanimoto", "hybrid"], required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--block-col", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    scores_dir = root / args.scores_dir
    outdir = root / args.outdir
    out_scores = outdir / "scores"
    out_scores.mkdir(parents=True, exist_ok=True)
    block_col = args.block_col or f"{args.mode}{int(round(args.threshold * 100)):02d}_block"
    score_paths = sorted(scores_dir.glob("*_scores.csv"))
    if not score_paths:
        raise RuntimeError(f"No score files found under {scores_dir}")

    all_stats: list[dict] = []
    for path in tqdm(score_paths, desc=f"{args.mode}{args.threshold:g} blocks"):
        df = pd.read_csv(path)
        if block_col in df.columns:
            raise ValueError(f"{path} already contains {block_col!r}")
        out_df, stats = add_block_column(df, mode=args.mode, threshold=args.threshold, block_col=block_col)
        out_df.to_csv(out_scores / path.name, index=False)
        target_id = str(df["target_chembl_id"].iloc[0])
        pref_name = str(df["pref_name"].iloc[0])
        for row in stats:
            row["target_chembl_id"] = target_id
            row["pref_name"] = pref_name
            all_stats.append(row)

    stats_df = pd.DataFrame(all_stats)
    stats_path = outdir / "block_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scores_dir": args.scores_dir,
        "outdir": args.outdir,
        "mode": args.mode,
        "threshold": args.threshold,
        "block_col": block_col,
        "targets": len(score_paths),
        "block_definition": (
            "connected components of ECFP4 Tanimoto graph"
            if args.mode == "tanimoto"
            else "connected components of ECFP4 Tanimoto graph plus original Murcko scaffold edges"
        ),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(stats_df[stats_df["split"] == "testpool"].to_markdown(index=False, floatfmt=".4f"))
    print(f"Wrote {out_scores}")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
