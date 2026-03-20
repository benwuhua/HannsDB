#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a tiny deterministic VectorDBBench custom dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where train.parquet/test.parquet/neighbors.parquet will be written.",
    )
    parser.add_argument("--dimension", type=int, default=32, help="Embedding dimension.")
    parser.add_argument("--train-size", type=int, default=1000, help="Train set size.")
    parser.add_argument("--test-size", type=int, default=100, help="Test set size.")
    parser.add_argument(
        "--metric",
        choices=["l2", "cosine", "ip"],
        default="l2",
        help="Distance metric used for ground-truth neighbors.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of ground-truth neighbors to store per test vector.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260319,
        help="Random seed for deterministic generation.",
    )
    return parser.parse_args()


def generate_vectors(
    train_size: int, test_size: int, dim: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if train_size <= 0 or test_size <= 0 or dim <= 0:
        raise ValueError("train-size, test-size, and dimension must be > 0")

    rng = np.random.default_rng(seed)
    train = rng.standard_normal((train_size, dim), dtype=np.float32)

    # Mix "near-train" queries with fresh random queries for a practical smoke dataset.
    copy_count = min(test_size // 2, train_size)
    copied = train[:copy_count] + (0.01 * rng.standard_normal((copy_count, dim), dtype=np.float32))
    random_count = test_size - copy_count
    random_queries = rng.standard_normal((random_count, dim), dtype=np.float32)
    test = np.vstack([copied, random_queries]).astype(np.float32, copy=False)

    return train, test


def topk_neighbors(
    train: np.ndarray, test: np.ndarray, metric: str, top_k: int
) -> np.ndarray:
    if top_k <= 0:
        raise ValueError("top-k must be > 0")
    k = min(top_k, train.shape[0])

    if metric == "l2":
        # Brute-force squared L2 (monotonic with L2).
        diffs = test[:, None, :] - train[None, :, :]
        scores = np.sum(diffs * diffs, axis=2)  # smaller is better
        neighbor_idx = np.argsort(scores, axis=1)[:, :k]
    elif metric == "ip":
        scores = test @ train.T  # larger is better
        neighbor_idx = np.argsort(-scores, axis=1)[:, :k]
    elif metric == "cosine":
        train_norm = np.linalg.norm(train, axis=1, keepdims=True)
        test_norm = np.linalg.norm(test, axis=1, keepdims=True)
        # Stable normalization: keep zero vectors as zero.
        train_safe = np.divide(train, train_norm, out=np.zeros_like(train), where=train_norm > 0)
        test_safe = np.divide(test, test_norm, out=np.zeros_like(test), where=test_norm > 0)
        scores = test_safe @ train_safe.T  # larger is better
        neighbor_idx = np.argsort(-scores, axis=1)[:, :k]
    else:
        raise ValueError(f"unsupported metric: {metric}")

    return neighbor_idx


def write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def to_arrow_list_f32(vectors: np.ndarray) -> pa.Array:
    lists = [row.tolist() for row in vectors.astype(np.float32, copy=False)]
    return pa.array(lists, type=pa.list_(pa.float32()))


def main() -> None:
    args = parse_args()
    train, test = generate_vectors(args.train_size, args.test_size, args.dimension, args.seed)
    neighbor_idx = topk_neighbors(train, test, args.metric, args.top_k)

    train_ids = np.arange(args.train_size, dtype=np.int64)
    test_ids = np.arange(args.test_size, dtype=np.int64)
    neighbor_ids = [[int(j) for j in row.tolist()] for row in neighbor_idx]

    train_table = pa.table(
        {
            "id": pa.array(train_ids, type=pa.int64()),
            "emb": to_arrow_list_f32(train),
        }
    )
    test_table = pa.table(
        {
            "id": pa.array(test_ids, type=pa.int64()),
            "emb": to_arrow_list_f32(test),
        }
    )
    neighbors_table = pa.table(
        {
            "id": pa.array(test_ids, type=pa.int64()),
            "neighbors_id": pa.array(neighbor_ids, type=pa.list_(pa.int64())),
        }
    )

    out_dir = args.output_dir
    write_parquet(out_dir / "train.parquet", train_table)
    write_parquet(out_dir / "test.parquet", test_table)
    write_parquet(out_dir / "neighbors.parquet", neighbors_table)

    print(
        f"dataset generated at {out_dir} "
        f"(train={args.train_size}, test={args.test_size}, dim={args.dimension}, metric={args.metric}, top_k={min(args.top_k, args.train_size)})"
    )


if __name__ == "__main__":
    main()
