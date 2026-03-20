# Tiny Custom Dataset For VectorDBBench Smoke

Use this helper to create a small local `PerformanceCustomDataset` directory without downloading the 50K default dataset.

## Generate

```bash
/Users/ryan/Code/HannsDB/.venv-hannsdb/bin/python \
  /Users/ryan/Code/HannsDB/python/generate_custom_vdbb_dataset.py \
  --output-dir /tmp/hannsdb-custom-ds \
  --dimension 16 \
  --train-size 500 \
  --test-size 50 \
  --metric l2 \
  --top-k 10
```

## Output

The output directory contains:

- `train.parquet` with columns: `id` (`int64`), `emb` (`list<float32>`)
- `test.parquet` with columns: `id` (`int64`), `emb` (`list<float32>`)
- `neighbors.parquet` with columns: `id` (`int64`), `neighbors_id` (`list<int64>`)

Ground-truth neighbors are computed by brute force according to `--metric` (`l2`, `ip`, or `cosine`).
