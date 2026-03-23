//! Standalone performance benchmark entries.
//!
//! Each entry is a regular `#[test]` function that prints structured timing
//! lines.  Run with `-- --nocapture` to see the output.
//!
//! Environment variables (override defaults):
//!
//! | Variable              | Default | Meaning                                  |
//! |-----------------------|---------|------------------------------------------|
//! | `HANNSDB_BENCH_N`     | 5000    | Number of vectors to insert              |
//! | `HANNSDB_BENCH_DIM`   | 64      | Vector dimensionality                    |
//! | `HANNSDB_BENCH_SEGS`  | 5       | Number of segments for compaction bench  |
//!
//! Timing output format (one line per phase, parseable by scripts):
//!
//! ```text
//! BENCH_<NAME> phase=<phase> n=<n> dim=<dim> ms=<elapsed_ms>
//! ```

use std::time::Instant;

use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{Document, FieldValue};
use hannsdb_core::segment::{
    append_payloads, append_record_ids, append_records, SegmentMetadata, SegmentSet, TombstoneMask,
};

// ── helpers ──────────────────────────────────────────────────────────────────

fn read_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

/// Deterministic synthetic vector (same formula as optimize_benchmark_entry).
fn synthetic_value(i: usize, j: usize) -> f32 {
    let x = (i as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add((j as u64).wrapping_mul(1442695040888963407))
        .wrapping_add(1);
    ((((x >> 16) as u32) as f32) / (u32::MAX as f32)) * 2.0 - 1.0
}

fn build_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; n * dim];
    for i in 0..n {
        for (j, cell) in v[i * dim..(i + 1) * dim].iter_mut().enumerate() {
            *cell = synthetic_value(i, j);
        }
    }
    v
}

fn doc_empty(id: i64, vector: Vec<f32>) -> Document {
    Document::new(id, Vec::<(String, FieldValue)>::new(), vector)
}

fn write_segment_for_bench(
    segment_dir: &std::path::Path,
    segment_id: &str,
    dimension: usize,
    documents: &[Document],
) {
    std::fs::create_dir_all(segment_dir).expect("create segment dir");
    let mut ids = Vec::with_capacity(documents.len());
    let mut vectors = Vec::with_capacity(documents.len() * dimension);
    let mut payloads = Vec::with_capacity(documents.len());
    for d in documents {
        ids.push(d.id);
        vectors.extend_from_slice(&d.vector);
        payloads.push(d.fields.clone());
    }
    append_records(&segment_dir.join("records.bin"), dimension, &vectors).expect("write records");
    let _ = append_record_ids(&segment_dir.join("ids.bin"), &ids).expect("write ids");
    let _ = append_payloads(&segment_dir.join("payloads.jsonl"), &payloads).expect("write payloads");
    TombstoneMask::new(documents.len())
        .save_to_path(&segment_dir.join("tombstones.json"))
        .expect("write tombstones");
    SegmentMetadata::new(segment_id, dimension, documents.len(), 0)
        .save_to_path(&segment_dir.join("segment.json"))
        .expect("write segment metadata");
}

// ── bench: insert throughput ──────────────────────────────────────────────────

/// Measures raw insert throughput for a single flat-file collection.
///
/// Reports rows/sec for batch inserts of N vectors at DIM dimensions.
/// This baseline captures the disk-write path (records.bin + ids.bin +
/// payloads.jsonl + segment.json + tombstones.json) with no ANN indexing.
#[test]
fn bench_insert_throughput() {
    let n = read_env_usize("HANNSDB_BENCH_N", 5_000);
    let dim = read_env_usize("HANNSDB_BENCH_DIM", 64);

    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();
    let mut db = HannsDb::open(root).expect("open db");
    db.create_collection("bench", dim, "l2").expect("create collection");

    let ids: Vec<i64> = (0..n as i64).collect();
    let vectors = build_vectors(n, dim);

    let t = Instant::now();
    let inserted = db.insert("bench", &ids, &vectors).expect("insert");
    let ms = t.elapsed().as_millis();
    assert_eq!(inserted, n);

    let rows_per_sec = if ms == 0 { n as u128 * 1000 } else { n as u128 * 1000 / ms };
    println!("BENCH_INSERT_THROUGHPUT phase=insert n={n} dim={dim} ms={ms} rows_per_sec={rows_per_sec}");
}

// ── bench: brute-force search scaling ────────────────────────────────────────

/// Measures single-query brute-force search latency at three collection sizes.
///
/// This entry grows the collection incrementally so all three sizes are tested
/// in one pass.  Each search returns top-10 results; no ANN index is built.
#[test]
fn bench_brute_force_search_scaling() {
    let dim = read_env_usize("HANNSDB_BENCH_DIM", 64);
    let checkpoints: &[usize] = &[1_000, 10_000, 50_000];

    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();
    let mut db = HannsDb::open(root).expect("open db");
    db.create_collection("bench", dim, "l2").expect("create collection");

    let query = build_vectors(1, dim);
    let mut inserted_so_far: i64 = 0;

    for &target_n in checkpoints {
        let batch = target_n - inserted_so_far as usize;
        let ids: Vec<i64> = (inserted_so_far..inserted_so_far + batch as i64).collect();
        let vectors = build_vectors(batch, dim);
        db.insert("bench", &ids, &vectors).expect("insert batch");
        inserted_so_far += batch as i64;

        let t = Instant::now();
        let hits = db.search("bench", &query, 10).expect("search");
        let ms = t.elapsed().as_millis();
        let us = t.elapsed().as_micros();
        assert!(!hits.is_empty());
        println!(
            "BENCH_BRUTE_FORCE_SEARCH phase=search n={target_n} dim={dim} ms={ms} us={us}"
        );
    }
}

// ── bench: compaction throughput ──────────────────────────────────────────────

/// Measures the time to compact K immutable segments containing N total rows.
///
/// Segments are written directly (bypassing the DB insert path) to isolate the
/// compaction read+merge+write cost from the insert cost.
#[test]
fn bench_compaction_timing() {
    let n = read_env_usize("HANNSDB_BENCH_N", 5_000);
    let dim = read_env_usize("HANNSDB_BENCH_DIM", 64);
    let k_segs = read_env_usize("HANNSDB_BENCH_SEGS", 5);

    assert!(k_segs >= 2, "HANNSDB_BENCH_SEGS must be >= 2 to have at least one immutable");
    let rows_per_seg = n / k_segs;
    assert!(rows_per_seg > 0, "n must be >= k_segs");

    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path();

    {
        let mut db = HannsDb::open(root).expect("open db");
        db.create_collection("bench", dim, "l2").expect("create collection");
    }
    std::fs::remove_file(root.join("wal.jsonl")).expect("remove wal");

    let segs_dir = root
        .join("collections")
        .join("bench")
        .join("segments");

    let mut immutable_ids: Vec<String> = Vec::new();
    let mut global_id: i64 = 0;
    for seg_idx in 0..k_segs {
        let seg_id = format!("seg-{:06}", seg_idx + 1);
        let seg_dir = segs_dir.join(&seg_id);
        let docs: Vec<Document> = (0..rows_per_seg)
            .map(|_row| {
                let id = global_id;
                global_id += 1;
                doc_empty(id, build_vectors(1, dim))
            })
            .collect();
        write_segment_for_bench(&seg_dir, &seg_id, dim, &docs);
        immutable_ids.push(seg_id);
    }
    // The last segment is "active"; the rest are immutable.
    let active_id = immutable_ids.pop().expect("at least one segment");
    SegmentSet {
        active_segment_id: active_id.clone(),
        immutable_segment_ids: immutable_ids.clone(),
    }
    .save_to_path(
        &root
            .join("collections")
            .join("bench")
            .join("segment_set.json"),
    )
    .expect("write segment_set");

    let mut db = HannsDb::open(root).expect("reopen db");

    let total_rows = k_segs * rows_per_seg;
    let immutable_rows = immutable_ids.len() * rows_per_seg;

    let t = Instant::now();
    db.compact_collection("bench").expect("compact");
    let ms = t.elapsed().as_millis();

    println!(
        "BENCH_COMPACTION phase=compact k_segs={} immutable_rows={immutable_rows} \
         total_rows={total_rows} dim={dim} ms={ms}",
        immutable_ids.len()
    );

    // Sanity: search still works after compaction.
    let query = build_vectors(1, dim);
    let hits = db.search("bench", &query, 1).expect("search after compact");
    assert!(!hits.is_empty(), "search must return results after compaction");
}
