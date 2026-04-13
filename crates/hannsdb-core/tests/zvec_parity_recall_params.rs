/// Port of zvec recall calculation and query parameter tests.
///
/// Mirrors patterns from:
///   - zvec/python/tests/detail/test_collection_recall.py
///   - zvec/python/tests/detail/test_collection_dql.py
///   - zvec/python/tests/detail/distance_helper.py
///
/// Distance conventions (matching HannsDB internals):
///   L2     = sqrt(sum((a-b)^2))
///   Cosine = 1 - dot(a,b)/(norm(a)*norm(b))
///   IP     = -dot(a,b)   (negated so smaller = more similar)
use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema,
};
use hannsdb_core::query::{QueryContext, QueryVector, VectorQuery, VectorQueryParam};

// ---------------------------------------------------------------------------
// Inline distance helpers (mirroring zvec distance_helper.py / search.rs)
// ---------------------------------------------------------------------------

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    1.0 - dot / (norm_a * norm_b)
}

fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute distance matching the metric string.  For IP we negate the
/// dot-product so that the sort order (ascending) puts the most-similar
/// vectors first, consistent with HannsDB's convention.
fn distance_by_metric(a: &[f32], b: &[f32], metric: &str) -> f32 {
    match metric {
        "l2" => l2_distance(a, b),
        "cosine" => cosine_distance(a, b),
        "ip" => -inner_product(a, b),
        _ => panic!("unsupported metric: {metric}"),
    }
}

// ---------------------------------------------------------------------------
// Recall helpers
// ---------------------------------------------------------------------------

/// Compute recall@k as |expected_ids ∩ actual_ids| / min(k, |expected_ids|).
fn compute_recall(expected: &[(i64, f32)], actual: &[(i64, f32)], k: usize) -> f64 {
    let expected_ids: std::collections::HashSet<i64> =
        expected.iter().take(k).map(|(id, _)| *id).collect();
    let actual_ids: std::collections::HashSet<i64> =
        actual.iter().take(k).map(|(id, _)| *id).collect();
    let intersection = expected_ids.intersection(&actual_ids).count();
    intersection as f64 / k.min(expected_ids.len()) as f64
}

/// Brute-force ground truth search.  Returns (id, distance) sorted ascending
/// by distance (matching HannsDB convention where smaller = closer).
fn brute_force_search(
    vectors: &[(i64, Vec<f32>)],
    query: &[f32],
    k: usize,
    metric: &str,
) -> Vec<(i64, f32)> {
    let mut scored: Vec<(i64, f32)> = vectors
        .iter()
        .map(|(id, vec)| {
            let dist = distance_by_metric(query, vec, metric);
            (*id, dist)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.truncate(k);
    scored
}

// ---------------------------------------------------------------------------
// Test data helpers
// ---------------------------------------------------------------------------

/// Generate `n` deterministic random-ish vectors of dimension `dim`.
/// Uses a simple linear congruential generator so tests are reproducible.
fn generate_test_vectors(n: usize, dim: usize) -> Vec<(i64, Vec<f32>)> {
    let mut state: u64 = 42;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let f = (state as f32) / (u64::MAX as f32); // (0, 1)
            v.push(f * 2.0 - 1.0); // map to [-1, 1]
        }
        out.push((i as i64, v));
    }
    out
}

/// Insert test vectors into a collection, returning the documents for later
/// querying.
fn insert_test_data(
    db: &mut HannsDb,
    collection: &str,
    vectors: &[(i64, Vec<f32>)],
    metric: &str,
    dim: usize,
) {
    let schema = CollectionSchema::new("vector", dim, metric, vec![]);
    db.create_collection_with_schema(collection, &schema)
        .expect("create collection");
    let docs: Vec<Document> = vectors
        .iter()
        .map(|(id, vec)| Document::new(*id, [], vec.clone()))
        .collect();
    db.insert_documents(collection, &docs)
        .expect("insert documents");
}

// ---------------------------------------------------------------------------
// Tests: Recall across metrics
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_recall_l2_metric_brute_force_ground_truth() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let n = 100;
    let dim = 8;
    let topk = 10;
    let metric = "l2";
    let vectors = generate_test_vectors(n, dim);

    insert_test_data(&mut db, "recall_l2", &vectors, metric, dim);

    // Use the first 5 vectors as queries
    let mut total_recall = 0.0;
    let query_count = 5;
    for q in 0..query_count {
        let query = &vectors[q].1;
        let expected = brute_force_search(&vectors, query, topk, metric);
        let hits = db.search("recall_l2", query, topk).expect("search");
        let actual: Vec<(i64, f32)> = hits.iter().map(|h| (h.id, h.distance)).collect();
        total_recall += compute_recall(&expected, &actual, topk);
    }

    let avg_recall = total_recall / query_count as f64;
    assert!(
        avg_recall >= 0.8,
        "L2 recall after optimize should be >= 0.8, got {avg_recall}"
    );
}

#[test]
fn zvec_parity_recall_cosine_metric_brute_force_ground_truth() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let n = 100;
    let dim = 8;
    let topk = 10;
    let metric = "cosine";
    let vectors = generate_test_vectors(n, dim);

    insert_test_data(&mut db, "recall_cosine", &vectors, metric, dim);

    let mut total_recall = 0.0;
    let query_count = 5;
    for q in 0..query_count {
        let query = &vectors[q].1;
        let expected = brute_force_search(&vectors, query, topk, metric);
        let hits = db.search("recall_cosine", query, topk).expect("search");
        let actual: Vec<(i64, f32)> = hits.iter().map(|h| (h.id, h.distance)).collect();
        total_recall += compute_recall(&expected, &actual, topk);
    }

    let avg_recall = total_recall / query_count as f64;
    assert!(
        avg_recall >= 0.8,
        "Cosine recall should be >= 0.8, got {avg_recall}"
    );
}

#[test]
fn zvec_parity_recall_ip_metric_brute_force_ground_truth() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let n = 100;
    let dim = 8;
    let topk = 10;
    let metric = "ip";
    let vectors = generate_test_vectors(n, dim);

    insert_test_data(&mut db, "recall_ip", &vectors, metric, dim);

    let mut total_recall = 0.0;
    let query_count = 5;
    for q in 0..query_count {
        let query = &vectors[q].1;
        let expected = brute_force_search(&vectors, query, topk, metric);
        let hits = db.search("recall_ip", query, topk).expect("search");
        let actual: Vec<(i64, f32)> = hits.iter().map(|h| (h.id, h.distance)).collect();
        total_recall += compute_recall(&expected, &actual, topk);
    }

    let avg_recall = total_recall / query_count as f64;
    assert!(
        avg_recall >= 0.8,
        "IP recall should be >= 0.8, got {avg_recall}"
    );
}

// ---------------------------------------------------------------------------
// Tests: Recall improves after optimize
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_recall_improves_after_optimize() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let n = 100;
    let dim = 8;
    let topk = 10;
    let metric = "l2";
    let vectors = generate_test_vectors(n, dim);

    insert_test_data(&mut db, "recall_opt", &vectors, metric, dim);

    // Measure recall before optimize (brute force, should be perfect already
    // since without knowhere-backend the search is brute-force).
    let query = &vectors[0].1;
    let expected = brute_force_search(&vectors, query, topk, metric);
    let hits_pre = db.search("recall_opt", query, topk).expect("search pre");
    let actual_pre: Vec<(i64, f32)> = hits_pre.iter().map(|h| (h.id, h.distance)).collect();
    let recall_pre = compute_recall(&expected, &actual_pre, topk);

    // Optimize (builds ANN index when knowhere-backend is active)
    db.optimize_collection("recall_opt").expect("optimize");

    let hits_post = db.search("recall_opt", query, topk).expect("search post");
    let actual_post: Vec<(i64, f32)> = hits_post.iter().map(|h| (h.id, h.distance)).collect();
    let recall_post = compute_recall(&expected, &actual_post, topk);

    // Post-optimize recall should be at least as good as pre-optimize
    assert!(
        recall_post >= recall_pre,
        "recall after optimize ({recall_post}) should not degrade vs before ({recall_pre})"
    );
    assert!(
        recall_post >= 0.8,
        "recall after optimize should be >= 0.8, got {recall_post}"
    );
}

// ---------------------------------------------------------------------------
// Tests: ef_search parameter impact
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_ef_search_higher_ef_gives_equal_or_better_recall() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let n = 100;
    let dim = 8;
    let topk = 10;
    let metric = "l2";
    let vectors = generate_test_vectors(n, dim);

    insert_test_data(&mut db, "ef_compare", &vectors, metric, dim);
    db.optimize_collection("ef_compare").expect("optimize");

    let query = &vectors[0].1;
    let expected = brute_force_search(&vectors, query, topk, metric);

    // Search with low ef=10
    let hits_low = db
        .search_with_ef("ef_compare", query, topk, 10)
        .expect("search ef=10");
    let actual_low: Vec<(i64, f32)> = hits_low.iter().map(|h| (h.id, h.distance)).collect();
    let recall_low = compute_recall(&expected, &actual_low, topk);

    // Search with high ef=200
    let hits_high = db
        .search_with_ef("ef_compare", query, topk, 200)
        .expect("search ef=200");
    let actual_high: Vec<(i64, f32)> = hits_high.iter().map(|h| (h.id, h.distance)).collect();
    let recall_high = compute_recall(&expected, &actual_high, topk);

    assert!(
        recall_high >= recall_low,
        "ef=200 recall ({recall_high}) should be >= ef=10 recall ({recall_low})"
    );
}

#[test]
fn zvec_parity_ef_search_via_query_context_matches_search_with_ef() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let n = 50;
    let dim = 4;
    let topk = 5;
    let metric = "l2";
    let vectors = generate_test_vectors(n, dim);

    insert_test_data(&mut db, "ef_context", &vectors, metric, dim);

    let query = &vectors[0].1;

    let legacy_hits = db
        .search_with_ef("ef_context", query, topk, 64)
        .expect("search_with_ef");

    let typed_hits = db
        .query_with_context(
            "ef_context",
            &QueryContext {
                top_k: topk,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(query.clone()),
                    param: Some(VectorQueryParam {
                        ef_search: Some(64),
                        nprobe: None,
                    }),
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("query_with_context ef=64");

    let legacy_ids: Vec<i64> = legacy_hits.iter().map(|h| h.id).collect();
    let typed_ids: Vec<i64> = typed_hits.iter().map(|h| h.id).collect();
    assert_eq!(
        legacy_ids, typed_ids,
        "typed ef_search query should match legacy search_with_ef results"
    );

    let legacy_dists: Vec<f32> = legacy_hits.iter().map(|h| h.distance).collect();
    let typed_dists: Vec<f32> = typed_hits.iter().map(|h| h.distance).collect();
    assert_eq!(
        legacy_dists, typed_dists,
        "distances from typed ef_search should match legacy path"
    );
}

// ---------------------------------------------------------------------------
// Tests: query_by_id (port of zvec test_query_by_id)
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_query_by_id_returns_results_using_existing_vector() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let dim = 4;
    let metric = "l2";
    let schema = CollectionSchema::new("vector", dim, metric, vec![]);
    db.create_collection_with_schema("qbyid", &schema)
        .expect("create collection");

    let vectors = generate_test_vectors(10, dim);
    let docs: Vec<Document> = vectors
        .iter()
        .map(|(id, vec)| Document::new(*id, [], vec.clone()))
        .collect();
    db.insert_documents("qbyid", &docs).expect("insert");

    // query_by_id for doc id=3 — should use doc 3's vector as the query
    let hits = db
        .query_with_context(
            "qbyid",
            &QueryContext {
                top_k: 3,
                queries: Vec::new(),
                query_by_id: Some(vec![3]),
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("query_by_id");

    assert!(!hits.is_empty(), "query_by_id should return results");
    // The first result should be the document itself (distance 0)
    assert_eq!(hits[0].id, 3);
    assert!(
        hits[0].distance.abs() < 1e-6,
        "self-distance should be ~0, got {}",
        hits[0].distance
    );
}

// ---------------------------------------------------------------------------
// Tests: Query consistency (port of zvec test_query_consistency)
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_query_consistency_same_results_five_times() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let n = 20;
    let dim = 4;
    let metric = "l2";
    let vectors = generate_test_vectors(n, dim);
    insert_test_data(&mut db, "consistency", &vectors, metric, dim);

    let query = &vectors[0].1;
    let topk = 5;

    let mut rounds: Vec<Vec<(i64, f32)>> = Vec::new();
    for _ in 0..5 {
        let hits = db.search("consistency", query, topk).expect("search");
        let round: Vec<(i64, f32)> = hits.iter().map(|h| (h.id, h.distance)).collect();
        rounds.push(round);
    }

    // All 5 rounds must be identical
    for i in 1..5 {
        assert_eq!(
            rounds[0], rounds[i],
            "round {i} should match round 0 for deterministic query"
        );
    }
}

#[test]
fn zvec_parity_typed_query_consistency_same_results_five_times() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let n = 20;
    let dim = 4;
    let metric = "l2";
    let vectors = generate_test_vectors(n, dim);
    insert_test_data(&mut db, "typed_consistency", &vectors, metric, dim);

    let query = vectors[0].1.clone();
    let topk = 5;

    let mut rounds: Vec<Vec<(i64, f32)>> = Vec::new();
    for _ in 0..5 {
        let hits = db
            .query_with_context(
                "typed_consistency",
                &QueryContext {
                    top_k: topk,
                    queries: vec![VectorQuery {
                        field_name: "vector".to_string(),
                        vector: QueryVector::Dense(query.clone()),
                        param: None,
                    }],
                    query_by_id: None,
                    query_by_id_field_name: None,
                    filter: None,
                    output_fields: None,
                    include_vector: false,
                    group_by: None,
                    reranker: None,
                    order_by: None,
                },
            )
            .expect("typed query");
        let round: Vec<(i64, f32)> = hits.iter().map(|h| (h.id, h.distance)).collect();
        rounds.push(round);
    }

    for i in 1..5 {
        assert_eq!(
            rounds[0], rounds[i],
            "typed query round {i} should match round 0"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests: Topk edge cases
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_topk_one_returns_single_result() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let n = 20;
    let dim = 4;
    let metric = "l2";
    let vectors = generate_test_vectors(n, dim);
    insert_test_data(&mut db, "topk1", &vectors, metric, dim);

    let query = &vectors[0].1;
    let hits = db.search("topk1", query, 1).expect("search topk=1");
    assert_eq!(hits.len(), 1, "topk=1 should return exactly 1 result");

    // The nearest neighbor should be the query vector itself (distance 0)
    assert_eq!(hits[0].id, vectors[0].0);
    assert!(hits[0].distance.abs() < 1e-6);
}

#[test]
fn zvec_parity_topk_exceeds_record_count_returns_all_records() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let n = 15;
    let dim = 4;
    let metric = "l2";
    let vectors = generate_test_vectors(n, dim);
    insert_test_data(&mut db, "topk_overflow", &vectors, metric, dim);

    let query = &vectors[0].1;
    let topk = 1000; // far more than n=15
    let hits = db
        .search("topk_overflow", query, topk)
        .expect("search topk >> n");
    assert_eq!(
        hits.len(),
        n,
        "topk > record_count should return all {n} records"
    );

    // Verify sorted by distance ascending
    for window in hits.windows(2) {
        assert!(
            window[0].distance <= window[1].distance,
            "results should be sorted by distance: {} !<= {}",
            window[0].distance,
            window[1].distance
        );
    }
}

// ---------------------------------------------------------------------------
// Tests: Filter that excludes all documents
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_filter_excludes_all_returns_empty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let dim = 4;
    let metric = "l2";
    let schema = CollectionSchema::new(
        "vector",
        dim,
        metric,
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    db.create_collection_with_schema("filter_empty", &schema)
        .expect("create collection");

    let vectors = generate_test_vectors(10, dim);
    let docs: Vec<Document> = vectors
        .iter()
        .map(|(id, vec)| {
            Document::new(
                *id,
                [("group".to_string(), FieldValue::Int64(*id % 3))],
                vec.clone(),
            )
        })
        .collect();
    db.insert_documents("filter_empty", &docs).expect("insert");

    // Filter that matches no documents: group == 999
    let hits = db
        .query_with_context(
            "filter_empty",
            &QueryContext {
                top_k: 10,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vectors[0].1.clone()),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("group == 999".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("query with exclusive filter");

    assert!(
        hits.is_empty(),
        "filter that excludes all should return empty result"
    );
}

// ---------------------------------------------------------------------------
// Tests: Recall with filter — filtered ground truth matches
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_filtered_recall_matches_ground_truth() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let n = 50;
    let dim = 8;
    let topk = 5;
    let metric = "l2";
    let schema = CollectionSchema::new(
        "vector",
        dim,
        metric,
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    db.create_collection_with_schema("filtered_recall", &schema)
        .expect("create collection");

    let vectors = generate_test_vectors(n, dim);
    let docs: Vec<Document> = vectors
        .iter()
        .map(|(id, vec)| {
            Document::new(
                *id,
                [("group".to_string(), FieldValue::Int64(*id % 3))],
                vec.clone(),
            )
        })
        .collect();
    db.insert_documents("filtered_recall", &docs)
        .expect("insert");

    // Query: only group == 0 documents
    let query = &vectors[0].1;
    let filtered_vectors: Vec<(i64, Vec<f32>)> = vectors
        .iter()
        .filter(|(id, _)| id % 3 == 0)
        .map(|(id, vec)| (*id, vec.clone()))
        .collect();

    let expected = brute_force_search(&filtered_vectors, query, topk, metric);

    let hits = db
        .query_with_context(
            "filtered_recall",
            &QueryContext {
                top_k: topk,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(query.clone()),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("group == 0".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("filtered query");

    let actual: Vec<(i64, f32)> = hits.iter().map(|h| (h.id, h.distance)).collect();
    let recall = compute_recall(&expected, &actual, topk);

    // All returned docs should have group == 0
    for hit in &hits {
        assert_eq!(
            hit.fields.get("group"),
            Some(&FieldValue::Int64(0)),
            "all hits should be in group 0"
        );
    }

    assert!(
        recall >= 0.8,
        "filtered recall should be >= 0.8, got {recall}"
    );
}

// ---------------------------------------------------------------------------
// Tests: Distance ordering correctness across metrics
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_distances_sorted_ascending_l2() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let vectors = generate_test_vectors(30, 8);
    insert_test_data(&mut db, "sorted_l2", &vectors, "l2", 8);

    let query = &vectors[0].1;
    let hits = db.search("sorted_l2", query, 10).expect("search");

    for window in hits.windows(2) {
        assert!(
            window[0].distance <= window[1].distance + 1e-6,
            "L2 distances should be sorted ascending: {} > {}",
            window[0].distance,
            window[1].distance
        );
    }
}

#[test]
fn zvec_parity_distances_sorted_ascending_cosine() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let vectors = generate_test_vectors(30, 8);
    insert_test_data(&mut db, "sorted_cosine", &vectors, "cosine", 8);

    let query = &vectors[0].1;
    let hits = db.search("sorted_cosine", query, 10).expect("search");

    for window in hits.windows(2) {
        assert!(
            window[0].distance <= window[1].distance + 1e-6,
            "Cosine distances should be sorted ascending: {} > {}",
            window[0].distance,
            window[1].distance
        );
    }
}

#[test]
fn zvec_parity_distances_sorted_ascending_ip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let vectors = generate_test_vectors(30, 8);
    insert_test_data(&mut db, "sorted_ip", &vectors, "ip", 8);

    let query = &vectors[0].1;
    let hits = db.search("sorted_ip", query, 10).expect("search");

    // IP: distance = -dot, so ascending distance means descending dot product
    for window in hits.windows(2) {
        assert!(
            window[0].distance <= window[1].distance + 1e-6,
            "IP distances should be sorted ascending (most similar first): {} > {}",
            window[0].distance,
            window[1].distance
        );
    }
}

// ---------------------------------------------------------------------------
// Tests: Brute-force ground truth matches exactly for small dataset
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_brute_force_matches_search_for_l2() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let vectors = generate_test_vectors(20, 4);
    insert_test_data(&mut db, "exact_l2", &vectors, "l2", 4);

    let query = &vectors[5].1;
    let topk = 5;
    let expected = brute_force_search(&vectors, query, topk, "l2");
    let hits = db.search("exact_l2", query, topk).expect("search");

    let actual: Vec<(i64, f32)> = hits.iter().map(|h| (h.id, h.distance)).collect();

    assert_eq!(
        expected, actual,
        "brute-force L2 ground truth should match search results exactly"
    );
}

#[test]
fn zvec_parity_brute_force_matches_search_for_cosine() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let vectors = generate_test_vectors(20, 4);
    insert_test_data(&mut db, "exact_cosine", &vectors, "cosine", 4);

    let query = &vectors[5].1;
    let topk = 5;
    let expected = brute_force_search(&vectors, query, topk, "cosine");
    let hits = db.search("exact_cosine", query, topk).expect("search");

    let actual: Vec<(i64, f32)> = hits.iter().map(|h| (h.id, h.distance)).collect();

    assert_eq!(
        expected, actual,
        "brute-force Cosine ground truth should match search results exactly"
    );
}

#[test]
fn zvec_parity_brute_force_matches_search_for_ip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");

    let vectors = generate_test_vectors(20, 4);
    insert_test_data(&mut db, "exact_ip", &vectors, "ip", 4);

    let query = &vectors[5].1;
    let topk = 5;
    let expected = brute_force_search(&vectors, query, topk, "ip");
    let hits = db.search("exact_ip", query, topk).expect("search");

    let actual: Vec<(i64, f32)> = hits.iter().map(|h| (h.id, h.distance)).collect();

    assert_eq!(
        expected, actual,
        "brute-force IP ground truth should match search results exactly"
    );
}
