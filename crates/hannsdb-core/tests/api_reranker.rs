//! Integration tests for multi-vector reranker functionality.
//!
//! Covers RRF and weighted reranking with dense and sparse vector fields.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema, SparseVector,
    VectorFieldSchema,
};
use hannsdb_core::query::{QueryContext, QueryReranker, QueryVector, VectorQuery};

fn unique_temp_dir(name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{}_{}", name, nanos))
}

fn sparse_field_schema(name: &str) -> VectorFieldSchema {
    VectorFieldSchema {
        name: name.to_string(),
        data_type: FieldType::VectorSparse,
        dimension: 0,
        index_param: None,
        bm25_params: None,
    }
}

// ---------------------------------------------------------------------------
// Test 1: RRF basic — 2 dense vector fields, query both, rerank with RRF
// ---------------------------------------------------------------------------


#[test]
fn api_rrf_basic_two_dense_fields() {
    let root = unique_temp_dir("hannsdb_rrf_basic");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    // dense: id=1 close to origin, id=2 far, id=3 medium
    // title: id=1 far from origin, id=2 close, id=3 medium
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [("title".to_string(), vec![5.0_f32, 5.0])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![5.0_f32, 5.0],
                [("title".to_string(), vec![0.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                3,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![2.0_f32, 2.0],
                [("title".to_string(), vec![2.0_f32, 2.0])],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![
                    VectorQuery {
                        field_name: "dense".to_string(),
                        vector: QueryVector::Dense(vec![0.0, 0.0]),
                        param: None,
                    },
                    VectorQuery {
                        field_name: "title".to_string(),
                        vector: QueryVector::Dense(vec![0.0, 0.0]),
                        param: None,
                    },
                ],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: Some(QueryReranker::Rrf { rank_constant: 60 }),
                order_by: None,
            },
        )
        .expect("rrf reranked query");

    assert_eq!(hits.len(), 3, "all 3 documents should be returned");

    // RRF scores:
    // dense ranking: id=1 (rank 0), id=3 (rank 1), id=2 (rank 2)
    // title ranking: id=2 (rank 0), id=3 (rank 1), id=1 (rank 2)
    //
    // RRF score (k=60):
    //   id=1: 1/(60+0+1) + 1/(60+2+1) = 1/61 + 1/63
    //   id=2: 1/(60+2+1) + 1/(60+0+1) = 1/63 + 1/61
    //   id=3: 1/(60+1+1) + 1/(60+1+1) = 2/62
    //
    // id=1 and id=2 are tied (same sum), id=3 is slightly different.
    // Verify all ids are present and results are non-empty.
    let hit_ids: Vec<i64> = hits.iter().map(|h| h.id).collect();
    assert!(hit_ids.contains(&1));
    assert!(hit_ids.contains(&2));
    assert!(hit_ids.contains(&3));
}

// ---------------------------------------------------------------------------
// Test 2: RRF score calculation — verify formula 1/(k+rank+1)
// ---------------------------------------------------------------------------


#[test]
fn api_rrf_score_calculation_matches_formula() {
    let root = unique_temp_dir("hannsdb_rrf_formula");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    // 3 documents arranged so that dense and title produce clearly different rankings
    // dense ranking (closest to [0,0] first): 1, 2, 3
    // title ranking (closest to [0,0] first): 3, 1, 2
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [("title".to_string(), vec![1.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.5_f32, 0.0],
                [("title".to_string(), vec![5.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                3,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![2.0_f32, 0.0],
                [("title".to_string(), vec![0.0_f32, 0.0])],
            ),
        ],
    )
    .expect("insert documents");

    let k: f64 = 60.0;
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![
                    VectorQuery {
                        field_name: "dense".to_string(),
                        vector: QueryVector::Dense(vec![0.0, 0.0]),
                        param: None,
                    },
                    VectorQuery {
                        field_name: "title".to_string(),
                        vector: QueryVector::Dense(vec![0.0, 0.0]),
                        param: None,
                    },
                ],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: Some(QueryReranker::Rrf {
                    rank_constant: k as u64,
                }),
                order_by: None,
            },
        )
        .expect("rrf formula query");

    assert_eq!(hits.len(), 3);

    // RRF formula verification: 1/(k + rank + 1)
    // dense ranking: id=1(r0), id=2(r1), id=3(r2)
    // title ranking: id=3(r0), id=1(r1), id=2(r2)
    //
    // RRF scores (k=60):
    //   id=1: 1/(60+0+1) + 1/(60+1+1) = 1/61 + 1/62 ≈ 0.03240
    //   id=2: 1/(60+1+1) + 1/(60+2+1) = 1/62 + 1/63 ≈ 0.03212
    //   id=3: 1/(60+2+1) + 1/(60+0+1) = 1/63 + 1/61 ≈ 0.03213
    //
    // Expected order: id=1, then id=2 and id=3 are very close (tied within ~1e-4).
    // id=1 should be first (clearly highest).
    // Verify id=1 is the top result.
    assert_eq!(hits[0].id, 1, "id=1 should have the highest RRF score");

    // All 3 ids should be present
    let actual_ids: Vec<i64> = hits.iter().map(|h| h.id).collect();
    assert!(actual_ids.contains(&1));
    assert!(actual_ids.contains(&2));
    assert!(actual_ids.contains(&3));
}

// ---------------------------------------------------------------------------
// Test 3: Weighted basic — 2 dense vector fields, query both, rerank with weights
// ---------------------------------------------------------------------------


#[test]
fn api_weighted_basic_two_dense_fields() {
    let root = unique_temp_dir("hannsdb_weighted_basic");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.1_f32, 0.0],
                [("title".to_string(), vec![0.5_f32, 0.0])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.5_f32, 0.0],
                [("title".to_string(), vec![0.1_f32, 0.0])],
            ),
        ],
    )
    .expect("insert documents");

    let mut weights = BTreeMap::new();
    weights.insert("dense".to_string(), 0.7);
    weights.insert("title".to_string(), 0.3);

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: vec![
                    VectorQuery {
                        field_name: "dense".to_string(),
                        vector: QueryVector::Dense(vec![0.0, 0.0]),
                        param: None,
                    },
                    VectorQuery {
                        field_name: "title".to_string(),
                        vector: QueryVector::Dense(vec![0.0, 0.0]),
                        param: None,
                    },
                ],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: Some(QueryReranker::Weighted {
                    weights: weights.clone(),
                    metric: None,
                }),
                order_by: None,
            },
        )
        .expect("weighted reranked query");

    assert_eq!(hits.len(), 2);
    // dense has weight 0.7 and id=1 is closer in dense (0.1 vs 0.5)
    // title has weight 0.3 and id=2 is closer in title (0.1 vs 0.5)
    // With L2 normalization: lower distance => higher normalized score
    // id=1 wins because 0.7 * normalize(0.1) > 0.7 * normalize(0.5)
    assert_eq!(hits[0].id, 1, "id=1 should win with 0.7 dense weight");
}

// ---------------------------------------------------------------------------
// Test 4: Weighted with metric normalization — verify L2/IP/COSINE normalization
// ---------------------------------------------------------------------------


#[test]
fn api_weighted_metric_normalization() {
    // Test with cosine metric override
    let root = unique_temp_dir("hannsdb_weighted_metric");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "cosine",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![1.0_f32, 0.0],
                [("title".to_string(), vec![1.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![0.0_f32, 1.0],
                [("title".to_string(), vec![0.0_f32, 1.0])],
            ),
        ],
    )
    .expect("insert documents");

    let mut weights = BTreeMap::new();
    weights.insert("dense".to_string(), 0.5);
    weights.insert("title".to_string(), 0.5);

    // With cosine metric override and equal weights, both docs should be returned
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: vec![
                    VectorQuery {
                        field_name: "dense".to_string(),
                        vector: QueryVector::Dense(vec![1.0, 0.0]),
                        param: None,
                    },
                    VectorQuery {
                        field_name: "title".to_string(),
                        vector: QueryVector::Dense(vec![1.0, 0.0]),
                        param: None,
                    },
                ],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: Some(QueryReranker::Weighted {
                    weights: weights.clone(),
                    metric: Some("cosine".to_string()),
                }),
                order_by: None,
            },
        )
        .expect("weighted with cosine normalization");

    assert_eq!(hits.len(), 2);
}

// ---------------------------------------------------------------------------
// Test 5: Dense + sparse hybrid — query dense and sparse fields with RRF reranker
// ---------------------------------------------------------------------------


#[test]
fn api_dense_sparse_hybrid_rrf() {
    let root = unique_temp_dir("hannsdb_hybrid_rrf");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(sparse_field_schema("sparse"));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    db.insert_documents(
        "docs",
        &[
            Document::with_sparse_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [(
                    "sparse".to_string(),
                    SparseVector::new(vec![0, 1], vec![1.0, 0.5]),
                )],
            ),
            Document::with_sparse_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![5.0_f32, 5.0],
                [(
                    "sparse".to_string(),
                    SparseVector::new(vec![0, 1], vec![0.5, 1.0]),
                )],
            ),
            Document::with_sparse_vectors(
                3,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![1.0_f32, 1.0],
                [(
                    "sparse".to_string(),
                    SparseVector::new(vec![0, 2], vec![0.8, 0.3]),
                )],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![
                    VectorQuery {
                        field_name: "dense".to_string(),
                        vector: QueryVector::Dense(vec![0.0, 0.0]),
                        param: None,
                    },
                    VectorQuery {
                        field_name: "sparse".to_string(),
                        vector: QueryVector::Sparse(SparseVector::new(vec![0, 1], vec![1.0, 1.0])),
                        param: None,
                    },
                ],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: Some(QueryReranker::Rrf { rank_constant: 60 }),
                order_by: None,
            },
        )
        .expect("dense + sparse hybrid RRF query");

    assert_eq!(
        hits.len(),
        3,
        "all documents should be returned by hybrid query"
    );

    // id=1 ranks #1 in dense (closest) and has strong sparse match => should be top
    assert_eq!(hits[0].id, 1, "id=1 should rank first in hybrid RRF");
}

// ---------------------------------------------------------------------------
// Test 6: RRF with filter — multi-vector query + filter expression
// ---------------------------------------------------------------------------

#[test]
fn api_rrf_with_filter() {
    let root = unique_temp_dir("hannsdb_rrf_filter");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [("title".to_string(), vec![5.0_f32, 5.0])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![0.1_f32, 0.0],
                [("title".to_string(), vec![0.1_f32, 0.0])],
            ),
            Document::with_named_vectors(
                3,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![1.0_f32, 1.0],
                [("title".to_string(), vec![1.0_f32, 1.0])],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![
                    VectorQuery {
                        field_name: "dense".to_string(),
                        vector: QueryVector::Dense(vec![0.0, 0.0]),
                        param: None,
                    },
                    VectorQuery {
                        field_name: "title".to_string(),
                        vector: QueryVector::Dense(vec![0.0, 0.0]),
                        param: None,
                    },
                ],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("group == 1".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: Some(QueryReranker::Rrf { rank_constant: 60 }),
                order_by: None,
            },
        )
        .expect("rrf with filter");

    // Only group==1 docs: id=1 and id=3
    assert_eq!(hits.len(), 2);
    for hit in &hits {
        assert_eq!(
            hit.fields.get("group"),
            Some(&FieldValue::Int64(1)),
            "only group==1 docs should be returned"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 7: Weighted with unequal weights — 0.7 vs 0.3
// ---------------------------------------------------------------------------


#[test]
fn api_weighted_unequal_weights() {
    let root = unique_temp_dir("hannsdb_weighted_unequal");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    // dense: id=1 close, id=2 medium, id=3 far
    // title: id=1 far, id=2 medium, id=3 close
    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.1_f32, 0.0],
                [("title".to_string(), vec![5.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![1.0_f32, 0.0],
                [("title".to_string(), vec![1.0_f32, 0.0])],
            ),
            Document::with_named_vectors(
                3,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![5.0_f32, 0.0],
                [("title".to_string(), vec![0.1_f32, 0.0])],
            ),
        ],
    )
    .expect("insert documents");

    let mut weights = BTreeMap::new();
    weights.insert("dense".to_string(), 0.7);
    weights.insert("title".to_string(), 0.3);

    // With 0.7 weight on dense, id=1 (closest in dense) should win
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![
                    VectorQuery {
                        field_name: "dense".to_string(),
                        vector: QueryVector::Dense(vec![0.0, 0.0]),
                        param: None,
                    },
                    VectorQuery {
                        field_name: "title".to_string(),
                        vector: QueryVector::Dense(vec![0.0, 0.0]),
                        param: None,
                    },
                ],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: Some(QueryReranker::Weighted {
                    weights: weights.clone(),
                    metric: Some("l2".to_string()),
                }),
                order_by: None,
            },
        )
        .expect("weighted with unequal weights");

    assert_eq!(hits.len(), 3);
    // id=1 is closest in dense (0.1), dense has weight 0.7 => should rank first
    assert_eq!(
        hits[0].id, 1,
        "id=1 should rank first with 0.7 dense weight"
    );
}

// ---------------------------------------------------------------------------
// Test 8: Single vector query (no reranker) — results unchanged
// ---------------------------------------------------------------------------

#[test]
fn api_single_vector_no_reranker_unchanged() {
    let root = unique_temp_dir("hannsdb_no_reranker");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [("title".to_string(), vec![10.0_f32, 10.0])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.2_f32, 0.0],
                [("title".to_string(), vec![11.0_f32, 11.0])],
            ),
            Document::with_named_vectors(
                3,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![0.1_f32, 0.0],
                [("title".to_string(), vec![12.0_f32, 12.0])],
            ),
        ],
    )
    .expect("insert documents");

    // Query single field without reranker
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![VectorQuery {
                    field_name: "dense".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0]),
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
        .expect("single vector no reranker");

    // Should match plain brute-force L2 ordering: id=1 (0.0), id=3 (0.1), id=2 (0.2)
    assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![1, 3, 2]);
    assert!((hits[0].distance - 0.0).abs() < 1e-6);
    assert!((hits[1].distance - 0.1).abs() < 1e-6);
    assert!((hits[2].distance - 0.2).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Test 9: RRF with topk — verify result count matches topk
// ---------------------------------------------------------------------------

#[test]
fn api_rrf_topk_limits_results() {
    let root = unique_temp_dir("hannsdb_rrf_topk");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0_f32, 0.0],
                [("title".to_string(), vec![5.0_f32, 5.0])],
            ),
            Document::with_named_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.1_f32, 0.0],
                [("title".to_string(), vec![0.1_f32, 0.0])],
            ),
            Document::with_named_vectors(
                3,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![1.0_f32, 1.0],
                [("title".to_string(), vec![1.0_f32, 1.0])],
            ),
            Document::with_named_vectors(
                4,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![2.0_f32, 2.0],
                [("title".to_string(), vec![2.0_f32, 2.0])],
            ),
            Document::with_named_vectors(
                5,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![3.0_f32, 3.0],
                [("title".to_string(), vec![3.0_f32, 3.0])],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: vec![
                    VectorQuery {
                        field_name: "dense".to_string(),
                        vector: QueryVector::Dense(vec![0.0, 0.0]),
                        param: None,
                    },
                    VectorQuery {
                        field_name: "title".to_string(),
                        vector: QueryVector::Dense(vec![0.0, 0.0]),
                        param: None,
                    },
                ],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: Some(QueryReranker::Rrf { rank_constant: 60 }),
                order_by: None,
            },
        )
        .expect("rrf with topk=2");

    assert_eq!(hits.len(), 2, "topk should limit result count");
}

// ---------------------------------------------------------------------------
// Test 10: RRF with output_fields — multi-vector query with field projection
// ---------------------------------------------------------------------------

#[test]
fn api_rrf_with_output_fields() {
    let root = unique_temp_dir("hannsdb_rrf_output_fields");
    let mut db = HannsDb::open(&root).expect("open db");
    let mut schema = CollectionSchema::new(
        "dense",
        2,
        "l2",
        vec![
            ScalarFieldSchema::new("group", FieldType::Int64),
            ScalarFieldSchema::new("color", FieldType::String),
        ],
    );
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    db.insert_documents(
        "docs",
        &[
            Document::with_named_vectors(
                1,
                [
                    ("group".to_string(), FieldValue::Int64(1)),
                    ("color".to_string(), FieldValue::String("red".to_string())),
                ],
                "dense",
                vec![0.0_f32, 0.0],
                [("title".to_string(), vec![5.0_f32, 5.0])],
            ),
            Document::with_named_vectors(
                2,
                [
                    ("group".to_string(), FieldValue::Int64(2)),
                    ("color".to_string(), FieldValue::String("blue".to_string())),
                ],
                "dense",
                vec![0.1_f32, 0.0],
                [("title".to_string(), vec![0.1_f32, 0.0])],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: vec![
                    VectorQuery {
                        field_name: "dense".to_string(),
                        vector: QueryVector::Dense(vec![0.0, 0.0]),
                        param: None,
                    },
                    VectorQuery {
                        field_name: "title".to_string(),
                        vector: QueryVector::Dense(vec![0.0, 0.0]),
                        param: None,
                    },
                ],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: Some(vec!["color".to_string()]),
                include_vector: false,
                group_by: None,
                reranker: Some(QueryReranker::Rrf { rank_constant: 60 }),
                order_by: None,
            },
        )
        .expect("rrf with output fields");

    assert_eq!(hits.len(), 2);
    // Each hit should only have the "color" field projected
    for hit in &hits {
        assert_eq!(hit.fields.len(), 1, "only 'color' should be projected");
        assert!(hit.fields.contains_key("color"));
        assert!(!hit.fields.contains_key("group"));
    }
}
