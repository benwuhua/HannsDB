/// Integration tests ported from zvec's filter expression and query tests.
///
/// Source references:
///   zvec/python/tests/detail/test_collection_dql.py  (AND/OR, parentheses, IN, NOT IN, topk, output_fields)
///   zvec/python/tests/test_collection.py              (TestCollectionQuery)
///
/// Adapted to HannsDB's Rust API: `query_with_context` with `QueryContext`, and
/// `query_documents` for filter-only queries.

use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema,
};
use hannsdb_core::query::{QueryContext, QueryVector, VectorQuery};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build the standard schema used by most tests in this file.
///
/// Fields: `id` (Int64), `name` (String), `age` (Int64), `active` (Bool),
/// `score` (Float64).
fn standard_schema() -> CollectionSchema {
    CollectionSchema::new(
        "vector",
        4,
        "l2",
        vec![
            ScalarFieldSchema::new("id", FieldType::Int64),
            ScalarFieldSchema::new("name", FieldType::String),
            ScalarFieldSchema::new("age", FieldType::Int64),
            ScalarFieldSchema::new("active", FieldType::Bool),
            ScalarFieldSchema::new("score", FieldType::Float64),
        ],
    )
}

/// Insert a standard set of 15 documents with varied field values.
fn insert_standard_docs(db: &mut HannsDb, coll: &str) {
    let docs: Vec<Document> = (0..15)
        .map(|i| {
            let name = if i % 3 == 0 {
                "alice"
            } else if i % 3 == 1 {
                "bob"
            } else {
                "carol"
            };
            Document::new(
                i as i64,
                [
                    ("id".to_string(), FieldValue::Int64(i)),
                    ("name".to_string(), FieldValue::String(name.to_string())),
                    ("age".to_string(), FieldValue::Int64(20 + i as i64)),
                    ("active".to_string(), FieldValue::Bool(i % 2 == 0)),
                    ("score".to_string(), FieldValue::Float64(i as f64 * 0.5)),
                ],
                vec![i as f32 * 0.1, 0.0, 0.0, 0.0],
            )
        })
        .collect();
    db.insert_documents(coll, &docs).expect("insert docs");
}

// ---------------------------------------------------------------------------
// 1. Filter with AND
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_filter_and_two_conditions() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // age > 25 and active == true
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("age > 25 and active == true".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("filter AND query");

    for hit in &hits {
        let age = hit.fields.get("age").expect("age field");
        let active = hit.fields.get("active").expect("active field");
        assert!(
            matches!(age, FieldValue::Int64(v) if *v > 25),
            "age should be > 25, got {:?}",
            age
        );
        assert!(
            matches!(active, FieldValue::Bool(true)),
            "active should be true, got {:?}",
            active
        );
    }
    // age > 25 means i >= 6. active == true means i % 2 == 0.
    // i >= 6 and even: 6, 8, 10, 12, 14 => 5 hits
    assert_eq!(hits.len(), 5);
}

// ---------------------------------------------------------------------------
// 2. Filter with OR
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_filter_or_two_conditions() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // age < 22 or age > 33
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("age < 22 or age > 33".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("filter OR query");

    // age < 22 means i < 2 => i=0,1. age > 33 means i > 13 => i=14.
    assert_eq!(hits.len(), 3);
    let hit_ids: std::collections::HashSet<i64> =
        hits.iter().map(|h| h.id).collect();
    assert!(hit_ids.contains(&0));
    assert!(hit_ids.contains(&1));
    assert!(hit_ids.contains(&14));
}

// ---------------------------------------------------------------------------
// 3. Filter with parentheses (AND + OR combined)
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_filter_parentheses_and_or_combined() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // (age < 22 or age > 33) and active == false
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some(
                    "(age < 22 or age > 33) and active == false".to_string(),
                ),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("filter parentheses query");

    // OR matches: i=0,1,14. active==false means i%2==1 => i=1 only.
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 1);
}

#[test]
fn zvec_parity_filter_parentheses_or_and_combined() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // (age > 22 and age < 26) or (age > 32 and active == true)
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some(
                    "(age > 22 and age < 26) or (age > 32 and active == true)"
                        .to_string(),
                ),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("filter parentheses or-and query");

    // First group: age>22 and age<26 => 22<age<26 => i in {3,4,5}.
    // Second group: age>32 and active==true => i>12 and i%2==0 => i=14.
    // Total: 3,4,5,14 => 4 hits
    assert_eq!(hits.len(), 4);
}

// ---------------------------------------------------------------------------
// 4. Filter with IN
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_filter_in_list_integers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // id in (1, 2, 3)
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("id in (1, 2, 3)".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("filter IN query");

    assert_eq!(hits.len(), 3);
    let hit_ids: std::collections::HashSet<i64> =
        hits.iter().map(|h| h.id).collect();
    assert!(hit_ids.contains(&1));
    assert!(hit_ids.contains(&2));
    assert!(hit_ids.contains(&3));
}

// ---------------------------------------------------------------------------
// 5. Filter with NOT IN
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_filter_not_in_list_integers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // id not in (0, 1, 2) with topk=5
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 5,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("id not in (0, 1, 2)".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("filter NOT IN query");

    // 15 docs total, 3 excluded by NOT IN => 12 remain, but topk=5
    assert_eq!(hits.len(), 5);
    for hit in &hits {
        assert!(hit.id > 2, "id should not be 0,1,2, got {}", hit.id);
    }
}

// ---------------------------------------------------------------------------
// 6. Filter with has_prefix / has_suffix
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_filter_has_prefix() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // name has_prefix "al" => only "alice" docs (i % 3 == 0)
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("name has_prefix \"al\"".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("filter has_prefix query");

    // i % 3 == 0 for i in 0..15: 0,3,6,9,12 => 5 hits
    assert_eq!(hits.len(), 5);
    for hit in &hits {
        let name = hit.fields.get("name").unwrap();
        assert!(
            matches!(name, FieldValue::String(s) if s.starts_with("al")),
            "name should start with 'al', got {:?}",
            name
        );
    }
}

#[test]
fn zvec_parity_filter_has_suffix() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // name has_suffix "ol" => "carol" docs (i % 3 == 2)
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("name has_suffix \"ol\"".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("filter has_suffix query");

    // i % 3 == 2 for i in 0..15: 2,5,8,11,14 => 5 hits
    assert_eq!(hits.len(), 5);
    for hit in &hits {
        let name = hit.fields.get("name").unwrap();
        assert!(
            matches!(name, FieldValue::String(s) if s.ends_with("ol")),
            "name should end with 'ol', got {:?}",
            name
        );
    }
}

// ---------------------------------------------------------------------------
// 7. Filter with LIKE (pattern matching with %)
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_filter_like_contains() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // name like "%li%" => "alice" docs (contains "li")
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("name like \"%li%\"".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("filter LIKE query");

    // "alice" contains "li" => 5 hits (i % 3 == 0)
    assert_eq!(hits.len(), 5);
}

#[test]
fn zvec_parity_filter_like_suffix_pattern() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // name like "%b" => "bob" ends with "b"
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("name like \"%b\"".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("filter LIKE suffix query");

    // "bob" ends with "b" => i % 3 == 1: 1,4,7,10,13 => 5 hits
    assert_eq!(hits.len(), 5);
}

#[test]
fn zvec_parity_filter_not_like() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // name not like "%li%" => excludes "alice" (5 docs) => 10 remaining
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("name not like \"%li%\"".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("filter NOT LIKE query");

    assert_eq!(hits.len(), 10);
}

// ---------------------------------------------------------------------------
// 8. Query with output_fields (select specific fields)
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_query_output_fields_projection() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: Some(vec!["name".to_string(), "age".to_string()]),
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("output fields query");

    assert_eq!(hits.len(), 3);
    for hit in &hits {
        // Should only contain "name" and "age"
        assert_eq!(hit.fields.len(), 2);
        assert!(hit.fields.contains_key("name"));
        assert!(hit.fields.contains_key("age"));
        // Should NOT contain "id", "active", "score"
        assert!(!hit.fields.contains_key("id"));
        assert!(!hit.fields.contains_key("active"));
        assert!(!hit.fields.contains_key("score"));
    }
}

// ---------------------------------------------------------------------------
// 9. Query with include_vector=true
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_query_include_vector() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: None,
                output_fields: None,
                include_vector: true,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("include_vector query");

    assert_eq!(hits.len(), 3);
    for hit in &hits {
        let vec = hit
            .vectors
            .get("vector")
            .unwrap_or_else(|| panic!("expected vector for id {}", hit.id));
        assert_eq!(vec.len(), 4, "vector dimension should be 4");
    }
}

#[test]
fn zvec_parity_query_exclude_vector_by_default() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
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
        .expect("exclude_vector query");

    for hit in &hits {
        assert!(
            hit.vectors.is_empty(),
            "vectors should be empty when include_vector=false, got {:?} for id {}",
            hit.vectors,
            hit.id
        );
    }
}

// ---------------------------------------------------------------------------
// 10. Query with topk=1, topk=5, topk=100
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_query_topk_variations() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // topk=1
    let hits1 = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 1,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
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
        .expect("topk=1 query");
    assert_eq!(hits1.len(), 1);

    // topk=5
    let hits5 = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 5,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
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
        .expect("topk=5 query");
    assert_eq!(hits5.len(), 5);

    // topk=100 (more than total docs)
    let hits100 = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
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
        .expect("topk=100 query");
    assert_eq!(hits100.len(), 15); // capped at total doc count
}

// ---------------------------------------------------------------------------
// 11. Query returning empty result (no match)
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_query_empty_result_impossible_and() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // age >= 10 and age <= 5 => impossible range
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("age >= 10 and age <= 5".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("empty result query");

    assert!(hits.is_empty(), "should return no results for impossible filter");
}

#[test]
fn zvec_parity_query_empty_result_incompatible_equals() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // age == 3 and age == 8 => impossible
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("age == 3 and age == 8".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("empty result incompatible equals");

    assert!(hits.is_empty());
}

// ---------------------------------------------------------------------------
// 12. Query consistency (same query returns same results)
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_query_consistency_repeated_identical_queries() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    let query = QueryContext {
        top_k: 5,
        queries: vec![VectorQuery {
            field_name: "vector".to_string(),
            vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
            param: None,
        }],
        query_by_id: None,
        query_by_id_field_name: None,
        filter: Some("age >= 23 and age <= 27".to_string()),
        output_fields: None,
        include_vector: false,
        group_by: None,
        reranker: None,
        order_by: None,
    };

    let mut all_results = Vec::new();
    for _ in 0..5 {
        let hits = db
            .query_with_context("docs", &query)
            .expect("consistency query");
        all_results.push(hits);
    }

    let expected_count = all_results[0].len();
    for (i, result) in all_results.iter().enumerate() {
        assert_eq!(
            result.len(),
            expected_count,
            "result count mismatch at iteration {}",
            i
        );
    }

    // All iterations should return the same set of ids
    let expected_ids: std::collections::HashSet<i64> =
        all_results[0].iter().map(|h| h.id).collect();
    for (i, result) in all_results.iter().enumerate() {
        let ids: std::collections::HashSet<i64> =
            result.iter().map(|h| h.id).collect();
        assert_eq!(ids, expected_ids, "id set mismatch at iteration {}", i);
    }
}

// ---------------------------------------------------------------------------
// 13. Filter with all comparison operators (==, !=, <, <=, >, >=)
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_filter_all_comparison_operators() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    let query_vec = vec![VectorQuery {
        field_name: "vector".to_string(),
        vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
        param: None,
    }];

    // == : id == 7
    let hits_eq = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: query_vec.clone(),
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("id == 7".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("eq query");
    assert_eq!(hits_eq.len(), 1);
    assert_eq!(hits_eq[0].id, 7);

    // != : id != 0 (topk=3 to spot-check)
    let hits_ne = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: query_vec.clone(),
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("id != 0".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("ne query");
    assert_eq!(hits_ne.len(), 3);
    assert!(hits_ne.iter().all(|h| h.id != 0));

    // < : age < 22 => i < 2 => 0,1 => 2 hits
    let hits_lt = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: query_vec.clone(),
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("age < 22".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("lt query");
    assert_eq!(hits_lt.len(), 2);

    // <= : age <= 22 => i <= 2 => 0,1,2 => 3 hits
    let hits_lte = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: query_vec.clone(),
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("age <= 22".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("lte query");
    assert_eq!(hits_lte.len(), 3);

    // > : age > 33 => i > 13 => 14 => 1 hit
    let hits_gt = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: query_vec.clone(),
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("age > 33".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("gt query");
    assert_eq!(hits_gt.len(), 1);
    assert_eq!(hits_gt[0].id, 14);

    // >= : age >= 33 => i >= 13 => 13,14 => 2 hits
    let hits_gte = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: query_vec,
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("age >= 33".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("gte query");
    assert_eq!(hits_gte.len(), 2);
}

// ---------------------------------------------------------------------------
// 14. Filter on bool field
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_filter_bool_field_equals_true() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("active == true".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("bool filter query");

    // active == true when i % 2 == 0: 0,2,4,6,8,10,12,14 => 8 hits
    assert_eq!(hits.len(), 8);
    for hit in &hits {
        assert!(
            matches!(hit.fields.get("active"), Some(FieldValue::Bool(true))),
            "active should be true for id {}",
            hit.id
        );
    }
}

#[test]
fn zvec_parity_filter_bool_field_equals_false() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("active == false".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("bool false filter query");

    // active == false when i % 2 == 1: 1,3,5,7,9,11,13 => 7 hits
    assert_eq!(hits.len(), 7);
}

// ---------------------------------------------------------------------------
// 15. Filter on float field
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_filter_float_field_range() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // score >= 2.0 and score <= 4.0
    // score = i * 0.5, so 2.0 <= i*0.5 <= 4.0 => 4 <= i <= 8
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("score >= 2.0 and score <= 4.0".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("float range filter query");

    // 4 <= i <= 8: 4,5,6,7,8 => 5 hits
    assert_eq!(hits.len(), 5);
    for hit in &hits {
        let score = hit.fields.get("score").unwrap();
        match score {
            FieldValue::Float64(v) => {
                assert!(
                    *v >= 2.0 && *v <= 4.0,
                    "score should be in [2.0, 4.0], got {}",
                    v
                );
            }
            other => panic!("expected Float64, got {:?}", other),
        }
    }
}

#[test]
fn zvec_parity_filter_float_field_equals() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // score == 1.5 => i = 3
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("score == 1.5".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("float equals filter query");

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 3);
}

// ---------------------------------------------------------------------------
// Bonus: filter-only query (no vector recall, just scan + filter)
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_filter_only_query_returns_all_matching_docs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // filter-only: no queries, no query_by_id
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("active == true".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("filter-only query");

    // active == true: 8 hits (even i values 0..14)
    assert_eq!(hits.len(), 8);
    // filter-only queries return distance 0.0
    for hit in &hits {
        assert_eq!(hit.distance, 0.0);
    }
}

// ---------------------------------------------------------------------------
// Bonus: filter-only with output_fields projection
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_filter_only_query_with_output_fields() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 3,
                queries: vec![],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("active == true".to_string()),
                output_fields: Some(vec!["name".to_string()]),
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("filter-only output fields query");

    assert_eq!(hits.len(), 3);
    for hit in &hits {
        assert_eq!(hit.fields.len(), 1);
        assert!(hit.fields.contains_key("name"));
    }
}

// ---------------------------------------------------------------------------
// Bonus: combined output_fields + include_vector + filter
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_query_output_fields_include_vector_with_filter() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("active == false".to_string()),
                output_fields: Some(vec!["name".to_string()]),
                include_vector: true,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("combined output_fields + include_vector + filter");

    assert_eq!(hits.len(), 2);
    for hit in &hits {
        // output_fields projects to "name" only
        assert_eq!(hit.fields.len(), 1);
        assert!(hit.fields.contains_key("name"));
        // include_vector should populate vectors
        assert!(
            !hit.vectors.is_empty(),
            "vectors should be populated for id {}",
            hit.id
        );
        // "active" is not in output_fields, so it won't be in hit.fields.
        // The filter already ensures active==false at query time.
    }
}

// ---------------------------------------------------------------------------
// Bonus: IN list with string values
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_filter_in_list_strings() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // name in ("alice", "carol")
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("name in (\"alice\", \"carol\")".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("IN list strings query");

    // alice: i%3==0 (5 docs), carol: i%3==2 (5 docs) => 10 total
    assert_eq!(hits.len(), 10);
    for hit in &hits {
        let name = hit.fields.get("name").unwrap();
        match name {
            FieldValue::String(s) => {
                assert!(
                    s == "alice" || s == "carol",
                    "name should be alice or carol, got {}",
                    s
                );
            }
            other => panic!("expected String, got {:?}", other),
        }
    }
}

// ---------------------------------------------------------------------------
// Bonus: NOT operator
// ---------------------------------------------------------------------------

#[test]
fn zvec_parity_filter_not_operator() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = HannsDb::open(dir.path()).expect("open db");
    db.create_collection_with_schema("docs", &standard_schema())
        .expect("create collection");
    insert_standard_docs(&mut db, "docs");

    // not active == true => same as active != true
    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 100,
                queries: vec![VectorQuery {
                    field_name: "vector".to_string(),
                    vector: QueryVector::Dense(vec![0.0, 0.0, 0.0, 0.0]),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("not active == true".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("NOT operator query");

    // active != true: 7 hits (odd i values)
    assert_eq!(hits.len(), 7);
    for hit in &hits {
        assert!(
            matches!(hit.fields.get("active"), Some(FieldValue::Bool(false))),
            "active should be false for id {}",
            hit.id
        );
    }
}
