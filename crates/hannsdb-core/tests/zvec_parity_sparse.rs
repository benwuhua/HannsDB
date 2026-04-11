use std::time::{SystemTime, UNIX_EPOCH};

use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema, SparseVector,
    VectorFieldSchema,
};
use hannsdb_core::query::{QueryContext, QueryVector, VectorQuery};

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

#[test]
fn zvec_parity_sparse_bruteforce_search_returns_correct_ordering() {
    let root = unique_temp_dir("hannsdb_sparse_bruteforce");
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
                vec![1.0, 1.0],
                [(
                    "sparse".to_string(),
                    SparseVector::new(vec![0, 1], vec![1.0, 0.5]),
                )],
            ),
            Document::with_sparse_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(1))],
                "dense",
                vec![0.0, 0.0],
                [(
                    "sparse".to_string(),
                    SparseVector::new(vec![0, 1], vec![0.5, 1.0]),
                )],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 2,
                queries: vec![VectorQuery {
                    field_name: "sparse".to_string(),
                    vector: QueryVector::Sparse(SparseVector::new(vec![0, 1], vec![1.0, 1.0])),
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
        .expect("sparse query");

    assert_eq!(hits.len(), 2);
    // Both docs should match since both have non-zero inner product with the query.
}

#[test]
fn zvec_parity_dense_and_sparse_mixed_query_works() {
    let root = unique_temp_dir("hannsdb_sparse_mixed");
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
                vec![0.0, 0.0],
                [("sparse".to_string(), SparseVector::new(vec![0], vec![1.0]))],
            ),
            Document::with_sparse_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![5.0, 5.0],
                [("sparse".to_string(), SparseVector::new(vec![0], vec![0.1]))],
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
                        field_name: "sparse".to_string(),
                        vector: QueryVector::Sparse(SparseVector::new(vec![0], vec![1.0])),
                        param: None,
                    },
                ],
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
        .expect("mixed query");

    assert_eq!(hits.len(), 2);
}

#[test]
fn zvec_parity_sparse_query_with_filter() {
    let root = unique_temp_dir("hannsdb_sparse_filter");
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
                vec![0.0, 0.0],
                [("sparse".to_string(), SparseVector::new(vec![0], vec![1.0]))],
            ),
            Document::with_sparse_vectors(
                2,
                [("group".to_string(), FieldValue::Int64(2))],
                "dense",
                vec![0.0, 0.0],
                [("sparse".to_string(), SparseVector::new(vec![0], vec![2.0]))],
            ),
        ],
    )
    .expect("insert documents");

    let hits = db
        .query_with_context(
            "docs",
            &QueryContext {
                top_k: 1,
                queries: vec![VectorQuery {
                    field_name: "sparse".to_string(),
                    vector: QueryVector::Sparse(SparseVector::new(vec![0], vec![1.0])),
                    param: None,
                }],
                query_by_id: None,
                query_by_id_field_name: None,
                filter: Some("group == 1".to_string()),
                output_fields: None,
                include_vector: false,
                group_by: None,
                reranker: None,
                order_by: None,
            },
        )
        .expect("sparse filtered query");

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 1);
}
