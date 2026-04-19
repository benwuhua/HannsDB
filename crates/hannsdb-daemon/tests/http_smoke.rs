use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema, VectorFieldSchema,
};
use hannsdb_core::segment::{
    append_payloads, append_record_ids, append_records, SegmentMetadata, SegmentSet, TombstoneMask,
};
#[cfg(feature = "lance-storage")]
use hannsdb_core::storage::lance_store::LanceCollection;
use hannsdb_core::HannsDb;
use hannsdb_daemon::routes::build_router;
use serde_json::Value;
use std::fs;
use tower::ServiceExt;

fn seed_non_primary_query_by_id_collection(root: &std::path::Path) {
    let mut db = HannsDb::open(root).expect("open db");
    let mut schema = CollectionSchema::new("vector", 2, "l2", Vec::new());
    schema.vectors.push(VectorFieldSchema::new("title", 2));
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    db.insert_documents(
        "docs",
        &[
            Document::with_vectors(
                42,
                Vec::new(),
                vec![0.0, 0.0],
                [("title".to_string(), vec![10.0, 10.0])],
            ),
            Document::with_vectors(
                43,
                Vec::new(),
                vec![0.1, 0.0],
                [("title".to_string(), vec![50.0, 50.0])],
            ),
            Document::with_vectors(
                44,
                Vec::new(),
                vec![0.2, 0.0],
                [("title".to_string(), vec![60.0, 60.0])],
            ),
            Document::with_vectors(
                84,
                Vec::new(),
                vec![50.0, 50.0],
                [("title".to_string(), vec![10.0, 10.0])],
            ),
        ],
    )
    .expect("insert documents");
}

fn seed_delete_by_filter_collection(root: &std::path::Path) {
    let mut db = HannsDb::open(root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![ScalarFieldSchema::new("group", FieldType::Int64)],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    db.insert_documents(
        "docs",
        &[
            Document::new(
                42,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![0.0, 0.0],
            ),
            Document::new(
                43,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![0.1, 0.0],
            ),
            Document::new(
                44,
                [("group".to_string(), FieldValue::Int64(2))],
                vec![1.0, 1.0],
            ),
        ],
    )
    .expect("insert documents");
}

// This rewrite only supports the scalar-only layout used by the latest-live
// delete_by_filter smoke fixture below.
fn rewrite_delete_by_filter_latest_live_scalar_fixture_to_two_segments(
    root: &std::path::Path,
    collection: &str,
    dimension: usize,
    second_segment_documents: &[Document],
    deleted_second_segment_rows: &[usize],
) {
    let collection_dir = root.join("collections").join(collection);
    let segments_dir = collection_dir.join("segments");
    let seg1_dir = segments_dir.join("seg-0001");
    let seg2_dir = segments_dir.join("seg-0002");
    fs::create_dir_all(&seg1_dir).expect("create seg-0001 dir");
    fs::create_dir_all(&seg2_dir).expect("create seg-0002 dir");

    for file in [
        "segment.json",
        "records.bin",
        "ids.bin",
        "payloads.jsonl",
        "payloads.arrow",
        "vectors.arrow",
        "forward_store.json",
        "forward_store.arrow",
        "forward_store.parquet",
        "tombstones.json",
    ] {
        let source = collection_dir.join(file);
        if source.exists() {
            fs::rename(&source, seg1_dir.join(file)).expect("move seg-0001 file");
        }
    }

    let mut ids = Vec::with_capacity(second_segment_documents.len());
    let mut vectors = Vec::with_capacity(second_segment_documents.len() * dimension);
    let mut payloads = Vec::with_capacity(second_segment_documents.len());
    for document in second_segment_documents {
        ids.push(document.id);
        vectors.extend_from_slice(document.primary_vector());
        payloads.push(document.fields.clone());
    }

    let inserted =
        append_records(&seg2_dir.join("records.bin"), dimension, &vectors).expect("append records");
    assert_eq!(inserted, second_segment_documents.len());
    let _ = append_record_ids(&seg2_dir.join("ids.bin"), &ids).expect("append ids");
    let _ = append_payloads(&seg2_dir.join("payloads.jsonl"), &payloads).expect("append payloads");

    let mut seg2_tombstone = TombstoneMask::new(second_segment_documents.len());
    for row_idx in deleted_second_segment_rows {
        let marked = seg2_tombstone.mark_deleted(*row_idx);
        assert!(marked, "row index must be valid in seg-0002 tombstone");
    }
    seg2_tombstone
        .save_to_path(&seg2_dir.join("tombstones.json"))
        .expect("save seg-0002 tombstones");

    SegmentMetadata::new(
        "seg-0002",
        dimension,
        second_segment_documents.len(),
        seg2_tombstone.deleted_count(),
    )
    .save_to_path(&seg2_dir.join("segment.json"))
    .expect("save seg-0002 metadata");

    SegmentSet {
        active_segment_id: "seg-0002".to_string(),
        immutable_segment_ids: vec!["seg-0001".to_string()],
    }
    .save_to_path(&collection_dir.join("segment_set.json"))
    .expect("save segment set");
}

fn seed_delete_by_filter_latest_live_shadowing_collection(root: &std::path::Path) {
    let mut db = HannsDb::open(root).expect("open db");
    let schema = CollectionSchema::new(
        "vector",
        2,
        "l2",
        vec![
            ScalarFieldSchema::new("group", FieldType::Int64),
            ScalarFieldSchema::new("version", FieldType::String),
        ],
    );
    db.create_collection_with_schema("docs", &schema)
        .expect("create collection");

    db.insert_documents(
        "docs",
        &[
            Document::new(
                42,
                [
                    ("group".to_string(), FieldValue::Int64(1)),
                    (
                        "version".to_string(),
                        FieldValue::String("old-match".to_string()),
                    ),
                ],
                vec![10.0, 10.0],
            ),
            Document::new(
                43,
                [
                    ("group".to_string(), FieldValue::Int64(1)),
                    (
                        "version".to_string(),
                        FieldValue::String("stable-match".to_string()),
                    ),
                ],
                vec![0.1, 0.0],
            ),
            Document::new(
                44,
                [
                    ("group".to_string(), FieldValue::Int64(2)),
                    (
                        "version".to_string(),
                        FieldValue::String("control".to_string()),
                    ),
                ],
                vec![0.2, 0.0],
            ),
        ],
    )
    .expect("insert seg-0001 documents");

    rewrite_delete_by_filter_latest_live_scalar_fixture_to_two_segments(
        root,
        "docs",
        2,
        &[Document::new(
            42,
            [
                ("group".to_string(), FieldValue::Int64(2)),
                (
                    "version".to_string(),
                    FieldValue::String("latest-live".to_string()),
                ),
            ],
            vec![0.0, 0.0],
        )],
        &[],
    );

    let wal_path = root.join("wal.jsonl");
    if wal_path.exists() {
        fs::remove_file(wal_path).expect("remove wal after segment rewrite");
    }
}

#[tokio::test]
async fn health_route_returns_ok_json() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    assert_eq!(
        std::str::from_utf8(&body).expect("utf8 body"),
        "{\"status\":\"ok\"}"
    );
}

#[tokio::test]
async fn create_collection_route_persists_collection_metadata() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::CREATED);
    assert!(tempdir
        .path()
        .join("collections")
        .join("docs")
        .join("collection.json")
        .exists());
}

#[tokio::test]
async fn create_collection_route_returns_conflict_for_existing_collection() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let request = || {
        Request::builder()
            .method("POST")
            .uri("/collections")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
            .expect("build request")
    };

    let first = app.clone().oneshot(request()).await.expect("send request");
    assert_eq!(first.status(), StatusCode::CREATED);

    let second = app.oneshot(request()).await.expect("send request");
    assert_eq!(second.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn records_insert_search_delete_flow_works() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42","84"],"vectors":[[0.0,0.0],[1.0,1.0]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);
    let insert_body = to_bytes(insert.into_body(), usize::MAX)
        .await
        .expect("read insert body");
    let insert_json: Value = serde_json::from_slice(&insert_body).expect("parse insert json");
    assert_eq!(insert_json["inserted"], 2);

    let search = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"vector":[0.0,0.0],"top_k":1}"#))
                .expect("build request"),
        )
        .await
        .expect("send search request");
    assert_eq!(search.status(), StatusCode::OK);
    let search_body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let search_json: Value = serde_json::from_slice(&search_body).expect("parse search json");
    assert_eq!(search_json["hits"][0]["id"], "42");

    let delete = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send delete request");
    assert_eq!(delete.status(), StatusCode::OK);
    let delete_body = to_bytes(delete.into_body(), usize::MAX)
        .await
        .expect("read delete body");
    let delete_json: Value = serde_json::from_slice(&delete_body).expect("parse delete json");
    assert_eq!(delete_json["deleted"], 1);

    let search_after_delete = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"vector":[0.0,0.0],"top_k":2}"#))
                .expect("build request"),
        )
        .await
        .expect("send search request");
    assert_eq!(search_after_delete.status(), StatusCode::OK);
    let body = to_bytes(search_after_delete.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&body).expect("parse json");
    assert_eq!(json["hits"][0]["id"], "84");
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_records_insert_fetch_search_delete_flow_works() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"name":"docs","dimension":2,"metric":"cosine","storage":"lance"}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);
    assert!(tempdir
        .path()
        .join("collections")
        .join("docs.lance")
        .exists());

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42","84"],"vectors":[[0.0,0.0],[1.0,1.0]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let fetch = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/fetch")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42","84"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send fetch request");
    assert_eq!(fetch.status(), StatusCode::OK);
    let fetch_body = to_bytes(fetch.into_body(), usize::MAX)
        .await
        .expect("read fetch body");
    let fetch_json: Value = serde_json::from_slice(&fetch_body).expect("parse fetch json");
    assert_eq!(fetch_json["documents"][0]["id"], "42");
    assert_eq!(fetch_json["documents"][1]["id"], "84");

    let search = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"vector":[0.0,0.0],"top_k":1}"#))
                .expect("build request"),
        )
        .await
        .expect("send search request");
    assert_eq!(search.status(), StatusCode::OK);
    let search_body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let search_json: Value = serde_json::from_slice(&search_body).expect("parse search json");
    assert_eq!(search_json["hits"][0]["id"], "42");

    let delete = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send delete request");
    assert_eq!(delete.status(), StatusCode::OK);

    let fetch_after_delete = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/fetch")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42","84"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send fetch request");
    assert_eq!(fetch_after_delete.status(), StatusCode::OK);
    let body = to_bytes(fetch_after_delete.into_body(), usize::MAX)
        .await
        .expect("read fetch body");
    let json: Value = serde_json::from_slice(&body).expect("parse fetch json");
    assert_eq!(json["documents"].as_array().unwrap().len(), 1);
    assert_eq!(json["documents"][0]["id"], "84");
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_admin_routes_list_get_and_drop_collection() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"name":"docs","dimension":2,"metric":"cosine","storage":"lance"}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42"],"vectors":[[0.0,0.0]]}"#))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let list = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send list request");
    assert_eq!(list.status(), StatusCode::OK);
    let list_body = to_bytes(list.into_body(), usize::MAX)
        .await
        .expect("read list body");
    let list_json: Value = serde_json::from_slice(&list_body).expect("parse list json");
    assert_eq!(list_json["collections"], serde_json::json!(["docs"]));

    let info = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections/docs")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send info request");
    assert_eq!(info.status(), StatusCode::OK);
    let info_body = to_bytes(info.into_body(), usize::MAX)
        .await
        .expect("read info body");
    let info_json: Value = serde_json::from_slice(&info_body).expect("parse info json");
    assert_eq!(info_json["name"], "docs");
    assert_eq!(info_json["dimension"], 2);
    assert_eq!(info_json["metric"], "cosine");
    assert_eq!(info_json["record_count"], 1);
    assert_eq!(info_json["deleted_count"], 0);
    assert_eq!(info_json["live_count"], 1);

    let drop = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/collections/docs")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send drop request");
    assert_eq!(drop.status(), StatusCode::OK);
    assert!(!tempdir
        .path()
        .join("collections")
        .join("docs.lance")
        .exists());

    let list_after_drop = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send list request");
    assert_eq!(list_after_drop.status(), StatusCode::OK);
    let body = to_bytes(list_after_drop.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&body).expect("parse json");
    assert_eq!(json["collections"], serde_json::json!([]));
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_stats_route_returns_lance_collection_info() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"name":"docs","dimension":2,"metric":"cosine","storage":"lance"}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42","84"],"vectors":[[0.0,0.0],[1.0,1.0]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let stats = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections/docs/stats")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send stats request");
    assert_eq!(stats.status(), StatusCode::OK);
    let body = to_bytes(stats.into_body(), usize::MAX)
        .await
        .expect("read stats body");
    let json: Value = serde_json::from_slice(&body).expect("parse stats json");
    assert_eq!(json["name"], "docs");
    assert_eq!(json["dimension"], 2);
    assert_eq!(json["metric"], "cosine");
    assert_eq!(json["record_count"], 2);
    assert_eq!(json["deleted_count"], 0);
    assert_eq!(json["live_count"], 2);
    assert!(json.get("index_completeness").is_none());
}

#[cfg(all(feature = "lance-storage", feature = "hanns-backend"))]
#[tokio::test]
async fn lance_storage_admin_optimize_builds_hanns_sidecar() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"name":"docs","dimension":2,"metric":"l2","storage":"lance"}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42","84"],"vectors":[[0.0,0.0],[1.0,1.0]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let optimize = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/admin/optimize")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send optimize request");
    assert_eq!(optimize.status(), StatusCode::OK);
    assert!(tempdir
        .path()
        .join("collections")
        .join("docs.lance")
        .join("_hannsdb")
        .join("ann")
        .join("vector.hanns")
        .exists());
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_legacy_search_uses_daemon_metric_metadata() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"name":"docs","dimension":2,"metric":"cosine","storage":"lance"}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42","84"],"vectors":[[10.0,0.0],[0.9,0.4]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"vector":[1.0,0.0],"top_k":1}"#))
                .expect("build request"),
        )
        .await
        .expect("send search request");
    assert_eq!(search.status(), StatusCode::OK);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&body).expect("parse json");
    assert_eq!(json["hits"][0]["id"], "42");
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_legacy_search_applies_filter() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let schema = CollectionSchema {
        primary_vector: "vector".to_string(),
        fields: vec![ScalarFieldSchema::new("group", FieldType::Int64)],
        vectors: vec![VectorFieldSchema::new("vector", 2)],
    };
    LanceCollection::create(
        tempdir.path(),
        "docs",
        schema,
        &[
            Document::new(
                42,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![1.0, 1.0],
            ),
            Document::new(
                43,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![2.0, 2.0],
            ),
            Document::new(
                84,
                [("group".to_string(), FieldValue::Int64(2))],
                vec![0.0, 0.0],
            ),
        ],
    )
    .await
    .expect("seed Lance collection");
    let app = build_router(tempdir.path()).expect("build router");

    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"vector":[0.0,0.0],"top_k":3,"filter":"group == 1","output_fields":["group"]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send filtered search request");
    assert_eq!(search.status(), StatusCode::OK);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let json: Value = serde_json::from_slice(&body).expect("parse search json");
    let hit_ids = json["hits"]
        .as_array()
        .expect("hits array")
        .iter()
        .map(|hit| hit["id"].as_str().expect("hit id").to_string())
        .collect::<Vec<_>>();
    assert_eq!(hit_ids, vec!["42", "43"]);
    assert!(hit_ids.iter().all(|id| id != "84"));
    assert_eq!(json["hits"][0]["fields"], serde_json::json!({"group":1}));
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_typed_search_routes_to_lance() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"name":"docs","dimension":2,"metric":"l2","storage":"lance"}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42","84"],"vectors":[[0.0,0.0],[1.0,1.0]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let search = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"top_k":1,"queries":[{"field_name":"vector","vector":[0.0,0.0]}],"include_vector":true,"output_fields":[]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send search request");
    assert_eq!(search.status(), StatusCode::OK);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&body).expect("parse json");
    assert_eq!(json["hits"][0]["id"], "42");
    assert_eq!(json["hits"][0]["vector"], serde_json::json!([0.0, 0.0]));

    let unsupported = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"top_k":1,"queries":[{"field_name":"vector","vector":[0.0,0.0]}],"group_by":{"field_name":"group","group_topk":1}}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send unsupported search request");
    assert_eq!(unsupported.status(), StatusCode::BAD_REQUEST);
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_typed_search_applies_filter() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let schema = CollectionSchema {
        primary_vector: "vector".to_string(),
        fields: vec![ScalarFieldSchema::new("group", FieldType::Int64)],
        vectors: vec![VectorFieldSchema::new("vector", 2)],
    };
    LanceCollection::create(
        tempdir.path(),
        "docs",
        schema,
        &[
            Document::new(
                42,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![1.0, 1.0],
            ),
            Document::new(
                43,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![2.0, 2.0],
            ),
            Document::new(
                84,
                [("group".to_string(), FieldValue::Int64(2))],
                vec![0.0, 0.0],
            ),
        ],
    )
    .await
    .expect("seed Lance collection");
    let app = build_router(tempdir.path()).expect("build router");

    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"top_k":3,"queries":[{"field_name":"vector","vector":[0.0,0.0]}],"filter":"group == 1","output_fields":["group"]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send filtered typed search request");
    assert_eq!(search.status(), StatusCode::OK);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let json: Value = serde_json::from_slice(&body).expect("parse search json");
    let hit_ids = json["hits"]
        .as_array()
        .expect("hits array")
        .iter()
        .map(|hit| hit["id"].as_str().expect("hit id").to_string())
        .collect::<Vec<_>>();
    assert_eq!(hit_ids, vec!["42", "43"]);
    assert!(hit_ids.iter().all(|id| id != "84"));
    assert_eq!(json["hits"][0]["fields"], serde_json::json!({"group":1}));
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_typed_search_groups_by_scalar_field() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let schema = CollectionSchema {
        primary_vector: "vector".to_string(),
        fields: vec![ScalarFieldSchema::new("group", FieldType::Int64)],
        vectors: vec![VectorFieldSchema::new("vector", 2)],
    };
    LanceCollection::create(
        tempdir.path(),
        "docs",
        schema,
        &[
            Document::new(
                42,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![0.0, 0.0],
            ),
            Document::new(
                43,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![0.01, 0.0],
            ),
            Document::new(
                84,
                [("group".to_string(), FieldValue::Int64(2))],
                vec![0.02, 0.0],
            ),
            Document::new(
                85,
                [("group".to_string(), FieldValue::Int64(2))],
                vec![0.03, 0.0],
            ),
            Document::new(
                126,
                [("group".to_string(), FieldValue::Int64(3))],
                vec![0.04, 0.0],
            ),
        ],
    )
    .await
    .expect("seed Lance collection");
    let app = build_router(tempdir.path()).expect("build router");

    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"top_k":2,"queries":[{"field_name":"vector","vector":[0.0,0.0]}],"group_by":{"field_name":"group","group_topk":1,"group_count":2}}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send grouped typed search request");
    assert_eq!(search.status(), StatusCode::OK);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let json: Value = serde_json::from_slice(&body).expect("parse search json");
    let hits = json["hits"].as_array().expect("hits array");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0]["id"], "42");
    assert_eq!(hits[0]["group_key"], serde_json::json!(1));
    assert_eq!(hits[1]["id"], "84");
    assert_eq!(hits[1]["group_key"], serde_json::json!(2));
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_typed_search_orders_by_scalar_field() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let schema = CollectionSchema {
        primary_vector: "vector".to_string(),
        fields: vec![ScalarFieldSchema::new("rank", FieldType::Int64)],
        vectors: vec![VectorFieldSchema::new("vector", 2)],
    };
    LanceCollection::create(
        tempdir.path(),
        "docs",
        schema,
        &[
            Document::new(
                42,
                [("rank".to_string(), FieldValue::Int64(3))],
                vec![0.0, 0.0],
            ),
            Document::new(
                43,
                [("rank".to_string(), FieldValue::Int64(2))],
                vec![0.1, 0.0],
            ),
            Document::new(
                84,
                [("rank".to_string(), FieldValue::Int64(1))],
                vec![10.0, 10.0],
            ),
        ],
    )
    .await
    .expect("seed Lance collection");
    let app = build_router(tempdir.path()).expect("build router");

    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"top_k":2,"queries":[{"field_name":"vector","vector":[0.0,0.0]}],"order_by":{"field_name":"rank"}}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send ordered typed search request");
    assert_eq!(search.status(), StatusCode::OK);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let json: Value = serde_json::from_slice(&body).expect("parse search json");
    let hit_ids = json["hits"]
        .as_array()
        .expect("hits array")
        .iter()
        .map(|hit| hit["id"].as_str().expect("hit id").to_string())
        .collect::<Vec<_>>();
    assert_eq!(hit_ids, vec!["84", "43"]);
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_typed_search_query_by_id_routes_to_lance() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"name":"docs","dimension":2,"metric":"l2","storage":"lance"}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42","84"],"vectors":[[0.0,0.0],[1.0,1.0]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let search = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"top_k":1,"query_by_id":["42"],"include_vector":true}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send search request");
    assert_eq!(search.status(), StatusCode::OK);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&body).expect("parse json");
    assert_eq!(json["hits"][0]["id"], "42");
    assert_eq!(json["hits"][0]["vector"], serde_json::json!([0.0, 0.0]));

    let multi_id = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"top_k":1,"query_by_id":["42","84"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send multi-id search request");
    assert_eq!(multi_id.status(), StatusCode::BAD_REQUEST);
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_typed_search_query_by_id_field_name_routes_to_lance() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let schema = CollectionSchema {
        primary_vector: "vector".to_string(),
        fields: Vec::new(),
        vectors: vec![
            VectorFieldSchema::new("vector", 2),
            VectorFieldSchema::new("title", 2),
        ],
    };
    LanceCollection::create(
        tempdir.path(),
        "docs",
        schema,
        &[
            Document::with_vectors(
                42,
                Vec::new(),
                vec![0.0, 0.0],
                [("title".to_string(), vec![10.0, 10.0])],
            ),
            Document::with_vectors(
                43,
                Vec::new(),
                vec![0.1, 0.0],
                [("title".to_string(), vec![50.0, 50.0])],
            ),
            Document::with_vectors(
                44,
                Vec::new(),
                vec![0.2, 0.0],
                [("title".to_string(), vec![60.0, 60.0])],
            ),
            Document::with_vectors(
                84,
                Vec::new(),
                vec![50.0, 50.0],
                [("title".to_string(), vec![10.0, 10.0])],
            ),
        ],
    )
    .await
    .expect("seed Lance collection");
    let app = build_router(tempdir.path()).expect("build router");

    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"top_k":3,"query_by_id":["42"],"query_by_id_field_name":"title"}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send search request");
    assert_eq!(search.status(), StatusCode::OK);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let json: Value = serde_json::from_slice(&body).expect("parse search json");
    let hit_ids = json["hits"]
        .as_array()
        .expect("hits array")
        .iter()
        .map(|hit| hit["id"].as_str().expect("hit id").to_string())
        .collect::<Vec<_>>();
    assert_eq!(hit_ids, vec!["42", "84", "43"]);
    assert!(!hit_ids.iter().any(|id| id == "44"));
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_typed_search_secondary_vector_field_routes_to_lance() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let schema = CollectionSchema {
        primary_vector: "vector".to_string(),
        fields: Vec::new(),
        vectors: vec![
            VectorFieldSchema::new("vector", 2),
            VectorFieldSchema::new("title", 2),
        ],
    };
    LanceCollection::create(
        tempdir.path(),
        "docs",
        schema,
        &[
            Document::with_vectors(
                42,
                Vec::new(),
                vec![0.0, 0.0],
                [("title".to_string(), vec![10.0, 10.0])],
            ),
            Document::with_vectors(
                43,
                Vec::new(),
                vec![0.1, 0.0],
                [("title".to_string(), vec![50.0, 50.0])],
            ),
            Document::with_vectors(
                44,
                Vec::new(),
                vec![0.2, 0.0],
                [("title".to_string(), vec![60.0, 60.0])],
            ),
            Document::with_vectors(
                84,
                Vec::new(),
                vec![50.0, 50.0],
                [("title".to_string(), vec![10.0, 10.0])],
            ),
        ],
    )
    .await
    .expect("seed Lance collection");
    let app = build_router(tempdir.path()).expect("build router");

    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"top_k":3,"queries":[{"field_name":"title","vector":[10.0,10.0]}]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send search request");
    assert_eq!(search.status(), StatusCode::OK);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let json: Value = serde_json::from_slice(&body).expect("parse search json");
    let hit_ids = json["hits"]
        .as_array()
        .expect("hits array")
        .iter()
        .map(|hit| hit["id"].as_str().expect("hit id").to_string())
        .collect::<Vec<_>>();
    assert_eq!(hit_ids, vec!["42", "84", "43"]);
    assert!(!hit_ids.iter().any(|id| id == "44"));
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_typed_search_projects_output_fields() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let schema = CollectionSchema {
        primary_vector: "dense".to_string(),
        fields: vec![
            ScalarFieldSchema::new("title", FieldType::String),
            ScalarFieldSchema::new("group", FieldType::Int64),
        ],
        vectors: vec![VectorFieldSchema::new("dense", 2)],
    };
    LanceCollection::create(
        tempdir.path(),
        "docs",
        schema,
        &[
            Document::with_primary_vector_name(
                42,
                [
                    ("title".to_string(), FieldValue::String("alpha".to_string())),
                    ("group".to_string(), FieldValue::Int64(7)),
                ],
                "dense",
                vec![0.0, 0.0],
            ),
            Document::with_primary_vector_name(
                84,
                [
                    ("title".to_string(), FieldValue::String("beta".to_string())),
                    ("group".to_string(), FieldValue::Int64(9)),
                ],
                "dense",
                vec![1.0, 1.0],
            ),
        ],
    )
    .await
    .expect("seed Lance collection");
    let app = build_router(tempdir.path()).expect("build router");

    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"top_k":1,"queries":[{"field_name":"dense","vector":[0.0,0.0]}],"output_fields":["title"],"include_vector":true}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send search request");
    assert_eq!(search.status(), StatusCode::OK);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&body).expect("parse json");
    assert_eq!(json["hits"][0]["id"], "42");
    assert_eq!(
        json["hits"][0]["fields"],
        serde_json::json!({"title":"alpha"})
    );
    assert_eq!(json["hits"][0]["vector"], serde_json::json!([0.0, 0.0]));
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_fetch_uses_lance_primary_vector_name() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let schema = CollectionSchema {
        primary_vector: "dense".to_string(),
        fields: Vec::new(),
        vectors: vec![VectorFieldSchema::new("dense", 2)],
    };
    LanceCollection::create(
        tempdir.path(),
        "docs",
        schema,
        &[Document::with_primary_vector_name(
            42,
            Vec::new(),
            "dense",
            vec![0.0, 1.0],
        )],
    )
    .await
    .expect("seed Lance collection");
    let app = build_router(tempdir.path()).expect("build router");

    let fetch = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/fetch")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send fetch request");
    assert_eq!(fetch.status(), StatusCode::OK);
    let body = to_bytes(fetch.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&body).expect("parse json");
    assert_eq!(
        json["documents"][0]["vector"],
        serde_json::json!([0.0, 1.0])
    );
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_create_rejects_native_name_collision() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let native = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send native create");
    assert_eq!(native.status(), StatusCode::CREATED);

    let lance = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"name":"docs","dimension":2,"metric":"l2","storage":"lance"}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send lance create");
    assert_eq!(lance.status(), StatusCode::CONFLICT);
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn native_create_rejects_lance_name_collision() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let lance = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"name":"docs","dimension":2,"metric":"l2","storage":"lance"}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send lance create");
    assert_eq!(lance.status(), StatusCode::CREATED);

    let native = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send native create");
    assert_eq!(native.status(), StatusCode::CONFLICT);
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_insert_rejects_duplicate_ids_and_fields() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"name":"docs","dimension":2,"metric":"l2","storage":"lance"}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send create");
    assert_eq!(create.status(), StatusCode::CREATED);

    let first_insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42"],"vectors":[[0.0,0.0]]}"#))
                .expect("build request"),
        )
        .await
        .expect("send first insert");
    assert_eq!(first_insert.status(), StatusCode::OK);

    let duplicate_insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42"],"vectors":[[1.0,1.0]]}"#))
                .expect("build request"),
        )
        .await
        .expect("send duplicate insert");
    assert_eq!(duplicate_insert.status(), StatusCode::BAD_REQUEST);

    let duplicate_batch_insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["100","100"],"vectors":[[1.0,1.0],[2.0,2.0]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send duplicate batch insert");
    assert_eq!(duplicate_batch_insert.status(), StatusCode::BAD_REQUEST);

    let field_insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["84"],"vectors":[[1.0,1.0]],"fields":[{"title":"lost"}]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send field insert");
    assert_eq!(field_insert.status(), StatusCode::BAD_REQUEST);

    let extra_named_vector_insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["85"],"vectors":[[1.0,1.0]],"named_vectors":[{"extra":[2.0,2.0]}]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send named vector insert");
    assert_eq!(extra_named_vector_insert.status(), StatusCode::BAD_REQUEST);

    let sparse_vector_insert = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["86"],"vectors":[[1.0,1.0]],"sparse_vectors":[{"sparse":{"indices":[1],"values":[1.0]}}]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send sparse vector insert");
    assert_eq!(sparse_vector_insert.status(), StatusCode::BAD_REQUEST);
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_concurrent_insert_rejects_duplicate_ids() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"name":"docs","dimension":2,"metric":"l2","storage":"lance"}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send create");
    assert_eq!(create.status(), StatusCode::CREATED);

    let attempts = 8usize;
    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(attempts));
    let mut handles = Vec::with_capacity(attempts);
    for attempt in 0..attempts {
        let app = app.clone();
        let barrier = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier.wait().await;
            let body = format!(r#"{{"ids":["42"],"vectors":[[{attempt}.0,{attempt}.0]]}}"#);
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections/docs/records")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .expect("build request"),
            )
            .await
            .expect("send concurrent insert")
            .status()
        }));
    }

    let mut statuses = Vec::with_capacity(attempts);
    for handle in handles {
        statuses.push(handle.await.expect("join concurrent insert"));
    }
    assert_eq!(
        statuses
            .iter()
            .filter(|status| **status == StatusCode::OK)
            .count(),
        1
    );
    assert_eq!(
        statuses
            .iter()
            .filter(|status| **status == StatusCode::BAD_REQUEST)
            .count(),
        attempts - 1
    );

    let fetch = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/fetch")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send fetch");
    assert_eq!(fetch.status(), StatusCode::OK);
    let body = to_bytes(fetch.into_body(), usize::MAX)
        .await
        .expect("read fetch body");
    let json: Value = serde_json::from_slice(&body).expect("parse fetch json");
    assert_eq!(json["documents"].as_array().expect("documents").len(), 1);
}

#[cfg(feature = "lance-storage")]
#[tokio::test]
async fn lance_storage_delete_by_filter_deletes_matching_rows() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let schema = CollectionSchema {
        primary_vector: "vector".to_string(),
        fields: vec![ScalarFieldSchema::new("group", FieldType::Int64)],
        vectors: vec![VectorFieldSchema::new("vector", 2)],
    };
    LanceCollection::create(
        tempdir.path(),
        "docs",
        schema,
        &[
            Document::new(
                42,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![0.0, 0.0],
            ),
            Document::new(
                43,
                [("group".to_string(), FieldValue::Int64(1))],
                vec![0.1, 0.0],
            ),
            Document::new(
                84,
                [("group".to_string(), FieldValue::Int64(2))],
                vec![1.0, 1.0],
            ),
        ],
    )
    .await
    .expect("seed Lance collection");
    let app = build_router(tempdir.path()).expect("build router");

    let delete = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/delete_by_filter")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"filter":"group == 1"}"#))
                .expect("build request"),
        )
        .await
        .expect("send delete-by-filter request");
    assert_eq!(delete.status(), StatusCode::OK);
    let body = to_bytes(delete.into_body(), usize::MAX)
        .await
        .expect("read delete body");
    let json: Value = serde_json::from_slice(&body).expect("parse delete json");
    assert_eq!(json["deleted"], 2);

    let fetch = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/fetch")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42","43","84"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send fetch");
    assert_eq!(fetch.status(), StatusCode::OK);
    let body = to_bytes(fetch.into_body(), usize::MAX)
        .await
        .expect("read fetch body");
    let json: Value = serde_json::from_slice(&body).expect("parse fetch json");
    assert_eq!(json["documents"].as_array().expect("documents").len(), 1);
    assert_eq!(json["documents"][0]["id"], "84");
}

#[tokio::test]
async fn delete_by_filter_route_deletes_matching_rows() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    seed_delete_by_filter_collection(tempdir.path());
    let app = build_router(tempdir.path()).expect("build router");

    let delete = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/delete_by_filter")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"filter":"group == 1"}"#))
                .expect("build request"),
        )
        .await
        .expect("send delete-by-filter request");

    assert_eq!(delete.status(), StatusCode::OK);
    let body = to_bytes(delete.into_body(), usize::MAX)
        .await
        .expect("read delete body");
    let json: Value = serde_json::from_slice(&body).expect("parse delete json");
    assert_eq!(json["deleted"], 2);

    let fetch = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/fetch")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42","43","44"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send fetch request");

    assert_eq!(fetch.status(), StatusCode::OK);
    let body = to_bytes(fetch.into_body(), usize::MAX)
        .await
        .expect("read fetch body");
    let json: Value = serde_json::from_slice(&body).expect("parse fetch json");
    let documents = json["documents"].as_array().expect("documents array");
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0]["id"], "44");
}

#[tokio::test]
async fn delete_by_filter_route_uses_latest_live_view() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    seed_delete_by_filter_latest_live_shadowing_collection(tempdir.path());
    let app = build_router(tempdir.path()).expect("build router");

    let delete = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/delete_by_filter")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"filter":"group == 1"}"#))
                .expect("build request"),
        )
        .await
        .expect("send delete-by-filter request");

    assert_eq!(delete.status(), StatusCode::OK);
    let body = to_bytes(delete.into_body(), usize::MAX)
        .await
        .expect("read delete body");
    let json: Value = serde_json::from_slice(&body).expect("parse delete json");
    assert_eq!(json["deleted"], 1);

    let fetch = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/fetch")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42","43","44"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send fetch request");

    assert_eq!(fetch.status(), StatusCode::OK);
    let body = to_bytes(fetch.into_body(), usize::MAX)
        .await
        .expect("read fetch body");
    let json: Value = serde_json::from_slice(&body).expect("parse fetch json");
    let documents = json["documents"].as_array().expect("documents array");
    assert_eq!(documents.len(), 2);
    assert_eq!(documents[0]["id"], "42");
    assert_eq!(documents[0]["fields"]["group"], 2);
    assert_eq!(documents[0]["fields"]["version"], "latest-live");
    assert_eq!(documents[1]["id"], "44");

    let second_delete = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/delete_by_filter")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"filter":"group == 1"}"#))
                .expect("build request"),
        )
        .await
        .expect("send second delete-by-filter request");

    assert_eq!(second_delete.status(), StatusCode::OK);
    let body = to_bytes(second_delete.into_body(), usize::MAX)
        .await
        .expect("read second delete body");
    let json: Value = serde_json::from_slice(&body).expect("parse second delete json");
    assert_eq!(json["deleted"], 0);
}

#[tokio::test]
async fn delete_by_filter_route_returns_zero_for_no_matches() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    seed_delete_by_filter_collection(tempdir.path());
    let app = build_router(tempdir.path()).expect("build router");

    let delete = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/delete_by_filter")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"filter":"group == 99"}"#))
                .expect("build request"),
        )
        .await
        .expect("send delete-by-filter request");

    assert_eq!(delete.status(), StatusCode::OK);
    let body = to_bytes(delete.into_body(), usize::MAX)
        .await
        .expect("read delete body");
    let json: Value = serde_json::from_slice(&body).expect("parse delete json");
    assert_eq!(json["deleted"], 0);
}

#[tokio::test]
async fn delete_by_filter_route_second_call_returns_zero() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    seed_delete_by_filter_collection(tempdir.path());
    let app = build_router(tempdir.path()).expect("build router");

    let first = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/delete_by_filter")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"filter":"group == 1"}"#))
                .expect("build request"),
        )
        .await
        .expect("send first delete-by-filter request");
    assert_eq!(first.status(), StatusCode::OK);
    let first_body = to_bytes(first.into_body(), usize::MAX)
        .await
        .expect("read first delete body");
    let first_json: Value = serde_json::from_slice(&first_body).expect("parse first delete json");
    assert_eq!(first_json["deleted"], 2);

    let second = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/delete_by_filter")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"filter":"group == 1"}"#))
                .expect("build request"),
        )
        .await
        .expect("send second delete-by-filter request");

    assert_eq!(second.status(), StatusCode::OK);
    let second_body = to_bytes(second.into_body(), usize::MAX)
        .await
        .expect("read second delete body");
    let second_json: Value =
        serde_json::from_slice(&second_body).expect("parse second delete json");
    assert_eq!(second_json["deleted"], 0);
}

#[tokio::test]
async fn delete_by_filter_route_returns_bad_request_for_invalid_filter() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    seed_delete_by_filter_collection(tempdir.path());
    let app = build_router(tempdir.path()).expect("build router");

    let delete = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/delete_by_filter")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"filter":"group ??? 1"}"#))
                .expect("build request"),
        )
        .await
        .expect("send delete-by-filter request");

    assert_eq!(delete.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(delete.into_body(), usize::MAX)
        .await
        .expect("read delete body");
    let json: Value = serde_json::from_slice(&body).expect("parse delete json");
    assert!(
        json.get("error").is_some(),
        "daemon error envelope should be returned"
    );
}

#[tokio::test]
async fn delete_by_filter_route_returns_bad_request_for_empty_filter() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    seed_delete_by_filter_collection(tempdir.path());
    let app = build_router(tempdir.path()).expect("build router");

    let delete = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/delete_by_filter")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"filter":""}"#))
                .expect("build request"),
        )
        .await
        .expect("send delete-by-filter request");

    assert_eq!(delete.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(delete.into_body(), usize::MAX)
        .await
        .expect("read delete body");
    let json: Value = serde_json::from_slice(&body).expect("parse delete json");
    assert!(
        json.get("error").is_some(),
        "daemon error envelope should be returned"
    );
}

#[tokio::test]
async fn delete_by_filter_route_returns_bad_request_for_whitespace_only_filter() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    seed_delete_by_filter_collection(tempdir.path());
    let app = build_router(tempdir.path()).expect("build router");

    let delete = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/delete_by_filter")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::json!({"filter":"   "}).to_string()))
                .expect("build request"),
        )
        .await
        .expect("send delete-by-filter request");

    assert_eq!(delete.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(delete.into_body(), usize::MAX)
        .await
        .expect("read delete body");
    let json: Value = serde_json::from_slice(&body).expect("parse delete json");
    assert!(
        json.get("error").is_some(),
        "daemon error envelope should be returned"
    );
}

#[tokio::test]
async fn delete_by_filter_route_wraps_malformed_json_in_daemon_error_envelope() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    seed_delete_by_filter_collection(tempdir.path());
    let app = build_router(tempdir.path()).expect("build router");

    let delete = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/delete_by_filter")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"filter":1"#))
                .expect("build request"),
        )
        .await
        .expect("send delete-by-filter request");

    assert_eq!(delete.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(delete.into_body(), usize::MAX)
        .await
        .expect("read delete body");
    let json: Value = serde_json::from_slice(&body).expect("parse delete json");
    assert!(
        json.get("error").is_some(),
        "daemon error envelope should be returned"
    );
}

#[tokio::test]
async fn delete_by_filter_route_returns_bad_request_for_missing_filter_field() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    seed_delete_by_filter_collection(tempdir.path());
    let app = build_router(tempdir.path()).expect("build router");

    let delete = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/delete_by_filter")
                .header("content-type", "application/json")
                .body(Body::from(r#"{}"#))
                .expect("build request"),
        )
        .await
        .expect("send delete-by-filter request");

    assert_eq!(delete.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(delete.into_body(), usize::MAX)
        .await
        .expect("read delete body");
    let json: Value = serde_json::from_slice(&body).expect("parse delete json");
    assert!(
        json.get("error").is_some(),
        "daemon error envelope should be returned"
    );
}

#[tokio::test]
async fn delete_by_filter_route_rejects_unknown_fields() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    seed_delete_by_filter_collection(tempdir.path());
    let app = build_router(tempdir.path()).expect("build router");

    let delete = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/delete_by_filter")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "filter": "group == 1",
                        "ids": ["42"],
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("send delete-by-filter request");

    assert_eq!(delete.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(delete.into_body(), usize::MAX)
        .await
        .expect("read delete body");
    let json: Value = serde_json::from_slice(&body).expect("parse delete json");
    assert!(
        json.get("error").is_some(),
        "daemon error envelope should be returned"
    );
}

#[tokio::test]
async fn delete_by_filter_route_returns_not_found_for_missing_collection() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let delete = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs_does_not_exist_123/records/delete_by_filter")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"filter":"group == 1"}"#))
                .expect("build request"),
        )
        .await
        .expect("send delete-by-filter request");

    assert_eq!(delete.status(), StatusCode::NOT_FOUND);
    let body = to_bytes(delete.into_body(), usize::MAX)
        .await
        .expect("read delete body");
    let json: Value = serde_json::from_slice(&body).expect("parse delete json");
    let error = json["error"].as_str().expect("daemon error string");
    assert!(
        error.contains("docs_does_not_exist_123"),
        "error text should mention the missing collection"
    );
}

#[tokio::test]
async fn delete_by_filter_route_returns_internal_error_for_corrupted_collection_state() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    seed_delete_by_filter_collection(tempdir.path());
    fs::remove_file(tempdir.path().join("wal.jsonl")).expect("remove wal");
    fs::remove_file(
        tempdir
            .path()
            .join("collections")
            .join("docs")
            .join("collection.json"),
    )
    .expect("remove collection metadata");
    let app = build_router(tempdir.path()).expect("build router");

    let delete = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/delete_by_filter")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"filter":"group == 1"}"#))
                .expect("build request"),
        )
        .await
        .expect("send delete-by-filter request");

    assert_eq!(delete.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = to_bytes(delete.into_body(), usize::MAX)
        .await
        .expect("read delete body");
    let json: Value = serde_json::from_slice(&body).expect("parse delete json");
    let error = json["error"].as_str().expect("daemon error string");
    assert!(
        !error.contains("collection not found: docs"),
        "collection-internal corruption should not be misreported as a missing collection"
    );
}

#[tokio::test]
async fn delete_records_route_still_deletes_by_explicit_ids() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42","84"],"vectors":[[0.0,0.0],[1.0,1.0]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let delete = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send delete request");
    assert_eq!(delete.status(), StatusCode::OK);
    let delete_body = to_bytes(delete.into_body(), usize::MAX)
        .await
        .expect("read delete body");
    let delete_json: Value = serde_json::from_slice(&delete_body).expect("parse delete json");
    assert_eq!(delete_json["deleted"], 1);

    let fetch = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/fetch")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42","84"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send fetch request");
    assert_eq!(fetch.status(), StatusCode::OK);
    let body = to_bytes(fetch.into_body(), usize::MAX)
        .await
        .expect("read fetch body");
    let json: Value = serde_json::from_slice(&body).expect("parse fetch json");
    let documents = json["documents"].as_array().expect("documents array");
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0]["id"], "84");
}

#[tokio::test]
async fn search_route_accepts_typed_primary_vector_query_and_projects_output_fields() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "ids":["42","84"],
                        "vectors":[[0.0,0.0],[0.2,0.0]],
                        "fields":[
                            {"color":"red","shape":"circle"},
                            {"color":"blue","shape":"square"}
                        ]
                    }"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    // Typed daemon parity slice stays within the currently supported primary vector field.
    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "top_k":1,
                        "queries":[{"field_name":"vector","vector":[0.0,0.0]}],
                        "output_fields":["color"]
                    }"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send search request");

    assert_eq!(search.status(), StatusCode::OK);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let json: Value = serde_json::from_slice(&body).expect("parse search json");
    assert_eq!(json["hits"][0]["id"], "42");
    assert_eq!(json["hits"][0]["fields"]["color"], "red");
    assert!(json["hits"][0]["fields"].get("shape").is_none());
}

#[tokio::test]
async fn search_route_accepts_typed_query_by_id_field_name_on_non_primary_vector_collection() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    seed_non_primary_query_by_id_collection(tempdir.path());
    let app = build_router(tempdir.path()).expect("build router");

    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "top_k":3,
                        "queries":[{"field_name":"vector","vector":[0.0,0.0]}],
                        "query_by_id":["42"],
                        "query_by_id_field_name":"title"
                    }"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send search request");

    assert_eq!(search.status(), StatusCode::OK);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let json: Value = serde_json::from_slice(&body).expect("parse search json");
    let hit_ids = json["hits"]
        .as_array()
        .expect("hits array")
        .iter()
        .map(|hit| hit["id"].as_str().expect("hit id").to_string())
        .collect::<Vec<_>>();
    assert_eq!(hit_ids, vec!["42", "84", "43"]);
    assert!(hit_ids.iter().any(|id| id == "84"));
    assert!(!hit_ids.iter().any(|id| id == "44"));
}

#[tokio::test]
async fn search_route_returns_daemon_error_for_invalid_query_by_id_field_name() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "ids":["42"],
                        "vectors":[[0.0,0.0]]
                    }"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "top_k":1,
                        "query_by_id":["42"],
                        "query_by_id_field_name":"missing"
                    }"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send search request");

    assert_eq!(search.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let json: Value = serde_json::from_slice(&body).expect("parse search json");
    assert!(json["error"]
        .as_str()
        .expect("error string")
        .contains("query_by_id field"));
}

#[tokio::test]
async fn search_route_rejects_legacy_request_mixing_vector_with_query_by_id_field_name() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42"],"vectors":[[0.0,0.0]]}"#))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "vector":[0.0,0.0],
                        "top_k":1,
                        "query_by_id_field_name":"vector"
                    }"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send search request");

    assert_eq!(search.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let json: Value = serde_json::from_slice(&body).expect("parse search json");
    assert!(json["error"]
        .as_str()
        .expect("error string")
        .contains("cannot mix legacy and typed query keys"));
}

#[tokio::test]
async fn search_route_include_vector_returns_hits_with_vectors() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "top_k":1,
                        "queries":[{"field_name":"vector","vector":[0.0,0.0]}],
                        "include_vector":true
                    }"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send search request");

    assert_eq!(search.status(), StatusCode::OK);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let json: Value = serde_json::from_slice(&body).expect("parse search json");
    assert!(json["hits"].is_array());
}

#[tokio::test]
async fn search_route_returns_daemon_error_envelope_for_malformed_typed_body() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "top_k":"oops",
                        "queries":[{"field_name":"vector","vector":[0.0,0.0]}]
                    }"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send search request");

    assert_eq!(search.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let json: Value = serde_json::from_slice(&body).expect("parse search json");
    assert!(
        json.get("error").is_some(),
        "daemon error envelope should be returned"
    );
}

#[tokio::test]
async fn fetch_route_projects_selected_output_fields() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "ids":["42"],
                        "vectors":[[0.0,0.0]],
                        "fields":[{"color":"red","shape":"circle"}]
                    }"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let fetch = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/fetch")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42"],"output_fields":["shape"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send fetch request");

    assert_eq!(fetch.status(), StatusCode::OK);
    let body = to_bytes(fetch.into_body(), usize::MAX)
        .await
        .expect("read fetch body");
    let json: Value = serde_json::from_slice(&body).expect("parse fetch json");
    assert_eq!(json["documents"][0]["id"], "42");
    assert_eq!(json["documents"][0]["fields"]["shape"], "circle");
    assert!(json["documents"][0]["fields"].get("color").is_none());
    assert_eq!(
        json["documents"][0]["vector"],
        serde_json::json!([0.0, 0.0])
    );
}

#[tokio::test]
async fn fetch_route_returns_empty_fields_for_explicit_empty_output_fields() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "ids":["42"],
                        "vectors":[[0.0,0.0]],
                        "fields":[{"color":"red","shape":"circle"}]
                    }"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let fetch = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/fetch")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42"],"output_fields":[]}"#))
                .expect("build request"),
        )
        .await
        .expect("send fetch request");

    assert_eq!(fetch.status(), StatusCode::OK);
    let body = to_bytes(fetch.into_body(), usize::MAX)
        .await
        .expect("read fetch body");
    let json: Value = serde_json::from_slice(&body).expect("parse fetch json");
    assert_eq!(json["documents"][0]["fields"], serde_json::json!({}));
}

#[tokio::test]
async fn legacy_search_route_ignores_extra_unknown_keys() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42"],"vectors":[[0.0,0.0]],"fields":[{"color":"red"}]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"vector":[0.0,0.0],"top_k":1,"unexpected":"ignored"}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send search request");

    assert_eq!(search.status(), StatusCode::OK);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let json: Value = serde_json::from_slice(&body).expect("parse search json");
    assert_eq!(json["hits"][0]["id"], "42");
}

#[tokio::test]
async fn search_route_rejects_mixed_legacy_and_typed_body_shapes() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "vector":[0.0,0.0],
                        "top_k":1,
                        "queries":[{"field_name":"vector","vector":[0.0,0.0]}],
                        "include_vector":true
                    }"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send search request");

    assert_eq!(search.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let json: Value = serde_json::from_slice(&body).expect("parse search json");
    assert!(
        json.get("error").is_some(),
        "daemon error envelope should be returned"
    );
}

#[tokio::test]
async fn collections_list_and_drop_flow_works() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    for name in ["docs_a", "docs_b"] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/collections")
                    .header("content-type", "application/json")
                    .body(Body::from(format!(
                        r#"{{"name":"{name}","dimension":2,"metric":"l2"}}"#
                    )))
                    .expect("build request"),
            )
            .await
            .expect("send request");
        assert_eq!(response.status(), StatusCode::CREATED);
    }

    let list = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send list request");
    assert_eq!(list.status(), StatusCode::OK);
    let list_body = to_bytes(list.into_body(), usize::MAX)
        .await
        .expect("read list body");
    let list_json: Value = serde_json::from_slice(&list_body).expect("parse list json");
    let names = list_json["collections"]
        .as_array()
        .expect("collections array");
    assert!(names.iter().any(|v| v == "docs_a"));
    assert!(names.iter().any(|v| v == "docs_b"));

    let drop_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/collections/docs_a")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send drop request");
    assert!(drop_response.status().is_success());

    let list_after_drop = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send list request");
    assert_eq!(list_after_drop.status(), StatusCode::OK);
    let body = to_bytes(list_after_drop.into_body(), usize::MAX)
        .await
        .expect("read list body");
    let json: Value = serde_json::from_slice(&body).expect("parse list json");
    let names = json["collections"].as_array().expect("collections array");
    assert!(!names.iter().any(|v| v == "docs_a"));
    assert!(names.iter().any(|v| v == "docs_b"));
}

#[tokio::test]
async fn index_ddl_routes_create_list_and_drop_indexes() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create_collection = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create_collection.status(), StatusCode::CREATED);

    let collection_meta_path = tempdir
        .path()
        .join("collections")
        .join("docs")
        .join("collection.json");
    let collection_meta = serde_json::json!({
        "format_version": 1,
        "name": "docs",
        "primary_vector": "title",
        "fields": [
            {
                "name": "session_id",
                "data_type": "String",
                "nullable": false,
                "array": false
            }
        ],
        "vectors": [
            {
                "name": "title",
                "data_type": "VectorFp32",
                "dimension": 2
            }
        ]
    });
    fs::write(
        &collection_meta_path,
        serde_json::to_vec_pretty(&collection_meta).expect("serialize metadata"),
    )
    .expect("write metadata");

    let create_vector = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/indexes/vector")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"field_name":"title","kind":"ivf","metric":"l2","params":{"nlist":8}}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send vector index request");
    let create_vector_status = create_vector.status();
    if !create_vector_status.is_success() {
        let body = to_bytes(create_vector.into_body(), usize::MAX)
            .await
            .expect("read create vector body");
        panic!(
            "vector index create failed: status={} body={}",
            create_vector_status,
            std::str::from_utf8(&body).expect("utf8 body")
        );
    }

    let create_scalar = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/indexes/scalar")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"field_name":"session_id","kind":"inverted","params":{"tokenizer":"keyword"}}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send scalar index request");
    assert!(create_scalar.status().is_success());

    let vector_indexes = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections/docs/indexes/vector")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send vector list request");
    assert_eq!(vector_indexes.status(), StatusCode::OK);
    let vector_body = to_bytes(vector_indexes.into_body(), usize::MAX)
        .await
        .expect("read vector body");
    let vector_json: Value = serde_json::from_slice(&vector_body).expect("parse vector json");
    assert_eq!(vector_json["vector_indexes"][0]["field_name"], "title");
    assert_eq!(vector_json["vector_indexes"][0]["kind"], "ivf");

    let scalar_indexes = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections/docs/indexes/scalar")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send scalar list request");
    assert_eq!(scalar_indexes.status(), StatusCode::OK);
    let scalar_body = to_bytes(scalar_indexes.into_body(), usize::MAX)
        .await
        .expect("read scalar body");
    let scalar_json: Value = serde_json::from_slice(&scalar_body).expect("parse scalar json");
    assert_eq!(scalar_json["scalar_indexes"][0]["field_name"], "session_id");
    assert_eq!(scalar_json["scalar_indexes"][0]["kind"], "inverted");

    let drop_vector = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/collections/docs/indexes/vector/title")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send vector drop request");
    assert!(drop_vector.status().is_success());

    let drop_scalar = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/collections/docs/indexes/scalar/session_id")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send scalar drop request");
    assert!(drop_scalar.status().is_success());
}

#[tokio::test]
async fn index_ddl_routes_reject_invalid_kind_without_panic() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create_collection = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create_collection.status(), StatusCode::CREATED);

    let create_vector = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/indexes/vector")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"field_name":"title","kind":"bogus","params":{"nlist":8}}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send invalid vector index request");
    assert_eq!(create_vector.status(), StatusCode::BAD_REQUEST);

    let create_scalar = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/indexes/scalar")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"field_name":"session_id","kind":"bogus","params":{}}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send invalid scalar index request");
    assert_eq!(create_scalar.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn drop_missing_collection_returns_not_found() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let response = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/collections/missing")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send drop request");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn flush_collection_returns_ok_for_existing_collection() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let flush = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/admin/flush")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send flush request");
    assert_eq!(flush.status(), StatusCode::OK);
}

#[tokio::test]
async fn flush_collection_returns_not_found_for_missing_collection() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let flush = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/missing/admin/flush")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send flush request");
    assert_eq!(flush.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn flush_collection_returns_not_found_if_manifest_entry_exists_but_collection_is_gone() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    fs::remove_dir_all(tempdir.path().join("collections").join("docs"))
        .expect("remove collection dir directly");

    let flush = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/admin/flush")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send flush request");
    assert_eq!(flush.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn collection_info_route_reports_index_complete_after_optimize() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42","84"],"vectors":[[0.0,0.0],[1.0,1.0]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let optimize = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/admin/optimize")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send optimize request");
    assert_eq!(optimize.status(), StatusCode::OK);

    let info = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections/docs")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send info request");
    assert_eq!(info.status(), StatusCode::OK);
    let body = to_bytes(info.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&body).expect("parse json");
    assert_eq!(json["index_completeness"]["vector"], serde_json::json!(1.0));
}

#[cfg(feature = "hanns-backend")]
#[tokio::test]
async fn collection_info_route_preserves_index_complete_after_rebuild_router() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42","84"],"vectors":[[0.0,0.0],[1.0,1.0]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let optimize = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/admin/optimize")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send optimize request");
    assert_eq!(optimize.status(), StatusCode::OK);

    let rebuilt_app = build_router(tempdir.path()).expect("rebuild router");
    let info = rebuilt_app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections/docs")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send info request");
    assert_eq!(info.status(), StatusCode::OK);
    let body = to_bytes(info.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&body).expect("parse json");
    assert_eq!(json["index_completeness"]["vector"], serde_json::json!(1.0));
}

#[cfg(feature = "hanns-backend")]
#[tokio::test]
async fn collection_info_route_clears_index_complete_after_subsequent_write() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42","84"],"vectors":[[0.0,0.0],[1.0,1.0]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let optimize = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/admin/optimize")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send optimize request");
    assert_eq!(optimize.status(), StatusCode::OK);

    let insert_again = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["126"],"vectors":[[2.0,2.0]]}"#))
                .expect("build request"),
        )
        .await
        .expect("send second insert request");
    assert_eq!(insert_again.status(), StatusCode::OK);

    let rebuilt_app = build_router(tempdir.path()).expect("rebuild router");
    let info = rebuilt_app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections/docs")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send info request");
    assert_eq!(info.status(), StatusCode::OK);
    let body = to_bytes(info.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&body).expect("parse json");
    assert_eq!(json["index_completeness"]["vector"], serde_json::json!(0.0));
}

#[tokio::test]
async fn collection_info_route_returns_stats() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42","84"],"vectors":[[0.0,0.0],[1.0,1.0]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let delete = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send delete request");
    assert_eq!(delete.status(), StatusCode::OK);

    let info = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections/docs")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send info request");
    assert_eq!(info.status(), StatusCode::OK);
    let body = to_bytes(info.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&body).expect("parse json");
    assert_eq!(json["name"], "docs");
    assert_eq!(json["dimension"], 2);
    assert_eq!(json["metric"], "l2");
    assert_eq!(json["record_count"], 2);
    assert_eq!(json["deleted_count"], 1);
    assert_eq!(json["live_count"], 1);
}

#[tokio::test]
async fn collection_stats_route_returns_stats() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42","84"],"vectors":[[0.0,0.0],[1.0,1.0]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let delete = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send delete request");
    assert_eq!(delete.status(), StatusCode::OK);

    let stats = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections/docs/stats")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send stats request");
    assert_eq!(stats.status(), StatusCode::OK);
    let body = to_bytes(stats.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&body).expect("parse json");
    assert_eq!(json["name"], "docs");
    assert_eq!(json["dimension"], 2);
    assert_eq!(json["metric"], "l2");
    assert_eq!(json["record_count"], 2);
    assert_eq!(json["deleted_count"], 1);
    assert_eq!(json["live_count"], 1);
}

#[tokio::test]
async fn collection_info_route_returns_not_found_for_missing_collection() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let info = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections/missing")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send info request");
    assert_eq!(info.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn records_upsert_fetch_and_filtered_search_work() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "ids":["42","84","126"],
                        "vectors":[[100.0,100.0],[0.2,0.2],[0.05,0.05]],
                        "fields":[
                            {"session_id":"s1","turn":1,"active":true},
                            {"session_id":"s1","turn":2,"active":true},
                            {"session_id":"s2","turn":3,"active":false}
                        ]
                    }"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let search = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"vector":[0.0,0.0],"top_k":1,"filter":"session_id == \"s1\" and turn >= 2","output_fields":["session_id","turn"]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send search request");
    assert_eq!(search.status(), StatusCode::OK);
    let search_body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let search_json: Value = serde_json::from_slice(&search_body).expect("parse search json");
    assert_eq!(search_json["hits"][0]["id"], "84");
    assert_eq!(search_json["hits"][0]["fields"]["session_id"], "s1");
    assert_eq!(search_json["hits"][0]["fields"]["turn"], 2);
    assert!(search_json["hits"][0]["fields"].get("active").is_none());

    let fetch = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/fetch")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["84"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send fetch request");
    assert_eq!(fetch.status(), StatusCode::OK);
    let fetch_body = to_bytes(fetch.into_body(), usize::MAX)
        .await
        .expect("read fetch body");
    let fetch_json: Value = serde_json::from_slice(&fetch_body).expect("parse fetch json");
    assert_eq!(fetch_json["documents"][0]["id"], "84");
    assert_eq!(fetch_json["documents"][0]["fields"]["session_id"], "s1");
    assert_eq!(fetch_json["documents"][0]["fields"]["turn"], 2);
    assert_eq!(
        fetch_json["documents"][0]["vector"],
        serde_json::json!([0.2, 0.2])
    );

    let upsert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/upsert")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "ids":["84"],
                        "vectors":[[0.0,0.0]],
                        "fields":[{"session_id":"s1","turn":4,"active":true}]
                    }"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send upsert request");
    assert_eq!(upsert.status(), StatusCode::OK);
    let upsert_body = to_bytes(upsert.into_body(), usize::MAX)
        .await
        .expect("read upsert body");
    let upsert_json: Value = serde_json::from_slice(&upsert_body).expect("parse upsert json");
    assert_eq!(upsert_json["upserted"], 1);

    let fetch_after_upsert = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records/fetch")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["84"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send fetch request");
    assert_eq!(fetch_after_upsert.status(), StatusCode::OK);
    let body = to_bytes(fetch_after_upsert.into_body(), usize::MAX)
        .await
        .expect("read fetch body");
    let json: Value = serde_json::from_slice(&body).expect("parse fetch json");
    assert_eq!(json["documents"][0]["fields"]["turn"], 4);
    assert_eq!(
        json["documents"][0]["vector"],
        serde_json::json!([0.0, 0.0])
    );
}

#[tokio::test]
async fn admin_segments_route_returns_segment_stats_for_single_segment() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42","84"],"vectors":[[0.0,0.0],[1.0,1.0]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let delete = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["42"]}"#))
                .expect("build request"),
        )
        .await
        .expect("send delete request");
    assert_eq!(delete.status(), StatusCode::OK);

    let segments = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections/docs/admin/segments")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send segments request");
    assert_eq!(segments.status(), StatusCode::OK);

    let body = to_bytes(segments.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&body).expect("parse segments json");
    assert_eq!(
        json["segments"].as_array().expect("segments array").len(),
        1
    );
    assert_eq!(json["segments"][0]["id"], "seg-0001");
    assert_eq!(json["segments"][0]["live"], 1);
    assert_eq!(json["segments"][0]["dead"], 1);
    assert_eq!(json["segments"][0]["ann_ready"], false);
}

#[cfg(feature = "hanns-backend")]
#[tokio::test]
async fn admin_segments_route_reports_ann_ready_after_optimize_then_false_after_write() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42","84"],"vectors":[[0.0,0.0],[1.0,1.0]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let optimize = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/admin/optimize")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send optimize request");
    assert_eq!(optimize.status(), StatusCode::OK);

    let segments_ready = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections/docs/admin/segments")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send segments request");
    assert_eq!(segments_ready.status(), StatusCode::OK);
    let body = to_bytes(segments_ready.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&body).expect("parse segments json");
    assert_eq!(json["segments"][0]["ann_ready"], true);

    let insert_again = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ids":["126"],"vectors":[[2.0,2.0]]}"#))
                .expect("build request"),
        )
        .await
        .expect("send second insert request");
    assert_eq!(insert_again.status(), StatusCode::OK);

    let segments_stale = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections/docs/admin/segments")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send stale segments request");
    assert_eq!(segments_stale.status(), StatusCode::OK);
    let body = to_bytes(segments_stale.into_body(), usize::MAX)
        .await
        .expect("read stale body");
    let json: Value = serde_json::from_slice(&body).expect("parse stale segments json");
    assert_eq!(json["segments"][0]["ann_ready"], false);
}

#[cfg(feature = "hanns-backend")]
#[tokio::test]
async fn admin_segments_route_preserves_ann_ready_after_rebuild_router() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let insert = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/records")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ids":["42","84"],"vectors":[[0.0,0.0],[1.0,1.0]]}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("send insert request");
    assert_eq!(insert.status(), StatusCode::OK);

    let optimize = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/admin/optimize")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send optimize request");
    assert_eq!(optimize.status(), StatusCode::OK);

    let rebuilt_app = build_router(tempdir.path()).expect("rebuild router");
    let segments = rebuilt_app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/collections/docs/admin/segments")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send segments request");
    assert_eq!(segments.status(), StatusCode::OK);
    let body = to_bytes(segments.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&body).expect("parse json");
    assert_eq!(json["segments"][0]["ann_ready"], true);
}

#[tokio::test]
async fn admin_compact_route_dispatches_to_core() {
    let tempdir = tempfile::tempdir().expect("create tempdir");
    let app = build_router(tempdir.path()).expect("build router");

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"docs","dimension":2,"metric":"l2"}"#))
                .expect("build request"),
        )
        .await
        .expect("send create request");
    assert_eq!(create.status(), StatusCode::CREATED);

    let compact = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/collections/docs/admin/compact")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send compact request");
    assert_eq!(compact.status(), StatusCode::OK);

    let body = to_bytes(compact.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: Value = serde_json::from_slice(&body).expect("parse compact json");
    assert_eq!(json["compacted"], true);
}
