use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use hannsdb_daemon::routes::build_router;
use serde_json::Value;
use std::fs;
use tower::ServiceExt;

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
async fn search_route_returns_bad_request_for_unsupported_typed_shape() {
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

    assert_eq!(search.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(search.into_body(), usize::MAX)
        .await
        .expect("read search body");
    let json: Value = serde_json::from_slice(&body).expect("parse search json");
    assert!(json["error"]
        .as_str()
        .expect("error string")
        .contains("include_vector"));
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
