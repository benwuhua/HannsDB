use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex};

use axum::extract::{rejection::JsonRejection, Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{Document, FieldValue};
use hannsdb_core::query::{
    QueryContext, QueryGroupBy, QueryReranker, VectorQuery, VectorQueryParam,
};

use crate::api::{
    CollectionInfoResponse, CompactCollectionResponse, CreateCollectionRequest,
    CreateCollectionResponse, CreateIndexResponse, DeleteByFilterRequest, DeleteRecordsRequest,
    DeleteRecordsResponse, DropCollectionResponse, DropIndexResponse, ErrorResponse,
    FetchRecordResponse, FetchRecordsRequest, FetchRecordsResponse, FlushCollectionResponse,
    HealthResponse, InsertRecordsRequest, InsertRecordsResponse, LegacySearchRequest,
    ListCollectionsResponse, ScalarIndexRequest, ScalarIndexesResponse, SearchHitResponse,
    SearchRequest, SearchResponse, SegmentsResponse, TypedSearchRequest, UpsertRecordsResponse,
    VectorIndexRequest, VectorIndexesResponse,
};

#[derive(Clone)]
struct DaemonState {
    db: Arc<Mutex<HannsDb>>,
}

pub fn build_router(root: &Path) -> io::Result<Router> {
    let db = HannsDb::open(root)?;
    let state = DaemonState {
        db: Arc::new(Mutex::new(db)),
    };

    Ok(Router::new()
        .route("/health", get(health))
        .route(
            "/collections",
            get(list_collections).post(create_collection),
        )
        .route(
            "/collections/:collection",
            get(get_collection).delete(drop_collection),
        )
        .route("/collections/:collection/stats", get(get_collection_stats))
        .route(
            "/collections/:collection/admin/flush",
            post(flush_collection),
        )
        .route(
            "/collections/:collection/admin/compact",
            post(compact_collection),
        )
        .route(
            "/collections/:collection/admin/segments",
            get(get_collection_segments),
        )
        .route(
            "/collections/:collection/records",
            post(insert_records).delete(delete_records),
        )
        .route(
            "/collections/:collection/records/delete_by_filter",
            post(delete_records_by_filter),
        )
        .route(
            "/collections/:collection/records/upsert",
            post(upsert_records),
        )
        .route(
            "/collections/:collection/records/fetch",
            post(fetch_records),
        )
        .route("/collections/:collection/search", post(search_records))
        .route(
            "/collections/:collection/indexes/vector",
            get(list_vector_indexes).post(create_vector_index),
        )
        .route(
            "/collections/:collection/indexes/vector/:field_name",
            delete(drop_vector_index),
        )
        .route(
            "/collections/:collection/indexes/scalar",
            get(list_scalar_indexes).post(create_scalar_index),
        )
        .route(
            "/collections/:collection/indexes/scalar/:field_name",
            delete(drop_scalar_index),
        )
        .with_state(state))
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse::ok())
}

async fn create_collection(
    State(state): State<DaemonState>,
    Json(request): Json<CreateCollectionRequest>,
) -> Response {
    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .create_collection(&request.name, request.dimension, &request.metric);

    match result {
        Ok(()) => (
            StatusCode::CREATED,
            Json(CreateCollectionResponse { name: request.name }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => (
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn list_collections(State(state): State<DaemonState>) -> Response {
    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .list_collections();

    match result {
        Ok(collections) => (
            StatusCode::OK,
            Json(ListCollectionsResponse { collections }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn drop_collection(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
) -> Response {
    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .drop_collection(&collection);

    match result {
        Ok(()) => (
            StatusCode::OK,
            Json(DropCollectionResponse {
                dropped: collection,
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn get_collection(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
) -> Response {
    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .get_collection_info(&collection);

    collection_info_response(result)
}

async fn get_collection_stats(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
) -> Response {
    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .get_collection_info(&collection);

    collection_info_response(result)
}

fn collection_info_response(result: io::Result<hannsdb_core::db::CollectionInfo>) -> Response {
    match result {
        Ok(info) => (
            StatusCode::OK,
            Json(CollectionInfoResponse {
                name: info.name,
                dimension: info.dimension,
                metric: info.metric,
                record_count: info.record_count,
                deleted_count: info.deleted_count,
                live_count: info.live_count,
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn flush_collection(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
) -> Response {
    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .flush_collection(&collection);

    match result {
        Ok(()) => (
            StatusCode::OK,
            Json(FlushCollectionResponse {
                flushed: collection,
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn compact_collection(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
) -> Response {
    let mut db = state.db.lock().expect("daemon state mutex poisoned");
    let result = db.compact_collection(&collection);

    match result {
        Ok(()) => (
            StatusCode::OK,
            Json(CompactCollectionResponse { compacted: true }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::Unsupported => (
            StatusCode::NOT_IMPLEMENTED,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn get_collection_segments(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
) -> Response {
    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .list_collection_segments(&collection);

    match result {
        Ok(segments) => (
            StatusCode::OK,
            Json(SegmentsResponse {
                segments: segments
                    .into_iter()
                    .map(|segment| crate::api::SegmentInfoResponse {
                        id: segment.id,
                        live: segment.live_count,
                        dead: segment.dead_count,
                        ann_ready: segment.ann_ready,
                    })
                    .collect(),
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn insert_records(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
    Json(request): Json<InsertRecordsRequest>,
) -> Response {
    let documents = match build_documents(request) {
        Ok(documents) => documents,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response()
        }
    };

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .insert_documents(&collection, &documents);

    match result {
        Ok(inserted) => (
            StatusCode::OK,
            Json(InsertRecordsResponse {
                inserted: inserted as u64,
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::InvalidInput => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn upsert_records(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
    Json(request): Json<InsertRecordsRequest>,
) -> Response {
    let documents = match build_documents(request) {
        Ok(documents) => documents,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response()
        }
    };

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .upsert_documents(&collection, &documents);

    match result {
        Ok(upserted) => (
            StatusCode::OK,
            Json(UpsertRecordsResponse {
                upserted: upserted as u64,
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::InvalidInput => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn delete_records(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
    Json(request): Json<DeleteRecordsRequest>,
) -> Response {
    let external_ids = match parse_external_ids(&request.ids) {
        Ok(ids) => ids,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response()
        }
    };

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .delete(&collection, &external_ids);

    match result {
        Ok(deleted) => (
            StatusCode::OK,
            Json(DeleteRecordsResponse {
                deleted: deleted as u64,
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::InvalidInput => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn delete_records_by_filter(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
    request: Result<Json<DeleteByFilterRequest>, JsonRejection>,
) -> Response {
    let request = match request {
        Ok(Json(request)) => request,
        Err(rejection) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: rejection.body_text(),
                }),
            )
                .into_response()
        }
    };

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .delete_by_filter(&collection, &request.filter);

    match result {
        Ok(deleted) => (
            StatusCode::OK,
            Json(DeleteRecordsResponse {
                deleted: deleted as u64,
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("collection not found: {collection}"),
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::InvalidInput => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn fetch_records(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
    Json(request): Json<FetchRecordsRequest>,
) -> Response {
    let output_fields = request.output_fields.as_deref();
    let external_ids = match parse_external_ids(&request.ids) {
        Ok(ids) => ids,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response()
        }
    };

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .fetch_documents(&collection, &external_ids);

    match result {
        Ok(documents) => (
            StatusCode::OK,
            Json(FetchRecordsResponse {
                documents: documents
                    .into_iter()
                    .map(|document| FetchRecordResponse {
                        id: document.id.to_string(),
                        fields: select_fetch_output_fields(&document.fields, output_fields),
                        vector: document.vector,
                    })
                    .collect(),
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::InvalidInput => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn search_records(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
    request: Result<Json<serde_json::Value>, JsonRejection>,
) -> Response {
    let request = match request {
        Ok(Json(request)) => request,
        Err(rejection) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: rejection.body_text(),
                }),
            )
                .into_response()
        }
    };
    let request = match classify_search_request(request) {
        Ok(request) => request,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response()
        }
    };

    let result = match request {
        SearchRequest::Legacy(request) => search_records_legacy(&state, &collection, request),
        SearchRequest::Typed(request) => search_records_typed(&state, &collection, request),
    };

    match result {
        Ok(hits) => (StatusCode::OK, Json(SearchResponse { hits })).into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::InvalidInput => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::Unsupported => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

fn classify_search_request(payload: serde_json::Value) -> Result<SearchRequest, String> {
    let object = payload
        .as_object()
        .ok_or_else(|| "search request body must be a JSON object".to_string())?;

    let has_legacy_marker = object.contains_key("vector");
    let has_typed_marker = [
        "queries",
        "query_by_id",
        "query_by_id_field_name",
        "include_vector",
        "group_by",
        "reranker",
    ]
    .iter()
    .any(|key| object.contains_key(*key));

    if has_legacy_marker && has_typed_marker {
        return Err("search request body cannot mix legacy and typed query keys".to_string());
    }

    if has_legacy_marker {
        return serde_json::from_value::<LegacySearchRequest>(payload)
            .map(SearchRequest::Legacy)
            .map_err(|error| error.to_string());
    }

    serde_json::from_value::<TypedSearchRequest>(payload)
        .map(SearchRequest::Typed)
        .map_err(|error| error.to_string())
}

fn search_records_legacy(
    state: &DaemonState,
    collection: &str,
    request: LegacySearchRequest,
) -> io::Result<Vec<SearchHitResponse>> {
    let output_fields = request.output_fields.as_deref();
    let include_fields = matches!(output_fields, Some(fields) if !fields.is_empty());

    if let Some(filter) = request
        .filter
        .as_deref()
        .map(str::trim)
        .filter(|f| !f.is_empty())
    {
        return state
            .db
            .lock()
            .expect("daemon state mutex poisoned")
            .query_documents(collection, &request.vector, request.top_k, Some(filter))
            .map(|hits| {
                hits.into_iter()
                    .map(|hit| SearchHitResponse {
                        id: hit.id.to_string(),
                        distance: hit.distance,
                        fields: select_output_fields(&hit.fields, output_fields),
                    })
                    .collect()
            });
    }

    if include_fields {
        return state
            .db
            .lock()
            .expect("daemon state mutex poisoned")
            .query_documents(collection, &request.vector, request.top_k, None)
            .map(|hits| {
                hits.into_iter()
                    .map(|hit| SearchHitResponse {
                        id: hit.id.to_string(),
                        distance: hit.distance,
                        fields: select_output_fields(&hit.fields, output_fields),
                    })
                    .collect()
            });
    }

    state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .search(collection, &request.vector, request.top_k)
        .map(|hits| {
            hits.into_iter()
                .map(|hit| SearchHitResponse {
                    id: hit.id.to_string(),
                    distance: hit.distance,
                    fields: BTreeMap::new(),
                })
                .collect()
        })
}

fn search_records_typed(
    state: &DaemonState,
    collection: &str,
    request: TypedSearchRequest,
) -> io::Result<Vec<SearchHitResponse>> {
    if request.include_vector {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "include_vector is not supported for typed search",
        ));
    }

    let include_fields =
        matches!(request.output_fields.as_deref(), Some(fields) if !fields.is_empty());
    let context = build_query_context(request)?;

    state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .query_with_context(collection, &context)
        .map(|hits| {
            hits.into_iter()
                .map(|hit| SearchHitResponse {
                    id: hit.id.to_string(),
                    distance: hit.distance,
                    fields: if include_fields {
                        field_values_to_json(hit.fields)
                    } else {
                        BTreeMap::new()
                    },
                })
                .collect()
        })
}

async fn create_vector_index(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
    Json(request): Json<VectorIndexRequest>,
) -> Response {
    let VectorIndexRequest {
        field_name,
        kind,
        metric,
        params,
    } = request;
    let descriptor = serde_json::json!({
        "field_name": field_name,
        "kind": kind,
        "metric": metric,
        "params": params,
    });
    let descriptor = match serde_json::from_value(descriptor) {
        Ok(descriptor) => descriptor,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: error.to_string(),
                }),
            )
                .into_response()
        }
    };
    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .create_vector_index(&collection, descriptor);

    match result {
        Ok(()) => (
            StatusCode::CREATED,
            Json(CreateIndexResponse { field_name }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::InvalidInput => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn list_vector_indexes(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
) -> Response {
    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .list_vector_indexes(&collection);

    match result {
        Ok(vector_indexes) => (
            StatusCode::OK,
            Json(VectorIndexesResponse {
                vector_indexes: vector_indexes
                    .into_iter()
                    .map(|descriptor| serde_json::to_value(descriptor).expect("descriptor json"))
                    .collect(),
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn drop_vector_index(
    State(state): State<DaemonState>,
    AxumPath((collection, field_name)): AxumPath<(String, String)>,
) -> Response {
    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .drop_vector_index(&collection, &field_name);

    match result {
        Ok(()) => (
            StatusCode::OK,
            Json(DropIndexResponse {
                dropped: field_name,
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn create_scalar_index(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
    Json(request): Json<ScalarIndexRequest>,
) -> Response {
    let ScalarIndexRequest {
        field_name,
        kind,
        params,
    } = request;
    let descriptor = serde_json::json!({
        "field_name": field_name,
        "kind": kind,
        "params": params,
    });
    let descriptor = match serde_json::from_value(descriptor) {
        Ok(descriptor) => descriptor,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: error.to_string(),
                }),
            )
                .into_response()
        }
    };
    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .create_scalar_index(&collection, descriptor);

    match result {
        Ok(()) => (
            StatusCode::CREATED,
            Json(CreateIndexResponse { field_name }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::InvalidInput => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn list_scalar_indexes(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
) -> Response {
    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .list_scalar_indexes(&collection);

    match result {
        Ok(scalar_indexes) => (
            StatusCode::OK,
            Json(ScalarIndexesResponse {
                scalar_indexes: scalar_indexes
                    .into_iter()
                    .map(|descriptor| serde_json::to_value(descriptor).expect("descriptor json"))
                    .collect(),
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn drop_scalar_index(
    State(state): State<DaemonState>,
    AxumPath((collection, field_name)): AxumPath<(String, String)>,
) -> Response {
    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .drop_scalar_index(&collection, &field_name);

    match result {
        Ok(()) => (
            StatusCode::OK,
            Json(DropIndexResponse {
                dropped: field_name,
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

fn parse_external_ids(ids: &[String]) -> Result<Vec<i64>, String> {
    ids.iter()
        .map(|id| {
            id.parse::<i64>()
                .map_err(|_| format!("invalid id, expected i64 string: {id}"))
        })
        .collect()
}

fn build_documents(request: InsertRecordsRequest) -> Result<Vec<Document>, String> {
    let external_ids = parse_external_ids(&request.ids)?;
    if request.vectors.len() != external_ids.len() {
        return Err("vector count must match id count".to_string());
    }

    let fields = if request.fields.is_empty() {
        vec![BTreeMap::new(); external_ids.len()]
    } else if request.fields.len() == external_ids.len() {
        request
            .fields
            .into_iter()
            .map(json_fields_to_field_values)
            .collect::<Result<Vec<_>, _>>()?
    } else {
        return Err("fields count must match id count".to_string());
    };

    external_ids
        .into_iter()
        .zip(request.vectors)
        .zip(fields)
        .map(|((id, vector), fields)| Ok(Document::new(id, fields, vector)))
        .collect()
}

fn json_fields_to_field_values(
    fields: BTreeMap<String, serde_json::Value>,
) -> Result<BTreeMap<String, FieldValue>, String> {
    fields
        .into_iter()
        .map(|(name, value)| Ok((name, json_value_to_field_value(value)?)))
        .collect()
}

fn json_value_to_field_value(value: serde_json::Value) -> Result<FieldValue, String> {
    match value {
        serde_json::Value::String(value) => Ok(FieldValue::String(value)),
        serde_json::Value::Bool(value) => Ok(FieldValue::Bool(value)),
        serde_json::Value::Number(value) => value
            .as_i64()
            .map(FieldValue::Int64)
            .or_else(|| value.as_f64().map(FieldValue::Float64))
            .ok_or_else(|| "unsupported numeric field value".to_string()),
        other => Err(format!("unsupported field value: {other}")),
    }
}

fn field_values_to_json(
    fields: BTreeMap<String, FieldValue>,
) -> BTreeMap<String, serde_json::Value> {
    fields
        .into_iter()
        .map(|(name, value)| (name, field_value_to_json(value)))
        .collect()
}

fn select_output_fields(
    fields: &BTreeMap<String, FieldValue>,
    output_fields: Option<&[String]>,
) -> BTreeMap<String, serde_json::Value> {
    match output_fields {
        Some(names) if !names.is_empty() => names
            .iter()
            .filter_map(|name| {
                fields
                    .get(name)
                    .map(|value| (name.clone(), field_value_to_json(value.clone())))
            })
            .collect(),
        _ => BTreeMap::new(),
    }
}

fn select_fetch_output_fields(
    fields: &BTreeMap<String, FieldValue>,
    output_fields: Option<&[String]>,
) -> BTreeMap<String, serde_json::Value> {
    match output_fields {
        Some(names) => select_output_fields(fields, Some(names)),
        None => field_values_to_json(fields.clone()),
    }
}

fn build_query_context(request: TypedSearchRequest) -> io::Result<QueryContext> {
    let query_by_id = request
        .query_by_id
        .map(|ids| {
            parse_external_ids(&ids)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))
        })
        .transpose()?;

    Ok(QueryContext {
        top_k: request.top_k,
        queries: request
            .queries
            .into_iter()
            .map(|query| VectorQuery {
                field_name: query.field_name,
                vector: query.vector,
                param: query.param.map(|param| VectorQueryParam {
                    ef_search: param.ef_search,
                }),
            })
            .collect(),
        query_by_id,
        query_by_id_field_name: request.query_by_id_field_name,
        filter: request.filter,
        output_fields: request.output_fields,
        include_vector: request.include_vector,
        group_by: request.group_by.map(|group_by| QueryGroupBy {
            field_name: group_by.field_name,
        }),
        reranker: request.reranker.map(|reranker| QueryReranker {
            model: reranker.model,
        }),
    })
}

fn field_value_to_json(value: FieldValue) -> serde_json::Value {
    match value {
        FieldValue::String(value) => serde_json::Value::String(value),
        FieldValue::Int64(value) => serde_json::Value::Number(value.into()),
        FieldValue::Float64(value) => serde_json::Value::Number(
            serde_json::Number::from_f64(value)
                .expect("finite float field values should serialize"),
        ),
        FieldValue::Bool(value) => serde_json::Value::Bool(value),
    }
}
