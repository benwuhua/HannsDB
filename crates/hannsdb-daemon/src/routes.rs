use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[cfg(feature = "lance-storage")]
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(feature = "lance-storage")]
use tokio::sync::Mutex as AsyncMutex;

use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use hannsdb_core::document::{
    Document, FieldType, FieldValue, ScalarFieldSchema, SparseVector, VectorFieldSchema,
    VectorIndexSchema,
};
use hannsdb_core::query::{
    QueryContext, QueryGroupBy, QueryReranker, QueryVector, VectorQuery, VectorQueryParam,
};
use hannsdb_core::HannsDb;
#[cfg(feature = "lance-storage")]
use hannsdb_core::{storage::lance_store::LanceCollection, CollectionSchema};

use crate::api::{
    AddColumnRequest, AddColumnResponse, AddVectorFieldRequest, AddVectorFieldResponse,
    AlterColumnRequest, AlterColumnResponse, CollectionInfoResponse, CompactCollectionResponse,
    CreateCollectionRequest, CreateCollectionResponse, CreateIndexResponse, DropCollectionResponse,
    DropColumnResponse, DropIndexResponse, DropVectorFieldResponse, ErrorResponse,
    FlushCollectionResponse, HealthResponse, InsertRecordsRequest, ListCollectionsResponse,
    OptimizeCollectionResponse, ScalarIndexRequest, ScalarIndexesResponse, SegmentsResponse,
    VectorIndexRequest, VectorIndexesResponse,
};

use super::routes_mutation;
use super::routes_search;

#[derive(Clone)]
pub(crate) struct DaemonState {
    #[cfg_attr(not(feature = "lance-storage"), allow(dead_code))]
    pub(crate) root: PathBuf,
    pub(crate) db: Arc<Mutex<HannsDb>>,
    #[cfg(feature = "lance-storage")]
    pub(crate) lance_insert_lock: Arc<AsyncMutex<()>>,
}

#[cfg(feature = "lance-storage")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LanceDaemonCollectionMetadata {
    metric: String,
}

pub fn build_router(root: &Path) -> io::Result<Router> {
    let db = HannsDb::open(root)?;
    let state = DaemonState {
        root: root.to_path_buf(),
        db: Arc::new(Mutex::new(db)),
        #[cfg(feature = "lance-storage")]
        lance_insert_lock: Arc::new(AsyncMutex::new(())),
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
            "/collections/:collection/admin/optimize",
            post(optimize_collection),
        )
        .route(
            "/collections/:collection/records",
            post(routes_mutation::insert_records).delete(routes_mutation::delete_records),
        )
        .route(
            "/collections/:collection/records/delete_by_filter",
            post(routes_mutation::delete_records_by_filter),
        )
        .route(
            "/collections/:collection/records/upsert",
            post(routes_mutation::upsert_records),
        )
        .route(
            "/collections/:collection/records/update",
            post(routes_mutation::update_records),
        )
        .route(
            "/collections/:collection/records/fetch",
            post(routes_search::fetch_records),
        )
        .route(
            "/collections/:collection/search",
            post(routes_search::search_records),
        )
        .route(
            "/collections/:collection/schema/columns",
            post(add_column_to_collection),
        )
        .route(
            "/collections/:collection/schema/columns/:field_name",
            delete(drop_column_from_collection).patch(alter_column_in_collection),
        )
        .route(
            "/collections/:collection/schema/vectors",
            post(add_vector_field_to_collection),
        )
        .route(
            "/collections/:collection/schema/vectors/:field_name",
            delete(drop_vector_field_from_collection),
        )
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
    match request
        .storage
        .as_deref()
        .map(normalize_storage_backend)
        .transpose()
    {
        Ok(Some("lance")) => return create_lance_collection(state, request).await,
        Ok(Some("hannsdb")) | Ok(None) => {}
        Ok(Some(_)) => unreachable!("normalize_storage_backend returns known values"),
        Err(error) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response()
        }
    }

    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &request.name) {
        return (
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                error: format!("collection already exists: {}", request.name),
            }),
        )
            .into_response();
    }

    let Some(dimension) = request.dimension else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "dimension is required for native collection create".to_string(),
            }),
        )
            .into_response();
    };
    let Some(metric) = request.metric.as_deref() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "metric is required for native collection create".to_string(),
            }),
        )
            .into_response();
    };

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .create_collection(&request.name, dimension, metric);

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

fn normalize_storage_backend(value: &str) -> Result<&'static str, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "hannsdb" | "default" | "" => Ok("hannsdb"),
        "lance" => Ok("lance"),
        other => Err(format!("unsupported storage backend: {other}")),
    }
}

#[cfg(feature = "lance-storage")]
async fn create_lance_collection(state: DaemonState, request: CreateCollectionRequest) -> Response {
    let native_exists = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .list_collections()
        .map(|collections| collections.iter().any(|name| name == &request.name))
        .unwrap_or(false);
    if native_exists {
        return (
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                error: format!("collection already exists: {}", request.name),
            }),
        )
            .into_response();
    }

    let metric = request
        .schema
        .as_ref()
        .map(|schema| schema.metric().to_string());
    let metric = match (metric, request.metric.as_deref()) {
        (Some(metric), _) => metric,
        (None, Some(metric)) => metric.to_string(),
        (None, None) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "metric is required for Lance create without schema".to_string(),
                }),
            )
                .into_response()
        }
    };
    let schema = match request.schema {
        Some(schema) => schema,
        None => {
            let Some(dimension) = request.dimension else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: "dimension is required for Lance create without schema".to_string(),
                    }),
                )
                    .into_response();
            };
            CollectionSchema::new(
                "vector",
                dimension,
                metric.clone(),
                Vec::<ScalarFieldSchema>::new(),
            )
        }
    };
    match LanceCollection::create(&state.root, request.name.clone(), schema, &[]).await {
        Ok(_) => match write_lance_daemon_metadata(&state, &request.name, &metric) {
            Ok(()) => (
                StatusCode::CREATED,
                Json(CreateCollectionResponse { name: request.name }),
            )
                .into_response(),
            Err(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: error.to_string(),
                }),
            )
                .into_response(),
        },
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => (
            StatusCode::CONFLICT,
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

#[cfg(not(feature = "lance-storage"))]
async fn create_lance_collection(
    _state: DaemonState,
    _request: CreateCollectionRequest,
) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: "Lance storage requires the lance-storage feature".to_string(),
        }),
    )
        .into_response()
}

#[cfg(feature = "lance-storage")]
pub(crate) fn lance_collection_exists(state: &DaemonState, collection: &str) -> bool {
    state
        .root
        .join("collections")
        .join(format!("{collection}.lance"))
        .exists()
}

#[cfg(feature = "lance-storage")]
pub(crate) async fn open_lance_collection(
    state: &DaemonState,
    collection: &str,
) -> io::Result<LanceCollection> {
    LanceCollection::open_inferred(&state.root, collection).await
}

#[cfg(feature = "lance-storage")]
fn lance_daemon_metadata_path(state: &DaemonState, collection: &str) -> PathBuf {
    state
        .root
        .join("collections")
        .join(format!("{collection}.lance"))
        .join("_hannsdb")
        .join("collection.json")
}

#[cfg(feature = "lance-storage")]
fn write_lance_daemon_metadata(
    state: &DaemonState,
    collection: &str,
    metric: &str,
) -> io::Result<()> {
    let path = lance_daemon_metadata_path(state, collection);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let metadata = LanceDaemonCollectionMetadata {
        metric: metric.to_string(),
    };
    let bytes = serde_json::to_vec_pretty(&metadata)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    std::fs::write(path, bytes)
}

#[cfg(feature = "lance-storage")]
fn read_lance_daemon_metadata(
    state: &DaemonState,
    collection: &str,
) -> io::Result<Option<LanceDaemonCollectionMetadata>> {
    let path = lance_daemon_metadata_path(state, collection);
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

#[cfg(feature = "lance-storage")]
pub(crate) fn lance_collection_metric(
    state: &DaemonState,
    collection: &str,
    fallback: &str,
) -> io::Result<String> {
    Ok(read_lance_daemon_metadata(state, collection)?
        .map(|metadata| metadata.metric)
        .unwrap_or_else(|| fallback.to_string()))
}

async fn list_collections(State(state): State<DaemonState>) -> Response {
    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .list_collections();

    match result {
        Ok(mut collections) => {
            #[cfg(feature = "lance-storage")]
            match list_lance_collections(&state) {
                Ok(lance_collections) => collections.extend(lance_collections),
                Err(error) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: error.to_string(),
                        }),
                    )
                        .into_response()
                }
            }
            collections.sort();
            collections.dedup();
            (
                StatusCode::OK,
                Json(ListCollectionsResponse { collections }),
            )
                .into_response()
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

#[cfg(feature = "lance-storage")]
fn list_lance_collections(state: &DaemonState) -> io::Result<Vec<String>> {
    let collections_dir = state.root.join("collections");
    let entries = match std::fs::read_dir(&collections_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if let Some(name) = file_name.strip_suffix(".lance") {
            names.push(name.to_string());
        }
    }
    Ok(names)
}

async fn drop_collection(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
) -> Response {
    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &collection) {
        let path = state
            .root
            .join("collections")
            .join(format!("{collection}.lance"));
        return match std::fs::remove_dir_all(path) {
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
        };
    }

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
    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &collection) {
        let result = lance_collection_info(&state, &collection).await;
        return collection_info_response(result);
    }

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .get_collection_info(&collection);

    collection_info_response(result)
}

#[cfg(feature = "lance-storage")]
async fn lance_collection_info(
    state: &DaemonState,
    collection: &str,
) -> io::Result<hannsdb_core::CollectionInfo> {
    let lance = open_lance_collection(state, collection).await?;
    let live_count = lance.count_rows().await?;
    let metric = lance_collection_metric(state, collection, lance.schema().metric())?;
    let index_completeness = lance_index_completeness(&lance, live_count);
    Ok(hannsdb_core::CollectionInfo {
        name: collection.to_string(),
        dimension: lance.schema().dimension(),
        metric,
        record_count: live_count,
        deleted_count: 0,
        live_count,
        index_completeness,
    })
}

#[cfg(all(feature = "lance-storage", feature = "hanns-backend"))]
fn lance_index_completeness(lance: &LanceCollection, live_count: usize) -> BTreeMap<String, f64> {
    let mut index_completeness = BTreeMap::new();
    let field_name = lance.schema().primary_vector_name().to_string();
    let completeness = if live_count == 0 || lance.hanns_index_path(&field_name).exists() {
        1.0
    } else {
        0.0
    };
    index_completeness.insert(field_name, completeness);
    index_completeness
}

#[cfg(all(feature = "lance-storage", not(feature = "hanns-backend")))]
fn lance_index_completeness(_lance: &LanceCollection, _live_count: usize) -> BTreeMap<String, f64> {
    BTreeMap::new()
}

async fn get_collection_stats(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
) -> Response {
    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &collection) {
        let result = lance_collection_info(&state, &collection).await;
        return collection_info_response(result);
    }

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .get_collection_info(&collection);

    collection_info_response(result)
}

fn collection_info_response(result: io::Result<hannsdb_core::CollectionInfo>) -> Response {
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
                index_completeness: info.index_completeness,
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
    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &collection) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Lance compact is not supported by this daemon route yet".to_string(),
            }),
        )
            .into_response();
    }

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

async fn optimize_collection(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
) -> Response {
    #[cfg(all(feature = "lance-storage", feature = "hanns-backend"))]
    if lance_collection_exists(&state, &collection) {
        let result = match open_lance_collection(&state, &collection).await {
            Ok(lance) => {
                let field_name = lance.schema().primary_vector_name().to_string();
                let metric = lance_collection_metric(&state, &collection, lance.schema().metric());
                match metric {
                    Ok(metric) => lance.optimize_hanns(&field_name, &metric).await,
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error),
        };
        return optimize_collection_response(result, collection);
    }

    #[cfg(all(feature = "lance-storage", not(feature = "hanns-backend")))]
    if lance_collection_exists(&state, &collection) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Lance optimize requires the hanns-backend feature".to_string(),
            }),
        )
            .into_response();
    }

    let db = state.db.lock().expect("daemon state mutex poisoned");
    let result = db.optimize_collection(&collection);

    optimize_collection_response(result, collection)
}

fn optimize_collection_response(result: io::Result<()>, collection: String) -> Response {
    match result {
        Ok(()) => (
            StatusCode::OK,
            Json(OptimizeCollectionResponse {
                optimized: collection,
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
    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &collection) {
        let result = lance_collection_segments(&state, &collection).await;
        return collection_segments_response(result);
    }

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .list_collection_segments(&collection);

    collection_segments_response(result.map(|segments| {
        segments
            .into_iter()
            .map(|segment| crate::api::SegmentInfoResponse {
                id: segment.id,
                live: segment.live_count,
                dead: segment.dead_count,
                ann_ready: segment.ann_ready,
            })
            .collect()
    }))
}

#[cfg(feature = "lance-storage")]
async fn lance_collection_segments(
    state: &DaemonState,
    collection: &str,
) -> io::Result<Vec<crate::api::SegmentInfoResponse>> {
    let lance = open_lance_collection(state, collection).await?;
    let live = lance.count_rows().await?;
    Ok(vec![crate::api::SegmentInfoResponse {
        id: "lance".to_string(),
        live,
        dead: 0,
        ann_ready: lance_primary_ann_ready(&lance),
    }])
}

#[cfg(all(feature = "lance-storage", feature = "hanns-backend"))]
fn lance_primary_ann_ready(lance: &LanceCollection) -> bool {
    let field_name = lance.schema().primary_vector_name();
    lance.hanns_index_path(field_name).exists()
}

#[cfg(all(feature = "lance-storage", not(feature = "hanns-backend")))]
fn lance_primary_ann_ready(_lance: &LanceCollection) -> bool {
    false
}

fn collection_segments_response(
    result: io::Result<Vec<crate::api::SegmentInfoResponse>>,
) -> Response {
    match result {
        Ok(segments) => (StatusCode::OK, Json(SegmentsResponse { segments })).into_response(),
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

#[cfg(feature = "lance-storage")]
fn daemon_storage_error_response(error: io::Error) -> Response {
    match error.kind() {
        io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        io::ErrorKind::AlreadyExists => (
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        io::ErrorKind::InvalidInput | io::ErrorKind::Unsupported => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn create_vector_index(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
    Json(request): Json<VectorIndexRequest>,
) -> Response {
    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &collection) {
        return create_lance_vector_index(state, collection, request).await;
    }

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
    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &collection) {
        return list_lance_vector_indexes(state, collection).await;
    }

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
    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &collection) {
        return drop_lance_vector_index(state, collection, field_name).await;
    }

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
    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &collection) {
        return create_lance_scalar_index(state, collection, request).await;
    }

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
    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &collection) {
        return list_lance_scalar_indexes(state, collection).await;
    }

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
    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &collection) {
        return drop_lance_scalar_index(state, collection, field_name).await;
    }

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

#[cfg(all(feature = "lance-storage", feature = "hanns-backend"))]
async fn create_lance_vector_index(
    state: DaemonState,
    collection: String,
    request: VectorIndexRequest,
) -> Response {
    let kind = request.kind.trim().to_ascii_lowercase();
    if matches!(kind.as_str(), "bm25" | "sparse_inverted" | "sparse_wand") {
        if request
            .params
            .as_object()
            .is_some_and(|params| !params.is_empty())
        {
            return daemon_storage_error_response(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Lance sparse sidecar index supports only default params",
            ));
        }
        let lance = match open_lance_collection(&state, &collection).await {
            Ok(lance) => lance,
            Err(error) => return daemon_storage_error_response(error),
        };
        let metric = request.metric.unwrap_or_else(|| {
            if kind == "bm25" {
                "bm25".to_string()
            } else {
                "ip".to_string()
            }
        });
        return match lance.optimize_sparse(&request.field_name, &metric).await {
            Ok(()) => (
                StatusCode::CREATED,
                Json(CreateIndexResponse {
                    field_name: request.field_name,
                }),
            )
                .into_response(),
            Err(error) => daemon_storage_error_response(error),
        };
    }
    if kind != "hnsw" {
        return daemon_storage_error_response(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Lance Hanns sidecar vector index supports only hnsw kind, got: {}",
                request.kind
            ),
        ));
    }
    if request
        .params
        .as_object()
        .is_some_and(|params| !params.is_empty())
    {
        return daemon_storage_error_response(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Lance sidecar vector indexes support only default params",
        ));
    }

    let lance = match open_lance_collection(&state, &collection).await {
        Ok(lance) => lance,
        Err(error) => return daemon_storage_error_response(error),
    };
    let vector_schema = match lance
        .schema()
        .vectors
        .iter()
        .find(|vector| vector.name == request.field_name)
    {
        Some(vector) => vector,
        None => {
            return daemon_storage_error_response(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("vector field not found: {}", request.field_name),
            ))
        }
    };
    let kind = request.kind.trim().to_ascii_lowercase();
    let metric = match request.metric {
        Some(metric) => metric,
        None => match lance_collection_metric(&state, &collection, lance.schema().metric()) {
            Ok(metric) => metric,
            Err(error) => return daemon_storage_error_response(error),
        },
    };

    let result = if matches!(vector_schema.data_type, FieldType::VectorSparse) {
        if kind != "bm25" {
            return daemon_storage_error_response(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Lance sparse sidecar vector index supports only bm25 kind, got: {}",
                    request.kind
                ),
            ));
        }
        lance.optimize_sparse(&request.field_name, &metric).await
    } else {
        if kind != "hnsw" {
            return daemon_storage_error_response(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Lance Hanns sidecar vector index supports only hnsw kind, got: {}",
                    request.kind
                ),
            ));
        }
        lance.optimize_hanns(&request.field_name, &metric).await
    };

    match result {
        Ok(()) => (
            StatusCode::CREATED,
            Json(CreateIndexResponse {
                field_name: request.field_name,
            }),
        )
            .into_response(),
        Err(error) => daemon_storage_error_response(error),
    }
}

#[cfg(all(feature = "lance-storage", not(feature = "hanns-backend")))]
async fn create_lance_vector_index(
    _state: DaemonState,
    _collection: String,
    _request: VectorIndexRequest,
) -> Response {
    daemon_storage_error_response(io::Error::new(
        io::ErrorKind::Unsupported,
        "Lance vector index DDL requires the hanns-backend feature",
    ))
}

#[cfg(all(feature = "lance-storage", feature = "hanns-backend"))]
async fn list_lance_vector_indexes(state: DaemonState, collection: String) -> Response {
    match open_lance_collection(&state, &collection).await {
        Ok(lance) => match lance.list_hanns_indexes() {
            Ok(indexes) => (
                StatusCode::OK,
                Json(VectorIndexesResponse {
                    vector_indexes: indexes
                        .into_iter()
                        .map(|descriptor| {
                            serde_json::to_value(descriptor).expect("descriptor json")
                        })
                        .collect(),
                }),
            )
                .into_response(),
            Err(error) => daemon_storage_error_response(error),
        },
        Err(error) => daemon_storage_error_response(error),
    }
}

#[cfg(all(feature = "lance-storage", not(feature = "hanns-backend")))]
async fn list_lance_vector_indexes(_state: DaemonState, _collection: String) -> Response {
    (
        StatusCode::OK,
        Json(VectorIndexesResponse {
            vector_indexes: Vec::new(),
        }),
    )
        .into_response()
}

#[cfg(all(feature = "lance-storage", feature = "hanns-backend"))]
async fn drop_lance_vector_index(
    state: DaemonState,
    collection: String,
    field_name: String,
) -> Response {
    match open_lance_collection(&state, &collection).await {
        Ok(lance) => match lance.drop_hanns_index(&field_name) {
            Ok(()) => (
                StatusCode::OK,
                Json(DropIndexResponse {
                    dropped: field_name,
                }),
            )
                .into_response(),
            Err(error) => daemon_storage_error_response(error),
        },
        Err(error) => daemon_storage_error_response(error),
    }
}

#[cfg(all(feature = "lance-storage", not(feature = "hanns-backend")))]
async fn drop_lance_vector_index(
    _state: DaemonState,
    _collection: String,
    field_name: String,
) -> Response {
    let _ = field_name;
    daemon_storage_error_response(io::Error::new(
        io::ErrorKind::Unsupported,
        "Lance vector index DDL requires the hanns-backend feature",
    ))
}

#[cfg(feature = "lance-storage")]
async fn create_lance_scalar_index(
    state: DaemonState,
    collection: String,
    request: ScalarIndexRequest,
) -> Response {
    match open_lance_collection(&state, &collection).await {
        Ok(lance) => match lance.create_scalar_index_descriptor(
            &request.field_name,
            &request.kind,
            request.params,
        ) {
            Ok(()) => (
                StatusCode::CREATED,
                Json(CreateIndexResponse {
                    field_name: request.field_name,
                }),
            )
                .into_response(),
            Err(error) => daemon_storage_error_response(error),
        },
        Err(error) => daemon_storage_error_response(error),
    }
}

#[cfg(feature = "lance-storage")]
async fn list_lance_scalar_indexes(state: DaemonState, collection: String) -> Response {
    match open_lance_collection(&state, &collection).await {
        Ok(lance) => match lance.list_scalar_index_descriptors() {
            Ok(indexes) => (
                StatusCode::OK,
                Json(ScalarIndexesResponse {
                    scalar_indexes: indexes
                        .into_iter()
                        .map(|descriptor| {
                            serde_json::to_value(descriptor).expect("descriptor json")
                        })
                        .collect(),
                }),
            )
                .into_response(),
            Err(error) => daemon_storage_error_response(error),
        },
        Err(error) => daemon_storage_error_response(error),
    }
}

#[cfg(feature = "lance-storage")]
async fn drop_lance_scalar_index(
    state: DaemonState,
    collection: String,
    field_name: String,
) -> Response {
    match open_lance_collection(&state, &collection).await {
        Ok(lance) => match lance.drop_scalar_index_descriptor(&field_name) {
            Ok(()) => (
                StatusCode::OK,
                Json(DropIndexResponse {
                    dropped: field_name,
                }),
            )
                .into_response(),
            Err(error) => daemon_storage_error_response(error),
        },
        Err(error) => daemon_storage_error_response(error),
    }
}

async fn add_column_to_collection(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
    Json(request): Json<AddColumnRequest>,
) -> Response {
    let data_type = match parse_daemon_field_type(&request.data_type) {
        Ok(dt) => dt,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response()
        }
    };
    let field = ScalarFieldSchema::new(request.name.clone(), data_type)
        .with_flags(request.nullable, request.array);

    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &collection) {
        return match open_lance_collection(&state, &collection).await {
            Ok(lance) => match lance.add_scalar_column(field).await {
                Ok(()) => (
                    StatusCode::OK,
                    Json(AddColumnResponse {
                        added: request.name,
                    }),
                )
                    .into_response(),
                Err(error) => daemon_storage_error_response(error),
            },
            Err(error) => daemon_storage_error_response(error),
        };
    }

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .add_column(&collection, field);

    match result {
        Ok(()) => (
            StatusCode::OK,
            Json(AddColumnResponse {
                added: request.name,
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

async fn drop_column_from_collection(
    State(state): State<DaemonState>,
    AxumPath((collection, field_name)): AxumPath<(String, String)>,
) -> Response {
    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &collection) {
        return match open_lance_collection(&state, &collection).await {
            Ok(lance) => match lance.drop_scalar_column(&field_name).await {
                Ok(()) => (
                    StatusCode::OK,
                    Json(DropColumnResponse {
                        dropped: field_name,
                    }),
                )
                    .into_response(),
                Err(error) => daemon_storage_error_response(error),
            },
            Err(error) => daemon_storage_error_response(error),
        };
    }

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .drop_column(&collection, &field_name);

    match result {
        Ok(()) => (
            StatusCode::OK,
            Json(DropColumnResponse {
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

async fn alter_column_in_collection(
    State(state): State<DaemonState>,
    AxumPath((collection, field_name)): AxumPath<(String, String)>,
    Json(request): Json<AlterColumnRequest>,
) -> Response {
    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &collection) {
        return match open_lance_collection(&state, &collection).await {
            Ok(lance) => match lance
                .rename_scalar_column(&field_name, &request.new_name)
                .await
            {
                Ok(()) => (
                    StatusCode::OK,
                    Json(AlterColumnResponse {
                        old_name: field_name,
                        new_name: request.new_name,
                    }),
                )
                    .into_response(),
                Err(error) => daemon_storage_error_response(error),
            },
            Err(error) => daemon_storage_error_response(error),
        };
    }

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .alter_column(&collection, &field_name, &request.new_name);

    match result {
        Ok(()) => (
            StatusCode::OK,
            Json(AlterColumnResponse {
                old_name: field_name,
                new_name: request.new_name,
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

async fn add_vector_field_to_collection(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
    Json(request): Json<AddVectorFieldRequest>,
) -> Response {
    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &collection) {
        return lance_vector_add_unsupported_response();
    }

    let data_type = match parse_daemon_vector_field_type(&request.data_type) {
        Ok(dt) => dt,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response()
        }
    };
    let index_param = match parse_optional_index_param(request.index_param.as_ref()) {
        Ok(p) => p,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response()
        }
    };
    let field = VectorFieldSchema {
        name: request.name.clone(),
        data_type,
        dimension: request.dimension,
        index_param,
        bm25_params: None,
    };

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .add_vector_field(&collection, field);

    match result {
        Ok(()) => (
            StatusCode::OK,
            Json(AddVectorFieldResponse {
                added: request.name,
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
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => (
            StatusCode::CONFLICT,
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

async fn drop_vector_field_from_collection(
    State(state): State<DaemonState>,
    AxumPath((collection, field_name)): AxumPath<(String, String)>,
) -> Response {
    #[cfg(feature = "lance-storage")]
    if lance_collection_exists(&state, &collection) {
        return match open_lance_collection(&state, &collection).await {
            Ok(lance) => match lance.drop_vector_field(&field_name).await {
                Ok(()) => (
                    StatusCode::OK,
                    Json(DropVectorFieldResponse {
                        dropped: field_name,
                    }),
                )
                    .into_response(),
                Err(error) => daemon_storage_error_response(error),
            },
            Err(error) => daemon_storage_error_response(error),
        };
    }

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .drop_vector_field(&collection, &field_name);

    match result {
        Ok(()) => (
            StatusCode::OK,
            Json(DropVectorFieldResponse {
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

#[cfg(feature = "lance-storage")]
fn lance_vector_add_unsupported_response() -> Response {
    daemon_storage_error_response(io::Error::new(
        io::ErrorKind::Unsupported,
        "Lance add vector field is not supported by this daemon route yet",
    ))
}

// ---------------------------------------------------------------------------
// Shared helpers (used by routes_mutation and routes_search via super::routes)
// ---------------------------------------------------------------------------

pub(crate) fn parse_daemon_vector_field_type(value: &str) -> Result<FieldType, String> {
    match value.to_ascii_lowercase().as_str() {
        "vector_fp32" | "vectorfp32" => Ok(FieldType::VectorFp32),
        "vector_fp16" | "vectorfp16" => Ok(FieldType::VectorFp16),
        "vector_sparse" | "vectorsparse" => Ok(FieldType::VectorSparse),
        other => Err(format!("unsupported vector data type: {other}")),
    }
}

pub(crate) fn parse_optional_index_param(
    value: Option<&Value>,
) -> Result<Option<VectorIndexSchema>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let obj = value
        .as_object()
        .ok_or_else(|| "index_param must be a JSON object".to_string())?;
    let kind = obj
        .get("kind")
        .and_then(|v: &Value| v.as_str())
        .unwrap_or("hnsw");
    let metric = obj
        .get("metric")
        .and_then(|v: &Value| v.as_str())
        .map(str::to_string);
    match kind {
        "hnsw" => {
            let m = obj.get("m").and_then(|v: &Value| v.as_u64()).unwrap_or(16) as usize;
            let ef_construction = obj
                .get("ef_construction")
                .and_then(|v: &Value| v.as_u64())
                .unwrap_or(128) as usize;
            Ok(Some(VectorIndexSchema::hnsw(
                metric.as_deref(),
                m,
                ef_construction,
            )))
        }
        "ivf" => {
            let nlist = obj
                .get("nlist")
                .and_then(|v: &Value| v.as_u64())
                .unwrap_or(1024) as usize;
            Ok(Some(VectorIndexSchema::ivf(metric.as_deref(), nlist)))
        }
        other => Err(format!("unsupported index kind: {other}")),
    }
}

pub(crate) fn parse_daemon_field_type(value: &str) -> Result<FieldType, String> {
    match value.to_ascii_lowercase().as_str() {
        "string" => Ok(FieldType::String),
        "int64" => Ok(FieldType::Int64),
        "int32" => Ok(FieldType::Int32),
        "uint32" => Ok(FieldType::UInt32),
        "uint64" => Ok(FieldType::UInt64),
        "float" => Ok(FieldType::Float),
        "float64" => Ok(FieldType::Float64),
        "bool" => Ok(FieldType::Bool),
        other => Err(format!("unsupported data type: {other}")),
    }
}

pub(crate) fn parse_external_ids(ids: &[String]) -> Result<Vec<i64>, String> {
    ids.iter()
        .map(|id| {
            id.parse::<i64>()
                .map_err(|_| format!("invalid id, expected i64 string: {id}"))
        })
        .collect()
}

pub(crate) fn build_documents(request: InsertRecordsRequest) -> Result<Vec<Document>, String> {
    let external_ids = parse_external_ids(&request.ids)?;
    if request.vectors.len() != external_ids.len()
        && !(request.vectors.is_empty() && request.named_vectors.is_some())
    {
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

    let mut documents = Vec::with_capacity(external_ids.len());
    for (i, id) in external_ids.into_iter().enumerate() {
        let doc_fields = fields[i].clone();

        // Build dense vectors map
        let mut vectors = BTreeMap::new();
        if !request.vectors.is_empty() {
            vectors.insert("vector".to_string(), request.vectors[i].clone());
        }
        if let Some(named) = &request.named_vectors {
            if i < named.len() {
                for (name, vec) in &named[i] {
                    vectors.insert(name.clone(), vec.clone());
                }
            }
        }

        // Build sparse vectors map
        let sparse_vectors = if let Some(sparse_vecs) = &request.sparse_vectors {
            if i < sparse_vecs.len() {
                sparse_vecs[i]
                    .iter()
                    .map(|(name, sv)| {
                        (
                            name.clone(),
                            SparseVector::new(sv.indices.clone(), sv.values.clone()),
                        )
                    })
                    .collect()
            } else {
                BTreeMap::new()
            }
        } else {
            BTreeMap::new()
        };

        documents.push(Document {
            id,
            fields: doc_fields,
            vectors,
            sparse_vectors,
        });
    }
    Ok(documents)
}

pub(crate) fn json_fields_to_field_values(
    fields: BTreeMap<String, serde_json::Value>,
) -> Result<BTreeMap<String, FieldValue>, String> {
    fields
        .into_iter()
        .map(|(name, value)| Ok((name, json_value_to_field_value(value)?)))
        .collect()
}

pub(crate) fn json_value_to_field_value(value: serde_json::Value) -> Result<FieldValue, String> {
    match value {
        serde_json::Value::String(value) => Ok(FieldValue::String(value)),
        serde_json::Value::Bool(value) => Ok(FieldValue::Bool(value)),
        serde_json::Value::Number(value) => {
            // Try integer types first (narrowest to widest), then floats.
            if let Some(v) = value.as_i64() {
                if let Ok(v32) = i32::try_from(v) {
                    Ok(FieldValue::Int32(v32))
                } else {
                    Ok(FieldValue::Int64(v))
                }
            } else if let Some(v) = value.as_u64() {
                if let Ok(v32) = u32::try_from(v) {
                    Ok(FieldValue::UInt32(v32))
                } else {
                    Ok(FieldValue::UInt64(v))
                }
            } else if let Some(v) = value.as_f64() {
                Ok(FieldValue::Float64(v))
            } else {
                Err("unsupported numeric field value".to_string())
            }
        }
        other => Err(format!("unsupported field value: {other}")),
    }
}

pub(crate) fn field_values_to_json(
    fields: BTreeMap<String, FieldValue>,
) -> BTreeMap<String, serde_json::Value> {
    fields
        .into_iter()
        .map(|(name, value)| (name, field_value_to_json(value)))
        .collect()
}

pub(crate) fn select_output_fields(
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

pub(crate) fn select_fetch_output_fields(
    fields: &BTreeMap<String, FieldValue>,
    output_fields: Option<&[String]>,
) -> BTreeMap<String, serde_json::Value> {
    match output_fields {
        Some(names) => select_output_fields(fields, Some(names)),
        None => field_values_to_json(fields.clone()),
    }
}

pub(crate) fn build_query_context(
    request: crate::api::TypedSearchRequest,
) -> io::Result<QueryContext> {
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
            .map(|query| {
                let vector = if let Some(vector) = query.vector {
                    QueryVector::Dense(vector)
                } else if let Some(sparse) = query.sparse_vector {
                    QueryVector::Sparse(SparseVector::new(sparse.indices, sparse.values))
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "query must have either vector or sparse_vector",
                    ));
                };
                Ok(VectorQuery {
                    field_name: query.field_name,
                    vector,
                    param: query.param.map(|param| VectorQueryParam {
                        ef_search: param.ef_search,
                        nprobe: param.nprobe,
                    }),
                })
            })
            .collect::<io::Result<Vec<_>>>()?,
        query_by_id,
        query_by_id_field_name: request.query_by_id_field_name,
        filter: request.filter,
        output_fields: request.output_fields,
        include_vector: request.include_vector,
        group_by: request.group_by.map(|group_by| QueryGroupBy {
            field_name: group_by.field_name,
            group_topk: group_by.group_topk.unwrap_or(0),
            group_count: group_by.group_count.unwrap_or(0),
        }),
        reranker: request.reranker.and_then(|reranker| {
            // Map daemon reranker request to core QueryReranker enum.
            // For now, only RRF is supported via daemon.
            if let Some(rank_constant) = reranker.rank_constant {
                Some(QueryReranker::Rrf { rank_constant })
            } else if !reranker.weights.is_empty() {
                Some(QueryReranker::Weighted {
                    weights: reranker.weights,
                    metric: reranker.metric,
                })
            } else {
                None
            }
        }),
        order_by: request.order_by.map(|ob| hannsdb_core::query::OrderBy {
            field_name: ob.field_name,
            descending: ob.descending,
        }),
    })
}

pub(crate) fn field_value_to_json(value: FieldValue) -> serde_json::Value {
    match value {
        FieldValue::Null => serde_json::Value::Null,
        FieldValue::String(value) => serde_json::Value::String(value),
        FieldValue::Int64(value) => serde_json::Value::Number(value.into()),
        FieldValue::Int32(value) => serde_json::Value::Number(value.into()),
        FieldValue::UInt32(value) => serde_json::Value::Number(value.into()),
        FieldValue::UInt64(value) => serde_json::Value::Number(value.into()),
        FieldValue::Float(value) => serde_json::Value::Number(
            serde_json::Number::from_f64(value as f64)
                .expect("finite float field values should serialize"),
        ),
        FieldValue::Float64(value) => serde_json::Value::Number(
            serde_json::Number::from_f64(value)
                .expect("finite float field values should serialize"),
        ),
        FieldValue::Bool(value) => serde_json::Value::Bool(value),
        FieldValue::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(field_value_to_json).collect())
        }
    }
}
