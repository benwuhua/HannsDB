use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex};

use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{Document, FieldValue};

use crate::api::{
    CollectionInfoResponse, CreateCollectionRequest, CreateCollectionResponse,
    DeleteRecordsRequest, DeleteRecordsResponse, DropCollectionResponse, ErrorResponse,
    FetchRecordResponse, FetchRecordsRequest, FetchRecordsResponse, FlushCollectionResponse,
    HealthResponse, InsertRecordsRequest, InsertRecordsResponse, ListCollectionsResponse,
    SearchHitResponse, SearchRequest, SearchResponse, UpsertRecordsResponse,
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
            "/collections/:collection/records",
            post(insert_records).delete(delete_records),
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

async fn fetch_records(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
    Json(request): Json<FetchRecordsRequest>,
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
        .fetch_documents(&collection, &external_ids);

    match result {
        Ok(documents) => (
            StatusCode::OK,
            Json(FetchRecordsResponse {
                documents: documents
                    .into_iter()
                    .map(|document| FetchRecordResponse {
                        id: document.id.to_string(),
                        fields: field_values_to_json(document.fields),
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
    Json(request): Json<SearchRequest>,
) -> Response {
    let output_fields = request.output_fields.as_deref();
    let include_fields = matches!(output_fields, Some(fields) if !fields.is_empty());
    let result = if let Some(filter) = request
        .filter
        .as_deref()
        .map(str::trim)
        .filter(|f| !f.is_empty())
    {
        state
            .db
            .lock()
            .expect("daemon state mutex poisoned")
            .query_documents(&collection, &request.vector, request.top_k, Some(filter))
            .map(|hits| {
                hits.into_iter()
                    .map(|hit| SearchHitResponse {
                        id: hit.id.to_string(),
                        distance: hit.distance,
                        fields: select_output_fields(&hit.fields, output_fields),
                    })
                    .collect::<Vec<_>>()
            })
    } else if include_fields {
        state
            .db
            .lock()
            .expect("daemon state mutex poisoned")
            .query_documents(&collection, &request.vector, request.top_k, None)
            .map(|hits| {
                hits.into_iter()
                    .map(|hit| SearchHitResponse {
                        id: hit.id.to_string(),
                        distance: hit.distance,
                        fields: select_output_fields(&hit.fields, output_fields),
                    })
                    .collect::<Vec<_>>()
            })
    } else {
        state
            .db
            .lock()
            .expect("daemon state mutex poisoned")
            .search(&collection, &request.vector, request.top_k)
            .map(|hits| {
                hits.into_iter()
                    .map(|hit| SearchHitResponse {
                        id: hit.id.to_string(),
                        distance: hit.distance,
                        fields: BTreeMap::new(),
                    })
                    .collect::<Vec<_>>()
            })
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
