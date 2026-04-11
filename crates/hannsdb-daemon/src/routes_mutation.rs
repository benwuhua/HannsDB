use std::collections::BTreeMap;
use std::io;

use axum::extract::{rejection::JsonRejection, Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use hannsdb_core::document::DocumentUpdate;

use crate::api::{
    DeleteByFilterRequest, DeleteRecordsRequest, DeleteRecordsResponse, ErrorResponse,
    InsertRecordsRequest, InsertRecordsResponse, UpdateRecordsRequest, UpdateRecordsResponse,
    UpsertRecordsResponse,
};

use super::routes::{build_documents, parse_external_ids, DaemonState};

pub(crate) async fn insert_records(
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

pub(crate) async fn upsert_records(
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

pub(crate) async fn update_records(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
    Json(request): Json<UpdateRecordsRequest>,
) -> Response {
    let external_ids = match parse_external_ids(&request.ids) {
        Ok(ids) => ids,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response()
        }
    };

    if !request.fields.is_empty() && request.fields.len() != external_ids.len() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "fields count must match id count".to_string(),
            }),
        )
            .into_response();
    }

    let mut updates = Vec::with_capacity(external_ids.len());
    for (i, id) in external_ids.iter().enumerate() {
        let fields = if request.fields.is_empty() {
            BTreeMap::new()
        } else {
            match request.fields[i]
                .iter()
                .map(|(k, v)| {
                    v.as_ref()
                        .map(|val| {
                            super::routes::json_value_to_field_value(val.clone())
                                .map(|fv| (k.clone(), Some(fv)))
                        })
                        .unwrap_or(Ok((k.clone(), None)))
                })
                .collect::<Result<BTreeMap<_, _>, _>>()
            {
                Ok(f) => f,
                Err(error) => {
                    return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response()
                }
            }
        };

        let vectors = request
            .vectors
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        updates.push(DocumentUpdate {
            id: *id,
            fields,
            vectors,
            sparse_vectors: Default::default(),
        });
    }

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .update_documents(&collection, &updates);

    match result {
        Ok(updated) => (
            StatusCode::OK,
            Json(UpdateRecordsResponse {
                updated: updated as u64,
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

pub(crate) async fn delete_records(
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

pub(crate) async fn delete_records_by_filter(
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

    let mut db = state.db.lock().expect("daemon state mutex poisoned");
    let result = db.delete_by_filter(&collection, &request.filter);

    match result {
        Ok(deleted) => (
            StatusCode::OK,
            Json(DeleteRecordsResponse {
                deleted: deleted as u64,
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let collection_missing = db
                .list_collections()
                .map(|collections| !collections.iter().any(|name| name == &collection))
                .unwrap_or(false);

            if collection_missing {
                (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: format!("collection not found: {collection}"),
                    }),
                )
                    .into_response()
            } else {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: error.to_string(),
                    }),
                )
                    .into_response()
            }
        }
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
