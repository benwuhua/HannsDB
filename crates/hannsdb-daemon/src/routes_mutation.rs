use std::collections::BTreeMap;
#[cfg(feature = "lance-storage")]
use std::collections::HashSet;
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

    #[cfg(feature = "lance-storage")]
    if super::routes::lance_collection_exists(&state, &collection) {
        if let Err(error) = validate_lance_daemon_documents(&documents, true) {
            return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response();
        }
        let _insert_guard = state.lance_insert_lock.lock().await;
        let result = match super::routes::open_lance_collection(&state, &collection).await {
            Ok(collection) => {
                let ids = documents
                    .iter()
                    .map(|document| document.id)
                    .collect::<Vec<_>>();
                match collection.fetch_documents(&ids).await {
                    Ok(existing) if !existing.is_empty() => Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "duplicate document id for Lance insert",
                    )),
                    Ok(_) => collection
                        .insert_documents(&documents)
                        .await
                        .map(|_| documents.len()),
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error),
        };
        return insert_response(result);
    }

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .insert_documents(&collection, &documents);

    insert_response(result)
}

fn insert_response(result: io::Result<usize>) -> Response {
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

    #[cfg(feature = "lance-storage")]
    if super::routes::lance_collection_exists(&state, &collection) {
        if let Err(error) = validate_lance_daemon_documents(&documents, false) {
            return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })).into_response();
        }
        let result = match super::routes::open_lance_collection(&state, &collection).await {
            Ok(collection) => collection.upsert_documents(&documents).await,
            Err(error) => Err(error),
        };
        return upsert_response(result);
    }

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .upsert_documents(&collection, &documents);

    upsert_response(result)
}

fn upsert_response(result: io::Result<usize>) -> Response {
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

#[cfg(feature = "lance-storage")]
fn validate_lance_daemon_documents(
    documents: &[hannsdb_core::document::Document],
    reject_duplicate_ids: bool,
) -> Result<(), String> {
    if reject_duplicate_ids {
        let mut seen = HashSet::with_capacity(documents.len());
        for document in documents {
            if !seen.insert(document.id) {
                return Err(format!(
                    "duplicate document id for Lance insert: {}",
                    document.id
                ));
            }
        }
    }

    for document in documents {
        if !document.fields.is_empty() {
            return Err(
                "daemon Lance collections created through this API do not support scalar fields yet"
                    .to_string(),
            );
        }
        if !document.sparse_vectors.is_empty() {
            return Err(
                "daemon Lance collections created through this API do not support sparse vectors yet"
                    .to_string(),
            );
        }
        if document.vectors.keys().any(|name| name != "vector") {
            return Err(
                "daemon Lance collections created through this API support only the primary vector"
                    .to_string(),
            );
        }
    }
    Ok(())
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

    #[cfg(feature = "lance-storage")]
    if super::routes::lance_collection_exists(&state, &collection) {
        let result = match super::routes::open_lance_collection(&state, &collection).await {
            Ok(collection) => collection.delete_documents(&external_ids).await,
            Err(error) => Err(error),
        };
        return delete_response(result);
    }

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .delete(&collection, &external_ids);

    delete_response(result)
}

fn delete_response(result: io::Result<usize>) -> Response {
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

    #[cfg(feature = "lance-storage")]
    if super::routes::lance_collection_exists(&state, &collection) {
        let result = match hannsdb_core::query::parse_filter(&request.filter) {
            Ok(filter) => match super::routes::open_lance_collection(&state, &collection).await {
                Ok(collection) => collection.delete_by_filter(&filter).await,
                Err(error) => Err(error),
            },
            Err(error) => Err(error),
        };
        return delete_by_filter_response(result, None);
    }

    let mut db = state.db.lock().expect("daemon state mutex poisoned");
    let result = db.delete_by_filter(&collection, &request.filter);
    let collection_missing = db
        .list_collections()
        .map(|collections| !collections.iter().any(|name| name == &collection))
        .unwrap_or(false);

    delete_by_filter_response(result, collection_missing.then_some(collection))
}

fn delete_by_filter_response(
    result: io::Result<usize>,
    missing_collection: Option<String>,
) -> Response {
    match result {
        Ok(deleted) => (
            StatusCode::OK,
            Json(DeleteRecordsResponse {
                deleted: deleted as u64,
            }),
        )
            .into_response(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => match missing_collection {
            Some(collection) => (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("collection not found: {collection}"),
                }),
            )
                .into_response(),
            None => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: error.to_string(),
                }),
            )
                .into_response(),
        },
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
