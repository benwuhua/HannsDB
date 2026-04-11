use std::collections::BTreeMap;
use std::io;

use axum::extract::{rejection::JsonRejection, Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::api::{
    ErrorResponse, FetchRecordResponse, FetchRecordsRequest, FetchRecordsResponse,
    LegacySearchRequest, SearchHitResponse, SearchRequest, SearchResponse, TypedSearchRequest,
};

use super::routes::{
    build_query_context, parse_external_ids, select_fetch_output_fields, select_output_fields,
    DaemonState,
};

pub(crate) async fn fetch_records(
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
                        vector: document.vectors.get("vector").cloned().unwrap_or_default(),
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

pub(crate) async fn search_records(
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
                        super::routes::field_values_to_json(hit.fields)
                    } else {
                        BTreeMap::new()
                    },
                })
                .collect()
        })
}
