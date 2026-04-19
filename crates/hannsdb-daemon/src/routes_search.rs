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

    #[cfg(feature = "lance-storage")]
    if super::routes::lance_collection_exists(&state, &collection) {
        let (result, primary_vector_name) =
            match super::routes::open_lance_collection(&state, &collection).await {
                Ok(collection) => {
                    let primary_vector_name = collection.schema().primary_vector_name().to_string();
                    (
                        collection.fetch_documents(&external_ids).await,
                        primary_vector_name,
                    )
                }
                Err(error) => (Err(error), "vector".to_string()),
            };
        return fetch_response(result, output_fields, &primary_vector_name);
    }

    let result = state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .fetch_documents(&collection, &external_ids);

    fetch_response(result, output_fields, "vector")
}

fn fetch_response(
    result: io::Result<Vec<hannsdb_core::document::Document>>,
    output_fields: Option<&[String]>,
    primary_vector_name: &str,
) -> Response {
    match result {
        Ok(documents) => (
            StatusCode::OK,
            Json(FetchRecordsResponse {
                documents: documents
                    .into_iter()
                    .map(|document| FetchRecordResponse {
                        id: document.id.to_string(),
                        fields: select_fetch_output_fields(&document.fields, output_fields),
                        vector: document
                            .vectors
                            .get(primary_vector_name)
                            .cloned()
                            .unwrap_or_default(),
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
        SearchRequest::Legacy(request) => search_records_legacy(&state, &collection, request).await,
        SearchRequest::Typed(request) => search_records_typed(&state, &collection, request).await,
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

async fn search_records_legacy(
    state: &DaemonState,
    collection: &str,
    request: LegacySearchRequest,
) -> io::Result<Vec<SearchHitResponse>> {
    let output_fields = request.output_fields.as_deref();
    let include_fields = matches!(output_fields, Some(fields) if !fields.is_empty());

    #[cfg(feature = "lance-storage")]
    if super::routes::lance_collection_exists(state, collection) {
        let filter = request
            .filter
            .as_deref()
            .map(str::trim)
            .filter(|filter| !filter.is_empty())
            .map(hannsdb_core::query::parse_filter)
            .transpose()?;
        let lance = super::routes::open_lance_collection(state, collection).await?;
        let metric =
            super::routes::lance_collection_metric(state, collection, lance.schema().metric())?;
        let hits = lance
            .search_filtered(&request.vector, request.top_k, &metric, filter.as_ref())
            .await?;
        let ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
        let scores = hits
            .iter()
            .map(|hit| (hit.id, hit.distance))
            .collect::<BTreeMap<_, _>>();
        let docs = lance.fetch_documents(&ids).await?;
        return Ok(docs
            .into_iter()
            .map(|document| SearchHitResponse {
                id: document.id.to_string(),
                distance: scores.get(&document.id).copied().unwrap_or_default(),
                fields: if include_fields {
                    select_output_fields(&document.fields, output_fields)
                } else {
                    BTreeMap::new()
                },
                vector: None,
                sparse_vectors: None,
                group_key: None,
            })
            .collect());
    }

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
                        vector: None,
                        sparse_vectors: None,
                        group_key: None,
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
                        vector: None,
                        sparse_vectors: None,
                        group_key: None,
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
                    vector: None,
                    sparse_vectors: None,
                    group_key: None,
                })
                .collect()
        })
}

async fn search_records_typed(
    state: &DaemonState,
    collection: &str,
    request: TypedSearchRequest,
) -> io::Result<Vec<SearchHitResponse>> {
    let include_vector = request.include_vector;
    let include_fields =
        matches!(request.output_fields.as_deref(), Some(fields) if !fields.is_empty());

    #[cfg(feature = "lance-storage")]
    if super::routes::lance_collection_exists(state, collection) {
        return search_records_typed_lance(state, collection, request).await;
    }

    let context = build_query_context(request)?;

    state
        .db
        .lock()
        .expect("daemon state mutex poisoned")
        .query_with_context(collection, &context)
        .map(|hits| {
            hits.into_iter()
                .map(|hit| {
                    let (vector, sparse_vectors) = if include_vector {
                        let primary_vec = hit.vectors.values().next().cloned();
                        let sparse = if hit.sparse_vectors.is_empty() {
                            None
                        } else {
                            Some(
                                hit.sparse_vectors
                                    .into_iter()
                                    .map(|(name, sv)| {
                                        (
                                            name,
                                            crate::api::SparseVectorResponse {
                                                indices: sv.indices,
                                                values: sv.values,
                                            },
                                        )
                                    })
                                    .collect(),
                            )
                        };
                        (primary_vec, sparse)
                    } else {
                        (None, None)
                    };
                    SearchHitResponse {
                        id: hit.id.to_string(),
                        distance: hit.distance,
                        fields: if include_fields {
                            super::routes::field_values_to_json(hit.fields)
                        } else {
                            BTreeMap::new()
                        },
                        vector,
                        sparse_vectors,
                        group_key: hit.group_key.map(super::routes::field_value_to_json),
                    }
                })
                .collect()
        })
}

#[cfg(feature = "lance-storage")]
async fn search_records_typed_lance(
    state: &DaemonState,
    collection: &str,
    request: TypedSearchRequest,
) -> io::Result<Vec<SearchHitResponse>> {
    if request.group_by.is_some() || request.reranker.is_some() || request.order_by.is_some() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Lance daemon typed search supports only a single dense vector query or query_by_id",
        ));
    }

    let filter = request
        .filter
        .as_deref()
        .map(str::trim)
        .filter(|filter| !filter.is_empty())
        .map(hannsdb_core::query::parse_filter)
        .transpose()?;
    let output_fields = request.output_fields.as_deref();
    let include_fields = matches!(output_fields, Some(fields) if !fields.is_empty());
    let lance = super::routes::open_lance_collection(state, collection).await?;
    let metric =
        super::routes::lance_collection_metric(state, collection, lance.schema().metric())?;
    let query = lance_typed_query_vector(&lance, &request).await?;
    let hits = lance
        .search_vector_field_filtered(
            &query.field_name,
            &query.vector,
            request.top_k,
            &metric,
            filter.as_ref(),
        )
        .await?;
    let ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    let scores = hits
        .iter()
        .map(|hit| (hit.id, hit.distance))
        .collect::<BTreeMap<_, _>>();
    let docs = lance.fetch_documents(&ids).await?;
    Ok(docs
        .into_iter()
        .map(|document| SearchHitResponse {
            id: document.id.to_string(),
            distance: scores.get(&document.id).copied().unwrap_or_default(),
            fields: if include_fields {
                select_output_fields(&document.fields, output_fields)
            } else {
                BTreeMap::new()
            },
            vector: if request.include_vector {
                document
                    .vectors
                    .get(lance.schema().primary_vector_name())
                    .cloned()
            } else {
                None
            },
            sparse_vectors: None,
            group_key: None,
        })
        .collect())
}

#[cfg(feature = "lance-storage")]
struct LanceTypedQueryVector {
    field_name: String,
    vector: Vec<f32>,
}

#[cfg(feature = "lance-storage")]
async fn lance_typed_query_vector(
    lance: &hannsdb_core::storage::lance_store::LanceCollection,
    request: &TypedSearchRequest,
) -> io::Result<LanceTypedQueryVector> {
    if let Some(ids) = request.query_by_id.as_ref() {
        if !request.queries.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Lance daemon typed search cannot mix queries with query_by_id",
            ));
        }
        let [id] = ids.as_slice() else {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Lance daemon typed search supports exactly one query_by_id",
            ));
        };
        let id = id.parse::<i64>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid id, expected i64 string: {id}"),
            )
        })?;
        let field_name = match request.query_by_id_field_name.as_deref() {
            Some(field_name) => {
                let field_name = field_name.trim();
                if field_name.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "query_by_id_field_name must not be empty",
                    ));
                }
                field_name.to_string()
            }
            None => lance.schema().primary_vector_name().to_string(),
        };
        let vector_schema = lance
            .schema()
            .vectors
            .iter()
            .find(|vector| vector.name == field_name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("query_by_id field '{field_name}' is not defined"),
                )
            })?;
        let docs = lance.fetch_documents(&[id]).await?;
        let document = docs.first().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("document not found for query_by_id: {id}"),
            )
        })?;
        let vector = document.vectors.get(&field_name).cloned().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("query_by_id document is missing vector field '{field_name}'"),
            )
        })?;
        if vector.len() != vector_schema.dimension {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "query_by_id vector dimension mismatch for field '{field_name}': expected {}, got {}",
                    vector_schema.dimension,
                    vector.len()
                ),
            ));
        }
        return Ok(LanceTypedQueryVector { field_name, vector });
    }

    if request.query_by_id_field_name.is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "query_by_id_field_name requires query_by_id",
        ));
    }

    let [query] = request.queries.as_slice() else {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Lance daemon typed search requires exactly one query",
        ));
    };
    if query.field_name != lance.schema().primary_vector_name()
        || query.sparse_vector.is_some()
        || query.param.is_some()
    {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Lance daemon typed search supports only the primary dense vector",
        ));
    }
    let vector = query.vector.clone().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Lance daemon typed search requires a dense vector",
        )
    })?;
    Ok(LanceTypedQueryVector {
        field_name: query.field_name.clone(),
        vector,
    })
}
