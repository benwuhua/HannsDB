#[cfg(feature = "lance-storage")]
use std::cmp::Ordering;
use std::collections::BTreeMap;
#[cfg(feature = "lance-storage")]
use std::collections::{BTreeSet, HashMap};
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
        let projection = lance_search_projection(lance.schema(), output_fields, None, None, None);
        let result = lance
            .search_vector_field_filtered_projected(
                lance.schema().primary_vector_name(),
                &request.vector,
                request.top_k,
                &metric,
                filter.as_ref(),
                projection,
            )
            .await?;
        return Ok(result
            .documents
            .into_iter()
            .map(|document| SearchHitResponse {
                id: document.document.id.to_string(),
                distance: document.distance,
                fields: if include_fields {
                    select_output_fields(&document.document.fields, output_fields)
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
    let default_metric =
        super::routes::lance_collection_metric(state, collection, lance.schema().metric())?;
    let recall_sources = lance_typed_recall_sources(&lance, &request, &default_metric).await?;
    let reranker = lance_reranker_from_request(request.reranker.as_ref(), &recall_sources)?;

    let needs_post_processing = request.group_by.is_some() || request.order_by.is_some();
    let raw_top_k = if needs_post_processing {
        lance.count_rows().await?
    } else {
        request.top_k
    };
    let projection = lance_search_projection(
        lance.schema(),
        output_fields,
        request
            .group_by
            .as_ref()
            .map(|group_by| group_by.field_name.as_str()),
        request
            .order_by
            .as_ref()
            .map(|order_by| order_by.field_name.as_str()),
        if request.include_vector {
            Some(lance.schema().primary_vector_name())
        } else {
            None
        },
    );

    let mut recall_results = Vec::with_capacity(recall_sources.len());
    for source in &recall_sources {
        let result = match &source.vector {
            LanceRecallVector::Dense(vector) => {
                lance
                    .search_vector_field_filtered_projected(
                        &source.field_name,
                        vector,
                        raw_top_k,
                        &source.metric,
                        filter.as_ref(),
                        projection.clone(),
                    )
                    .await?
            }
            LanceRecallVector::Sparse(sparse) => {
                if filter.is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "Lance sparse vector recall does not support filters yet",
                    ));
                }
                #[cfg(feature = "hanns-backend")]
                {
                    lance
                        .search_sparse_vector_field_projected(
                            &source.field_name,
                            sparse,
                            raw_top_k,
                            &source.metric,
                            projection.clone(),
                        )
                        .await?
                }
                #[cfg(not(feature = "hanns-backend"))]
                {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "Lance sparse vector recall requires hanns-backend",
                    ));
                }
            }
        };
        recall_results.push(LanceRecallResult {
            field_name: source.field_name.clone(),
            metric: source.metric.clone(),
            documents: result.documents,
        });
    }

    let should_fuse = reranker.is_some();
    let fusion_limit = if needs_post_processing {
        usize::MAX
    } else {
        request.top_k
    };
    let mut scored_documents = if should_fuse {
        fuse_lance_recall_results(
            recall_results,
            reranker.expect("reranker checked above"),
            fusion_limit,
        )
    } else {
        let mut recall_results = recall_results;
        let result = recall_results
            .pop()
            .expect("at least one recall source is required");
        result
            .documents
            .into_iter()
            .map(|document| LanceScoredDocument {
                distance: document.distance,
                document: document.document,
                group_key: None,
            })
            .collect()
    };

    let group_limit = if request.order_by.is_some() {
        usize::MAX
    } else {
        request.top_k
    };
    scored_documents = apply_lance_group_by(
        scored_documents,
        lance.schema(),
        request.group_by.as_ref(),
        group_limit,
    )?;
    apply_lance_order_by(
        &mut scored_documents,
        lance.schema(),
        request.order_by.as_ref(),
    )?;
    if scored_documents.len() > request.top_k {
        scored_documents.truncate(request.top_k);
    }

    Ok(scored_documents
        .into_iter()
        .map(|document| SearchHitResponse {
            id: document.document.id.to_string(),
            distance: document.distance,
            fields: if include_fields {
                select_output_fields(&document.document.fields, output_fields)
            } else {
                BTreeMap::new()
            },
            vector: if request.include_vector {
                document
                    .document
                    .vectors
                    .get(lance.schema().primary_vector_name())
                    .cloned()
            } else {
                None
            },
            sparse_vectors: None,
            group_key: document.group_key.map(super::routes::field_value_to_json),
        })
        .collect())
}

#[cfg(feature = "lance-storage")]
fn lance_search_projection(
    schema: &hannsdb_core::document::CollectionSchema,
    output_fields: Option<&[String]>,
    group_by_field: Option<&str>,
    order_by_field: Option<&str>,
    include_vector_field: Option<&str>,
) -> hannsdb_core::storage::lance_store::LanceSearchProjection {
    let mut output = BTreeSet::new();
    if let Some(fields) = output_fields {
        output.extend(fields.iter().cloned());
    }
    let mut projection =
        hannsdb_core::storage::lance_store::LanceSearchProjection::with_output_fields(output);
    if let Some(field) = group_by_field {
        if schema.fields.iter().any(|scalar| scalar.name == field) {
            projection.required_fields.insert(field.to_string());
        }
    }
    if let Some(field) = order_by_field {
        if schema.fields.iter().any(|scalar| scalar.name == field) {
            projection.required_fields.insert(field.to_string());
        }
    }
    if let Some(field) = include_vector_field {
        projection.required_vectors.insert(field.to_string());
    }
    projection
}

#[cfg(feature = "lance-storage")]
fn fuse_lance_recall_results(
    recall_results: Vec<LanceRecallResult>,
    reranker: LanceReranker,
    top_k: usize,
) -> Vec<LanceScoredDocument> {
    let mut by_id =
        std::collections::BTreeMap::<i64, (f64, hannsdb_core::document::Document)>::new();

    for result in recall_results {
        let weight = match &reranker {
            LanceReranker::Rrf { .. } => 1.0,
            LanceReranker::Weighted { weights, .. } => {
                weights.get(&result.field_name).copied().unwrap_or(1.0)
            }
        };
        let metric = match &reranker {
            LanceReranker::Rrf { .. } => result.metric.as_str(),
            LanceReranker::Weighted {
                metric_override, ..
            } => metric_override.as_deref().unwrap_or(result.metric.as_str()),
        };
        for (rank, document) in result.documents.into_iter().enumerate() {
            let contribution = match &reranker {
                LanceReranker::Rrf { rank_constant } => {
                    1.0 / (*rank_constant as f64 + rank as f64 + 1.0)
                }
                LanceReranker::Weighted { .. } => {
                    weight * normalize_lance_distance(document.distance, metric)
                }
            };
            let entry = by_id
                .entry(document.document.id)
                .or_insert_with(|| (0.0, document.document.clone()));
            entry.0 += contribution;
        }
    }

    let mut fused = by_id
        .into_iter()
        .map(|(_id, (score, document))| LanceScoredDocument {
            document,
            distance: -(score as f32),
            group_key: None,
        })
        .collect::<Vec<_>>();
    fused.sort_by(compare_lance_scored_documents);
    if fused.len() > top_k {
        fused.truncate(top_k);
    }
    fused
}

#[cfg(feature = "lance-storage")]
fn normalize_lance_distance(distance: f32, metric: &str) -> f64 {
    let distance = distance as f64;
    match metric {
        "l2" => 1.0 - 2.0 * distance.atan() / std::f64::consts::PI,
        "ip" => 0.5 + distance.atan() / std::f64::consts::PI,
        "cosine" => 1.0 - distance / 2.0,
        _ => 1.0 - distance,
    }
}

#[cfg(feature = "lance-storage")]
const DEFAULT_LANCE_RRF_RANK_CONSTANT: u64 = 60;

#[cfg(feature = "lance-storage")]
enum LanceReranker {
    Rrf {
        rank_constant: u64,
    },
    Weighted {
        weights: BTreeMap<String, f64>,
        metric_override: Option<String>,
    },
}

#[cfg(feature = "lance-storage")]
fn lance_reranker_from_request(
    reranker: Option<&crate::api::TypedQueryRerankerRequest>,
    recall_sources: &[LanceRecallSource],
) -> io::Result<Option<LanceReranker>> {
    let Some(reranker) = reranker else {
        return Ok((recall_sources.len() > 1).then_some(LanceReranker::Rrf {
            rank_constant: DEFAULT_LANCE_RRF_RANK_CONSTANT,
        }));
    };

    if !reranker.weights.is_empty() {
        if recall_sources.iter().any(LanceRecallSource::is_sparse) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "weighted reranker does not support sparse/BM25 Lance recall",
            ));
        }
        let metric_override = reranker
            .metric
            .as_deref()
            .map(validate_lance_metric_override)
            .transpose()?;
        return Ok(Some(LanceReranker::Weighted {
            weights: reranker.weights.clone(),
            metric_override,
        }));
    }

    if reranker.metric.is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "reranker metric override requires weighted reranker weights",
        ));
    }

    Ok(Some(LanceReranker::Rrf {
        rank_constant: reranker
            .rank_constant
            .unwrap_or(DEFAULT_LANCE_RRF_RANK_CONSTANT),
    }))
}

#[cfg(feature = "lance-storage")]
fn validate_lance_metric_override(metric: &str) -> io::Result<String> {
    let metric = metric.trim().to_ascii_lowercase();
    match metric.as_str() {
        "l2" | "ip" | "cosine" => Ok(metric),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsupported reranker metric override: {metric}"),
        )),
    }
}

#[cfg(feature = "lance-storage")]
struct LanceScoredDocument {
    document: hannsdb_core::document::Document,
    distance: f32,
    group_key: Option<hannsdb_core::document::FieldValue>,
}

#[cfg(feature = "lance-storage")]
fn apply_lance_order_by(
    documents: &mut [LanceScoredDocument],
    schema: &hannsdb_core::document::CollectionSchema,
    order_by: Option<&crate::api::TypedQueryOrderByRequest>,
) -> io::Result<()> {
    let Some(order_by) = order_by else {
        return Ok(());
    };
    let field_name = order_by.field_name.trim();
    if field_name.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "order_by field name must not be empty",
        ));
    }
    if schema
        .vectors
        .iter()
        .any(|vector| vector.name == field_name)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "order_by field '{field_name}' must reference a scalar field, not a vector field"
            ),
        ));
    }
    if schema
        .fields
        .iter()
        .any(|field| field.name == field_name && field.array)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("order_by field '{field_name}' must reference a scalar field, not an array"),
        ));
    }

    documents.sort_by(|left, right| {
        let left_value = left.document.fields.get(field_name);
        let right_value = right.document.fields.get(field_name);
        let ordering = match (left_value, right_value) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(left), Some(right)) => compare_lance_field_value_for_sort(left, right),
        };
        if order_by.descending {
            ordering.reverse()
        } else {
            ordering
        }
        .then_with(|| compare_lance_scored_documents(left, right))
    });
    Ok(())
}

#[cfg(feature = "lance-storage")]
fn compare_lance_scored_documents(
    left: &LanceScoredDocument,
    right: &LanceScoredDocument,
) -> Ordering {
    left.distance
        .total_cmp(&right.distance)
        .then_with(|| left.document.id.cmp(&right.document.id))
}

#[cfg(feature = "lance-storage")]
fn compare_lance_field_value_for_sort(
    left: &hannsdb_core::document::FieldValue,
    right: &hannsdb_core::document::FieldValue,
) -> Ordering {
    use hannsdb_core::document::FieldValue;

    match (left, right) {
        (FieldValue::String(left), FieldValue::String(right)) => left.cmp(right),
        (FieldValue::Null, FieldValue::Null) => Ordering::Equal,
        (FieldValue::Null, _) => Ordering::Less,
        (_, FieldValue::Null) => Ordering::Greater,
        (FieldValue::Int64(left), FieldValue::Int64(right)) => left.cmp(right),
        (FieldValue::Int32(left), FieldValue::Int32(right)) => left.cmp(right),
        (FieldValue::UInt32(left), FieldValue::UInt32(right)) => left.cmp(right),
        (FieldValue::UInt64(left), FieldValue::UInt64(right)) => left.cmp(right),
        (FieldValue::Float(left), FieldValue::Float(right)) => left.total_cmp(right),
        (FieldValue::Float64(left), FieldValue::Float64(right)) => left.total_cmp(right),
        (FieldValue::Bool(left), FieldValue::Bool(right)) => left.cmp(right),
        (FieldValue::Int64(left), FieldValue::Float64(right)) => (*left as f64).total_cmp(right),
        (FieldValue::Float64(left), FieldValue::Int64(right)) => left.total_cmp(&(*right as f64)),
        (FieldValue::Int32(left), FieldValue::Float64(right)) => (*left as f64).total_cmp(right),
        (FieldValue::Float64(left), FieldValue::Int32(right)) => left.total_cmp(&(*right as f64)),
        (FieldValue::UInt64(left), FieldValue::Float64(right)) => (*left as f64).total_cmp(right),
        (FieldValue::Float64(left), FieldValue::UInt64(right)) => left.total_cmp(&(*right as f64)),
        (FieldValue::Int64(left), FieldValue::Int32(right)) => left.cmp(&(*right as i64)),
        (FieldValue::Int32(left), FieldValue::Int64(right)) => (*left as i64).cmp(right),
        (FieldValue::Int64(left), FieldValue::UInt32(right)) => left.cmp(&(*right as i64)),
        (FieldValue::UInt32(left), FieldValue::Int64(right)) => (*left as i64).cmp(right),
        (FieldValue::Int64(left), FieldValue::UInt64(right)) => {
            if *left < 0 {
                Ordering::Less
            } else {
                (*left as u64).cmp(right)
            }
        }
        (FieldValue::UInt64(left), FieldValue::Int64(right)) => {
            if *right < 0 {
                Ordering::Greater
            } else {
                left.cmp(&(*right as u64))
            }
        }
        _ => format!("{left:?}").cmp(&format!("{right:?}")),
    }
}

#[cfg(feature = "lance-storage")]
fn apply_lance_group_by(
    documents: Vec<LanceScoredDocument>,
    schema: &hannsdb_core::document::CollectionSchema,
    group_by: Option<&crate::api::TypedQueryGroupByRequest>,
    top_k: usize,
) -> io::Result<Vec<LanceScoredDocument>> {
    let Some(group_by) = group_by else {
        return Ok(documents);
    };
    let field_name = group_by.field_name.trim();
    if field_name.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "group_by field name must not be empty",
        ));
    }
    let group_field = schema
        .fields
        .iter()
        .find(|field| field.name == field_name)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("group_by field '{field_name}' is not defined"),
            )
        })?;
    if group_field.array {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("group_by field '{field_name}' must reference a scalar field, not an array"),
        ));
    }

    let per_group_limit = group_by.group_topk.unwrap_or(0).max(1);
    let group_count = group_by.group_count.unwrap_or(0);
    let mut per_group_count = HashMap::<LanceGroupByValueKey, usize>::new();
    let mut groups_seen = 0usize;
    let mut grouped = Vec::new();

    for mut document in documents {
        let group_key = document.document.fields.get(field_name).cloned();
        let group_key_lookup = group_key
            .as_ref()
            .map(LanceGroupByValueKey::from_field_value)
            .unwrap_or(LanceGroupByValueKey::Missing);
        let count = per_group_count.entry(group_key_lookup).or_insert(0);
        if *count == 0 {
            if group_count > 0 && groups_seen >= group_count {
                continue;
            }
            groups_seen += 1;
        }
        if *count >= per_group_limit {
            continue;
        }
        *count += 1;
        document.group_key = group_key;
        grouped.push(document);
        if grouped.len() >= top_k {
            break;
        }
    }
    Ok(grouped)
}

#[cfg(feature = "lance-storage")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum LanceGroupByValueKey {
    Missing,
    String(String),
    Int64(i64),
    Float64(LanceFloatGroupKey),
    Bool(bool),
}

#[cfg(feature = "lance-storage")]
impl LanceGroupByValueKey {
    fn from_field_value(value: &hannsdb_core::document::FieldValue) -> Self {
        match value {
            hannsdb_core::document::FieldValue::Null => Self::Missing,
            hannsdb_core::document::FieldValue::String(value) => Self::String(value.clone()),
            hannsdb_core::document::FieldValue::Int64(value) => Self::Int64(*value),
            hannsdb_core::document::FieldValue::Int32(value) => Self::Int64(*value as i64),
            hannsdb_core::document::FieldValue::UInt32(value) => Self::Int64(*value as i64),
            hannsdb_core::document::FieldValue::UInt64(value) => Self::Int64(*value as i64),
            hannsdb_core::document::FieldValue::Float(value) => {
                Self::Float64(LanceFloatGroupKey::new(*value as f64))
            }
            hannsdb_core::document::FieldValue::Float64(value) => {
                Self::Float64(LanceFloatGroupKey::new(*value))
            }
            hannsdb_core::document::FieldValue::Bool(value) => Self::Bool(*value),
            hannsdb_core::document::FieldValue::Array(_) => Self::Missing,
        }
    }
}

#[cfg(feature = "lance-storage")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum LanceFloatGroupKey {
    Nan,
    Zero,
    Exact(u64),
}

#[cfg(feature = "lance-storage")]
impl LanceFloatGroupKey {
    fn new(value: f64) -> Self {
        if value.is_nan() {
            Self::Nan
        } else if value == 0.0 {
            Self::Zero
        } else {
            Self::Exact(value.to_bits())
        }
    }
}

#[cfg(feature = "lance-storage")]
struct LanceRecallSource {
    field_name: String,
    vector: LanceRecallVector,
    metric: String,
}

#[cfg(feature = "lance-storage")]
enum LanceRecallVector {
    Dense(Vec<f32>),
    Sparse(hannsdb_core::document::SparseVector),
}

#[cfg(feature = "lance-storage")]
impl LanceRecallSource {
    fn is_sparse(&self) -> bool {
        matches!(self.vector, LanceRecallVector::Sparse(_))
    }
}

#[cfg(feature = "lance-storage")]
struct LanceRecallResult {
    field_name: String,
    metric: String,
    documents: Vec<hannsdb_core::storage::lance_store::LanceSearchDocument>,
}

#[cfg(feature = "lance-storage")]
async fn lance_typed_recall_sources(
    lance: &hannsdb_core::storage::lance_store::LanceCollection,
    request: &TypedSearchRequest,
    default_metric: &str,
) -> io::Result<Vec<LanceRecallSource>> {
    let mut sources = Vec::new();
    for query in &request.queries {
        sources.push(lance_typed_query_from_request(
            lance,
            query,
            default_metric,
        )?);
    }

    if let Some(ids) = request.query_by_id.as_ref() {
        sources.extend(lance_typed_query_by_id_sources(lance, request, ids, default_metric).await?);
    } else if request.query_by_id_field_name.is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "query_by_id_field_name requires query_by_id",
        ));
    }

    if sources.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Lance daemon typed search requires at least one query or query_by_id",
        ));
    }
    Ok(sources)
}

#[cfg(feature = "lance-storage")]
fn lance_typed_query_from_request(
    lance: &hannsdb_core::storage::lance_store::LanceCollection,
    query: &crate::api::TypedVectorQueryRequest,
    default_metric: &str,
) -> io::Result<LanceRecallSource> {
    match (&query.vector, &query.sparse_vector) {
        (Some(_), Some(_)) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "typed query must provide either vector or sparse_vector, not both",
            ));
        }
        (Some(vector), None) => {
            validate_lance_query_param(query.param.as_ref())?;
            return lance_typed_dense_source(
                lance,
                query.field_name.trim(),
                vector.clone(),
                default_metric,
                "query vector",
            );
        }
        (None, Some(sparse)) => {
            if query.param.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Lance sparse/BM25 recall does not accept dense query params",
                ));
            }
            return lance_typed_sparse_source(
                lance,
                query.field_name.trim(),
                sparse,
                "query sparse vector",
            );
        }
        (None, None) => {}
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "Lance daemon typed search requires a vector or sparse_vector",
    ))
}

#[cfg(feature = "lance-storage")]
async fn lance_typed_query_by_id_sources(
    lance: &hannsdb_core::storage::lance_store::LanceCollection,
    request: &TypedSearchRequest,
    ids: &[String],
    default_metric: &str,
) -> io::Result<Vec<LanceRecallSource>> {
    if ids.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "query_by_id requires at least one id",
        ));
    }
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
    let parsed_ids = ids
        .iter()
        .map(|id| {
            id.parse::<i64>().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("invalid id, expected i64 string: {id}"),
                )
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    let documents = lance.fetch_documents(&parsed_ids).await?;
    let mut sources = Vec::with_capacity(parsed_ids.len());
    for id in &parsed_ids {
        let document = documents
            .iter()
            .find(|document| document.id == *id)
            .ok_or_else(|| {
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
        sources.push(lance_typed_dense_source(
            lance,
            &field_name,
            vector,
            default_metric,
            "query_by_id vector",
        )?);
    }
    Ok(sources)
}

#[cfg(feature = "lance-storage")]
fn lance_typed_dense_source(
    lance: &hannsdb_core::storage::lance_store::LanceCollection,
    field_name: &str,
    vector: Vec<f32>,
    default_metric: &str,
    label: &str,
) -> io::Result<LanceRecallSource> {
    if field_name.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "query vector field name must not be empty",
        ));
    }
    let vector_schema = lance
        .schema()
        .vectors
        .iter()
        .find(|vector| vector.name == field_name)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("query vector field '{field_name}' is not defined"),
            )
        })?;
    if vector.len() != vector_schema.dimension {
        return Err(io::Error::new(
            if label == "query_by_id vector" {
                io::ErrorKind::InvalidData
            } else {
                io::ErrorKind::InvalidInput
            },
            format!(
                "{label} dimension mismatch for field '{field_name}': expected {}, got {}",
                vector_schema.dimension,
                vector.len()
            ),
        ));
    }
    if !matches!(
        vector_schema.data_type,
        hannsdb_core::document::FieldType::VectorFp32
    ) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("dense query vector provided for non-dense field '{field_name}'"),
        ));
    }
    Ok(LanceRecallSource {
        field_name: field_name.to_string(),
        vector: LanceRecallVector::Dense(vector),
        metric: vector_schema
            .metric()
            .unwrap_or(default_metric)
            .to_ascii_lowercase(),
    })
}

#[cfg(feature = "lance-storage")]
fn lance_typed_sparse_source(
    lance: &hannsdb_core::storage::lance_store::LanceCollection,
    field_name: &str,
    sparse: &crate::api::SparseVectorRequest,
    label: &str,
) -> io::Result<LanceRecallSource> {
    if field_name.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "query sparse vector field name must not be empty",
        ));
    }
    let vector_schema = lance
        .schema()
        .vectors
        .iter()
        .find(|vector| vector.name == field_name)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("query sparse vector field '{field_name}' is not defined"),
            )
        })?;
    if !matches!(
        vector_schema.data_type,
        hannsdb_core::document::FieldType::VectorSparse
    ) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{label} provided for non-sparse field '{field_name}'"),
        ));
    }
    if sparse.indices.len() != sparse.values.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "{label} for field '{field_name}' has mismatched indices/values lengths: {} != {}",
                sparse.indices.len(),
                sparse.values.len()
            ),
        ));
    }
    let sparse =
        hannsdb_core::document::SparseVector::new(sparse.indices.clone(), sparse.values.clone());
    if !sparse.is_sorted() && !sparse.indices.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{label} indices for field '{field_name}' must be strictly sorted"),
        ));
    }
    Ok(LanceRecallSource {
        field_name: field_name.to_string(),
        vector: LanceRecallVector::Sparse(sparse),
        metric: "ip".to_string(),
    })
}

#[cfg(feature = "lance-storage")]
fn validate_lance_query_param(
    param: Option<&crate::api::TypedVectorQueryParamRequest>,
) -> io::Result<()> {
    let Some(param) = param else {
        return Ok(());
    };
    if matches!(param.ef_search, Some(0)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "query param ef_search must be positive",
        ));
    }
    if matches!(param.nprobe, Some(0)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "query param nprobe must be positive",
        ));
    }
    // Dense Lance projected scans do not consume ef_search/nprobe today.  Accept
    // positive values as explicit no-ops so typed API callers can use one query
    // shape across Lance and non-Lance collections without losing validation.
    Ok(())
}
