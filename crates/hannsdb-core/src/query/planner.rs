use std::io;

use crate::catalog::{CollectionMetadata, IndexCatalog};
use crate::document::{Document, VectorFieldSchema};
use hannsdb_index::descriptor::{VectorIndexDescriptor, VectorIndexKind};

use super::{parse_filter, FilterExpr, QueryContext, QueryGroupBy, VectorQuery};

#[derive(Debug, Clone)]
pub(crate) enum QueryPlan {
    LegacySingleVector(LegacySingleVectorPlan),
    BruteForce(BruteForceQueryPlan),
}

#[derive(Debug, Clone)]
pub(crate) struct LegacySingleVectorPlan {
    pub(crate) field_name: String,
    pub(crate) top_k: usize,
    pub(crate) vector: Vec<f32>,
    pub(crate) ef_search: Option<usize>,
    pub(crate) output_fields: Option<Vec<String>>,
    pub(crate) filter: Option<FilterExpr>,
}

#[derive(Debug, Clone)]
pub(crate) struct BruteForceQueryPlan {
    pub(crate) top_k: usize,
    pub(crate) filter: Option<FilterExpr>,
    pub(crate) mode: BruteForceExecutionMode,
    pub(crate) recall_sources: Vec<PlannedRecallSource>,
    pub(crate) group_by: Option<PlannedGroupBy>,
    pub(crate) output_fields: Option<Vec<String>>,
    pub(crate) reranker: Option<super::QueryReranker>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BruteForceExecutionMode {
    Recall,
    FilterOnlyScan,
}

#[derive(Debug, Clone)]
pub(crate) struct PlannedRecallSource {
    pub(crate) kind: RecallSourceKind,
    pub(crate) vector: Vec<f32>,
    pub(crate) metric: String,
}

impl PlannedRecallSource {
    /// Returns a unique key for this recall source (field_name for explicit, id_field for query_by_id).
    pub(crate) fn field_key(&self) -> String {
        match &self.kind {
            RecallSourceKind::ExplicitVector { field_name } => field_name.clone(),
            RecallSourceKind::QueryById { id, field_name } => format!("{}_{}", field_name, id),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PlannedGroupBy {
    pub(crate) field_name: String,
    pub(crate) group_topk: usize,
    pub(crate) group_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RecallSourceKind {
    ExplicitVector { field_name: String },
    QueryById { id: i64, field_name: String },
}

pub(crate) struct QueryPlanner;

impl QueryPlanner {
    pub(crate) fn build(
        collection: &CollectionMetadata,
        index_catalog: &IndexCatalog,
        context: &QueryContext,
        query_by_id_documents: &[Document],
    ) -> io::Result<QueryPlan> {
        let filter = context
            .filter
            .as_deref()
            .map(str::trim)
            .filter(|filter| !filter.is_empty())
            .map(str::to_owned);

        let parsed_filter = filter.as_deref().map(parse_filter).transpose()?;

        let uses_ef_search = context.queries.iter().any(|query| {
            query
                .param
                .as_ref()
                .and_then(|param| param.ef_search)
                .is_some()
        });
        let is_single_vector_fast_path = context.group_by.is_none()
            && context.queries.len() == 1
            && context.query_by_id.is_none()
            && resolve_vector_descriptor_for_field(
                collection,
                index_catalog,
                &context.queries[0].field_name,
            )?
            .is_some();
        if uses_ef_search && !is_single_vector_fast_path {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "vector query param ef_search is only supported on the typed single-vector fast path",
            ));
        }

        if is_single_vector_fast_path {
            let query = &context.queries[0];
            validate_vector_query(collection, index_catalog, query, true, false)?;
            return Ok(QueryPlan::LegacySingleVector(LegacySingleVectorPlan {
                field_name: query.field_name.clone(),
                top_k: context.top_k,
                vector: query.vector.clone(),
                ef_search: query.param.as_ref().and_then(|param| param.ef_search),
                output_fields: context.output_fields.clone(),
                filter: parsed_filter,
            }));
        }

        let filter = parsed_filter;

        let mut recall_sources = Vec::new();
        for query in &context.queries {
            let metric = validate_vector_query(collection, index_catalog, query, false, false)?;
            recall_sources.push(PlannedRecallSource {
                kind: RecallSourceKind::ExplicitVector {
                    field_name: query.field_name.clone(),
                },
                vector: query.vector.clone(),
                metric,
            });
        }

        if let Some(query_by_id) = context.query_by_id.as_ref() {
            let query_by_id_field_name = match context.query_by_id_field_name.as_deref() {
                Some(field_name) => {
                    let field_name = field_name.trim();
                    if field_name.is_empty() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "query_by_id_field_name must not be empty",
                        ));
                    }
                    field_name.to_owned()
                }
                None => collection.primary_vector.clone(),
            };
            let query_by_id_vector_schema = collection
                .vectors
                .iter()
                .find(|vector| vector.name == query_by_id_field_name)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "query_by_id field '{}' is not defined in collection '{}'",
                            query_by_id_field_name, collection.name
                        ),
                    )
                })?;
            if query_by_id.len() != query_by_id_documents.len() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "query_by_id resolved {} document(s) for {} requested id(s)",
                        query_by_id_documents.len(),
                        query_by_id.len()
                    ),
                ));
            }

            for (id, document) in query_by_id.iter().zip(query_by_id_documents) {
                let vector = document
                    .vectors
                    .get(&query_by_id_field_name)
                    .map(Vec::as_slice)
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "stored query_by_id document {} is missing vector field '{}'",
                                id, query_by_id_field_name
                            ),
                        )
                    })?;
                if vector.len() != query_by_id_vector_schema.dimension {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "stored query_by_id vector dimension mismatch for field '{}': expected {}, got {}",
                            query_by_id_field_name,
                            query_by_id_vector_schema.dimension,
                            vector.len()
                        ),
                    ));
                }
                recall_sources.push(PlannedRecallSource {
                    kind: RecallSourceKind::QueryById {
                        id: *id,
                        field_name: query_by_id_field_name.clone(),
                    },
                    vector: vector.to_vec(),
                    metric: resolve_vector_metric(
                        collection,
                        index_catalog,
                        query_by_id_vector_schema,
                    )?,
                });
            }
        }

        let mode = if recall_sources.is_empty() {
            if filter.is_some() {
                BruteForceExecutionMode::FilterOnlyScan
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "query context must include at least one recall source or a filter",
                ));
            }
        } else {
            BruteForceExecutionMode::Recall
        };

        let group_by = context
            .group_by
            .as_ref()
            .map(|group_by| validate_group_by(collection, group_by, mode))
            .transpose()?;

        Ok(QueryPlan::BruteForce(BruteForceQueryPlan {
            top_k: context.top_k,
            filter,
            mode,
            recall_sources,
            group_by,
            output_fields: context.output_fields.clone(),
            reranker: context.reranker.clone(),
        }))
    }
}

fn validate_vector_query(
    collection: &CollectionMetadata,
    index_catalog: &IndexCatalog,
    query: &VectorQuery,
    allow_ef_search: bool,
    require_primary: bool,
) -> io::Result<String> {
    if !allow_ef_search
        && query
            .param
            .as_ref()
            .is_some_and(|param| param.ef_search.is_some())
    {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "vector query param ef_search is not supported on the typed brute-force path yet",
        ));
    }

    let Some(vector_schema) = collection
        .vectors
        .iter()
        .find(|vector| vector.name == query.field_name)
    else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "query vector field '{}' is not defined in collection '{}'",
                query.field_name, collection.name
            ),
        ));
    };

    if require_primary && query.field_name != collection.primary_vector {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "typed query only supports the primary vector field '{}', got '{}'",
                collection.primary_vector, query.field_name
            ),
        ));
    }

    if query.vector.len() != vector_schema.dimension {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "query vector dimension mismatch: expected {}, got {}",
                vector_schema.dimension,
                query.vector.len()
            ),
        ));
    }

    resolve_vector_metric(collection, index_catalog, vector_schema)
}

pub(crate) fn resolve_vector_descriptor_for_field(
    collection: &CollectionMetadata,
    index_catalog: &IndexCatalog,
    field_name: &str,
) -> io::Result<Option<VectorIndexDescriptor>> {
    if field_name == collection.primary_vector {
        return Ok(Some(resolve_primary_vector_descriptor_for_planner(
            collection,
            index_catalog,
        )?));
    }

    let Some(vector_schema) = collection
        .vectors
        .iter()
        .find(|vector| vector.name == field_name)
    else {
        return Ok(None);
    };

    if let Some(descriptor) = index_catalog
        .vector_indexes
        .iter()
        .find(|descriptor| descriptor.field_name == field_name)
    {
        return Ok(Some(descriptor.clone()));
    }

    Ok(vector_schema
        .index_param
        .as_ref()
        .map(|index_param| VectorIndexDescriptor {
            field_name: vector_schema.name.clone(),
            kind: match index_param {
                crate::document::VectorIndexSchema::Ivf { .. } => VectorIndexKind::Ivf,
                crate::document::VectorIndexSchema::Hnsw { .. } => VectorIndexKind::Hnsw,
            },
            metric: index_param.metric().map(str::to_string),
            params: match index_param {
                crate::document::VectorIndexSchema::Ivf { nlist, .. } => {
                    serde_json::json!({ "nlist": nlist })
                }
                crate::document::VectorIndexSchema::Hnsw {
                    m, ef_construction, ..
                } => serde_json::json!({
                    "m": m,
                    "ef_construction": ef_construction,
                }),
            },
        }))
}

fn resolve_primary_vector_descriptor_for_planner(
    collection_meta: &CollectionMetadata,
    index_catalog: &IndexCatalog,
) -> io::Result<VectorIndexDescriptor> {
    if let Some(descriptor) = index_catalog
        .vector_indexes
        .iter()
        .find(|descriptor| descriptor.field_name == collection_meta.primary_vector)
    {
        return Ok(descriptor.clone());
    }

    let primary_vector = collection_meta
        .vectors
        .iter()
        .find(|vector| vector.name == collection_meta.primary_vector)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "primary vector '{}' is not defined in collection metadata",
                    collection_meta.primary_vector
                ),
            )
        })?;

    let (kind, metric, params) = match primary_vector.index_param.as_ref() {
        Some(crate::document::VectorIndexSchema::Ivf { metric, nlist }) => (
            VectorIndexKind::Ivf,
            metric.clone(),
            serde_json::json!({ "nlist": nlist }),
        ),
        Some(crate::document::VectorIndexSchema::Hnsw {
            metric,
            m,
            ef_construction,
            ..
        }) => (
            VectorIndexKind::Hnsw,
            metric.clone(),
            serde_json::json!({
                "m": m,
                "ef_construction": ef_construction
            }),
        ),
        None => (
            VectorIndexKind::Hnsw,
            Some(collection_meta.metric.clone()),
            serde_json::json!({
                "m": collection_meta.hnsw_m,
                "ef_construction": collection_meta.hnsw_ef_construction
            }),
        ),
    };

    Ok(VectorIndexDescriptor {
        field_name: collection_meta.primary_vector.clone(),
        kind,
        metric,
        params,
    })
}

fn resolve_vector_metric(
    collection: &CollectionMetadata,
    index_catalog: &IndexCatalog,
    vector_schema: &VectorFieldSchema,
) -> io::Result<String> {
    let metric = index_catalog
        .vector_indexes
        .iter()
        .find(|descriptor| descriptor.field_name == vector_schema.name)
        .and_then(|descriptor| descriptor.metric.as_deref())
        .or_else(|| vector_schema.metric())
        .unwrap_or(collection.metric.as_str());
    Ok(metric.to_ascii_lowercase())
}

fn validate_group_by(
    collection: &CollectionMetadata,
    group_by: &QueryGroupBy,
    mode: BruteForceExecutionMode,
) -> io::Result<PlannedGroupBy> {
    if mode == BruteForceExecutionMode::FilterOnlyScan {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "group_by requires at least one recall source on the typed query path",
        ));
    }

    let field_name = group_by.field_name.trim();
    if field_name.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "group_by field name must not be empty",
        ));
    }

    if let Some(field) = collection
        .fields
        .iter()
        .find(|field| field.name == field_name)
    {
        if field.array || field.data_type == crate::document::FieldType::VectorFp32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "group_by field '{}' must reference a scalar field in collection '{}'",
                    field_name, collection.name
                ),
            ));
        }
        return Ok(PlannedGroupBy {
            field_name: field_name.to_string(),
            group_topk: group_by.group_topk,
            group_count: group_by.group_count,
        });
    }

    if collection
        .vectors
        .iter()
        .any(|vector| vector.name == field_name)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "group_by field '{}' must reference a scalar field, not a vector field",
                field_name
            ),
        ));
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "group_by field '{}' is not defined in collection '{}'",
            field_name, collection.name
        ),
    ))
}
