use std::io;

use crate::catalog::CollectionMetadata;
use crate::document::Document;

use super::{parse_filter, FilterExpr, QueryContext, QueryGroupBy, VectorQuery};

#[derive(Debug, Clone)]
pub(crate) enum QueryPlan {
    LegacySingleVector(LegacySingleVectorPlan),
    BruteForce(BruteForceQueryPlan),
}

#[derive(Debug, Clone)]
pub(crate) struct LegacySingleVectorPlan {
    pub(crate) top_k: usize,
    pub(crate) filter: Option<String>,
    pub(crate) vector: Vec<f32>,
    pub(crate) ef_search: Option<usize>,
    pub(crate) output_fields: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub(crate) struct BruteForceQueryPlan {
    pub(crate) top_k: usize,
    pub(crate) filter: Option<FilterExpr>,
    pub(crate) mode: BruteForceExecutionMode,
    pub(crate) recall_sources: Vec<PlannedRecallSource>,
    pub(crate) group_by: Option<PlannedGroupBy>,
    pub(crate) output_fields: Option<Vec<String>>,
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
}

#[derive(Debug, Clone)]
pub(crate) struct PlannedGroupBy {
    pub(crate) field_name: String,
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
        context: &QueryContext,
        query_by_id_documents: &[Document],
    ) -> io::Result<QueryPlan> {
        if context.reranker.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "reranker is not supported on the typed query path yet",
            ));
        }

        let filter = context
            .filter
            .as_deref()
            .map(str::trim)
            .filter(|filter| !filter.is_empty())
            .map(str::to_owned);

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
            && filter.is_none()
            && context.queries[0].field_name == collection.primary_vector;
        if uses_ef_search && !is_single_vector_fast_path {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "vector query param ef_search is only supported on the typed single-vector fast path",
            ));
        }

        if is_single_vector_fast_path {
            let query = &context.queries[0];
            validate_vector_query(collection, query, true, true)?;
            if filter.is_some()
                && query
                    .param
                    .as_ref()
                    .and_then(|param| param.ef_search)
                    .is_some()
            {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "vector query param ef_search is not supported with filtered typed single-vector fast path queries",
                ));
            }
            return Ok(QueryPlan::LegacySingleVector(LegacySingleVectorPlan {
                top_k: context.top_k,
                filter,
                vector: query.vector.clone(),
                ef_search: query.param.as_ref().and_then(|param| param.ef_search),
                output_fields: context.output_fields.clone(),
            }));
        }

        let filter = filter.as_deref().map(parse_filter).transpose()?;

        let mut recall_sources = Vec::new();
        for query in &context.queries {
            validate_vector_query(collection, query, false, false)?;
            recall_sources.push(PlannedRecallSource {
                kind: RecallSourceKind::ExplicitVector {
                    field_name: query.field_name.clone(),
                },
                vector: query.vector.clone(),
            });
        }

        if let Some(query_by_id) = context.query_by_id.as_ref() {
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
                if document.vector.len() != collection.dimension {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "stored query_by_id vector dimension mismatch: expected {}, got {}",
                            collection.dimension,
                            document.vector.len()
                        ),
                    ));
                }
                recall_sources.push(PlannedRecallSource {
                    kind: RecallSourceKind::QueryById {
                        id: *id,
                        field_name: collection.primary_vector.clone(),
                    },
                    vector: document.vector.clone(),
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
        }))
    }
}

fn validate_vector_query(
    collection: &CollectionMetadata,
    query: &VectorQuery,
    allow_ef_search: bool,
    require_primary: bool,
) -> io::Result<()> {
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

    Ok(())
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
