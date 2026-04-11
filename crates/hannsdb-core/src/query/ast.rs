use crate::document::SparseVector;

#[derive(Debug, Clone, PartialEq)]
pub struct QueryContext {
    pub top_k: usize,
    pub queries: Vec<VectorQuery>,
    pub query_by_id: Option<Vec<i64>>,
    pub query_by_id_field_name: Option<String>,
    pub filter: Option<String>,
    pub output_fields: Option<Vec<String>>,
    pub include_vector: bool,
    pub group_by: Option<QueryGroupBy>,
    pub reranker: Option<QueryReranker>,
    pub order_by: Option<OrderBy>,
}

impl Default for QueryContext {
    fn default() -> Self {
        Self {
            top_k: 10,
            queries: Vec::new(),
            query_by_id: None,
            query_by_id_field_name: None,
            filter: None,
            output_fields: None,
            include_vector: false,
            group_by: None,
            reranker: None,
            order_by: None,
        }
    }
}

/// A query vector that can be either dense (f32 array) or sparse.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryVector {
    Dense(Vec<f32>),
    Sparse(SparseVector),
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorQuery {
    pub field_name: String,
    pub vector: QueryVector,
    pub param: Option<VectorQueryParam>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VectorQueryParam {
    pub ef_search: Option<usize>,
    pub nprobe: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGroupBy {
    pub field_name: String,
    pub group_topk: usize,
    pub group_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QueryReranker {
    Rrf {
        rank_constant: u64,
    },
    Weighted {
        weights: std::collections::BTreeMap<String, f64>,
        /// Override metric used for score normalization.
        /// When set, this metric is applied to all fields regardless of their
        /// individual metric. When None, per-field metrics from recall sources
        /// are used (metric-aware normalization already happens via the planner).
        /// Accepted values: "l2", "ip", "cosine".
        metric: Option<String>,
    },
}

/// Specifies how query results should be ordered.
///
/// When present, results are sorted by the given scalar field value instead of
/// by vector distance. A secondary sort by distance (then by id) breaks ties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderBy {
    pub field_name: String,
    pub descending: bool,
}
