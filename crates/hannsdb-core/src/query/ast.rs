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
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorQuery {
    pub field_name: String,
    pub vector: Vec<f32>,
    pub param: Option<VectorQueryParam>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VectorQueryParam {
    pub ef_search: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryGroupBy {
    pub field_name: String,
    pub group_topk: usize,
    pub group_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QueryReranker {
    Rrf { rank_constant: u64 },
    Weighted { weights: std::collections::BTreeMap<String, f64> },
}
