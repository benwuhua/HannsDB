#[derive(Debug, Clone, PartialEq)]
pub struct QueryContext {
    pub top_k: usize,
    pub queries: Vec<VectorQuery>,
    pub query_by_id: Option<Vec<i64>>,
    pub filter: Option<String>,
    pub group_by: Option<QueryGroupBy>,
    pub reranker: Option<QueryReranker>,
}

impl Default for QueryContext {
    fn default() -> Self {
        Self {
            top_k: 10,
            queries: Vec::new(),
            query_by_id: None,
            filter: None,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryReranker {
    pub model: String,
}
