use hannsdb_daemon::api::{SparseVectorRequest, TypedSearchRequest, TypedVectorQueryRequest};
use serde_json::json;

#[test]
fn typed_search_accepts_only_existing_query_completion_shape() {
    let accepted = json!({
        "top_k": 3,
        "queries": [{
            "field_name": "vector",
            "vector": [0.0, 1.0],
            "param": {"ef_search": 32, "nprobe": 4}
        }],
        "query_by_id": ["42", "84"],
        "query_by_id_field_name": "vector",
        "filter": "group = 1",
        "output_fields": ["group"],
        "include_vector": true,
        "group_by": {"field_name": "group", "group_topk": 1, "group_count": 2},
        "reranker": {"rank_constant": 60, "weights": {"vector": 1.0}, "metric": "l2"},
        "order_by": {"field_name": "rank", "descending": true}
    });

    serde_json::from_value::<TypedSearchRequest>(accepted)
        .expect("all in-scope typed query-completion fields should remain accepted");
}

#[test]
fn typed_search_rejects_out_of_scope_public_shape_drift() {
    for request in [
        json!({"top_k": 1, "runtime": "lance"}),
        json!({"top_k": 1, "index_metadata": {"kind": "bm25"}}),
        json!({"top_k": 1, "compact": true}),
        json!({"top_k": 1, "queries": [], "benchmark_gate": true}),
    ] {
        let error = serde_json::from_value::<TypedSearchRequest>(request.clone())
            .expect_err("typed search must reject out-of-scope public request fields");
        assert!(
            error.to_string().contains("unknown field"),
            "unexpected error for {request}: {error}"
        );
    }
}

#[test]
fn typed_query_params_remain_limited_to_existing_fields() {
    let accepted = json!({
        "field_name": "vector",
        "vector": [0.0, 1.0],
        "param": {"ef_search": 32, "nprobe": 4}
    });
    serde_json::from_value::<TypedVectorQueryRequest>(accepted)
        .expect("existing typed query params should remain accepted");

    for query in [
        json!({"field_name": "vector", "vector": [0.0, 1.0], "param": {"metric": "ip"}}),
        json!({"field_name": "vector", "vector": [0.0, 1.0], "param": {"bm25": {"k1": 1.2}}}),
        json!({"field_name": "vector", "vector": [0.0, 1.0], "param": {"beam_width": 8}}),
    ] {
        let error = serde_json::from_value::<TypedVectorQueryRequest>(query.clone())
            .expect_err("query params must not grow new public fields in the Lance query PR");
        assert!(
            error.to_string().contains("unknown field"),
            "unexpected error for {query}: {error}"
        );
    }
}

#[test]
fn sparse_vector_request_shape_stays_indices_and_values_only() {
    serde_json::from_value::<SparseVectorRequest>(json!({"indices": [1, 5], "values": [0.5, 1.5]}))
        .expect("existing sparse vector shape should remain accepted");

    for sparse in [
        json!({"indices": [1], "values": [1.0], "metric": "bm25"}),
        json!({"indices": [1], "values": [1.0], "bm25": {"k1": 1.2, "b": 0.75}}),
        json!({"indices": [1], "values": [1.0], "norm": 1.0}),
    ] {
        let error = serde_json::from_value::<SparseVectorRequest>(sparse.clone())
            .expect_err("sparse/BM25 internals must not add public sparse vector fields");
        assert!(
            error.to_string().contains("unknown field"),
            "unexpected error for {sparse}: {error}"
        );
    }
}
