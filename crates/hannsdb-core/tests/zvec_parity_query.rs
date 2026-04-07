use serde::Serialize;
use serde_json::{json, Value};

#[derive(Serialize)]
struct CurrentSingleVectorQueryRequest<'a> {
    collection: &'a str,
    query: &'a [f32],
    top_k: usize,
    filter: Option<&'a str>,
}

fn current_query_request_value() -> Value {
    serde_json::to_value(CurrentSingleVectorQueryRequest {
        collection: "docs",
        query: &[0.0_f32, 0.1],
        top_k: 5,
        filter: None,
    })
    .expect("serialize current single-vector query shape")
}

#[test]
fn zvec_parity_query_request_supports_query_by_id_and_multi_vector_batches() {
    let actual = current_query_request_value();
    let expected = json!({
        "query_by_id": [11, 22],
        "queries": [
            {
                "field": "dense",
                "vector": [0.0, 0.1],
                "top_k": 5
            },
            {
                "field": "title",
                "vector": [0.2, 0.3],
                "top_k": 5
            }
        ]
    });

    assert_eq!(actual, expected);
}
