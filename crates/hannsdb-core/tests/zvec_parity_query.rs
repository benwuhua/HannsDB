use hannsdb_core::db::{DocumentHit, HannsDb};
use serde_json::{json, Value};
use std::io;

type CurrentQueryDocumentsSignature =
    fn(&HannsDb, &str, &[f32], usize, Option<&str>) -> io::Result<Vec<DocumentHit>>;

fn current_query_documents_signature() -> CurrentQueryDocumentsSignature {
    HannsDb::query_documents
}

fn current_query_request_value() -> Value {
    let _query_fn = current_query_documents_signature();
    let collection = "docs";
    let query = vec![0.0_f32, 0.1];
    let top_k = 5;
    let filter: Option<&str> = None;

    json!({
        "collection": collection,
        "query": query,
        "top_k": top_k,
        "filter": filter
    })
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
