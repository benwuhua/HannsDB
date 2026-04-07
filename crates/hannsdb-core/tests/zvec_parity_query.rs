use std::any::type_name;
use std::io;

use hannsdb_core::db::{DocumentHit, HannsDb};

type CurrentQueryDocumentsSignature =
    fn(&HannsDb, &str, &[f32], usize, Option<&str>) -> io::Result<Vec<DocumentHit>>;

#[test]
fn zvec_parity_query_documents_surface_is_not_typed_batch_requests() {
    let _query_fn: CurrentQueryDocumentsSignature = HannsDb::query_documents;
    let current_signature = type_name::<CurrentQueryDocumentsSignature>();

    assert!(
        current_signature.contains("QueryContext"),
        "expected a typed batch/query-by-id query surface, got {current_signature}"
    );
}
