mod arrow_io;
pub(crate) mod index_runtime;
mod manager;
mod metadata;
mod payloads;
mod records;
pub mod segment_set;
mod sparse;
mod tombstone;
mod vectors;
mod version_set;
mod writer;

pub use arrow_io::{
    load_payloads_arrow, load_vectors_arrow, write_payloads_arrow, write_vectors_arrow,
};
pub use manager::{SegmentManager, SegmentPaths};
pub use metadata::SegmentMetadata;
pub use payloads::{
    append_payloads, ensure_payload_rows, load_payloads, load_payloads_jsonl,
    load_payloads_with_fields,
};
pub use records::{
    append_record_ids, append_records, append_records_f16, load_record_ids, load_records,
    load_records_f16,
};
pub use segment_set::SegmentSet;
pub use sparse::{append_sparse_vectors, load_sparse_vectors};
pub use tombstone::TombstoneMask;
pub use vectors::{append_vectors, ensure_vector_rows, load_vectors, load_vectors_jsonl};
pub use version_set::{atomic_write, VersionSet};
pub use writer::SegmentWriter;

pub const SEGMENT_FORMAT_VERSION: u32 = 1;
