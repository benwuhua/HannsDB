mod manager;
mod metadata;
mod payloads;
mod records;
mod vectors;
pub mod segment_set;
mod tombstone;
mod version_set;

pub use manager::{SegmentManager, SegmentPaths};
pub use metadata::SegmentMetadata;
pub use payloads::{append_payloads, load_payloads};
pub use records::{append_record_ids, append_records, load_record_ids, load_records};
pub use vectors::{append_vectors, load_vectors};
pub use segment_set::SegmentSet;
pub use tombstone::TombstoneMask;
pub use version_set::VersionSet;

pub const SEGMENT_FORMAT_VERSION: u32 = 1;
