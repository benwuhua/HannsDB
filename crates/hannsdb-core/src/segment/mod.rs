mod metadata;
mod payloads;
mod records;
pub mod segment_set;
mod tombstone;

pub use metadata::SegmentMetadata;
pub use payloads::{append_payloads, load_payloads};
pub use records::{append_record_ids, append_records, load_record_ids, load_records};
pub use tombstone::TombstoneMask;

pub const SEGMENT_FORMAT_VERSION: u32 = 1;
