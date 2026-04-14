//! Forward-store kernel primitives for the zvec-aligned storage rewrite.
//!
//! This module intentionally stays isolated from HannsDB's current runtime
//! authority (`db.rs`, query execution, recovery, compaction, and the existing
//! `SegmentMetadata` / `SegmentPaths` layout). The declared-fields-only rule
//! implemented here applies only to this new forward-store core; it does not
//! tighten existing HannsDB runtime semantics outside this module.

pub mod descriptor;
pub mod mem_store;
pub mod schema;

pub use descriptor::{
    ForwardFileArtifact, ForwardFileFormat, ForwardStoreDescriptor,
    FORWARD_STORE_DESCRIPTOR_VERSION,
};
pub use mem_store::{ForwardStoreRow, MemForwardStore};
pub use schema::{
    ForwardColumnKind, ForwardColumnSchema, ForwardStoreSchema, ForwardSystemColumnKind,
    FORWARD_STORE_INTERNAL_ID_COLUMN, FORWARD_STORE_IS_DELETED_COLUMN,
    FORWARD_STORE_OP_SEQ_COLUMN,
};
