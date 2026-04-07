pub mod catalog;
pub mod db;
pub mod document;
pub mod query;
pub mod segment;
pub mod wal;

pub use catalog::CollectionMetadata;
pub use document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema, VectorFieldSchema,
    VectorIndexSchema,
};

pub fn core_bootstrap_marker() -> &'static str {
    "hannsdb-core-bootstrap"
}

#[cfg(test)]
mod tests {
    #[test]
    fn exports_core_bootstrap_symbol() {
        let _ = super::core_bootstrap_marker();
    }
}
