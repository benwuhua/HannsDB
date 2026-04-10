pub mod adapter;
pub mod bitset;
pub mod descriptor;
pub mod factory;
pub mod flat;
pub mod hnsw;
pub mod ivf;
pub mod scalar;

#[cfg(feature = "knowhere-backend")]
pub use knowhere_rs::BitsetView;

pub fn index_bootstrap_marker() -> &'static str {
    "hannsdb-index-bootstrap"
}

#[cfg(test)]
mod tests {
    #[test]
    fn exports_index_bootstrap_symbol() {
        let _ = super::index_bootstrap_marker();
    }
}
