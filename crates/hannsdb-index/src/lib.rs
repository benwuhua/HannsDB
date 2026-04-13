pub mod adapter;
pub mod bitset;
pub mod descriptor;
pub mod factory;
pub mod flat;
pub mod hnsw;
pub mod hnsw_hvq;
pub mod ivf;
pub mod ivf_usq;
pub mod scalar;
pub mod sparse;

#[cfg(feature = "hanns-backend")]
pub use hanns::BitsetView;

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
