use hannsdb_index::adapter::{AdapterError, HnswAdapter, HnswSearchHit};
use hannsdb_index::hnsw::InMemoryHnswIndex;
#[cfg(feature = "knowhere-backend")]
use hannsdb_index::hnsw::KnowhereHnswIndex;

#[test]
fn hnsw_adapter_smoke_insert_and_search_roundtrip() {
    let backend = InMemoryHnswIndex::new(2, "l2").expect("in-memory backend should construct");
    let mut adapter = HnswAdapter::new(backend);

    adapter
        .insert(vec![
            (10_u64, vec![0.0_f32, 0.0_f32]),
            (20_u64, vec![5.0_f32, 5.0_f32]),
        ])
        .expect("insert should succeed");

    let hits = adapter
        .search(&[0.1_f32, -0.1_f32], 1)
        .expect("search should succeed");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 10_u64);
    assert!(hits[0].distance >= 0.0);
}

#[test]
fn hnsw_adapter_cosine_metric_ranking_smoke() {
    let backend = InMemoryHnswIndex::new(2, "cosine").expect("in-memory backend should construct");
    let mut adapter = HnswAdapter::new(backend);

    adapter
        .insert(vec![
            (1_u64, vec![1.0_f32, 0.0_f32]),
            (2_u64, vec![0.0_f32, 1.0_f32]),
        ])
        .expect("insert should succeed");

    let hits = adapter
        .search(&[1.0_f32, 0.0_f32], 2)
        .expect("search should succeed");

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, 1_u64);
    assert!(hits[0].distance <= hits[1].distance);
    assert_eq!(
        hits[0],
        HnswSearchHit {
            id: 1_u64,
            distance: 0.0
        }
    );
}

#[test]
fn hnsw_adapter_rejects_empty_insert() {
    let backend = InMemoryHnswIndex::new(2, "l2").expect("in-memory backend should construct");
    let mut adapter = HnswAdapter::new(backend);

    let err = adapter
        .insert(Vec::new())
        .expect_err("empty insert should fail");
    assert_eq!(err, AdapterError::EmptyInsert);
}

#[test]
fn hnsw_adapter_rejects_dimension_mismatch() {
    let backend = InMemoryHnswIndex::new(2, "l2").expect("in-memory backend should construct");
    let mut adapter = HnswAdapter::new(backend);

    let err = adapter
        .insert(vec![(1_u64, vec![1.0_f32, 2.0_f32, 3.0_f32])])
        .expect_err("dimension mismatch should fail");
    assert_eq!(
        err,
        AdapterError::InvalidDimension {
            expected: 2,
            got: 3
        }
    );
}

#[cfg(feature = "knowhere-backend")]
#[test]
fn knowhere_hnsw_adapter_roundtrip_insert_and_search() {
    let backend = KnowhereHnswIndex::new(2, "l2").expect("knowhere backend should construct");
    let mut adapter = HnswAdapter::new(backend);

    adapter
        .insert(vec![
            (101_u64, vec![0.0_f32, 0.0_f32]),
            (202_u64, vec![10.0_f32, 10.0_f32]),
        ])
        .expect("knowhere insert should succeed");

    let hits = adapter
        .search(&[0.05_f32, -0.05_f32], 1)
        .expect("knowhere search should succeed");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 101_u64);
}

#[cfg(feature = "knowhere-backend")]
#[test]
fn knowhere_hnsw_adapter_near_origin_l2_search_prefers_closest_point() {
    let backend = KnowhereHnswIndex::new(2, "l2").expect("knowhere backend should construct");
    let mut adapter = HnswAdapter::new(backend);

    adapter
        .insert(vec![
            (0_u64, vec![0.0_f32, 0.0_f32]),
            (1_u64, vec![10.0_f32, 10.0_f32]),
        ])
        .expect("knowhere insert should succeed");

    let hits = adapter
        .search(&[0.1_f32, -0.1_f32], 1)
        .expect("knowhere search should succeed");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 0_u64);
}

#[cfg(feature = "knowhere-backend")]
#[test]
fn knowhere_hnsw_adapter_l2_search_matches_expected_neighbor_for_small_fixture() {
    let backend = KnowhereHnswIndex::new(2, "l2").expect("knowhere backend should construct");
    let mut adapter = HnswAdapter::new(backend);

    adapter
        .insert(vec![
            (0_u64, vec![1.0_f32, 1.0_f32]),
            (1_u64, vec![4.0_f32, 5.0_f32]),
        ])
        .expect("knowhere insert should succeed");

    let hits = adapter
        .search(&[2.0_f32, 1.0_f32], 1)
        .expect("knowhere search should succeed");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 0_u64);
}
