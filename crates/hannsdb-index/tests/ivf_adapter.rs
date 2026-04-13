use hannsdb_index::descriptor::{VectorIndexDescriptor, VectorIndexKind};
use hannsdb_index::factory::DefaultIndexFactory;
use serde_json::json;

#[test]
fn ivf_factory_backend_returns_nearest_neighbors_for_small_fixture() {
    let factory = DefaultIndexFactory::default();
    let descriptor = VectorIndexDescriptor {
        field_name: "embedding".to_string(),
        kind: VectorIndexKind::Ivf,
        metric: Some("l2".to_string()),
        params: json!({
            "nlist": 1
        }),
    };

    let mut backend = factory
        .create_vector_index(2, &descriptor, None)
        .expect("create ivf backend");
    backend
        .insert(&[
            (10_u64, vec![0.0_f32, 0.0]),
            (20_u64, vec![5.0_f32, 5.0]),
            (30_u64, vec![10.0_f32, 10.0]),
        ])
        .expect("insert fixture");

    let hits = backend
        .search(&[0.2_f32, -0.1], 2, 1)
        .expect("search ivf backend");
    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![10, 20]);
}

#[test]
fn ivf_factory_backend_rejects_dimension_mismatch() {
    let factory = DefaultIndexFactory::default();
    let descriptor = VectorIndexDescriptor {
        field_name: "embedding".to_string(),
        kind: VectorIndexKind::Ivf,
        metric: Some("cosine".to_string()),
        params: json!({
            "nlist": 2
        }),
    };

    let mut backend = factory
        .create_vector_index(2, &descriptor, None)
        .expect("create ivf backend");
    let err = backend
        .insert(&[(1_u64, vec![1.0_f32, 2.0, 3.0])])
        .expect_err("dimension mismatch should fail");

    assert_eq!(
        format!("{err:?}"),
        "InvalidDimension { expected: 2, got: 3 }"
    );
}

#[cfg(feature = "hanns-backend")]
#[test]
fn ivf_usq_factory_backend_returns_nearest_neighbors_for_small_fixture() {
    let factory = DefaultIndexFactory::default();
    let descriptor = VectorIndexDescriptor {
        field_name: "embedding".to_string(),
        kind: VectorIndexKind::IvfUsq,
        metric: Some("l2".to_string()),
        params: json!({
            "nlist": 1,
            "bits_per_dim": 4,
            "rotation_seed": 42,
            "rerank_k": 64,
            "use_high_accuracy_scan": false
        }),
    };

    let mut backend = factory
        .create_vector_index(2, &descriptor, None)
        .expect("create ivf_usq backend");
    backend
        .insert(&[
            (10_u64, vec![0.0_f32, 0.0]),
            (20_u64, vec![5.0_f32, 5.0]),
            (30_u64, vec![10.0_f32, 10.0]),
        ])
        .expect("insert fixture");

    let hits = backend
        .search(&[0.2_f32, -0.1], 2, 1)
        .expect("search ivf_usq backend");
    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![10, 20]);
}

#[cfg(feature = "hanns-backend")]
#[test]
fn ivf_usq_factory_backend_search_with_bitset_prefilters_results() {
    let factory = DefaultIndexFactory::default();
    let descriptor = VectorIndexDescriptor {
        field_name: "embedding".to_string(),
        kind: VectorIndexKind::IvfUsq,
        metric: Some("l2".to_string()),
        params: json!({
            "nlist": 1,
            "bits_per_dim": 4,
            "rotation_seed": 42,
            "rerank_k": 64,
            "use_high_accuracy_scan": false
        }),
    };

    let mut backend = factory
        .create_vector_index(2, &descriptor, None)
        .expect("create ivf_usq backend");
    backend
        .insert(&[
            (0_u64, vec![0.0_f32, 0.0]),
            (1_u64, vec![0.1_f32, 0.0]),
            (2_u64, vec![10.0_f32, 10.0]),
        ])
        .expect("insert fixture");

    let bitset = hanns::BitsetView::from_vec(vec![0b01], 3);
    let hits = backend
        .search_with_bitset(&[0.0_f32, 0.0], 2, 1, &bitset)
        .expect("search ivf_usq backend with bitset");
    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![1, 2]);
}

#[cfg(feature = "hanns-backend")]
#[test]
fn hnsw_hvq_factory_backend_returns_nearest_neighbors_for_small_fixture() {
    let factory = DefaultIndexFactory::default();
    let descriptor = VectorIndexDescriptor {
        field_name: "embedding".to_string(),
        kind: VectorIndexKind::HnswHvq,
        metric: Some("ip".to_string()),
        params: json!({
            "m": 8,
            "m_max0": 16,
            "ef_construction": 32,
            "ef_search": 32,
            "nbits": 4
        }),
    };

    let mut backend = factory
        .create_vector_index(2, &descriptor, None)
        .expect("create hnsw_hvq backend");
    backend
        .insert(&[
            (0_u64, vec![1.0_f32, 0.0]),
            (1_u64, vec![0.9_f32, 0.0]),
            (2_u64, vec![0.0_f32, 1.0]),
        ])
        .expect("insert fixture");

    let hits = backend
        .search(&[1.0_f32, 0.0], 2, 32)
        .expect("search hnsw_hvq backend");
    let hit_ids = hits.iter().map(|hit| hit.id).collect::<Vec<_>>();
    assert_eq!(hit_ids, vec![0, 1]);
}
