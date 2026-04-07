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
            "nlist": 4
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
        .search(&[0.2_f32, -0.1], 2, 8)
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
