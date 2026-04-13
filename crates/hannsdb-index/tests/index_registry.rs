use hannsdb_index::descriptor::{
    ScalarIndexDescriptor, ScalarIndexKind, VectorIndexDescriptor, VectorIndexKind,
};
use hannsdb_index::factory::DefaultIndexFactory;
use serde_json::json;

#[test]
fn vector_and_scalar_descriptors_round_trip_through_json() {
    let vector = VectorIndexDescriptor {
        field_name: "embedding".to_string(),
        kind: VectorIndexKind::Ivf,
        metric: Some("cosine".to_string()),
        params: json!({
            "nlist": 32
        }),
    };
    let scalar = ScalarIndexDescriptor {
        field_name: "category".to_string(),
        kind: ScalarIndexKind::Inverted,
        params: json!({
            "tokenizer": "keyword"
        }),
    };

    let vector_json = serde_json::to_value(&vector).expect("serialize vector descriptor");
    let scalar_json = serde_json::to_value(&scalar).expect("serialize scalar descriptor");

    assert_eq!(vector_json["kind"], "ivf");
    assert_eq!(vector_json["metric"], "cosine");
    assert_eq!(vector_json["params"]["nlist"], 32);
    assert_eq!(scalar_json["kind"], "inverted");
    assert_eq!(scalar_json["params"]["tokenizer"], "keyword");

    let restored_vector: VectorIndexDescriptor =
        serde_json::from_value(vector_json).expect("deserialize vector descriptor");
    let restored_scalar: ScalarIndexDescriptor =
        serde_json::from_value(scalar_json).expect("deserialize scalar descriptor");

    assert_eq!(restored_vector, vector);
    assert_eq!(restored_scalar, scalar);
}

#[test]
fn ivf_usq_vector_descriptor_round_trips_through_json() {
    let vector = VectorIndexDescriptor {
        field_name: "embedding".to_string(),
        kind: VectorIndexKind::IvfUsq,
        metric: Some("l2".to_string()),
        params: json!({
            "nlist": 32,
            "bits_per_dim": 4,
            "rotation_seed": 42,
            "rerank_k": 64,
            "use_high_accuracy_scan": true
        }),
    };

    let vector_json = serde_json::to_value(&vector).expect("serialize vector descriptor");
    assert_eq!(vector_json["kind"], "ivf_usq");
    assert_eq!(vector_json["params"]["nlist"], 32);
    assert_eq!(vector_json["params"]["bits_per_dim"], 4);

    let restored_vector: VectorIndexDescriptor =
        serde_json::from_value(vector_json).expect("deserialize vector descriptor");
    assert_eq!(restored_vector, vector);
}

#[test]
fn hnsw_hvq_vector_descriptor_round_trips_through_json() {
    let vector = VectorIndexDescriptor {
        field_name: "embedding".to_string(),
        kind: VectorIndexKind::HnswHvq,
        metric: Some("ip".to_string()),
        params: json!({
            "m": 16,
            "m_max0": 32,
            "ef_construction": 100,
            "ef_search": 64,
            "nbits": 4
        }),
    };

    let vector_json = serde_json::to_value(&vector).expect("serialize vector descriptor");
    assert_eq!(vector_json["kind"], "hnsw_hvq");
    assert_eq!(vector_json["params"]["m"], 16);
    assert_eq!(vector_json["params"]["nbits"], 4);

    let restored_vector: VectorIndexDescriptor =
        serde_json::from_value(vector_json).expect("deserialize vector descriptor");
    assert_eq!(restored_vector, vector);
}

#[test]
fn factory_creates_expected_vector_backend_for_descriptor_kind() {
    let factory = DefaultIndexFactory::default();
    let descriptor = VectorIndexDescriptor {
        field_name: "embedding".to_string(),
        kind: VectorIndexKind::Flat,
        metric: Some("l2".to_string()),
        params: json!({}),
    };

    let mut backend = factory
        .create_vector_index(3, &descriptor, None)
        .expect("create flat backend");
    backend
        .insert(&[(7_u64, vec![0.0_f32, 0.0, 0.0])])
        .expect("insert vector");

    let hits = backend
        .search(&[0.1_f32, 0.0, 0.0], 1, 8)
        .expect("search flat backend");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 7);
}
