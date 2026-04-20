#![cfg(feature = "lance-storage")]

use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema, SparseVector,
    VectorFieldSchema,
};
use hannsdb_core::storage::lance_store::{
    documents_from_lance_batch, documents_to_lance_batch, LanceCollection, LanceDatasetStore,
    LanceSearchProjection, LanceSparseIndexPath,
};
use hannsdb_core::storage::selector::{StorageBackend, StorageCollection};

fn sample_schema() -> CollectionSchema {
    CollectionSchema {
        primary_vector: "dense".to_string(),
        fields: vec![
            ScalarFieldSchema::new("title", FieldType::String),
            ScalarFieldSchema::new("year", FieldType::Int64),
        ],
        vectors: vec![hannsdb_core::document::VectorFieldSchema::new("dense", 3)],
    }
}

fn sample_documents() -> Vec<Document> {
    vec![
        Document::with_primary_vector_name(
            10,
            vec![
                ("title".to_string(), FieldValue::String("alpha".to_string())),
                ("year".to_string(), FieldValue::Int64(2024)),
            ],
            "dense",
            vec![1.0, 2.0, 3.0],
        ),
        Document::with_primary_vector_name(
            20,
            vec![
                ("title".to_string(), FieldValue::String("beta".to_string())),
                ("year".to_string(), FieldValue::Int64(2025)),
            ],
            "dense",
            vec![4.0, 5.0, 6.0],
        ),
    ]
}

fn sparse_schema() -> CollectionSchema {
    CollectionSchema {
        primary_vector: "dense".to_string(),
        fields: vec![ScalarFieldSchema::new("title", FieldType::String)],
        vectors: vec![
            VectorFieldSchema::new("dense", 3),
            VectorFieldSchema {
                name: "sparse_title".to_string(),
                data_type: FieldType::VectorSparse,
                dimension: 0,
                index_param: None,
                bm25_params: None,
            },
        ],
    }
}

fn sparse_documents() -> Vec<Document> {
    vec![
        Document::with_sparse_vectors(
            10,
            vec![("title".to_string(), FieldValue::String("alpha".to_string()))],
            "dense",
            vec![1.0, 2.0, 3.0],
            vec![(
                "sparse_title".to_string(),
                SparseVector::new(vec![1, 5, 9], vec![0.5, 2.0, 1.5]),
            )],
        ),
        Document::with_sparse_vectors(
            20,
            vec![("title".to_string(), FieldValue::String("beta".to_string()))],
            "dense",
            vec![4.0, 5.0, 6.0],
            vec![(
                "sparse_title".to_string(),
                SparseVector::new(vec![2, 5], vec![1.25, 3.5]),
            )],
        ),
    ]
}

fn pushdown_schema() -> CollectionSchema {
    CollectionSchema {
        primary_vector: "dense".to_string(),
        fields: vec![
            ScalarFieldSchema::new("title", FieldType::String),
            ScalarFieldSchema::new("group", FieldType::Int64),
            ScalarFieldSchema::new("score", FieldType::Float64),
            ScalarFieldSchema::new("extra", FieldType::String),
        ],
        vectors: vec![hannsdb_core::document::VectorFieldSchema::new("dense", 2)],
    }
}

fn pushdown_documents() -> Vec<Document> {
    vec![
        Document::with_primary_vector_name(
            1,
            vec![
                ("title".to_string(), FieldValue::String("alpha".to_string())),
                ("group".to_string(), FieldValue::Int64(1)),
                ("score".to_string(), FieldValue::Float64(10.0)),
                (
                    "extra".to_string(),
                    FieldValue::String("hidden-a".to_string()),
                ),
            ],
            "dense",
            vec![0.0, 0.0],
        ),
        Document::with_primary_vector_name(
            2,
            vec![
                ("title".to_string(), FieldValue::String("beta".to_string())),
                ("group".to_string(), FieldValue::Int64(1)),
                ("score".to_string(), FieldValue::Float64(20.0)),
                (
                    "extra".to_string(),
                    FieldValue::String("hidden-b".to_string()),
                ),
            ],
            "dense",
            vec![1.0, 1.0],
        ),
        Document::with_primary_vector_name(
            3,
            vec![
                ("title".to_string(), FieldValue::String("gamma".to_string())),
                ("group".to_string(), FieldValue::Int64(2)),
                ("score".to_string(), FieldValue::Float64(30.0)),
                (
                    "extra".to_string(),
                    FieldValue::String("hidden-c".to_string()),
                ),
            ],
            "dense",
            vec![2.0, 2.0],
        ),
    ]
}

#[test]
fn lance_batch_conversion_preserves_ids_scalars_and_vectors() {
    let batch = documents_to_lance_batch(&sample_schema(), &sample_documents())
        .expect("convert documents to lance batch");

    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.schema().field(0).name(), "id");
    assert_eq!(
        batch.schema().field_with_name("title").unwrap().name(),
        "title"
    );
    assert_eq!(
        batch.schema().field_with_name("year").unwrap().name(),
        "year"
    );
    assert_eq!(
        batch.schema().field_with_name("dense").unwrap().name(),
        "dense"
    );
}

#[test]
fn lance_batch_conversion_preserves_sparse_vectors() {
    let schema = sparse_schema();
    let documents = sparse_documents();
    let batch = documents_to_lance_batch(&schema, &documents)
        .expect("convert sparse documents to lance batch");

    assert_eq!(batch.num_rows(), 2);
    assert!(batch.schema().field_with_name("sparse_title").is_ok());

    let round_tripped =
        documents_from_lance_batch(&schema, &batch).expect("decode sparse documents from batch");
    assert_eq!(
        round_tripped[0].sparse_vectors.get("sparse_title"),
        documents[0].sparse_vectors.get("sparse_title")
    );
    assert_eq!(
        round_tripped[1].sparse_vectors.get("sparse_title"),
        documents[1].sparse_vectors.get("sparse_title")
    );
    assert_eq!(
        round_tripped[0].vectors.get("dense"),
        Some(&vec![1.0, 2.0, 3.0])
    );
}

#[tokio::test]
async fn lance_collection_can_write_read_and_reopen_sparse_vectors() {
    let temp = tempfile::tempdir().expect("tempdir");
    let docs = sparse_documents();
    let collection = LanceCollection::create(temp.path(), "docs", sparse_schema(), &docs[..1])
        .await
        .expect("create sparse lance collection");
    collection
        .insert_documents(&docs[1..])
        .await
        .expect("append sparse lance documents");

    let external = lance::Dataset::open(collection.uri())
        .await
        .expect("open sparse dataset with upstream Lance");
    assert_eq!(external.count_rows(None).await.expect("count rows"), 2);
    let external_batch = external.scan().try_into_batch().await.expect("scan rows");
    assert!(external_batch
        .schema()
        .field_with_name("sparse_title")
        .is_ok());

    let fetched = collection
        .fetch_documents(&[20, 10])
        .await
        .expect("fetch sparse documents");
    assert_eq!(
        fetched[0].sparse_vectors.get("sparse_title"),
        docs[1].sparse_vectors.get("sparse_title")
    );
    assert_eq!(
        fetched[1].sparse_vectors.get("sparse_title"),
        docs[0].sparse_vectors.get("sparse_title")
    );

    let reopened = LanceCollection::open_inferred(temp.path(), "docs")
        .await
        .expect("reopen sparse collection with inferred schema");
    let sparse_vector = reopened
        .schema()
        .vectors
        .iter()
        .find(|vector| vector.name == "sparse_title")
        .expect("inferred sparse vector field");
    assert_eq!(sparse_vector.data_type, FieldType::VectorSparse);
    assert_eq!(sparse_vector.dimension, 0);

    let reopened_fetched = reopened
        .fetch_documents(&[10, 20])
        .await
        .expect("fetch sparse documents after reopen");
    assert_eq!(
        reopened_fetched[0].sparse_vectors.get("sparse_title"),
        docs[0].sparse_vectors.get("sparse_title")
    );
    assert_eq!(
        reopened_fetched[1].sparse_vectors.get("sparse_title"),
        docs[1].sparse_vectors.get("sparse_title")
    );

    let external = lance::Dataset::open(reopened.uri())
        .await
        .expect("open sparse dataset with upstream Lance");
    assert_eq!(external.count_rows(None).await.expect("count rows"), 2);
}

#[cfg(feature = "hanns-backend")]
#[tokio::test]
async fn lance_collection_sparse_sidecar_survives_reopen_and_invalidates_on_mutation() {
    let temp = tempfile::tempdir().expect("tempdir");
    let collection =
        LanceCollection::create(temp.path(), "docs", sparse_schema(), &sparse_documents())
            .await
            .expect("create sparse Lance collection");

    collection
        .optimize_sparse("sparse_title", "ip")
        .await
        .expect("build sparse/BM25 sidecar");
    let sparse_sidecar_dir = std::path::Path::new(collection.uri())
        .join("_hannsdb")
        .join("sparse");
    assert!(
        sparse_sidecar_dir.exists(),
        "sparse/BM25 sidecar directory should prove indexed path build"
    );

    let reopened = LanceCollection::open_inferred(temp.path(), "docs")
        .await
        .expect("reopen sparse Lance collection");
    assert!(
        std::path::Path::new(reopened.uri())
            .join("_hannsdb")
            .join("sparse")
            .exists(),
        "reopen should preserve or load the sparse/BM25 sidecar"
    );

    reopened
        .insert_documents(&[Document::with_sparse_vectors(
            30,
            [("title".to_string(), FieldValue::String("gamma".to_string()))],
            "dense",
            vec![0.5, 0.5, 0.5],
            [(
                "sparse_title".to_string(),
                SparseVector::new(vec![1], vec![4.0]),
            )],
        )])
        .await
        .expect("insert invalidates sparse sidecar");
    assert!(
        !std::path::Path::new(reopened.uri())
            .join("_hannsdb")
            .join("sparse")
            .exists(),
        "sparse/BM25 sidecar should be invalidated rather than silently stale"
    );
}

#[tokio::test]
async fn lance_dataset_store_writes_dataset_openable_by_lance() {
    let temp = tempfile::tempdir().expect("tempdir");
    let uri = temp.path().join("docs.lance");
    let store = LanceDatasetStore::new(uri.to_string_lossy(), sample_schema());

    store
        .create(&sample_documents())
        .await
        .expect("create lance dataset");

    let dataset = lance::Dataset::open(uri.to_str().unwrap())
        .await
        .expect("open with upstream lance");
    assert_eq!(dataset.count_rows(None).await.expect("count rows"), 2);
}

#[tokio::test]
async fn lance_dataset_store_can_create_empty_dataset() {
    let temp = tempfile::tempdir().expect("tempdir");
    let uri = temp.path().join("empty.lance");
    let store = LanceDatasetStore::new(uri.to_string_lossy(), sample_schema());

    store.create(&[]).await.expect("create empty dataset");

    let dataset = lance::Dataset::open(uri.to_str().unwrap())
        .await
        .expect("open empty dataset with upstream lance");
    assert_eq!(dataset.count_rows(None).await.expect("count rows"), 0);
}

#[tokio::test]
async fn lance_dataset_store_append_is_visible_to_lance_scan() {
    let temp = tempfile::tempdir().expect("tempdir");
    let uri = temp.path().join("docs.lance");
    let store = LanceDatasetStore::new(uri.to_string_lossy(), sample_schema());

    store
        .create(&sample_documents()[..1])
        .await
        .expect("create lance dataset");
    store
        .append(&sample_documents()[1..])
        .await
        .expect("append lance dataset");

    let dataset = LanceDatasetStore::new(uri.to_string_lossy(), sample_schema())
        .open_lance()
        .await
        .expect("open via store");
    assert_eq!(dataset.count_rows(None).await.expect("count rows"), 2);

    let batch = dataset.scan().try_into_batch().await.expect("scan rows");
    assert_eq!(batch.num_rows(), 2);
    assert!(batch.schema().field_with_name("title").is_ok());
    assert!(batch.schema().field_with_name("dense").is_ok());
}

#[tokio::test]
async fn lance_dataset_store_fetches_documents_from_lance_dataset() {
    let temp = tempfile::tempdir().expect("tempdir");
    let uri = temp.path().join("docs.lance");
    let store = LanceDatasetStore::new(uri.to_string_lossy(), sample_schema());
    store
        .create(&sample_documents())
        .await
        .expect("create lance dataset");

    let fetched = store.fetch(&[20, 10]).await.expect("fetch from lance");

    assert_eq!(
        fetched
            .iter()
            .map(|document| document.id)
            .collect::<Vec<_>>(),
        vec![20, 10]
    );
    assert_eq!(
        fetched[0].fields.get("title"),
        Some(&FieldValue::String("beta".to_string()))
    );
    assert_eq!(fetched[0].vectors.get("dense"), Some(&vec![4.0, 5.0, 6.0]));
}

#[tokio::test]
async fn lance_dataset_store_bruteforce_searches_lance_vectors() {
    let temp = tempfile::tempdir().expect("tempdir");
    let uri = temp.path().join("docs.lance");
    let store = LanceDatasetStore::new(uri.to_string_lossy(), sample_schema());
    store
        .create(&sample_documents())
        .await
        .expect("create lance dataset");

    let hits = store
        .search(&[4.0, 5.0, 6.0], 1, "l2")
        .await
        .expect("search lance dataset");

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 20);
    assert_eq!(hits[0].distance, 0.0);
}

#[tokio::test]
async fn lance_collection_facade_supports_basic_insert_fetch_and_search() {
    let temp = tempfile::tempdir().expect("tempdir");
    let docs = sample_documents();
    let collection = LanceCollection::create(temp.path(), "docs", sample_schema(), &docs[..1])
        .await
        .expect("create lance collection facade");

    collection
        .insert_documents(&docs[1..])
        .await
        .expect("insert second batch");

    let external = lance::Dataset::open(collection.uri())
        .await
        .expect("open facade data with upstream Lance");
    assert_eq!(external.count_rows(None).await.expect("count rows"), 2);

    let fetched = collection
        .fetch_documents(&[20, 10])
        .await
        .expect("fetch documents");
    assert_eq!(
        fetched
            .iter()
            .map(|document| document.id)
            .collect::<Vec<_>>(),
        vec![20, 10]
    );

    let hits = collection
        .search(&[1.0, 2.0, 3.0], 1, "l2")
        .await
        .expect("search collection");
    assert_eq!(hits[0].id, 10);
}

#[tokio::test]
async fn lance_collection_can_reopen_with_inferred_schema() {
    let temp = tempfile::tempdir().expect("tempdir");
    let docs = sample_documents();
    let created = LanceCollection::create(temp.path(), "docs", sample_schema(), &docs)
        .await
        .expect("create lance collection facade");

    let reopened = LanceCollection::open_inferred(temp.path(), "docs")
        .await
        .expect("reopen Lance collection with inferred schema");

    assert_eq!(reopened.name(), "docs");
    assert_eq!(reopened.uri(), created.uri());
    assert_eq!(reopened.schema().primary_vector_name(), "dense");
    assert_eq!(reopened.schema().fields.len(), 2);
    assert_eq!(reopened.schema().vectors[0].dimension, 3);
    let fetched = reopened
        .fetch_documents(&[10, 20])
        .await
        .expect("fetch rows");
    assert_eq!(
        fetched
            .iter()
            .map(|document| document.id)
            .collect::<Vec<_>>(),
        vec![10, 20]
    );
}

#[tokio::test]
async fn core_storage_selector_can_create_and_reopen_lance_collection() {
    let temp = tempfile::tempdir().expect("tempdir");
    let docs = sample_documents();

    let collection = StorageBackend::Lance
        .create_collection(temp.path(), "docs", sample_schema(), &docs[..1])
        .await
        .expect("create via storage selector");
    assert!(matches!(collection, StorageCollection::Lance(_)));
    assert_eq!(collection.name(), "docs");

    collection
        .insert_documents(&docs[1..])
        .await
        .expect("insert through selected collection");
    let fetched = collection
        .fetch_documents(&[10, 20])
        .await
        .expect("fetch through selected collection");
    assert_eq!(
        fetched
            .iter()
            .map(|document| document.id)
            .collect::<Vec<_>>(),
        vec![10, 20]
    );

    let reopened = StorageBackend::Lance
        .open_collection_inferred(temp.path(), "docs")
        .await
        .expect("reopen via storage selector");
    assert_eq!(reopened.schema().primary_vector_name(), "dense");
    assert_eq!(
        reopened
            .fetch_documents(&[20])
            .await
            .expect("fetch reopened")[0]
            .id,
        20
    );
}

#[tokio::test]
async fn lance_collection_delete_documents_hides_rows_from_lance_and_facade() {
    let temp = tempfile::tempdir().expect("tempdir");
    let collection =
        LanceCollection::create(temp.path(), "docs", sample_schema(), &sample_documents())
            .await
            .expect("create lance collection facade");

    let deleted = collection
        .delete_documents(&[10])
        .await
        .expect("delete document");

    assert_eq!(deleted, 1);
    assert!(collection.fetch_documents(&[10]).await.unwrap().is_empty());
    assert_eq!(collection.fetch_documents(&[20]).await.unwrap()[0].id, 20);

    let external = lance::Dataset::open(collection.uri())
        .await
        .expect("open facade data with upstream Lance");
    assert_eq!(external.count_rows(None).await.expect("count live rows"), 1);
}

#[tokio::test]
async fn lance_collection_upsert_replaces_existing_and_inserts_new_documents() {
    let temp = tempfile::tempdir().expect("tempdir");
    let collection =
        LanceCollection::create(temp.path(), "docs", sample_schema(), &sample_documents())
            .await
            .expect("create lance collection facade");
    let replacements = vec![
        Document::with_primary_vector_name(
            20,
            vec![
                (
                    "title".to_string(),
                    FieldValue::String("beta-v2".to_string()),
                ),
                ("year".to_string(), FieldValue::Int64(2026)),
            ],
            "dense",
            vec![7.0, 8.0, 9.0],
        ),
        Document::with_primary_vector_name(
            30,
            vec![
                ("title".to_string(), FieldValue::String("gamma".to_string())),
                ("year".to_string(), FieldValue::Int64(2027)),
            ],
            "dense",
            vec![10.0, 11.0, 12.0],
        ),
    ];

    let upserted = collection
        .upsert_documents(&replacements)
        .await
        .expect("upsert documents");

    assert_eq!(upserted, 2);
    assert!(collection.fetch_documents(&[20]).await.unwrap()[0]
        .fields
        .get("title")
        .is_some_and(|value| value == &FieldValue::String("beta-v2".to_string())));
    assert_eq!(collection.fetch_documents(&[30]).await.unwrap()[0].id, 30);

    let external = lance::Dataset::open(collection.uri())
        .await
        .expect("open facade data with upstream Lance");
    assert_eq!(external.count_rows(None).await.expect("count live rows"), 3);
}

#[cfg(feature = "hanns-backend")]
#[tokio::test]
async fn lance_collection_hanns_sidecar_optimize_creates_artifact_and_searches() {
    let temp = tempfile::tempdir().expect("tempdir");
    let collection =
        LanceCollection::create(temp.path(), "docs", sample_schema(), &sample_documents())
            .await
            .expect("create lance collection facade");

    collection
        .optimize_hanns("dense", "l2")
        .await
        .expect("build Hanns sidecar");

    assert!(
        collection.hanns_index_path("dense").exists(),
        "Hanns sidecar artifact should exist after optimize_hanns"
    );

    let hits = collection
        .search(&[4.0, 5.0, 6.0], 1, "l2")
        .await
        .expect("search with Hanns sidecar");

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 20);
    assert_eq!(hits[0].distance, 0.0);
}

#[cfg(feature = "hanns-backend")]
#[tokio::test]
async fn lance_collection_mutations_invalidate_hanns_sidecar() {
    let temp = tempfile::tempdir().expect("tempdir");
    let collection =
        LanceCollection::create(temp.path(), "docs", sample_schema(), &sample_documents())
            .await
            .expect("create lance collection facade");

    collection
        .optimize_hanns("dense", "l2")
        .await
        .expect("build Hanns sidecar");
    assert!(collection.hanns_index_path("dense").exists());

    collection
        .insert_documents(&[Document::with_primary_vector_name(
            30,
            vec![
                ("title".to_string(), FieldValue::String("gamma".to_string())),
                ("year".to_string(), FieldValue::Int64(2026)),
            ],
            "dense",
            vec![0.0, 0.0, 0.0],
        )])
        .await
        .expect("insert invalidates sidecar");
    assert!(
        !collection.hanns_index_path("dense").exists(),
        "insert should invalidate stale Hanns sidecar"
    );

    collection
        .optimize_hanns("dense", "l2")
        .await
        .expect("rebuild Hanns sidecar");
    assert!(collection.hanns_index_path("dense").exists());

    collection
        .delete_documents(&[30])
        .await
        .expect("delete invalidates sidecar");
    assert!(
        !collection.hanns_index_path("dense").exists(),
        "delete should invalidate stale Hanns sidecar"
    );

    collection
        .optimize_hanns("dense", "l2")
        .await
        .expect("rebuild Hanns sidecar");
    assert!(collection.hanns_index_path("dense").exists());

    collection
        .upsert_documents(&[Document::with_primary_vector_name(
            20,
            vec![
                (
                    "title".to_string(),
                    FieldValue::String("beta-v3".to_string()),
                ),
                ("year".to_string(), FieldValue::Int64(2027)),
            ],
            "dense",
            vec![9.0, 9.0, 9.0],
        )])
        .await
        .expect("upsert invalidates sidecar");
    assert!(
        !collection.hanns_index_path("dense").exists(),
        "upsert should invalidate stale Hanns sidecar"
    );
}

#[cfg(feature = "hanns-backend")]
#[tokio::test]
async fn lance_collection_sparse_sidecar_rebuilds_missing_then_loads_for_bm25_search() {
    let temp = tempfile::tempdir().expect("tempdir");
    let collection =
        LanceCollection::create(temp.path(), "docs", sparse_schema(), &sparse_documents())
            .await
            .expect("create sparse lance collection");

    let query = SparseVector::new(vec![5], vec![1.0]);
    let first = collection
        .search_sparse_vector_field_projected(
            "sparse_title",
            &query,
            2,
            "bm25",
            LanceSearchProjection::default(),
        )
        .await
        .expect("search sparse with missing sidecar rebuild");

    assert!(
        collection.sparse_index_path("sparse_title").exists(),
        "sparse sidecar artifact should exist after missing-sidecar rebuild"
    );
    assert_eq!(
        first
            .observation
            .sparse_index
            .as_ref()
            .expect("sparse observation")
            .path,
        LanceSparseIndexPath::RebuiltMissing
    );
    assert_eq!(
        first.hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![20, 10]
    );

    let second = collection
        .search_sparse_vector_field_projected(
            "sparse_title",
            &query,
            2,
            "bm25",
            LanceSearchProjection::default(),
        )
        .await
        .expect("search sparse with loaded sidecar");
    assert_eq!(
        second
            .observation
            .sparse_index
            .as_ref()
            .expect("sparse observation")
            .path,
        LanceSparseIndexPath::Loaded
    );
    assert_eq!(
        second.hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![20, 10]
    );
}

#[cfg(feature = "hanns-backend")]
#[tokio::test]
async fn lance_collection_sparse_sidecar_rebuilds_corrupt_artifact() {
    let temp = tempfile::tempdir().expect("tempdir");
    let collection =
        LanceCollection::create(temp.path(), "docs", sparse_schema(), &sparse_documents())
            .await
            .expect("create sparse lance collection");

    collection
        .optimize_sparse("sparse_title", "bm25")
        .await
        .expect("build sparse sidecar");
    std::fs::write(
        collection.sparse_index_path("sparse_title"),
        b"not a sparse index",
    )
    .expect("corrupt sparse sidecar");

    let result = collection
        .search_sparse_vector_field_projected(
            "sparse_title",
            &SparseVector::new(vec![5], vec![1.0]),
            2,
            "bm25",
            LanceSearchProjection::default(),
        )
        .await
        .expect("search sparse rebuilds corrupt sidecar");

    assert_eq!(
        result
            .observation
            .sparse_index
            .as_ref()
            .expect("sparse observation")
            .path,
        LanceSparseIndexPath::RebuiltCorrupt
    );
    assert_eq!(
        result.hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![20, 10]
    );
}

#[cfg(feature = "hanns-backend")]
#[tokio::test]
async fn lance_collection_sparse_sidecar_invalidates_on_mutation() {
    let temp = tempfile::tempdir().expect("tempdir");
    let collection =
        LanceCollection::create(temp.path(), "docs", sparse_schema(), &sparse_documents())
            .await
            .expect("create sparse lance collection");

    collection
        .optimize_sparse("sparse_title", "bm25")
        .await
        .expect("build sparse sidecar");
    assert!(collection.sparse_index_path("sparse_title").exists());

    collection
        .insert_documents(&[Document::with_sparse_vectors(
            30,
            vec![("title".to_string(), FieldValue::String("gamma".to_string()))],
            "dense",
            vec![7.0, 8.0, 9.0],
            vec![(
                "sparse_title".to_string(),
                SparseVector::new(vec![5, 11], vec![4.0, 1.0]),
            )],
        )])
        .await
        .expect("insert invalidates sparse sidecar");
    assert!(
        !collection.sparse_index_path("sparse_title").exists(),
        "insert should invalidate stale sparse sidecar"
    );

    collection
        .optimize_sparse("sparse_title", "bm25")
        .await
        .expect("rebuild sparse sidecar");
    assert!(collection.sparse_index_path("sparse_title").exists());

    collection
        .delete_documents(&[30])
        .await
        .expect("delete invalidates sparse sidecar");
    assert!(
        !collection.sparse_index_path("sparse_title").exists(),
        "delete should invalidate stale sparse sidecar"
    );

    collection
        .optimize_sparse("sparse_title", "bm25")
        .await
        .expect("rebuild sparse sidecar");
    assert!(collection.sparse_index_path("sparse_title").exists());

    collection
        .upsert_documents(&[Document::with_sparse_vectors(
            20,
            vec![(
                "title".to_string(),
                FieldValue::String("beta-v2".to_string()),
            )],
            "dense",
            vec![4.0, 5.0, 6.0],
            vec![(
                "sparse_title".to_string(),
                SparseVector::new(vec![2, 5], vec![1.25, 5.0]),
            )],
        )])
        .await
        .expect("upsert invalidates sparse sidecar");
    assert!(
        !collection.sparse_index_path("sparse_title").exists(),
        "upsert should invalidate stale sparse sidecar"
    );
}

#[tokio::test]
async fn lance_dataset_store_pushes_equality_filter_to_scan() {
    let temp = tempfile::tempdir().expect("tempdir");
    let uri = temp.path().join("docs.lance");
    let store = LanceDatasetStore::new(uri.to_string_lossy(), pushdown_schema());
    store
        .create(&pushdown_documents())
        .await
        .expect("create lance dataset");
    let filter = hannsdb_core::query::parse_filter("group == 1").expect("parse filter");

    let result = store
        .search_vector_field_filtered_projected(
            "dense",
            LanceSearchProjection::with_output_fields(["title".to_string()]),
            &[0.0, 0.0],
            10,
            "l2",
            Some(&filter),
        )
        .await
        .expect("search with pushdown");

    assert_eq!(
        result.hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(result.observation.predicate.as_deref(), Some("group = 1"));
    assert!(result.observation.fallback_reason.is_none());
}

#[tokio::test]
async fn lance_dataset_store_pushes_range_and_filter_to_scan() {
    let temp = tempfile::tempdir().expect("tempdir");
    let uri = temp.path().join("docs.lance");
    let store = LanceDatasetStore::new(uri.to_string_lossy(), pushdown_schema());
    store
        .create(&pushdown_documents())
        .await
        .expect("create lance dataset");
    let filter =
        hannsdb_core::query::parse_filter("group == 1 and score >= 15.0").expect("parse filter");

    let result = store
        .search_vector_field_filtered_projected(
            "dense",
            LanceSearchProjection::with_output_fields(["title".to_string()]),
            &[0.0, 0.0],
            10,
            "l2",
            Some(&filter),
        )
        .await
        .expect("search with and pushdown");

    assert_eq!(
        result.hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![2]
    );
    assert_eq!(
        result.observation.predicate.as_deref(),
        Some("(group = 1) AND (score >= 15)")
    );
    assert!(result.observation.fallback_reason.is_none());
}

#[tokio::test]
async fn lance_dataset_store_projects_requested_columns() {
    let temp = tempfile::tempdir().expect("tempdir");
    let uri = temp.path().join("docs.lance");
    let store = LanceDatasetStore::new(uri.to_string_lossy(), pushdown_schema());
    store
        .create(&pushdown_documents())
        .await
        .expect("create lance dataset");

    let result = store
        .search_vector_field_filtered_projected(
            "dense",
            LanceSearchProjection::with_output_fields(["title".to_string()]),
            &[0.0, 0.0],
            1,
            "l2",
            None,
        )
        .await
        .expect("search with projection");

    assert!(result
        .observation
        .projected_columns
        .contains(&"id".to_string()));
    assert!(result
        .observation
        .projected_columns
        .contains(&"dense".to_string()));
    assert!(result
        .observation
        .projected_columns
        .contains(&"title".to_string()));
    assert!(!result
        .observation
        .projected_columns
        .contains(&"extra".to_string()));
    assert_eq!(
        result.documents[0].document.fields.get("title"),
        Some(&FieldValue::String("alpha".to_string()))
    );
    assert!(!result.documents[0].document.fields.contains_key("extra"));
}

#[tokio::test]
async fn lance_dataset_store_falls_back_for_unsupported_filter() {
    let temp = tempfile::tempdir().expect("tempdir");
    let uri = temp.path().join("docs.lance");
    let store = LanceDatasetStore::new(uri.to_string_lossy(), pushdown_schema());
    store
        .create(&pushdown_documents())
        .await
        .expect("create lance dataset");
    let filter = hannsdb_core::query::parse_filter("title like \"%a\"").expect("parse filter");

    let result = store
        .search_vector_field_filtered_projected(
            "dense",
            LanceSearchProjection::with_output_fields(["title".to_string()]),
            &[0.0, 0.0],
            10,
            "l2",
            Some(&filter),
        )
        .await
        .expect("search with fallback");

    assert_eq!(
        result.hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert!(result.observation.predicate.is_none());
    assert!(result
        .observation
        .fallback_reason
        .as_deref()
        .is_some_and(|reason| reason.contains("like")));
}

#[tokio::test]
async fn lance_dataset_store_does_not_partially_push_mixed_filter() {
    let temp = tempfile::tempdir().expect("tempdir");
    let uri = temp.path().join("docs.lance");
    let store = LanceDatasetStore::new(uri.to_string_lossy(), pushdown_schema());
    store
        .create(&pushdown_documents())
        .await
        .expect("create lance dataset");
    let filter = hannsdb_core::query::parse_filter("group == 1 and title like \"%a\"")
        .expect("parse filter");

    let result = store
        .search_vector_field_filtered_projected(
            "dense",
            LanceSearchProjection::with_output_fields(["title".to_string()]),
            &[0.0, 0.0],
            10,
            "l2",
            Some(&filter),
        )
        .await
        .expect("search with mixed fallback");

    assert_eq!(
        result.hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert!(result.observation.predicate.is_none());
    assert!(result
        .observation
        .fallback_reason
        .as_deref()
        .is_some_and(|reason| reason.contains("like")));
}
