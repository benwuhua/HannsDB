#![cfg(feature = "lance-storage")]

use hannsdb_core::document::{
    CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema,
};
use hannsdb_core::storage::lance_store::{
    documents_to_lance_batch, LanceCollection, LanceDatasetStore,
};

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
