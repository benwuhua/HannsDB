use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use hannsdb_core::catalog::{CollectionMetadata, ManifestMetadata, CATALOG_FORMAT_VERSION};

fn unique_temp_file(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{}_{}.json", name, nanos))
}

#[test]
fn catalog_manifest_collection_roundtrip() {
    let path = unique_temp_file("hannsdb_collection_meta");
    let meta = CollectionMetadata::new("docs", 768, "cosine");

    meta.save_to_path(&path).expect("save collection metadata");
    let loaded = CollectionMetadata::load_from_path(&path).expect("load collection metadata");

    assert_eq!(loaded, meta);
    assert_eq!(loaded.format_version, CATALOG_FORMAT_VERSION);
}

#[test]
fn catalog_manifest_manifest_roundtrip() {
    let path = unique_temp_file("hannsdb_manifest_meta");
    let manifest = ManifestMetadata::new("hannsdb-local", vec!["docs".to_string()]);

    manifest
        .save_to_path(&path)
        .expect("save manifest metadata");
    let loaded = ManifestMetadata::load_from_path(&path).expect("load manifest metadata");

    assert_eq!(loaded, manifest);
    assert_eq!(loaded.format_version, CATALOG_FORMAT_VERSION);
}
