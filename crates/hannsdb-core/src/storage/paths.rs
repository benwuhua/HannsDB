use std::path::{Path, PathBuf};

const INDEX_CATALOG_FILE: &str = "indexes.json";

pub(crate) fn manifest_path(root: &Path) -> PathBuf {
    root.join("manifest.json")
}

pub(crate) fn wal_path(root: &Path) -> PathBuf {
    root.join("wal.jsonl")
}

pub(crate) fn collection_paths_for_root(root: &Path, name: &str) -> CollectionPaths {
    let dir = root.join("collections").join(name);
    collection_paths_for_dir(&dir)
}

pub(crate) fn collection_paths_for_dir(dir: &Path) -> CollectionPaths {
    let dir = dir.to_path_buf();
    CollectionPaths {
        dir: dir.clone(),
        collection_meta: dir.join("collection.json"),
        primary_keys: dir.join("primary_keys.json"),
        index_catalog: dir.join(INDEX_CATALOG_FILE),
        segment_set: dir.join("segment_set.json"),
        segments_dir: dir.join("segments"),
        segment_meta: dir.join("segment.json"),
        records: dir.join("records.bin"),
        external_ids: dir.join("ids.bin"),
        payloads: dir.join("payloads.jsonl"),
        vectors: dir.join("vectors.jsonl"),
        sparse_vectors: dir.join("sparse_vectors.jsonl"),
        tombstones: dir.join("tombstones.json"),
    }
}

pub(crate) struct CollectionPaths {
    pub(crate) dir: PathBuf,
    pub(crate) collection_meta: PathBuf,
    pub(crate) primary_keys: PathBuf,
    pub(crate) index_catalog: PathBuf,
    pub(crate) segment_set: PathBuf,
    pub(crate) segments_dir: PathBuf,
    pub(crate) segment_meta: PathBuf,
    pub(crate) records: PathBuf,
    pub(crate) external_ids: PathBuf,
    pub(crate) payloads: PathBuf,
    pub(crate) vectors: PathBuf,
    pub(crate) sparse_vectors: PathBuf,
    pub(crate) tombstones: PathBuf,
}
