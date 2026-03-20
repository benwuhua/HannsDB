use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use hannsdb_core::segment::{
    append_records, load_records, SegmentMetadata, TombstoneMask, SEGMENT_FORMAT_VERSION,
};

fn unique_temp_file(name: &str, ext: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{}_{}.{}", name, nanos, ext))
}

#[test]
fn segment_storage_metadata_roundtrip() {
    let path = unique_temp_file("hannsdb_segment_meta", "json");
    let meta = SegmentMetadata::new("seg-0001", 4, 12, 3);

    meta.save_to_path(&path).expect("save segment metadata");
    let loaded = SegmentMetadata::load_from_path(&path).expect("load segment metadata");

    assert_eq!(loaded, meta);
    assert_eq!(loaded.format_version, SEGMENT_FORMAT_VERSION);
}

#[test]
fn segment_storage_record_append_and_load() {
    let path = unique_temp_file("hannsdb_segment_records", "bin");
    let dim = 4;
    let batch1 = vec![1.0_f32, 2.0, 3.0, 4.0];
    let batch2 = vec![5.0_f32, 6.0, 7.0, 8.0];

    let added1 = append_records(&path, dim, &batch1).expect("append first batch");
    let added2 = append_records(&path, dim, &batch2).expect("append second batch");
    let all = load_records(&path, dim).expect("load records");

    assert_eq!(added1, 1);
    assert_eq!(added2, 1);
    assert_eq!(all, [batch1, batch2].concat());
}

#[test]
fn segment_storage_tombstone_masking() {
    let path = unique_temp_file("hannsdb_segment_tombstone", "json");
    let mut mask = TombstoneMask::new(5);
    mask.mark_deleted(1);
    mask.mark_deleted(3);
    mask.save_to_path(&path).expect("save tombstone");

    let loaded = TombstoneMask::load_from_path(&path).expect("load tombstone");
    let visible = loaded.visible_indices();

    assert_eq!(visible, vec![0, 2, 4]);
    assert!(loaded.is_deleted(1));
    assert!(loaded.is_deleted(3));
    assert!(!loaded.is_deleted(2));
}
