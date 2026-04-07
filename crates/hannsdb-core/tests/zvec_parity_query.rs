use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repo root")
        .to_path_buf()
}

#[test]
fn zvec_parity_schema_query_surface_compiles_against_typed_batch_request() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let crate_dir = tempdir.path().join("query-surface-check");
    let src_dir = crate_dir.join("src");
    fs::create_dir_all(&src_dir).expect("create src dir");

    let core_path = repo_root().join("crates/hannsdb-core");
    fs::write(
        crate_dir.join("Cargo.toml"),
        format!(
            r#"[package]
name = "query-surface-check"
version = "0.1.0"
edition = "2021"

[dependencies]
hannsdb_core = {{ package = "hannsdb-core", path = "{}" }}
"#,
            core_path.display()
        ),
    )
    .expect("write Cargo.toml");

    fs::write(
        src_dir.join("main.rs"),
        r#"use hannsdb_core::query::{QueryContext, VectorQuery};

fn main() {
    let query = VectorQuery {
        field_name: "dense".to_string(),
        vector: vec![0.0_f32, 0.1],
        param: None,
    };
    let _request = QueryContext {
        queries: vec![query],
        query_by_id: Some(vec![11, 22]),
    };
}
"#,
    )
    .expect("write main.rs");

    let output = Command::new("cargo")
        .arg("check")
        .arg("--quiet")
        .current_dir(&crate_dir)
        .output()
        .expect("run cargo check");

    assert!(
        output.status.success(),
        "cargo check failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
