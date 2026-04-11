//! Sparse vector storage: JSONL sidecar for sparse vector fields.
//!
//! Each row is a JSON map from field name to `{ "indices": [...], "values": [...] }`.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::document::SparseVector;

pub fn append_sparse_vectors(
    path: &Path,
    vectors: &[BTreeMap<String, SparseVector>],
) -> io::Result<usize> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = BufWriter::new(file);
    for vector_map in vectors {
        let line = serde_json::to_string(vector_map).map_err(json_to_io_error)?;
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(vectors.len())
}

pub fn load_sparse_vectors(path: &Path) -> io::Result<Vec<BTreeMap<String, SparseVector>>> {
    let file = OpenOptions::new().read(true).open(path)?;
    let reader = BufReader::new(file);
    let mut vectors = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let vector_map = serde_json::from_str(&line).map_err(json_to_io_error)?;
        vectors.push(vector_map);
    }
    Ok(vectors)
}

fn json_to_io_error(err: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}
