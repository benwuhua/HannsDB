use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

pub fn append_vectors(path: &Path, vectors: &[BTreeMap<String, Vec<f32>>]) -> io::Result<usize> {
    invalidate_arrow_snapshot(path.with_extension("arrow"))?;
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

/// Load vectors from Arrow IPC if available, otherwise JSONL.
pub fn load_vectors(path: &Path) -> io::Result<Vec<BTreeMap<String, Vec<f32>>>> {
    let arrow_path = path.with_extension("arrow");
    if arrow_path.exists() {
        return super::arrow_io::load_vectors_arrow(&arrow_path);
    }
    load_vectors_jsonl(path)
}

/// Load vectors from JSONL format (used by active segments and legacy data).
pub fn load_vectors_jsonl(path: &Path) -> io::Result<Vec<BTreeMap<String, Vec<f32>>>> {
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

fn invalidate_arrow_snapshot(path: std::path::PathBuf) -> io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

/// Ensure the vectors JSONL file has exactly `existing_rows` rows by padding
/// missing rows with empty maps.
pub fn ensure_vector_rows(path: &Path, existing_rows: usize) -> io::Result<()> {
    match super::payloads::count_jsonl_lines(path) {
        Ok(count) => {
            if count != existing_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "vector row count is misaligned with existing records",
                ));
            }
            Ok(())
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            if existing_rows > 0 {
                let _ = append_vectors(path, &vec![BTreeMap::new(); existing_rows])?;
            }
            Ok(())
        }
        Err(err) => Err(err),
    }
}
