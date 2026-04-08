use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

pub fn append_vectors(path: &Path, vectors: &[BTreeMap<String, Vec<f32>>]) -> io::Result<usize> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    for vector_map in vectors {
        let line = serde_json::to_string(vector_map).map_err(json_to_io_error)?;
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
    }
    Ok(vectors.len())
}

pub fn load_vectors(path: &Path) -> io::Result<Vec<BTreeMap<String, Vec<f32>>>> {
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
