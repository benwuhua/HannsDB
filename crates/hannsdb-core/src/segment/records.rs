use std::fs::OpenOptions;
use std::io;
use std::io::{Read, Write};
use std::path::Path;

pub fn append_records(path: &Path, dimension: usize, vectors: &[f32]) -> io::Result<usize> {
    if dimension == 0 || vectors.len() % dimension != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "vectors length must be divisible by dimension and dimension > 0",
        ));
    }

    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    for value in vectors {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(vectors.len() / dimension)
}

pub fn load_records(path: &Path, dimension: usize) -> io::Result<Vec<f32>> {
    if dimension == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "dimension must be > 0",
        ));
    }

    let mut file = OpenOptions::new().read(true).open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    if bytes.len() % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "record file byte length is not aligned to f32",
        ));
    }

    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    if values.len() % dimension != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "record count is not aligned to dimension",
        ));
    }

    Ok(values)
}

pub fn append_record_ids(path: &Path, ids: &[i64]) -> io::Result<usize> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    for id in ids {
        file.write_all(&id.to_le_bytes())?;
    }
    Ok(ids.len())
}

pub fn load_record_ids(path: &Path) -> io::Result<Vec<i64>> {
    let mut file = OpenOptions::new().read(true).open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    if bytes.len() % 8 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "ids file byte length is not aligned to i64",
        ));
    }

    let mut ids = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        ids.push(i64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    Ok(ids)
}
