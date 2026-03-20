use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::document::FieldValue;

pub fn append_payloads(
    path: &Path,
    payloads: &[BTreeMap<String, FieldValue>],
) -> io::Result<usize> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    for payload in payloads {
        let line = serde_json::to_string(payload).map_err(json_to_io_error)?;
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
    }
    Ok(payloads.len())
}

pub fn load_payloads(path: &Path) -> io::Result<Vec<BTreeMap<String, FieldValue>>> {
    let file = OpenOptions::new().read(true).open(path)?;
    let reader = BufReader::new(file);
    let mut payloads = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let payload = serde_json::from_str(&line).map_err(json_to_io_error)?;
        payloads.push(payload);
    }
    Ok(payloads)
}

fn json_to_io_error(err: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}
