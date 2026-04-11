use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::document::FieldValue;

pub fn append_payloads(
    path: &Path,
    payloads: &[BTreeMap<String, FieldValue>],
) -> io::Result<usize> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = BufWriter::new(file);
    for payload in payloads {
        let line = serde_json::to_string(payload).map_err(json_to_io_error)?;
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(payloads.len())
}

/// Load payloads from Arrow IPC if available, otherwise JSONL.
pub fn load_payloads(path: &Path) -> io::Result<Vec<BTreeMap<String, FieldValue>>> {
    load_payloads_with_fields(path, None)
}

/// Load payloads with optional column projection.
/// When `fields` is `Some`, only the listed fields are loaded (Arrow IPC only;
/// JSONL fallback loads all fields and filters in-memory).
pub fn load_payloads_with_fields(
    path: &Path,
    fields: Option<&[String]>,
) -> io::Result<Vec<BTreeMap<String, FieldValue>>> {
    let arrow_path = path.with_extension("arrow");
    if arrow_path.exists() {
        return super::arrow_io::load_payloads_arrow_with_projection(&arrow_path, fields);
    }
    // JSONL fallback: load all, then project if needed.
    let payloads = load_payloads_jsonl(path)?;
    if let Some(field_names) = fields {
        let keep: std::collections::BTreeSet<&str> =
            field_names.iter().map(String::as_str).collect();
        return Ok(payloads
            .into_iter()
            .map(|mut map| {
                map.retain(|k, _| keep.contains(k.as_str()));
                map
            })
            .collect());
    }
    Ok(payloads)
}

/// Load payloads from JSONL format (used by active segments and legacy data).
pub fn load_payloads_jsonl(path: &Path) -> io::Result<Vec<BTreeMap<String, FieldValue>>> {
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

/// Ensure the payloads JSONL file has exactly `expected_rows` rows by padding
/// missing rows with empty maps.
pub fn ensure_payload_rows(path: &Path, expected_rows: usize) -> io::Result<()> {
    match count_jsonl_lines(path) {
        Ok(count) => {
            if count > expected_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload row count exceeds record row count",
                ));
            }
            if count < expected_rows {
                let missing = vec![BTreeMap::new(); expected_rows - count];
                let _ = append_payloads(path, &missing)?;
            }
            Ok(())
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            if expected_rows > 0 {
                let _ = append_payloads(path, &vec![BTreeMap::new(); expected_rows])?;
            }
            Ok(())
        }
        Err(err) => Err(err),
    }
}

/// Count non-empty lines in a JSONL file without parsing the content.
pub fn count_jsonl_lines(path: &Path) -> io::Result<usize> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    Ok(reader
        .lines()
        .filter(|l| l.as_ref().map_or(false, |s| !s.is_empty()))
        .count())
}
