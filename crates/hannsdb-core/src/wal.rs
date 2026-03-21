use std::fs::OpenOptions;
use std::io;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::document::{CollectionSchema, Document};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WalRecord {
    CreateCollection {
        collection: String,
        schema: CollectionSchema,
    },
    DropCollection {
        collection: String,
    },
    Insert {
        collection: String,
        ids: Vec<i64>,
        vectors: Vec<f32>,
    },
    InsertDocuments {
        collection: String,
        documents: Vec<Document>,
    },
    UpsertDocuments {
        collection: String,
        documents: Vec<Document>,
    },
    Delete {
        collection: String,
        ids: Vec<i64>,
    },
}

pub fn append_wal_record(path: &Path, record: &WalRecord) -> io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    let encoded = serde_json::to_string(record).map_err(json_to_io_error)?;
    file.write_all(encoded.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

pub fn load_wal_records(path: &Path) -> io::Result<Vec<WalRecord>> {
    let file = OpenOptions::new().read(true).open(path)?;
    let reader = BufReader::new(file);
    let non_empty_lines = reader
        .lines()
        .filter_map(|line| match line {
            Ok(line) => {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(Ok(trimmed))
                }
            }
            Err(err) => Some(Err(err)),
        })
        .collect::<io::Result<Vec<_>>>()?;

    let mut records = Vec::with_capacity(non_empty_lines.len());
    for (idx, line) in non_empty_lines.iter().enumerate() {
        match serde_json::from_str::<WalRecord>(line) {
            Ok(record) => records.push(record),
            Err(_) if idx + 1 == non_empty_lines.len() => {
                // A truncated final line can happen on crash before newline/fsync.
                // Skip only this tail line, and keep strict parsing for earlier lines.
                break;
            }
            Err(err) => return Err(json_to_io_error(err)),
        }
    }
    Ok(records)
}

fn json_to_io_error(err: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}
