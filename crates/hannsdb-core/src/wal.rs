use std::fs::OpenOptions;
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::document::{
    CollectionSchema, Document, DocumentUpdate, FieldValue, ScalarFieldSchema, VectorFieldSchema,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AddColumnBackfill {
    Constant { value: Option<FieldValue> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlterColumnMigration {
    Int32ToInt64,
    UInt32ToUInt64,
    FloatToFloat64,
    RenameAndInt32ToInt64,
    RenameAndUInt32ToUInt64,
    RenameAndFloatToFloat64,
}

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
    CompactCollection {
        collection_name: String,
        compacted_segment_id: String,
    },
    UpdateDocuments {
        collection: String,
        updates: Vec<DocumentUpdate>,
    },
    AddColumn {
        collection: String,
        field: ScalarFieldSchema,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        backfill: Option<AddColumnBackfill>,
    },
    DropColumn {
        collection: String,
        field_name: String,
    },
    AlterColumn {
        collection: String,
        old_name: String,
        new_name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        field: Option<ScalarFieldSchema>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        migration: Option<AlterColumnMigration>,
    },
    AddVectorField {
        collection: String,
        field: VectorFieldSchema,
    },
    DropVectorField {
        collection: String,
        field_name: String,
    },
}

pub fn append_wal_record(path: &Path, record: &WalRecord) -> io::Result<()> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = BufWriter::new(file);
    let encoded = serde_json::to_string(record).map_err(json_to_io_error)?;
    writer.write_all(encoded.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    // fsync to ensure the WAL entry survives power loss / kernel panic.
    writer.get_ref().sync_all()?;
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
            Err(err) => {
                // A corrupted mid-file line can happen from bad sectors or partial page writes.
                // Rather than making the database unopenable, log a warning and stop reading.
                // We keep all records parsed so far and discard the rest of the WAL.
                eprintln!(
                    "warning: corrupted WAL line {} ({}), stopping replay at that point: {}",
                    idx + 1,
                    line.len().min(80),
                    err
                );
                break;
            }
        }
    }
    Ok(records)
}

/// Truncate the WAL file to zero bytes, removing all entries.
///
/// This is a checkpoint operation: after all data has been flushed/optimized
/// to segment files, the WAL entries are no longer needed and can be discarded.
/// The WAL file itself is kept (as an empty file) so that subsequent appends
/// work without needing to check for file existence.
pub fn truncate_wal(path: &Path) -> io::Result<()> {
    use std::io::Seek;
    let mut file = OpenOptions::new().write(true).open(path)?;
    file.set_len(0)?;
    file.seek(io::SeekFrom::Start(0))?;
    file.sync_all()?;
    Ok(())
}

fn json_to_io_error(err: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}
