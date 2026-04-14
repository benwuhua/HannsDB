use std::io;
use std::path::Path;

pub use crate::wal::{append_wal_record, load_wal_records, truncate_wal, WalRecord};

pub(crate) fn load_wal_records_or_empty(path: &Path) -> io::Result<Vec<WalRecord>> {
    match load_wal_records(path) {
        Ok(records) => Ok(records),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(err),
    }
}
