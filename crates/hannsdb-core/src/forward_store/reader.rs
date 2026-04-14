use std::collections::BTreeMap;
use std::fs::File;
use std::io;

use arrow::ipc::reader::FileReader;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::descriptor::{ForwardFileFormat, ForwardStoreDescriptor};
use super::schema::{project_row, ForwardRow};

#[derive(Debug, Clone)]
pub struct ForwardStoreReader {
    rows: Vec<ForwardRow>,
}

impl ForwardStoreReader {
    pub fn open(
        descriptor: &ForwardStoreDescriptor,
        format: ForwardFileFormat,
    ) -> io::Result<Self> {
        let artifact = descriptor.artifact(format).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("forward-store descriptor missing {}", format.label()),
            )
        })?;

        let rows = match format {
            ForwardFileFormat::ArrowIpc => {
                let file = File::open(&artifact.path)?;
                let reader = FileReader::try_new(file, None).map_err(arrow_to_io)?;
                read_batches(reader, descriptor)?
            }
            ForwardFileFormat::Parquet => {
                let file = File::open(&artifact.path)?;
                let reader = ParquetRecordBatchReaderBuilder::try_new(file)
                    .map_err(parquet_to_io)?
                    .with_batch_size(1024)
                    .build()
                    .map_err(parquet_to_io)?;
                read_batches(reader, descriptor)?
            }
        };

        Ok(Self { rows })
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub fn fetch_rows(
        &self,
        row_positions: &[usize],
        columns: Option<&[&str]>,
    ) -> io::Result<Vec<ForwardRow>> {
        let mut out = Vec::with_capacity(row_positions.len());
        for &position in row_positions {
            let row = self.rows.get(position).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("row position {} out of bounds", position),
                )
            })?;
            out.push(project_row(row, columns));
        }
        Ok(out)
    }

    pub fn scan_columns(&self, columns: Option<&[&str]>) -> io::Result<Vec<ForwardRow>> {
        Ok(self
            .rows
            .iter()
            .map(|row| project_row(row, columns))
            .collect())
    }

    pub fn latest_live_rows(&self) -> Vec<ForwardRow> {
        let mut latest = BTreeMap::<u64, &ForwardRow>::new();
        for row in &self.rows {
            match latest.get(&row.internal_id) {
                Some(existing) if existing.op_seq >= row.op_seq => {}
                _ => {
                    latest.insert(row.internal_id, row);
                }
            }
        }

        latest
            .into_values()
            .filter(|row| !row.is_deleted)
            .cloned()
            .collect()
    }
}

fn read_batches<I, E>(reader: I, descriptor: &ForwardStoreDescriptor) -> io::Result<Vec<ForwardRow>>
where
    I: IntoIterator<Item = Result<RecordBatch, E>>,
    E: IntoReaderError,
{
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|err| err.into_reader_error())?;
        rows.extend(descriptor.schema.rows_from_batch(&batch)?);
    }
    Ok(rows)
}

trait IntoReaderError {
    fn into_reader_error(self) -> io::Error;
}

impl IntoReaderError for arrow::error::ArrowError {
    fn into_reader_error(self) -> io::Error {
        arrow_to_io(self)
    }
}

impl IntoReaderError for parquet::errors::ParquetError {
    fn into_reader_error(self) -> io::Error {
        parquet_to_io(self)
    }
}

fn arrow_to_io(err: arrow::error::ArrowError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}

fn parquet_to_io(err: parquet::errors::ParquetError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}
