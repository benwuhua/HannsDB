use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use arrow::ipc::writer::FileWriter;
use parquet::arrow::ArrowWriter;

use super::descriptor::{ForwardFileFormat, ForwardStoreArtifact, ForwardStoreDescriptor};
use super::mem_store::MemForwardStore;

#[derive(Debug, Clone)]
pub struct ChunkedFileWriter {
    base_dir: PathBuf,
}

impl ChunkedFileWriter {
    pub fn new(base_dir: impl AsRef<Path>) -> Self {
        Self {
            base_dir: base_dir.as_ref().to_path_buf(),
        }
    }

    pub fn write(
        &self,
        stem: &str,
        store: &MemForwardStore,
        formats: &[ForwardFileFormat],
    ) -> io::Result<ForwardStoreDescriptor> {
        fs::create_dir_all(&self.base_dir)?;
        let batch = store.schema().to_record_batch(store.rows(), None)?;
        let mut artifacts = Vec::with_capacity(formats.len());

        for &format in formats {
            let path = self
                .base_dir
                .join(format!("{}.{}", stem, format.extension()));
            match format {
                ForwardFileFormat::ArrowIpc => write_arrow_ipc(&path, &batch)?,
                ForwardFileFormat::Parquet => write_parquet(&path, &batch)?,
            }
            artifacts.push(ForwardStoreArtifact { format, path });
        }

        Ok(ForwardStoreDescriptor {
            schema: store.schema().clone(),
            row_count: store.row_count(),
            artifacts,
        })
    }
}

fn write_arrow_ipc(path: &Path, batch: &arrow::record_batch::RecordBatch) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = FileWriter::try_new(file, batch.schema().as_ref()).map_err(arrow_to_io)?;
    writer.write(batch).map_err(arrow_to_io)?;
    writer.finish().map_err(arrow_to_io)?;
    Ok(())
}

fn write_parquet(path: &Path, batch: &arrow::record_batch::RecordBatch) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).map_err(parquet_to_io)?;
    writer.write(batch).map_err(parquet_to_io)?;
    writer.close().map_err(parquet_to_io)?;
    Ok(())
}

fn arrow_to_io(err: arrow::error::ArrowError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}

fn parquet_to_io(err: parquet::errors::ParquetError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}
