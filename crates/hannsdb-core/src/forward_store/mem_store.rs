use std::io;

use arrow::record_batch::RecordBatch;

use crate::document::CollectionSchema;

use super::schema::{estimate_row_bytes, ForwardRow, ForwardSchema};

#[derive(Debug, Clone)]
pub struct MemForwardStore {
    schema: ForwardSchema,
    rows: Vec<ForwardRow>,
    estimated_bytes: usize,
}

impl MemForwardStore {
    pub fn new(collection_schema: CollectionSchema) -> Self {
        Self {
            schema: ForwardSchema::new(collection_schema),
            rows: Vec::new(),
            estimated_bytes: 0,
        }
    }

    pub fn append(&mut self, row: ForwardRow) -> io::Result<()> {
        self.schema.validate_row(&row)?;
        self.estimated_bytes += estimate_row_bytes(&row);
        self.rows.push(row);
        Ok(())
    }

    pub fn append_rows<I>(&mut self, rows: I) -> io::Result<()>
    where
        I: IntoIterator<Item = ForwardRow>,
    {
        for row in rows {
            self.append(row)?;
        }
        Ok(())
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub fn estimated_bytes(&self) -> usize {
        self.estimated_bytes
    }

    pub fn is_full(&self, max_bytes: usize) -> bool {
        self.estimated_bytes >= max_bytes
    }

    pub fn schema(&self) -> &ForwardSchema {
        &self.schema
    }

    pub fn rows(&self) -> &[ForwardRow] {
        &self.rows
    }

    pub fn record_batch(&self) -> io::Result<RecordBatch> {
        self.schema.to_record_batch(&self.rows, None)
    }
}
