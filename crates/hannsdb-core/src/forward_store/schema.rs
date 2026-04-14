use std::io;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use serde::{Deserialize, Serialize};

use crate::document::{CollectionSchema, FieldType, FieldValue, VectorFieldSchema};

pub const FORWARD_STORE_INTERNAL_ID_COLUMN: &str = "internal_id";
pub const FORWARD_STORE_OP_SEQ_COLUMN: &str = "op_seq";
pub const FORWARD_STORE_IS_DELETED_COLUMN: &str = "is_deleted";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForwardSystemColumnKind {
    InternalId,
    OpSeq,
    IsDeleted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ForwardColumnKind {
    System { system: ForwardSystemColumnKind },
    Scalar,
    PrimaryVector,
    SecondaryVector,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForwardColumnSchema {
    pub name: String,
    pub data_type: FieldType,
    #[serde(default)]
    pub nullable: bool,
    #[serde(default)]
    pub array: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimension: Option<usize>,
    pub kind: ForwardColumnKind,
}

impl ForwardColumnSchema {
    pub fn new_scalar(
        name: impl Into<String>,
        data_type: FieldType,
        nullable: bool,
        array: bool,
    ) -> io::Result<Self> {
        reject_vector_data_type(&data_type, "scalar forward-store column")?;
        Ok(Self {
            name: name.into(),
            data_type,
            nullable,
            array,
            dimension: None,
            kind: ForwardColumnKind::Scalar,
        })
    }

    fn new_system(
        name: impl Into<String>,
        system: ForwardSystemColumnKind,
        data_type: FieldType,
    ) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable: false,
            array: false,
            dimension: None,
            kind: ForwardColumnKind::System { system },
        }
    }

    fn new_vector(
        name: impl Into<String>,
        data_type: FieldType,
        dimension: usize,
        nullable: bool,
        kind: ForwardColumnKind,
    ) -> io::Result<Self> {
        if dimension == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "forward-store vector columns must have dimension > 0",
            ));
        }
        if data_type == FieldType::VectorSparse {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "forward-store core does not yet support sparse vector columns",
            ));
        }
        if !matches!(kind, ForwardColumnKind::PrimaryVector | ForwardColumnKind::SecondaryVector) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "forward-store vector column kind must be primary_vector or secondary_vector",
            ));
        }
        Ok(Self {
            name: name.into(),
            data_type,
            nullable,
            array: false,
            dimension: Some(dimension),
            kind,
        })
    }

    pub fn arrow_field(&self) -> io::Result<Field> {
        Ok(Field::new(
            &self.name,
            self.arrow_data_type()?,
            self.nullable,
        ))
    }

    pub fn arrow_data_type(&self) -> io::Result<DataType> {
        match self.kind {
            ForwardColumnKind::System { .. } | ForwardColumnKind::Scalar => {
                scalar_arrow_data_type(&self.data_type, self.array)
            }
            ForwardColumnKind::PrimaryVector | ForwardColumnKind::SecondaryVector => {
                vector_arrow_data_type(&self.data_type, self.dimension)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForwardStoreSchema {
    pub primary_vector: String,
    pub system_columns: Vec<ForwardColumnSchema>,
    pub scalar_fields: Vec<ForwardColumnSchema>,
    pub primary_vector_column: ForwardColumnSchema,
    pub secondary_vector_fields: Vec<ForwardColumnSchema>,
}

impl ForwardStoreSchema {
    pub fn from_collection_schema(collection: &CollectionSchema) -> io::Result<Self> {
        let primary_vector = collection.primary_vector().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "collection schema is missing primary vector '{}'",
                    collection.primary_vector_name()
                ),
            )
        })?;

        let scalar_fields = collection
            .fields
            .iter()
            .map(|field| {
                ForwardColumnSchema::new_scalar(
                    field.name.clone(),
                    field.data_type.clone(),
                    field.nullable,
                    field.array,
                )
            })
            .collect::<io::Result<Vec<_>>>()?;

        let primary_vector_column = forward_vector_column(
            primary_vector,
            false,
            ForwardColumnKind::PrimaryVector,
        )?;

        let secondary_vector_fields = collection
            .vectors
            .iter()
            .filter(|vector| vector.name != collection.primary_vector)
            .map(|vector| {
                forward_vector_column(vector, true, ForwardColumnKind::SecondaryVector)
            })
            .collect::<io::Result<Vec<_>>>()?;

        Ok(Self {
            primary_vector: collection.primary_vector.clone(),
            system_columns: vec![
                ForwardColumnSchema::new_system(
                    FORWARD_STORE_INTERNAL_ID_COLUMN,
                    ForwardSystemColumnKind::InternalId,
                    FieldType::Int64,
                ),
                ForwardColumnSchema::new_system(
                    FORWARD_STORE_OP_SEQ_COLUMN,
                    ForwardSystemColumnKind::OpSeq,
                    FieldType::UInt64,
                ),
                ForwardColumnSchema::new_system(
                    FORWARD_STORE_IS_DELETED_COLUMN,
                    ForwardSystemColumnKind::IsDeleted,
                    FieldType::Bool,
                ),
            ],
            scalar_fields,
            primary_vector_column,
            secondary_vector_fields,
        })
    }

    pub fn columns(&self) -> Vec<&ForwardColumnSchema> {
        self.system_columns
            .iter()
            .chain(self.scalar_fields.iter())
            .chain(std::iter::once(&self.primary_vector_column))
            .chain(self.secondary_vector_fields.iter())
            .collect()
    }

    pub fn scalar_field(&self, name: &str) -> Option<&ForwardColumnSchema> {
        self.scalar_fields.iter().find(|field| field.name == name)
    }

    pub fn vector_field(&self, name: &str) -> Option<&ForwardColumnSchema> {
        if self.primary_vector_column.name == name {
            return Some(&self.primary_vector_column);
        }
        self.secondary_vector_fields
            .iter()
            .find(|field| field.name == name)
    }

    pub fn arrow_schema(&self) -> io::Result<ArrowSchema> {
        let fields = self
            .columns()
            .into_iter()
            .map(ForwardColumnSchema::arrow_field)
            .collect::<io::Result<Vec<_>>>()?;
        Ok(ArrowSchema::new(fields))
    }

    pub fn validate_scalar_value(&self, name: &str, value: &FieldValue) -> io::Result<()> {
        let column = self.scalar_field(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "forward-store row field '{}' is not declared in the collection schema",
                    name
                ),
            )
        })?;

        if field_value_matches_schema(value, &column.data_type, column.array) {
            return Ok(());
        }

        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "forward-store row field '{}' does not match declared type {:?}{}",
                name,
                column.data_type,
                if column.array { "[]" } else { "" }
            ),
        ))
    }

    pub fn vector_dimension(&self, name: &str) -> Option<usize> {
        self.vector_field(name).and_then(|column| column.dimension)
    }
}

pub(crate) fn field_value_matches_schema(
    value: &FieldValue,
    data_type: &FieldType,
    array: bool,
) -> bool {
    if array {
        let FieldValue::Array(items) = value else {
            return false;
        };
        return items
            .iter()
            .all(|item| field_value_matches_schema(item, data_type, false));
    }

    matches!(
        (value, data_type),
        (FieldValue::String(_), FieldType::String)
            | (FieldValue::Int64(_), FieldType::Int64)
            | (FieldValue::Int32(_), FieldType::Int32)
            | (FieldValue::UInt32(_), FieldType::UInt32)
            | (FieldValue::UInt64(_), FieldType::UInt64)
            | (FieldValue::Float(_), FieldType::Float)
            | (FieldValue::Float64(_), FieldType::Float64)
            | (FieldValue::Bool(_), FieldType::Bool)
    )
}

fn forward_vector_column(
    vector: &VectorFieldSchema,
    nullable: bool,
    kind: ForwardColumnKind,
) -> io::Result<ForwardColumnSchema> {
    ForwardColumnSchema::new_vector(
        vector.name.clone(),
        vector.data_type.clone(),
        vector.dimension,
        nullable,
        kind,
    )
}

fn reject_vector_data_type(data_type: &FieldType, subject: &str) -> io::Result<()> {
    if matches!(
        data_type,
        FieldType::VectorFp32 | FieldType::VectorFp16 | FieldType::VectorSparse
    ) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{subject} cannot use vector data type {:?}", data_type),
        ));
    }
    Ok(())
}

fn scalar_arrow_data_type(data_type: &FieldType, array: bool) -> io::Result<DataType> {
    let inner = match data_type {
        FieldType::String => DataType::Utf8,
        FieldType::Int64 => DataType::Int64,
        FieldType::Int32 => DataType::Int32,
        FieldType::UInt32 => DataType::UInt32,
        FieldType::UInt64 => DataType::UInt64,
        FieldType::Float => DataType::Float32,
        FieldType::Float64 => DataType::Float64,
        FieldType::Bool => DataType::Boolean,
        FieldType::VectorFp32 | FieldType::VectorFp16 | FieldType::VectorSparse => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "forward-store scalar columns cannot use vector data type {:?}",
                    data_type
                ),
            ))
        }
    };

    if array {
        Ok(DataType::List(Arc::new(Field::new("item", inner, true))))
    } else {
        Ok(inner)
    }
}

fn vector_arrow_data_type(data_type: &FieldType, dimension: Option<usize>) -> io::Result<DataType> {
    let dimension = dimension.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "forward-store vector columns require a dimension",
        )
    })?;

    match data_type {
        FieldType::VectorFp32 | FieldType::VectorFp16 => Ok(DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, false)),
            dimension as i32,
        )),
        FieldType::VectorSparse => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "forward-store core does not yet support sparse vectors",
        )),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "forward-store vector column must use a vector data type, got {:?}",
                data_type
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use crate::document::{FieldType, ScalarFieldSchema, VectorFieldSchema};

    use super::*;

    #[test]
    fn forward_store_schema_adds_system_columns_and_vectors() {
        let mut schema = CollectionSchema::new(
            "embedding",
            3,
            "l2",
            vec![
                ScalarFieldSchema::new("title", FieldType::String),
                ScalarFieldSchema::new("tags", FieldType::String).with_flags(true, true),
            ],
        );
        schema.vectors.push(VectorFieldSchema::new("image", 2));

        let forward = ForwardStoreSchema::from_collection_schema(&schema).expect("forward schema");
        let column_names = forward
            .columns()
            .into_iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            column_names,
            vec![
                FORWARD_STORE_INTERNAL_ID_COLUMN,
                FORWARD_STORE_OP_SEQ_COLUMN,
                FORWARD_STORE_IS_DELETED_COLUMN,
                "title",
                "tags",
                "embedding",
                "image",
            ]
        );
        assert_eq!(forward.primary_vector_column.dimension, Some(3));
        assert_eq!(forward.secondary_vector_fields[0].dimension, Some(2));
    }

    #[test]
    fn forward_store_schema_rejects_sparse_vector_columns() {
        let mut schema = CollectionSchema::new("embedding", 3, "l2", vec![]);
        let mut sparse = VectorFieldSchema::new("sparse_terms", 8);
        sparse.data_type = FieldType::VectorSparse;
        schema.vectors.push(sparse);

        let err = ForwardStoreSchema::from_collection_schema(&schema).expect_err("should reject");
        assert!(
            err.to_string().contains("does not yet support sparse vector columns"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn forward_store_schema_validates_declared_scalar_types() {
        let schema = ForwardStoreSchema::from_collection_schema(&CollectionSchema::new(
            "embedding",
            3,
            "l2",
            vec![ScalarFieldSchema::new("score", FieldType::Float)],
        ))
        .expect("forward schema");

        schema
            .validate_scalar_value("score", &FieldValue::Float(0.5))
            .expect("matching type");

        let err = schema
            .validate_scalar_value("score", &FieldValue::String("bad".to_string()))
            .expect_err("mismatched type");
        assert!(
            err.to_string().contains("does not match declared type"),
            "unexpected error: {err}"
        );
    }
}
