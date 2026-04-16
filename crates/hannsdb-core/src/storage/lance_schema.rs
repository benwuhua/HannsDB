use std::io;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};

use crate::document::{CollectionSchema, FieldType};

pub(crate) const LANCE_ID_COLUMN: &str = "id";

pub(crate) fn arrow_schema_for_lance(schema: &CollectionSchema) -> io::Result<Arc<Schema>> {
    let mut fields = Vec::with_capacity(1 + schema.fields.len() + schema.vectors.len());
    fields.push(Field::new(LANCE_ID_COLUMN, DataType::Int64, false));

    for scalar in &schema.fields {
        if scalar.array {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "array scalar fields are not supported by Lance storage P0: {}",
                    scalar.name
                ),
            ));
        }
        fields.push(Field::new(
            scalar.name.clone(),
            scalar_data_type(&scalar.data_type)?,
            scalar.nullable,
        ));
    }

    for vector in &schema.vectors {
        let data_type = match vector.data_type {
            FieldType::VectorFp32 => {
                if vector.dimension == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("dense vector dimension must be nonzero: {}", vector.name),
                    ));
                }
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    i32::try_from(vector.dimension).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("vector dimension exceeds i32 range: {}", vector.name),
                        )
                    })?,
                )
            }
            FieldType::VectorFp16 => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "fp16 vectors are not supported by Lance storage P0: {}",
                        vector.name
                    ),
                ));
            }
            FieldType::VectorSparse => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "sparse vectors are not supported by Lance storage P0: {}",
                        vector.name
                    ),
                ));
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("non-vector field registered as vector: {}", vector.name),
                ));
            }
        };
        fields.push(Field::new(vector.name.clone(), data_type, false));
    }

    Ok(Arc::new(Schema::new(fields)))
}

fn scalar_data_type(data_type: &FieldType) -> io::Result<DataType> {
    match data_type {
        FieldType::String => Ok(DataType::Utf8),
        FieldType::Int64 => Ok(DataType::Int64),
        FieldType::Int32 => Ok(DataType::Int32),
        FieldType::UInt32 => Ok(DataType::UInt32),
        FieldType::UInt64 => Ok(DataType::UInt64),
        FieldType::Float => Ok(DataType::Float32),
        FieldType::Float64 => Ok(DataType::Float64),
        FieldType::Bool => Ok(DataType::Boolean),
        FieldType::VectorFp32 | FieldType::VectorFp16 | FieldType::VectorSparse => {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "vector field type cannot be used as a Lance scalar column",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{CollectionSchema, FieldType, ScalarFieldSchema, VectorFieldSchema};

    #[test]
    fn lance_schema_maps_supported_scalar_and_vector_fields() {
        let schema = CollectionSchema {
            primary_vector: "dense".to_string(),
            fields: vec![
                ScalarFieldSchema::new("title", FieldType::String),
                ScalarFieldSchema::new("year", FieldType::Int64),
                ScalarFieldSchema::new("score", FieldType::Float64),
                ScalarFieldSchema::new("active", FieldType::Bool),
            ],
            vectors: vec![VectorFieldSchema::new("dense", 3)],
        };

        let arrow = arrow_schema_for_lance(&schema).expect("convert schema");
        assert_eq!(arrow.field(0).name(), "id");
        assert_eq!(arrow.field(0).data_type(), &arrow_schema::DataType::Int64);
        assert_eq!(
            arrow.field_with_name("title").unwrap().data_type(),
            &arrow_schema::DataType::Utf8
        );
        assert_eq!(
            arrow.field_with_name("year").unwrap().data_type(),
            &arrow_schema::DataType::Int64
        );
        assert_eq!(
            arrow.field_with_name("score").unwrap().data_type(),
            &arrow_schema::DataType::Float64
        );
        assert_eq!(
            arrow.field_with_name("active").unwrap().data_type(),
            &arrow_schema::DataType::Boolean
        );
        assert_eq!(
            arrow.field_with_name("dense").unwrap().data_type(),
            &arrow_schema::DataType::FixedSizeList(
                std::sync::Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::Float32,
                    false,
                )),
                3,
            )
        );
    }

    #[test]
    fn lance_schema_rejects_sparse_vectors_in_p0() {
        let schema = CollectionSchema {
            primary_vector: "dense".to_string(),
            fields: Vec::new(),
            vectors: vec![VectorFieldSchema {
                name: "dense".to_string(),
                data_type: FieldType::VectorSparse,
                dimension: 0,
                index_param: None,
                bm25_params: None,
            }],
        };

        let err = arrow_schema_for_lance(&schema).expect_err("sparse vector rejected");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("sparse vectors are not supported"));
    }
}
