use std::io;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};

use crate::document::{CollectionSchema, FieldType, ScalarFieldSchema, VectorFieldSchema};

pub(crate) const LANCE_ID_COLUMN: &str = "id";
pub(crate) const LANCE_SPARSE_INDICES_FIELD: &str = "indices";
pub(crate) const LANCE_SPARSE_VALUES_FIELD: &str = "values";

pub(crate) fn arrow_schema_for_lance(schema: &CollectionSchema) -> io::Result<Arc<Schema>> {
    let mut fields = Vec::with_capacity(1 + schema.fields.len() + schema.vectors.len());
    fields.push(Field::new(LANCE_ID_COLUMN, DataType::Int64, false));

    for scalar in &schema.fields {
        fields.push(Field::new(
            scalar.name.clone(),
            scalar_data_type_for_lance(&scalar.data_type, scalar.array)?,
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
                if vector.dimension != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "sparse vector dimension must be zero for Lance storage: {}",
                            vector.name
                        ),
                    ));
                }
                sparse_vector_data_type_for_lance()
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

pub(crate) fn collection_schema_from_lance_arrow(arrow: &Schema) -> io::Result<CollectionSchema> {
    let id = arrow.field_with_name(LANCE_ID_COLUMN).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "Lance dataset schema is missing HannsDB id column",
        )
    })?;
    if id.data_type() != &DataType::Int64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Lance dataset id column is not Int64",
        ));
    }

    let mut fields = Vec::new();
    let mut vectors = Vec::new();
    for field in arrow.fields() {
        if field.name() == LANCE_ID_COLUMN {
            continue;
        }
        if sparse_vector_shape(field.data_type())? {
            vectors.push(VectorFieldSchema {
                name: field.name().clone(),
                data_type: FieldType::VectorSparse,
                dimension: 0,
                index_param: None,
                bm25_params: None,
            });
            continue;
        }
        if let Some(dimension) = vector_dimension(field.data_type())? {
            vectors.push(VectorFieldSchema::new(field.name().clone(), dimension));
            continue;
        }
        let (field_type, is_array) = scalar_field_shape(field.data_type())?;
        fields.push(
            ScalarFieldSchema::new(field.name().clone(), field_type)
                .with_flags(field.is_nullable(), is_array),
        );
    }

    let primary_vector = vectors
        .iter()
        .find(|vector| vector.name == "dense")
        .or_else(|| vectors.first())
        .map(|vector| vector.name.clone())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Lance dataset schema does not contain a supported dense vector column",
            )
        })?;

    Ok(CollectionSchema {
        primary_vector,
        fields,
        vectors,
    })
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

pub(crate) fn scalar_data_type_for_lance(
    data_type: &FieldType,
    is_array: bool,
) -> io::Result<DataType> {
    let item_type = scalar_data_type(data_type)?;
    if is_array {
        Ok(DataType::List(Arc::new(Field::new(
            "item", item_type, true,
        ))))
    } else {
        Ok(item_type)
    }
}

pub(crate) fn sparse_vector_data_type_for_lance() -> DataType {
    DataType::Struct(
        vec![
            Field::new(
                LANCE_SPARSE_INDICES_FIELD,
                DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
                true,
            ),
            Field::new(
                LANCE_SPARSE_VALUES_FIELD,
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            ),
        ]
        .into(),
    )
}

fn scalar_field_type(data_type: &DataType) -> io::Result<FieldType> {
    match data_type {
        DataType::Utf8 => Ok(FieldType::String),
        DataType::Int64 => Ok(FieldType::Int64),
        DataType::Int32 => Ok(FieldType::Int32),
        DataType::UInt32 => Ok(FieldType::UInt32),
        DataType::UInt64 => Ok(FieldType::UInt64),
        DataType::Float32 => Ok(FieldType::Float),
        DataType::Float64 => Ok(FieldType::Float64),
        DataType::Boolean => Ok(FieldType::Bool),
        unsupported => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported Lance scalar column type: {unsupported:?}"),
        )),
    }
}

fn scalar_field_shape(data_type: &DataType) -> io::Result<(FieldType, bool)> {
    match data_type {
        DataType::List(item) => {
            scalar_field_type(item.data_type()).map(|field_type| (field_type, true))
        }
        _ => scalar_field_type(data_type).map(|field_type| (field_type, false)),
    }
}

fn vector_dimension(data_type: &DataType) -> io::Result<Option<usize>> {
    let DataType::FixedSizeList(item, dimension) = data_type else {
        return Ok(None);
    };
    if item.data_type() != &DataType::Float32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported Lance vector value type: {:?}",
                item.data_type()
            ),
        ));
    }
    usize::try_from(*dimension).map(Some).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "negative Lance vector dimension",
        )
    })
}

fn sparse_vector_shape(data_type: &DataType) -> io::Result<bool> {
    let DataType::Struct(fields) = data_type else {
        return Ok(false);
    };
    if fields.len() != 2 {
        return Ok(false);
    }
    let indices = fields
        .iter()
        .find(|field| field.name() == LANCE_SPARSE_INDICES_FIELD);
    let values = fields
        .iter()
        .find(|field| field.name() == LANCE_SPARSE_VALUES_FIELD);
    let (Some(indices), Some(values)) = (indices, values) else {
        return Ok(false);
    };
    if !matches!(
        indices.data_type(),
        DataType::List(item) if item.data_type() == &DataType::UInt32
    ) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported Lance sparse vector indices shape",
        ));
    }
    if !matches!(
        values.data_type(),
        DataType::List(item) if item.data_type() == &DataType::Float32
    ) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported Lance sparse vector values shape",
        ));
    }
    Ok(true)
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
    fn lance_schema_maps_sparse_vectors_to_arrow_struct_lists() {
        let schema = CollectionSchema {
            primary_vector: "dense".to_string(),
            fields: Vec::new(),
            vectors: vec![
                VectorFieldSchema::new("dense", 3),
                VectorFieldSchema {
                    name: "sparse_title".to_string(),
                    data_type: FieldType::VectorSparse,
                    dimension: 0,
                    index_param: None,
                    bm25_params: None,
                },
            ],
        };

        let arrow = arrow_schema_for_lance(&schema).expect("sparse vector schema");
        let sparse = arrow.field_with_name("sparse_title").unwrap();
        let DataType::Struct(fields) = sparse.data_type() else {
            panic!("expected sparse vector to be encoded as struct/list");
        };

        assert_eq!(fields[0].name(), "indices");
        assert_eq!(
            fields[0].data_type(),
            &DataType::List(Arc::new(Field::new("item", DataType::UInt32, true)))
        );
        assert_eq!(fields[1].name(), "values");
        assert_eq!(
            fields[1].data_type(),
            &DataType::List(Arc::new(Field::new("item", DataType::Float32, true)))
        );
    }

    #[test]
    fn lance_schema_can_be_inferred_from_arrow_schema() {
        let arrow = Schema::new(vec![
            Field::new(LANCE_ID_COLUMN, DataType::Int64, false),
            Field::new("title", DataType::Utf8, true),
            Field::new("year", DataType::Int32, false),
            Field::new(
                "dense",
                DataType::FixedSizeList(
                    std::sync::Arc::new(Field::new("item", DataType::Float32, false)),
                    2,
                ),
                false,
            ),
        ]);

        let schema = collection_schema_from_lance_arrow(&arrow).expect("infer schema");

        assert_eq!(schema.primary_vector, "dense");
        assert_eq!(schema.fields.len(), 2);
        assert_eq!(schema.fields[0].name, "title");
        assert_eq!(schema.fields[0].data_type, FieldType::String);
        assert!(schema.fields[0].nullable);
        assert_eq!(schema.fields[1].data_type, FieldType::Int32);
        assert_eq!(schema.vectors.len(), 1);
        assert_eq!(schema.vectors[0].name, "dense");
        assert_eq!(schema.vectors[0].dimension, 2);
    }

    #[test]
    fn lance_schema_infers_sparse_vector_fields_from_arrow_struct_lists() {
        let arrow = Schema::new(vec![
            Field::new(LANCE_ID_COLUMN, DataType::Int64, false),
            Field::new(
                "dense",
                DataType::FixedSizeList(
                    std::sync::Arc::new(Field::new("item", DataType::Float32, false)),
                    2,
                ),
                false,
            ),
            Field::new(
                "sparse_title",
                DataType::Struct(
                    vec![
                        Field::new(
                            "indices",
                            DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
                            true,
                        ),
                        Field::new(
                            "values",
                            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                            true,
                        ),
                    ]
                    .into(),
                ),
                false,
            ),
        ]);

        let schema = collection_schema_from_lance_arrow(&arrow).expect("infer schema");

        assert_eq!(schema.primary_vector, "dense");
        assert_eq!(schema.vectors.len(), 2);
        assert_eq!(schema.vectors[0].name, "dense");
        assert_eq!(schema.vectors[0].data_type, FieldType::VectorFp32);
        assert_eq!(schema.vectors[1].name, "sparse_title");
        assert_eq!(schema.vectors[1].data_type, FieldType::VectorSparse);
        assert_eq!(schema.vectors[1].dimension, 0);
    }
}
