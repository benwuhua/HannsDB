use std::io;

use crate::document::ScalarFieldSchema;

/// Validate and append a new field to a collection's schema.
///
/// Returns an error if a field with the same name already exists.
pub fn add_field_to_schema(
    fields: &mut Vec<ScalarFieldSchema>,
    new_field: ScalarFieldSchema,
) -> io::Result<()> {
    if fields.iter().any(|f| f.name == new_field.name) {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("field already exists: {}", new_field.name),
        ));
    }
    fields.push(new_field);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::FieldType;

    #[test]
    fn add_field_appends_to_empty_schema() {
        let mut fields = Vec::new();
        add_field_to_schema(&mut fields, ScalarFieldSchema::new("name", FieldType::String))
            .expect("add field should succeed");
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "name");
    }

    #[test]
    fn add_field_appends_to_existing_schema() {
        let mut fields = vec![ScalarFieldSchema::new("id", FieldType::Int64)];
        add_field_to_schema(&mut fields, ScalarFieldSchema::new("score", FieldType::Float64))
            .expect("add field should succeed");
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[1].name, "score");
    }

    #[test]
    fn add_field_rejects_duplicate_name() {
        let mut fields = vec![ScalarFieldSchema::new("id", FieldType::Int64)];
        let result =
            add_field_to_schema(&mut fields, ScalarFieldSchema::new("id", FieldType::String));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(fields.len(), 1, "original field list should be unchanged");
    }

    #[test]
    fn add_field_allows_same_name_different_type_after_removal() {
        let mut fields = vec![ScalarFieldSchema::new("x", FieldType::Int64)];
        fields.clear();
        add_field_to_schema(&mut fields, ScalarFieldSchema::new("x", FieldType::Float64))
            .expect("add field after clearing should succeed");
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].data_type, FieldType::Float64);
    }
}
