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

/// Remove a field from the schema. Returns NotFound if field doesn't exist.
pub fn remove_field_from_schema(
    fields: &mut Vec<ScalarFieldSchema>,
    field_name: &str,
) -> io::Result<()> {
    let original_len = fields.len();
    fields.retain(|f| f.name != field_name);
    if fields.len() == original_len {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("field not found: {field_name}"),
        ));
    }
    Ok(())
}

/// Rename a field in the schema. Returns NotFound if old_name doesn't exist,
/// AlreadyExists if new_name already exists.
pub fn rename_field_in_schema(
    fields: &mut Vec<ScalarFieldSchema>,
    old_name: &str,
    new_name: &str,
) -> io::Result<()> {
    if fields.iter().any(|f| f.name == new_name) {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("field already exists: {new_name}"),
        ));
    }
    let field = fields
        .iter_mut()
        .find(|f| f.name == old_name)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("field not found: {old_name}"),
            )
        })?;
    field.name = new_name.to_string();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::FieldType;

    #[test]
    fn add_field_appends_to_empty_schema() {
        let mut fields = Vec::new();
        add_field_to_schema(
            &mut fields,
            ScalarFieldSchema::new("name", FieldType::String),
        )
        .expect("add field should succeed");
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "name");
    }

    #[test]
    fn add_field_appends_to_existing_schema() {
        let mut fields = vec![ScalarFieldSchema::new("id", FieldType::Int64)];
        add_field_to_schema(
            &mut fields,
            ScalarFieldSchema::new("score", FieldType::Float64),
        )
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

    #[test]
    fn remove_field_deletes_existing_field() {
        let mut fields = vec![
            ScalarFieldSchema::new("id", FieldType::Int64),
            ScalarFieldSchema::new("score", FieldType::Float64),
        ];
        remove_field_from_schema(&mut fields, "score").expect("remove should succeed");
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "id");
    }

    #[test]
    fn remove_field_returns_not_found_for_missing_field() {
        let mut fields = vec![ScalarFieldSchema::new("id", FieldType::Int64)];
        let err = remove_field_from_schema(&mut fields, "missing").expect_err("should fail");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert_eq!(fields.len(), 1, "field list should be unchanged");
    }

    #[test]
    fn remove_field_on_empty_schema_returns_not_found() {
        let mut fields: Vec<ScalarFieldSchema> = Vec::new();
        let err = remove_field_from_schema(&mut fields, "anything").expect_err("should fail");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn rename_field_changes_name_in_place() {
        let mut fields = vec![ScalarFieldSchema::new("old", FieldType::String)];
        rename_field_in_schema(&mut fields, "old", "new").expect("rename should succeed");
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "new");
        assert_eq!(fields[0].data_type, FieldType::String);
    }

    #[test]
    fn rename_field_returns_not_found_for_missing_old_name() {
        let mut fields = vec![ScalarFieldSchema::new("id", FieldType::Int64)];
        let err = rename_field_in_schema(&mut fields, "missing", "new").expect_err("should fail");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert_eq!(fields[0].name, "id", "field name should be unchanged");
    }

    #[test]
    fn rename_field_returns_already_exists_for_conflicting_new_name() {
        let mut fields = vec![
            ScalarFieldSchema::new("a", FieldType::String),
            ScalarFieldSchema::new("b", FieldType::Int64),
        ];
        let err = rename_field_in_schema(&mut fields, "a", "b").expect_err("should fail");
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(fields[0].name, "a", "field name should be unchanged");
    }

    #[test]
    fn rename_field_noop_when_old_equals_new() {
        let mut fields = vec![ScalarFieldSchema::new("x", FieldType::Float64)];
        let err = rename_field_in_schema(&mut fields, "x", "x").expect_err("should fail");
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
    }
}
