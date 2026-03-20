use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::io;

use crate::document::FieldValue;

#[derive(Debug, Clone, PartialEq)]
pub struct FilterExpr {
    clauses: Vec<FilterClause>,
}

impl FilterExpr {
    pub fn matches(&self, fields: &BTreeMap<String, FieldValue>) -> bool {
        self.clauses.iter().all(|clause| clause.matches(fields))
    }
}

#[derive(Debug, Clone, PartialEq)]
struct FilterClause {
    field: String,
    op: ComparisonOp,
    value: FieldValue,
}

impl FilterClause {
    fn matches(&self, fields: &BTreeMap<String, FieldValue>) -> bool {
        let Some(actual) = fields.get(&self.field) else {
            return false;
        };

        match self.op {
            ComparisonOp::Eq => values_equal(actual, &self.value),
            ComparisonOp::Ne => !values_equal(actual, &self.value),
            ComparisonOp::Gt => compare_values(actual, &self.value) == Some(Ordering::Greater),
            ComparisonOp::Gte => matches!(
                compare_values(actual, &self.value),
                Some(Ordering::Greater | Ordering::Equal)
            ),
            ComparisonOp::Lt => compare_values(actual, &self.value) == Some(Ordering::Less),
            ComparisonOp::Lte => matches!(
                compare_values(actual, &self.value),
                Some(Ordering::Less | Ordering::Equal)
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComparisonOp {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
}

pub fn parse_filter(input: &str) -> io::Result<FilterExpr> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "filter expression cannot be empty",
        ));
    }

    let clauses = trimmed
        .split(" and ")
        .map(parse_clause)
        .collect::<io::Result<Vec<_>>>()?;
    Ok(FilterExpr { clauses })
}

fn parse_clause(input: &str) -> io::Result<FilterClause> {
    let trimmed = input.trim();
    for (token, op) in [
        (">=", ComparisonOp::Gte),
        ("<=", ComparisonOp::Lte),
        ("==", ComparisonOp::Eq),
        ("!=", ComparisonOp::Ne),
        (">", ComparisonOp::Gt),
        ("<", ComparisonOp::Lt),
    ] {
        if let Some((field, value)) = trimmed.split_once(token) {
            let field = field.trim();
            let value = value.trim();
            if field.is_empty() || value.is_empty() {
                break;
            }
            return Ok(FilterClause {
                field: field.to_string(),
                op,
                value: parse_value(value)?,
            });
        }
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("unsupported filter clause: {trimmed}"),
    ))
}

fn parse_value(input: &str) -> io::Result<FieldValue> {
    if input.starts_with('"') && input.ends_with('"') && input.len() >= 2 {
        let value = serde_json::from_str::<String>(input).map_err(json_to_io_error)?;
        return Ok(FieldValue::String(value));
    }

    match input {
        "true" => return Ok(FieldValue::Bool(true)),
        "false" => return Ok(FieldValue::Bool(false)),
        _ => {}
    }

    if let Ok(value) = input.parse::<i64>() {
        return Ok(FieldValue::Int64(value));
    }
    if let Ok(value) = input.parse::<f64>() {
        return Ok(FieldValue::Float64(value));
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("unsupported filter literal: {input}"),
    ))
}

fn values_equal(left: &FieldValue, right: &FieldValue) -> bool {
    match (left, right) {
        (FieldValue::String(a), FieldValue::String(b)) => a == b,
        (FieldValue::Bool(a), FieldValue::Bool(b)) => a == b,
        (FieldValue::Int64(a), FieldValue::Int64(b)) => a == b,
        (FieldValue::Float64(a), FieldValue::Float64(b)) => a == b,
        (FieldValue::Int64(a), FieldValue::Float64(b)) => (*a as f64) == *b,
        (FieldValue::Float64(a), FieldValue::Int64(b)) => *a == (*b as f64),
        _ => false,
    }
}

fn compare_values(left: &FieldValue, right: &FieldValue) -> Option<Ordering> {
    match (left, right) {
        (FieldValue::String(a), FieldValue::String(b)) => Some(a.cmp(b)),
        (FieldValue::Int64(a), FieldValue::Int64(b)) => Some(a.cmp(b)),
        (FieldValue::Float64(a), FieldValue::Float64(b)) => a.partial_cmp(b),
        (FieldValue::Int64(a), FieldValue::Float64(b)) => (*a as f64).partial_cmp(b),
        (FieldValue::Float64(a), FieldValue::Int64(b)) => a.partial_cmp(&(*b as f64)),
        _ => None,
    }
}

fn json_to_io_error(err: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}
