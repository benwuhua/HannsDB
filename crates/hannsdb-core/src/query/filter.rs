use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::io;

use crate::document::FieldValue;

// ---------------------------------------------------------------------------
// Recursive filter AST
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum FilterExpr {
    And(Vec<FilterExpr>),
    Or(Vec<FilterExpr>),
    Not(Box<FilterExpr>),
    Clause {
        field: String,
        op: ComparisonOp,
        value: FieldValue,
    },
    InList {
        field: String,
        negated: bool,
        values: Vec<FieldValue>,
    },
    NullCheck {
        field: String,
        negated: bool,
    },
}

impl FilterExpr {
    pub fn matches(&self, fields: &BTreeMap<String, FieldValue>) -> bool {
        match self {
            FilterExpr::And(exprs) => exprs.iter().all(|e| e.matches(fields)),
            FilterExpr::Or(exprs) => exprs.iter().any(|e| e.matches(fields)),
            FilterExpr::Not(expr) => !expr.matches(fields),
            FilterExpr::Clause { field, op, value } => {
                let Some(actual) = fields.get(field) else {
                    return false;
                };
                match op {
                    ComparisonOp::Eq => values_equal(actual, value),
                    ComparisonOp::Ne => !values_equal(actual, value),
                    ComparisonOp::Gt => compare_values(actual, value) == Some(Ordering::Greater),
                    ComparisonOp::Gte => matches!(
                        compare_values(actual, value),
                        Some(Ordering::Greater | Ordering::Equal)
                    ),
                    ComparisonOp::Lt => compare_values(actual, value) == Some(Ordering::Less),
                    ComparisonOp::Lte => matches!(
                        compare_values(actual, value),
                        Some(Ordering::Less | Ordering::Equal)
                    ),
                }
            }
            FilterExpr::InList { field, negated, values } => {
                let Some(actual) = fields.get(field) else {
                    return *negated; // missing field: non-negated → false, negated → true
                };
                let found = values.iter().any(|v| values_equal(actual, v));
                if *negated { !found } else { found }
            }
            FilterExpr::NullCheck { field, negated } => {
                let is_null = !fields.contains_key(field);
                if *negated { !is_null } else { is_null }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
}

// ---------------------------------------------------------------------------
// Recursive-descent parser
// ---------------------------------------------------------------------------
//
// Grammar:
//   expr     = or_expr
//   or_expr  = and_expr ("or" and_expr)*
//   and_expr = not_expr ("and" not_expr)*
//   not_expr = "not" not_expr | primary
//   primary  = "(" expr ")" | clause
//   clause   = field op value
//
// Tokenisation is done on-the-fly by peeking / consuming slices of the input
// string. Tokens are: "(", ")", "and", "or", "not", plus comparison ops and
// field/value literals consumed inside `parse_clause`.

pub fn parse_filter(input: &str) -> io::Result<FilterExpr> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "filter expression cannot be empty",
        ));
    }
    let mut pos = 0usize;
    let expr = parse_or_expr(trimmed, &mut pos)?;
    // trailing whitespace is fine, but trailing junk is an error
    let remaining = trimmed[pos..].trim();
    if !remaining.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unexpected trailing text in filter: {remaining}"),
        ));
    }
    Ok(expr)
}

// or_expr = and_expr ("or" and_expr)*
fn parse_or_expr(input: &str, pos: &mut usize) -> io::Result<FilterExpr> {
    let mut exprs = vec![parse_and_expr(input, pos)?];
    loop {
        skip_whitespace(input, pos);
        if peek_keyword(input, *pos, "or") {
            *pos += 2; // skip "or"
            exprs.push(parse_and_expr(input, pos)?);
        } else {
            break;
        }
    }
    Ok(if exprs.len() == 1 {
        exprs.pop().unwrap()
    } else {
        FilterExpr::Or(exprs)
    })
}

// and_expr = not_expr ("and" not_expr)*
fn parse_and_expr(input: &str, pos: &mut usize) -> io::Result<FilterExpr> {
    let mut exprs = vec![parse_not_expr(input, pos)?];
    loop {
        skip_whitespace(input, pos);
        if peek_keyword(input, *pos, "and") {
            *pos += 3; // skip "and"
            exprs.push(parse_not_expr(input, pos)?);
        } else {
            break;
        }
    }
    Ok(if exprs.len() == 1 {
        exprs.pop().unwrap()
    } else {
        FilterExpr::And(exprs)
    })
}

// not_expr = "not" not_expr | primary
fn parse_not_expr(input: &str, pos: &mut usize) -> io::Result<FilterExpr> {
    skip_whitespace(input, pos);
    if peek_keyword(input, *pos, "not") {
        *pos += 3; // skip "not"
        let inner = parse_not_expr(input, pos)?;
        Ok(FilterExpr::Not(Box::new(inner)))
    } else {
        parse_primary(input, pos)
    }
}

// primary = "(" expr ")" | clause
fn parse_primary(input: &str, pos: &mut usize) -> io::Result<FilterExpr> {
    skip_whitespace(input, pos);
    let rest = &input[*pos..];
    if rest.starts_with('(') {
        *pos += 1; // skip '('
        let inner = parse_or_expr(input, pos)?;
        skip_whitespace(input, pos);
        let rest = &input[*pos..];
        if !rest.starts_with(')') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "expected ')' in filter expression",
            ));
        }
        *pos += 1; // skip ')'
        Ok(inner)
    } else {
        parse_clause_expr(input, pos)
    }
}

// clause = field op value | field "in" "(" values ")" | field "not" "in" "(" values ")"
//        | field "is" "null" | field "is" "not" "null"
fn parse_clause_expr(input: &str, pos: &mut usize) -> io::Result<FilterExpr> {
    skip_whitespace(input, pos);

    // 1. Read field name. We need to look ahead to determine the clause type.
    //    Field names end at whitespace followed by a keyword (and/or/not/in/is)
    //    or at a comparison operator.
    let field = read_field_name(input, pos);
    if field.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "empty field name in filter clause",
        ));
    }

    skip_whitespace(input, pos);

    // 2. Look ahead to determine clause type.
    let rest = &input[*pos..];

    // Check for "is null" / "is not null"
    if peek_keyword(rest, 0, "is") {
        *pos += 2; // skip "is"
        skip_whitespace(input, pos);
        let rest2 = &input[*pos..];
        if peek_keyword(rest2, 0, "not") {
            *pos += 3; // skip "not"
            skip_whitespace(input, pos);
            expect_keyword(input, pos, "null")?;
            return Ok(FilterExpr::NullCheck { field, negated: true });
        }
        expect_keyword(input, pos, "null")?;
        return Ok(FilterExpr::NullCheck { field, negated: false });
    }

    // Check for "not in (...)"
    if peek_keyword(rest, 0, "not") {
        let saved = *pos;
        *pos += 3; // skip "not"
        skip_whitespace(input, pos);
        if peek_keyword(&input[*pos..], 0, "in") {
            *pos += 2; // skip "in"
            let values = parse_in_list_values(input, pos)?;
            return Ok(FilterExpr::InList { field, negated: true, values });
        }
        // Not "not in" — backtrack
        *pos = saved;
    }

    // Check for "in (...)"
    if peek_keyword(rest, 0, "in") {
        *pos += 2; // skip "in"
        let values = parse_in_list_values(input, pos)?;
        return Ok(FilterExpr::InList { field, negated: false, values });
    }

    // 3. Default: comparison clause (field op value)
    let (op, op_len) = parse_op(&input[*pos..])?;
    *pos += op_len;

    skip_whitespace(input, pos);
    let value_str = read_value_token(input, *pos);
    if value_str.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "missing value in filter clause",
        ));
    }
    *pos += value_str.len();
    let value = parse_value(value_str.trim())?;

    Ok(FilterExpr::Clause { field, op, value })
}

/// Read a field name: runs until whitespace or a comparison operator.
fn read_field_name(input: &str, pos: &mut usize) -> String {
    let bytes = input.as_bytes();
    let start = *pos;
    while *pos < bytes.len() {
        let c = bytes[*pos];
        if c.is_ascii_whitespace() || c == b'>' || c == b'<' || c == b'=' || c == b'!' {
            break;
        }
        *pos += 1;
    }
    let name = input[start..*pos].trim().to_string();

    // If the name itself is a keyword (in/is/not/and/or), it could be:
    // - a real field named that (unlikely but valid)
    // - But for our grammar, we never have fields named after keywords.
    // Since field names run to whitespace, "group in" will read "group" then
    // stop at the space before "in". This should be fine.
    name
}

/// Parse parenthesized comma-separated values for `in (...)`.
fn parse_in_list_values(input: &str, pos: &mut usize) -> io::Result<Vec<FieldValue>> {
    skip_whitespace(input, pos);
    let rest = &input[*pos..];
    if !rest.starts_with('(') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "expected '(' after 'in' in filter expression",
        ));
    }
    *pos += 1; // skip '('

    let mut values = Vec::new();
    loop {
        skip_whitespace(input, pos);
        let rest = &input[*pos..];
        if rest.starts_with(')') {
            *pos += 1;
            break;
        }
        if !values.is_empty() {
            if !rest.starts_with(',') {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "expected ',' or ')' in value list",
                ));
            }
            *pos += 1; // skip ','
            skip_whitespace(input, pos);
        }
        let token = read_list_value_token(input, *pos);
        if token.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "expected value in 'in' list",
            ));
        }
        *pos += token.len();
        values.push(parse_value(token.trim())?);
    }
    Ok(values)
}

/// Read a value token inside a parenthesized list.
/// Stops at ',' or ')' or end-of-input.
fn read_list_value_token(input: &str, pos: usize) -> &str {
    let bytes = input.as_bytes();
    if pos >= bytes.len() {
        return "";
    }

    // Quoted string
    if bytes[pos] == b'"' {
        let mut i = pos + 1;
        while i < bytes.len() {
            if bytes[i] == b'\\' {
                i += 2;
                continue;
            }
            if bytes[i] == b'"' {
                return &input[pos..=i];
            }
            i += 1;
        }
        return &input[pos..];
    }

    // Unquoted: scan until ',' or ')' or end
    let mut i = pos;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b',' || c == b')' {
            break;
        }
        i += 1;
    }
    &input[pos..i]
}

/// Expect a keyword at the current position and advance past it.
fn expect_keyword(input: &str, pos: &mut usize, kw: &str) -> io::Result<()> {
    let rest = &input[*pos..];
    if !rest.starts_with(kw) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("expected '{kw}' in filter expression"),
        ));
    }
    *pos += kw.len();
    Ok(())
}

// Parse comparison operator at the start of `input`.
fn parse_op(input: &str) -> io::Result<(ComparisonOp, usize)> {
    for (token, op) in [
        (">=", ComparisonOp::Gte),
        ("<=", ComparisonOp::Lte),
        ("==", ComparisonOp::Eq),
        ("=", ComparisonOp::Eq),
        ("!=", ComparisonOp::Ne),
        (">", ComparisonOp::Gt),
        ("<", ComparisonOp::Lt),
    ] {
        if input.starts_with(token) {
            return Ok((op, token.len()));
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "expected comparison operator, found: {}",
            &input[..input.len().min(20)]
        ),
    ))
}

/// Read a value token starting at `pos`.
///
/// If the value starts with `"` we read until the matching closing `"`,
/// respecting escapes. Otherwise we read until we hit a whitespace followed
/// by a keyword (`and`, `or`, `not`), a `)` or end-of-input.
fn read_value_token(input: &str, pos: usize) -> &str {
    let bytes = input.as_bytes();
    if pos >= bytes.len() {
        return "";
    }

    // Quoted string value.
    if bytes[pos] == b'"' {
        let mut i = pos + 1;
        while i < bytes.len() {
            if bytes[i] == b'\\' {
                i += 2; // skip escaped char
                continue;
            }
            if bytes[i] == b'"' {
                // include the closing quote
                return &input[pos..=i];
            }
            i += 1;
        }
        // unterminated string -- return rest
        return &input[pos..];
    }

    // Unquoted value: scan until whitespace followed by a keyword or ')' or end.
    let mut i = pos;
    while i < bytes.len() {
        if bytes[i] == b')' {
            break;
        }
        if bytes[i].is_ascii_whitespace() {
            // Peek ahead: is the next token a keyword?
            let after = input[i..].trim_start();
            if after.is_empty()
                || after.starts_with("and ")
                || after.starts_with("and\t")
                || after == "and"
                || after.starts_with("or ")
                || after.starts_with("or\t")
                || after == "or"
                || after.starts_with("not ")
                || after.starts_with("not\t")
                || after == "not"
                || after.starts_with(')')
            {
                break;
            }
            // Otherwise the whitespace is inside the token (unlikely for our
            // value types but be safe): keep going.
        }
        i += 1;
    }
    &input[pos..i]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn skip_whitespace(input: &str, pos: &mut usize) {
    let bytes = input.as_bytes();
    while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

/// Return true if `input[pos..]` starts with `kw` followed by a non-alnum
/// boundary (whitespace, paren, end-of-string).
fn peek_keyword(input: &str, pos: usize, kw: &str) -> bool {
    let rest = &input[pos..];
    if !rest.starts_with(kw) {
        return false;
    }
    let after = &rest[kw.len()..];
    after.is_empty()
        || after.as_bytes()[0].is_ascii_whitespace()
        || after.as_bytes()[0] == b'('
        || after.as_bytes()[0] == b')'
}

// ---------------------------------------------------------------------------
// Value parsing (unchanged)
// ---------------------------------------------------------------------------

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

    // Try integer types in order of narrowest to widest.
    if let Ok(value) = input.parse::<i32>() {
        return Ok(FieldValue::Int32(value));
    }
    if let Ok(value) = input.parse::<i64>() {
        return Ok(FieldValue::Int64(value));
    }
    if let Ok(value) = input.parse::<u64>() {
        return Ok(FieldValue::UInt64(value));
    }
    if let Ok(value) = input.parse::<f32>() {
        return Ok(FieldValue::Float(value));
    }
    if let Ok(value) = input.parse::<f64>() {
        return Ok(FieldValue::Float64(value));
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("unsupported filter literal: {input}"),
    ))
}

/// Promote any numeric FieldValue to f64 for cross-type comparison.
fn to_f64(value: &FieldValue) -> Option<f64> {
    match value {
        FieldValue::Int64(v) => Some(*v as f64),
        FieldValue::Int32(v) => Some(*v as f64),
        FieldValue::UInt32(v) => Some(*v as f64),
        FieldValue::UInt64(v) => Some(*v as f64),
        FieldValue::Float(v) => Some(*v as f64),
        FieldValue::Float64(v) => Some(*v),
        _ => None,
    }
}

/// Promote any integer FieldValue to i64 for cross-type integer comparison.
fn to_i64(value: &FieldValue) -> Option<i64> {
    match value {
        FieldValue::Int64(v) => Some(*v),
        FieldValue::Int32(v) => Some(*v as i64),
        FieldValue::UInt32(v) => Some(*v as i64),
        _ => None,
    }
}

fn values_equal(left: &FieldValue, right: &FieldValue) -> bool {
    match (left, right) {
        (FieldValue::String(a), FieldValue::String(b)) => a == b,
        (FieldValue::Bool(a), FieldValue::Bool(b)) => a == b,
        // Same-type numeric equality
        (FieldValue::Int64(a), FieldValue::Int64(b)) => a == b,
        (FieldValue::Int32(a), FieldValue::Int32(b)) => a == b,
        (FieldValue::UInt32(a), FieldValue::UInt32(b)) => a == b,
        (FieldValue::UInt64(a), FieldValue::UInt64(b)) => a == b,
        (FieldValue::Float(a), FieldValue::Float(b)) => a == b,
        (FieldValue::Float64(a), FieldValue::Float64(b)) => a == b,
        // Cross-type: try exact integer comparison first, then fall back to f64
        _ => {
            if let (Some(a), Some(b)) = (to_i64(left), to_i64(right)) {
                return a == b;
            }
            if let (Some(a), Some(b)) = (to_f64(left), to_f64(right)) {
                return a == b;
            }
            false
        }
    }
}

fn compare_values(left: &FieldValue, right: &FieldValue) -> Option<Ordering> {
    match (left, right) {
        (FieldValue::String(a), FieldValue::String(b)) => Some(a.cmp(b)),
        // Same-type numeric comparisons
        (FieldValue::Int64(a), FieldValue::Int64(b)) => Some(a.cmp(b)),
        (FieldValue::Int32(a), FieldValue::Int32(b)) => Some(a.cmp(b)),
        (FieldValue::UInt32(a), FieldValue::UInt32(b)) => Some(a.cmp(b)),
        (FieldValue::UInt64(a), FieldValue::UInt64(b)) => Some(a.cmp(b)),
        (FieldValue::Float(a), FieldValue::Float(b)) => a.partial_cmp(b),
        (FieldValue::Float64(a), FieldValue::Float64(b)) => a.partial_cmp(b),
        // Cross-type: try exact integer comparison first, then fall back to f64
        _ => {
            if let (Some(a), Some(b)) = (to_i64(left), to_i64(right)) {
                return Some(a.cmp(&b));
            }
            if let (Some(a), Some(b)) = (to_f64(left), to_f64(right)) {
                return a.partial_cmp(&b);
            }
            None
        }
    }
}

fn json_to_io_error(err: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn field_map(pairs: Vec<(&str, FieldValue)>) -> BTreeMap<String, FieldValue> {
        pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }

    // -- legacy AND-only tests (backward compat) --

    #[test]
    fn single_clause_eq() {
        let expr = parse_filter("status == \"active\"").unwrap();
        let fields = field_map(vec![("status", FieldValue::String("active".into()))]);
        assert!(expr.matches(&fields));
    }

    #[test]
    fn and_clauses() {
        let expr = parse_filter("price > 100 and category == \"premium\"").unwrap();
        let fields = field_map(vec![
            ("price", FieldValue::Int64(150)),
            ("category", FieldValue::String("premium".into())),
        ]);
        assert!(expr.matches(&fields));

        let fields2 = field_map(vec![
            ("price", FieldValue::Int64(50)),
            ("category", FieldValue::String("premium".into())),
        ]);
        assert!(!expr2_matches(&expr, &fields2));
    }

    #[test]
    fn empty_filter_error() {
        assert!(parse_filter("").is_err());
        assert!(parse_filter("   ").is_err());
    }

    // -- OR tests --

    #[test]
    fn or_clauses() {
        let expr = parse_filter("price > 100 or category == \"premium\"").unwrap();
        // both match
        let fields1 = field_map(vec![
            ("price", FieldValue::Int64(150)),
            ("category", FieldValue::String("premium".into())),
        ]);
        assert!(expr.matches(&fields1));

        // only left matches
        let fields2 = field_map(vec![
            ("price", FieldValue::Int64(150)),
            ("category", FieldValue::String("basic".into())),
        ]);
        assert!(expr.matches(&fields2));

        // only right matches
        let fields3 = field_map(vec![
            ("price", FieldValue::Int64(50)),
            ("category", FieldValue::String("premium".into())),
        ]);
        assert!(expr.matches(&fields3));

        // neither matches
        let fields4 = field_map(vec![
            ("price", FieldValue::Int64(50)),
            ("category", FieldValue::String("basic".into())),
        ]);
        assert!(!expr.matches(&fields4));
    }

    // -- NOT tests --

    #[test]
    fn not_clause() {
        let expr = parse_filter("not category == \"inactive\"").unwrap();
        let fields1 = field_map(vec![("category", FieldValue::String("active".into()))]);
        assert!(expr.matches(&fields1));

        let fields2 = field_map(vec![("category", FieldValue::String("inactive".into()))]);
        assert!(!expr.matches(&fields2));
    }

    // -- Parenthesized + mixed tests --

    #[test]
    fn paren_or_and() {
        // (price > 100 or category == "premium") and active == true
        let expr =
            parse_filter("(price > 100 or category == \"premium\") and active == true").unwrap();
        // both or-branches true, active true
        let f1 = field_map(vec![
            ("price", FieldValue::Int64(150)),
            ("category", FieldValue::String("premium".into())),
            ("active", FieldValue::Bool(true)),
        ]);
        assert!(expr.matches(&f1));

        // only left or-branch true, active true
        let f2 = field_map(vec![
            ("price", FieldValue::Int64(150)),
            ("category", FieldValue::String("basic".into())),
            ("active", FieldValue::Bool(true)),
        ]);
        assert!(expr.matches(&f2));

        // or is true but active is false
        let f3 = field_map(vec![
            ("price", FieldValue::Int64(150)),
            ("category", FieldValue::String("premium".into())),
            ("active", FieldValue::Bool(false)),
        ]);
        assert!(!expr.matches(&f3));

        // or is false, active true
        let f4 = field_map(vec![
            ("price", FieldValue::Int64(50)),
            ("category", FieldValue::String("basic".into())),
            ("active", FieldValue::Bool(true)),
        ]);
        assert!(!expr.matches(&f4));
    }

    #[test]
    fn not_paren_and() {
        // not (price < 50 and category == "basic")
        let expr = parse_filter("not (price < 50 and category == \"basic\")").unwrap();
        // both and-conditions true => not(true) = false
        let f1 = field_map(vec![
            ("price", FieldValue::Int64(30)),
            ("category", FieldValue::String("basic".into())),
        ]);
        assert!(!expr.matches(&f1));

        // and is false (price not < 50) => not(false) = true
        let f2 = field_map(vec![
            ("price", FieldValue::Int64(100)),
            ("category", FieldValue::String("basic".into())),
        ]);
        assert!(expr.matches(&f2));

        // and is false (category != basic) => not(false) = true
        let f3 = field_map(vec![
            ("price", FieldValue::Int64(30)),
            ("category", FieldValue::String("premium".into())),
        ]);
        assert!(expr.matches(&f3));
    }

    #[test]
    fn precedence_or_and() {
        // price > 100 or category == "premium" and active == true
        // Should parse as: price > 100 OR (category == "premium" AND active == true)
        let expr =
            parse_filter("price > 100 or category == \"premium\" and active == true").unwrap();
        // price > 100 is true => whole thing true regardless of and
        let f1 = field_map(vec![
            ("price", FieldValue::Int64(150)),
            ("category", FieldValue::String("basic".into())),
            ("active", FieldValue::Bool(false)),
        ]);
        assert!(expr.matches(&f1));

        // price <= 100, but and-branch is true
        let f2 = field_map(vec![
            ("price", FieldValue::Int64(50)),
            ("category", FieldValue::String("premium".into())),
            ("active", FieldValue::Bool(true)),
        ]);
        assert!(expr.matches(&f2));

        // price <= 100, and-branch false (active=false)
        let f3 = field_map(vec![
            ("price", FieldValue::Int64(50)),
            ("category", FieldValue::String("premium".into())),
            ("active", FieldValue::Bool(false)),
        ]);
        assert!(!expr.matches(&f3));
    }

    #[test]
    fn double_not() {
        // not not active == true  =>  active == true
        let expr = parse_filter("not not active == true").unwrap();
        let f1 = field_map(vec![("active", FieldValue::Bool(true))]);
        assert!(expr.matches(&f1));
        let f2 = field_map(vec![("active", FieldValue::Bool(false))]);
        assert!(!expr.matches(&f2));
    }

    // helper to avoid confusing variable names
    fn expr2_matches(expr: &FilterExpr, fields: &BTreeMap<String, FieldValue>) -> bool {
        expr.matches(fields)
    }

    // -- in / not in tests --

    #[test]
    fn in_list_matches() {
        let expr = parse_filter("status in (\"active\", \"pending\")").unwrap();
        let f1 = field_map(vec![("status", FieldValue::String("active".into()))]);
        assert!(expr.matches(&f1));
        let f2 = field_map(vec![("status", FieldValue::String("pending".into()))]);
        assert!(expr.matches(&f2));
        let f3 = field_map(vec![("status", FieldValue::String("closed".into()))]);
        assert!(!expr.matches(&f3));
    }

    #[test]
    fn in_list_integers() {
        let expr = parse_filter("group in (1, 2, 3)").unwrap();
        let f1 = field_map(vec![("group", FieldValue::Int64(2))]);
        assert!(expr.matches(&f1));
        let f2 = field_map(vec![("group", FieldValue::Int64(5))]);
        assert!(!expr.matches(&f2));
    }

    #[test]
    fn not_in_list() {
        let expr = parse_filter("status not in (\"closed\", \"archived\")").unwrap();
        let f1 = field_map(vec![("status", FieldValue::String("active".into()))]);
        assert!(expr.matches(&f1));
        let f2 = field_map(vec![("status", FieldValue::String("closed".into()))]);
        assert!(!expr.matches(&f2));
    }

    #[test]
    fn in_list_missing_field() {
        let expr = parse_filter("missing in (1, 2)").unwrap();
        let fields = field_map(vec![("other", FieldValue::Int64(1))]);
        assert!(!expr.matches(&fields));
    }

    #[test]
    fn not_in_list_missing_field() {
        let expr = parse_filter("missing not in (1, 2)").unwrap();
        let fields = field_map(vec![("other", FieldValue::Int64(1))]);
        assert!(expr.matches(&fields)); // negated: missing → true
    }

    #[test]
    fn in_list_combined_with_and() {
        let expr = parse_filter("status in (\"active\") and score > 10").unwrap();
        let f1 = field_map(vec![
            ("status", FieldValue::String("active".into())),
            ("score", FieldValue::Int64(20)),
        ]);
        assert!(expr.matches(&f1));
        let f2 = field_map(vec![
            ("status", FieldValue::String("active".into())),
            ("score", FieldValue::Int64(5)),
        ]);
        assert!(!expr.matches(&f2));
    }

    // -- is null / is not null tests --

    #[test]
    fn is_null_matches() {
        let expr = parse_filter("nickname is null").unwrap();
        let f1: BTreeMap<String, FieldValue> = BTreeMap::new();
        assert!(expr.matches(&f1));
        let f2 = field_map(vec![("nickname", FieldValue::String("bob".into()))]);
        assert!(!expr.matches(&f2));
    }

    #[test]
    fn is_not_null_matches() {
        let expr = parse_filter("nickname is not null").unwrap();
        let f1: BTreeMap<String, FieldValue> = BTreeMap::new();
        assert!(!expr.matches(&f1));
        let f2 = field_map(vec![("nickname", FieldValue::String("bob".into()))]);
        assert!(expr.matches(&f2));
    }

    #[test]
    fn is_null_combined_with_and() {
        let expr = parse_filter("nickname is null and score > 5").unwrap();
        let f1 = field_map(vec![("score", FieldValue::Int64(10))]);
        assert!(expr.matches(&f1));
        let f2 = field_map(vec![("nickname", FieldValue::String("x".into())), ("score", FieldValue::Int64(10))]);
        assert!(!expr.matches(&f2));
    }
}
