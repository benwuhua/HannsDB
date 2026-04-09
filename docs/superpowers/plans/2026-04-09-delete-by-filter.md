# Delete By Filter Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add core-backed `delete_by_filter` support to `hannsdb-core` and `hannsdb-py`, preserving latest-live segment semantics and existing `delete(ids)`/WAL behavior.

**Architecture:** Implement `delete_by_filter` as a thin selection layer over the existing segment-aware runtime view. The core path should parse the existing filter grammar, collect matching ids from the latest-live view, then delegate one time to the existing segment-aware `delete(ids)` mutation path. Python should stop raising `NotImplementedError` and forward directly to the core-backed implementation.

**Tech Stack:** Rust (`hannsdb-core`, `hannsdb-py`/PyO3), Python facade tests (`pytest`), existing filter parser and WAL/delete machinery.

---

## Chunk 1: Core Delete-By-Filter

### Task 1: Lock core behavior with failing tests

**Files:**
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- Modify: `crates/hannsdb-core/tests/wal_recovery.rs`

- [ ] **Step 1: Write the failing single-segment and no-match tests**

Add focused `collection_api` tests that express the contract directly:

```rust
#[test]
fn collection_api_delete_by_filter_deletes_matching_live_rows() {
    // create docs with scalar field values
    // delete_by_filter("group == 1")
    // assert return count == number of newly deleted rows
    // assert fetch/search only return surviving docs
}

#[test]
fn collection_api_delete_by_filter_no_match_returns_zero() {
    // delete_by_filter("group == 999")
    // assert return count == 0 and data remains unchanged
}
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:
- `cargo test -p hannsdb-core --test collection_api collection_api_delete_by_filter_deletes_matching_live_rows -- --nocapture`
- `cargo test -p hannsdb-core --test collection_api collection_api_delete_by_filter_no_match_returns_zero -- --nocapture`

Expected: FAIL because `delete_by_filter` does not exist yet.

- [ ] **Step 3: Write the failing multi-segment shadowing tests**

Add `collection_api` coverage for latest-live semantics:

```rust
#[test]
fn collection_api_delete_by_filter_uses_latest_live_view_across_segments() {
    // rewrite to two-segment layout
    // newer live row matches filter, older row also exists
    // assert only the newest visible row is deleted
}

#[test]
fn collection_api_delete_by_filter_newer_tombstone_shadows_older_matching_row() {
    // newer segment already tombstones the id
    // older row still matches filter
    // assert delete_by_filter returns 0 for that id and older row stays untouched
}
```

- [ ] **Step 4: Run the targeted shadowing tests to verify they fail**

Run:
- `cargo test -p hannsdb-core --test collection_api collection_api_delete_by_filter_uses_latest_live_view_across_segments -- --nocapture`
- `cargo test -p hannsdb-core --test collection_api collection_api_delete_by_filter_newer_tombstone_shadows_older_matching_row -- --nocapture`

Expected: FAIL because there is no segment-aware filter-delete path yet.

- [ ] **Step 5: Write the failing repeat-delete, invalid-filter, and replay/WAL tests**

Add:

```rust
#[test]
fn collection_api_delete_by_filter_second_call_returns_zero() {
    // same filter twice
    // first call deletes rows, second returns 0
}

#[test]
fn collection_api_delete_by_filter_invalid_filter_errors() {
    // delete_by_filter("group = 1")
    // assert InvalidInput-style error from parse_filter
}

#[test]
fn wal_recovery_replays_delete_by_filter_outcome() {
    // run delete_by_filter, remove tombstone metadata, reopen
    // assert latest-live deleted outcome is restored
}

#[test]
fn wal_recovery_delete_by_filter_appends_one_delete_record() {
    // run delete_by_filter once
    // assert WAL appended exactly one existing Delete { ids } record
    // assert no new WAL variant is introduced
}
```

- [ ] **Step 6: Run the targeted repeat/replay/WAL tests to verify they fail**

Run:
- `cargo test -p hannsdb-core --test collection_api collection_api_delete_by_filter_second_call_returns_zero -- --nocapture`
- `cargo test -p hannsdb-core --test collection_api collection_api_delete_by_filter_invalid_filter_errors -- --nocapture`
- `cargo test -p hannsdb-core --test wal_recovery wal_recovery_replays_delete_by_filter_outcome -- --nocapture`
- `cargo test -p hannsdb-core --test wal_recovery wal_recovery_delete_by_filter_appends_one_delete_record -- --nocapture`

Expected: FAIL because delete-by-filter is not implemented yet.

### Task 2: Implement minimal core delete-by-filter

**Files:**
- Modify: `crates/hannsdb-core/src/db.rs`
- Test: `crates/hannsdb-core/tests/collection_api.rs`
- Test: `crates/hannsdb-core/tests/wal_recovery.rs`

- [ ] **Step 1: Add a core API entry point**

Add:

```rust
pub fn delete_by_filter(&mut self, collection: &str, filter: &str) -> io::Result<usize>
```

This should remain a thin wrapper, not a new mutation subsystem.

- [ ] **Step 2: Implement latest-live id selection**

Use the same segment order and shadowing rule as current read paths. Keep the logic local to `db.rs`; if a helper is needed, make it a small private helper in `db.rs`, not a new shared planner/selector abstraction. The implementation should:

```rust
let filter_expr = parse_filter(filter)?;
let matching_ids = /* local latest-live id collection in db.rs */;
if matching_ids.is_empty() {
    return Ok(0);
}
self.delete(collection, &matching_ids)
```

Requirements:
- evaluate the filter only once per latest-live external id
- a newer tombstoned row must still suppress older matching rows
- invalid filters must return the parser error unchanged

- [ ] **Step 3: Keep WAL behavior unchanged**

Do **not** add a new WAL variant. `delete_by_filter` must reach the existing `delete(ids)` path so the WAL contract stays as the existing `Delete { ids }` record shape.

- [ ] **Step 4: Run the targeted core tests to verify they pass**

Run:
- `cargo test -p hannsdb-core --test collection_api collection_api_delete_by_filter_deletes_matching_live_rows -- --nocapture`
- `cargo test -p hannsdb-core --test collection_api collection_api_delete_by_filter_no_match_returns_zero -- --nocapture`
- `cargo test -p hannsdb-core --test collection_api collection_api_delete_by_filter_uses_latest_live_view_across_segments -- --nocapture`
- `cargo test -p hannsdb-core --test collection_api collection_api_delete_by_filter_newer_tombstone_shadows_older_matching_row -- --nocapture`
- `cargo test -p hannsdb-core --test collection_api collection_api_delete_by_filter_second_call_returns_zero -- --nocapture`
- `cargo test -p hannsdb-core --test collection_api collection_api_delete_by_filter_invalid_filter_errors -- --nocapture`
- `cargo test -p hannsdb-core --test wal_recovery wal_recovery_replays_delete_by_filter_outcome -- --nocapture`
- `cargo test -p hannsdb-core --test wal_recovery wal_recovery_delete_by_filter_appends_one_delete_record -- --nocapture`

Expected: PASS

- [ ] **Step 5: Run broader core regression checks**

Run:
- `cargo test -p hannsdb-core --test collection_api -- --nocapture`
- `cargo test -p hannsdb-core --test wal_recovery -- --nocapture`

Expected: PASS

- [ ] **Step 6: Commit the core slice**

```bash
git add crates/hannsdb-core/src/db.rs \
        crates/hannsdb-core/tests/collection_api.rs \
        crates/hannsdb-core/tests/wal_recovery.rs
git commit -m "feat(core): add delete-by-filter"
```

## Chunk 2: Python Surface

### Task 3: Replace unsupported Python surface with real delegation

**Files:**
- Modify: `crates/hannsdb-py/src/lib.rs`
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Modify: `crates/hannsdb-py/tests/test_collection_parity.py`
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`

- [ ] **Step 1: Write the failing native Python tests**

Replace the old unsupported expectations with real behavior:

```python
def test_native_collection_delete_by_filter_deletes_matching_rows(tmp_path):
    deleted = collection.delete_by_filter("group == 1")
    assert deleted == 2

def test_native_collection_delete_by_filter_invalid_filter_propagates(tmp_path):
    with pytest.raises(Exception, match="unsupported filter clause|filter"):
        collection.delete_by_filter("group = 1")
```

- [ ] **Step 2: Run the targeted native tests to verify they fail**

Run:
- `uv run --with pytest pytest crates/hannsdb-py/tests/test_collection_parity.py -q -k delete_by_filter`

Expected: FAIL because the native surface still raises `NotImplementedError`.

- [ ] **Step 3: Write the failing pure-facade tests**

Add real `Collection` coverage:

```python
def test_real_collection_delete_by_filter_deletes_matching_rows(tmp_path):
    assert collection.delete_by_filter("group == 1") == 2

def test_real_collection_delete_by_filter_second_call_returns_zero(tmp_path):
    assert collection.delete_by_filter("group == 1") == 2
    assert collection.delete_by_filter("group == 1") == 0

def test_real_collection_delete_by_filter_invalid_filter_propagates(tmp_path):
    with pytest.raises(Exception, match="unsupported filter clause|filter"):
        collection.delete_by_filter("group = 1")
```

Also assert supported calls no longer raise `NotImplementedError`.

- [ ] **Step 4: Run the targeted facade tests to verify they fail**

Run:
- `uv run --with pytest pytest crates/hannsdb-py/tests/test_collection_facade.py -q -k delete_by_filter`

Expected: FAIL because the pure facade still raises `NotImplementedError`.

- [ ] **Step 5: Implement native delegation**

In `crates/hannsdb-py/src/lib.rs`, replace:

```rust
fn delete_by_filter(&mut self, _filter: String) -> PyResult<()> {
    Err(PyNotImplementedError::new_err(...))
}
```

with a real integer-returning delegation to the core API.

- [ ] **Step 6: Implement pure-Python delegation**

In `crates/hannsdb-py/python/hannsdb/model/collection.py`, replace:

```python
def delete_by_filter(self, filter: str):
    raise NotImplementedError(...)
```

with direct delegation to `_core.delete_by_filter(filter)` while preserving the existing locking/concurrency style used by nearby mutating methods.

- [ ] **Step 7: Run the targeted Python tests to verify they pass**

Run:
- `uv run --with pytest pytest crates/hannsdb-py/tests/test_collection_parity.py -q -k delete_by_filter`
- `uv run --with pytest pytest crates/hannsdb-py/tests/test_collection_facade.py -q -k delete_by_filter`

Expected: PASS

- [ ] **Step 8: Run broader Python regression checks**

Run:
- `cargo test -p hannsdb-py --features python-binding --lib`
- `uv run --with pytest pytest crates/hannsdb-py/tests/test_collection_parity.py crates/hannsdb-py/tests/test_collection_facade.py -q`

Expected: PASS

- [ ] **Step 9: Sanity-check the diff and commit**

Run:
- `git diff --check`
- `git status --short`

Then commit:

```bash
git add crates/hannsdb-py/src/lib.rs \
        crates/hannsdb-py/python/hannsdb/model/collection.py \
        crates/hannsdb-py/tests/test_collection_parity.py \
        crates/hannsdb-py/tests/test_collection_facade.py
git commit -m "feat(py): expose delete-by-filter"
```
