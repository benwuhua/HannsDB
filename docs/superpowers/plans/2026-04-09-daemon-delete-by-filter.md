# Daemon Delete By Filter Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose core-backed `delete_by_filter` through `hannsdb-daemon` with stable HTTP semantics and smoke coverage.

**Architecture:** Keep this slice transport-only. Add one new daemon request type and one dedicated HTTP route that delegates directly to `HannsDb::delete_by_filter(...)`, then lock the HTTP contract with `http_smoke` tests for success, malformed requests, and latest-live semantics.

**Tech Stack:** Rust (`axum`, `serde`, `hannsdb-daemon`, `hannsdb-core`), daemon smoke tests (`tokio`, `tower::ServiceExt`).

---

## File Map

- Modify: `crates/hannsdb-daemon/src/api.rs`
  Adds `DeleteByFilterRequest` and reuses the existing delete response envelope.
- Modify: `crates/hannsdb-daemon/src/routes.rs`
  Adds the new route, a handler that wraps JSON extractor failures, and status mapping from core `io::ErrorKind`.
- Modify: `crates/hannsdb-daemon/tests/http_smoke.rs`
  Adds transport-level smoke coverage for positive delete-by-filter behavior, malformed input, and route non-regression.

## Chunk 1: Transport Wiring

### Task 1: Lock the new daemon contract with failing smoke tests

**Files:**
- Modify: `crates/hannsdb-daemon/tests/http_smoke.rs`

- [ ] **Step 1: Write the failing positive delete-by-filter smoke test**

Add a test like:

```rust
#[tokio::test]
async fn delete_by_filter_route_deletes_matching_rows() {
    // create collection
    // insert docs with scalar field `group`
    // POST /collections/docs/records/delete_by_filter {"filter":"group == 1"}
    // assert 200 and {"deleted": 2}
    // assert follow-up search/fetch shows only survivors
}
```

- [ ] **Step 2: Run the positive smoke test to verify it fails**

Run:

```bash
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_deletes_matching_rows -- --nocapture
```

Expected: FAIL because the route does not exist yet.

- [ ] **Step 3: Write the failing zero-match and repeat-delete smoke tests**

Add:

```rust
#[tokio::test]
async fn delete_by_filter_route_returns_zero_for_no_matches() {}

#[tokio::test]
async fn delete_by_filter_route_second_call_returns_zero() {}
```

Assertions:
- valid request returns `200`
- body is `{"deleted":0}` when nothing matches
- repeated delete of the same filter returns `0` on the second call

- [ ] **Step 4: Run the zero-match and repeat-delete smoke tests to verify they fail**

Run:

```bash
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_returns_zero_for_no_matches -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_second_call_returns_zero -- --nocapture
```

Expected: FAIL because the route does not exist yet.

- [ ] **Step 5: Write the failing malformed-body and invalid-filter smoke tests**

Add:

```rust
#[tokio::test]
async fn delete_by_filter_route_returns_bad_request_for_invalid_filter() {}

#[tokio::test]
async fn delete_by_filter_route_returns_bad_request_for_empty_filter() {}

#[tokio::test]
async fn delete_by_filter_route_returns_bad_request_for_whitespace_only_filter() {}

#[tokio::test]
async fn delete_by_filter_route_wraps_malformed_json_in_daemon_error_envelope() {}

#[tokio::test]
async fn delete_by_filter_route_returns_bad_request_for_missing_filter_field() {}
```

Assertions:
- all return `400`
- body is daemon `{"error": ...}` JSON
- no axum default rejection body leaks through

- [ ] **Step 6: Run the malformed/invalid smoke tests to verify they fail**

Run:

```bash
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_returns_bad_request_for_invalid_filter -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_returns_bad_request_for_empty_filter -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_returns_bad_request_for_whitespace_only_filter -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_wraps_malformed_json_in_daemon_error_envelope -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_returns_bad_request_for_missing_filter_field -- --nocapture
```

Expected: FAIL because the route and extractor wrapper do not exist yet.

- [ ] **Step 7: Write the failing not-found and route-regression smoke tests**

Add:

```rust
#[tokio::test]
async fn delete_by_filter_route_returns_not_found_for_missing_collection() {}

#[tokio::test]
async fn delete_records_route_still_deletes_by_explicit_ids() {}
```

The second test should explicitly hit:

```text
DELETE /collections/docs/records
```

and assert the existing id-delete path still behaves unchanged after the new route is added.

The missing-collection test must assert daemon semantics, not just bare `404`. Require:
- response status is `404`
- body parses as daemon `{"error": ...}` JSON
- error text refers to the missing collection, so a router-level path miss cannot false-pass

- [ ] **Step 8: Run the not-found and route-regression smoke tests to verify the route case fails and the regression case already passes**

Run:

```bash
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_returns_not_found_for_missing_collection -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke delete_records_route_still_deletes_by_explicit_ids -- --nocapture
```

Expected:
- `delete_by_filter_route_returns_not_found_for_missing_collection` fails because the route does not exist yet
- `delete_records_route_still_deletes_by_explicit_ids` already passes and must keep passing through the rest of the slice

### Task 2: Add daemon request type and route handler

**Files:**
- Modify: `crates/hannsdb-daemon/src/api.rs`
- Modify: `crates/hannsdb-daemon/src/routes.rs`
- Test: `crates/hannsdb-daemon/tests/http_smoke.rs`

- [ ] **Step 1: Add the request type**

In `crates/hannsdb-daemon/src/api.rs`, add:

```rust
#[derive(Debug, Deserialize)]
pub struct DeleteByFilterRequest {
    pub filter: String,
}
```

Do not add a new response type; reuse `DeleteRecordsResponse`.

- [ ] **Step 2: Register the route**

In `build_router(...)`, add:

```rust
.route(
    "/collections/:collection/records/delete_by_filter",
    post(delete_records_by_filter),
)
```

Keep the existing:

```rust
.route("/collections/:collection/records", post(insert_records).delete(delete_records))
```

unchanged.

- [ ] **Step 3: Implement a handler with explicit JSON rejection wrapping**

Implement a handler with this shape:

```rust
async fn delete_records_by_filter(
    State(state): State<DaemonState>,
    AxumPath(collection): AxumPath<String>,
    request: Result<Json<DeleteByFilterRequest>, JsonRejection>,
) -> Response
```

Inside the handler:
- map `JsonRejection` to `400` + `Json(ErrorResponse { error: rejection.body_text() })`
- call `db.delete_by_filter(&collection, &request.filter)`
- map:
  - `Ok(deleted)` -> `200` + `DeleteRecordsResponse`
  - `NotFound` -> `404`
  - `InvalidInput` -> `400`
  - other errors -> `500`

- [ ] **Step 4: Keep the implementation transport-only**

Do not:
- parse the filter in daemon
- special-case filter syntax beyond status mapping
- touch `hannsdb-core`
- refactor unrelated mutation routes

The only daemon-specific logic beyond delegation should be JSON rejection wrapping.

- [ ] **Step 5: Run the targeted route tests to verify they pass**

Run:

```bash
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_deletes_matching_rows -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_returns_zero_for_no_matches -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_second_call_returns_zero -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_returns_bad_request_for_invalid_filter -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_returns_bad_request_for_empty_filter -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_returns_bad_request_for_whitespace_only_filter -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_wraps_malformed_json_in_daemon_error_envelope -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_returns_bad_request_for_missing_filter_field -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_returns_not_found_for_missing_collection -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke delete_records_route_still_deletes_by_explicit_ids -- --nocapture
```

Expected: PASS

- [ ] **Step 6: Commit the transport slice**

```bash
git add crates/hannsdb-daemon/src/api.rs \
        crates/hannsdb-daemon/src/routes.rs \
        crates/hannsdb-daemon/tests/http_smoke.rs
git commit -m "feat(daemon): add delete-by-filter route"
```

## Chunk 2: Latest-Live Regression And Final Verification

### Task 3: Lock latest-live delegation semantics at the HTTP layer

**Files:**
- Modify: `crates/hannsdb-daemon/tests/http_smoke.rs`

- [ ] **Step 1: Write the failing latest-live shadowing smoke test**

Follow the existing style used for non-primary query-by-id setup helpers. Add either a helper or inline setup that creates a collection with multi-version data such that:
- an older row matches the filter
- a newer visible row for the same id does not match, or is already tombstoned

Then assert:

```rust
#[tokio::test]
async fn delete_by_filter_route_uses_latest_live_view() {
    // POST delete_by_filter
    // assert only latest-live matching rows count toward `deleted`
    // assert older shadowed rows are not resurrected or re-deleted
}
```

- [ ] **Step 2: Run the latest-live smoke test to verify it fails or is incomplete**

Run:

```bash
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_uses_latest_live_view -- --nocapture
```

Expected: FAIL until the test asserts the exact transport-level contract correctly. If it passes immediately, tighten the assertions so the route proves delegation to core latest-live semantics rather than merely exercising a flat happy path.

- [ ] **Step 3: Adjust the smoke fixture until it proves delegation, then rerun**

Make the fixture specific enough that a flat-layout delete would produce the wrong result. Good signals:
- response count differs between latest-live and naive flat scan
- a follow-up search/fetch proves the surviving row is the newest visible one

- [ ] **Step 4: Run the focused latest-live smoke test to verify it passes**

Run:

```bash
cargo test -p hannsdb-daemon --test http_smoke delete_by_filter_route_uses_latest_live_view -- --nocapture
```

Expected: PASS

### Task 4: Run the full daemon regression set and finish cleanly

**Files:**
- Modify: `crates/hannsdb-daemon/tests/http_smoke.rs`
- Verify: `crates/hannsdb-daemon/src/api.rs`
- Verify: `crates/hannsdb-daemon/src/routes.rs`

- [ ] **Step 1: Run the full daemon smoke suite**

Run:

```bash
cargo test -p hannsdb-daemon --test http_smoke -- --nocapture
```

Expected: PASS

- [ ] **Step 2: Commit the latest-live regression coverage if it changed after the first commit**

If Task 3 required additional test-only edits after the first transport commit:

```bash
git add crates/hannsdb-daemon/tests/http_smoke.rs
git commit -m "test(daemon): lock latest-live delete-by-filter semantics"
```

Otherwise skip this commit.

- [ ] **Step 3: Run formatting and diff hygiene checks**

Run:

```bash
git diff --check
git status --short --branch
```

Expected:
- `git diff --check` prints nothing
- `git status --short --branch` shows only the current feature branch and no unstaged changes

- [ ] **Step 4: Record final verification evidence**

Capture the exact commands and their passing outcomes in the implementation notes / final handoff:
- targeted `http_smoke` delete-by-filter tests
- full `http_smoke`
- `git diff --check`
- `git status --short --branch`
