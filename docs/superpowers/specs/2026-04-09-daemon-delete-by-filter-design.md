# Daemon Delete By Filter Transport Design

**Date:** 2026-04-09

**Goal:** Expose the existing core-backed `delete_by_filter` capability through `hannsdb-daemon` HTTP transport, while keeping the slice limited to request/response wiring and daemon smoke coverage.

## Context

HannsDB already has:

- segment-aware `delete(ids)` semantics
- core-backed `HannsDb::delete_by_filter(...)`
- Python/native `Collection.delete_by_filter(...)`

The remaining user-facing gap is the daemon layer. Today `hannsdb-daemon` only exposes record deletion by explicit id list on `/collections/:collection/records`, while filtered deletion is unavailable over HTTP.

This makes daemon transport the next smallest parity slice. The core mutation semantics already exist, so this slice should avoid inventing new delete logic and only delegate cleanly to core.

This spec assumes the current branch already contains:

- `HannsDb::delete_by_filter(&mut self, collection: &str, filter: &str) -> io::Result<usize>`

The daemon work in this slice is therefore transport-only. It does not introduce or redesign the core API.

## Chosen Scope

This slice will implement only:

1. `hannsdb-daemon` request/response types for filter delete
2. a new HTTP route that delegates to `HannsDb::delete_by_filter(...)`
3. daemon smoke coverage for success and error semantics

This slice will **not** implement:

- any new core delete behavior
- any Python changes
- daemon column DDL
- bulk mutation unification
- new filter grammar

## Route Shape

Add a dedicated route:

- `POST /collections/:collection/records/delete_by_filter`

Request body:

```json
{ "filter": "group == 1" }
```

Response body reuses the existing delete envelope:

```json
{ "deleted": 2 }
```

This route is intentionally separate from `DELETE /collections/:collection/records`:

- `DELETE /records` already means explicit id deletion and expects an `ids` payload
- overloading the same endpoint with multiple body shapes would increase ambiguity
- a dedicated sub-route keeps backward compatibility and makes HTTP semantics obvious

## API Types

In `crates/hannsdb-daemon/src/api.rs` add:

- `DeleteByFilterRequest { filter: String }`

Keep using the existing:

- `DeleteRecordsResponse { deleted: u64 }`
- `ErrorResponse { error: String }`

No new response envelope is needed.

## Route Behavior

In `crates/hannsdb-daemon/src/routes.rs` add a handler that:

1. extracts `collection` from the path
2. extracts `filter` from JSON body
3. calls `db.delete_by_filter(&collection, &request.filter)`
4. maps the result into the existing daemon response envelopes

The daemon must not parse or reinterpret the filter on its own. Core remains the only source of truth for filter syntax and latest-live delete semantics.

The status mapping depends on the existing core contract:

- `db.delete_by_filter(...)` must surface invalid filter syntax as `io::ErrorKind::InvalidInput`
- `db.delete_by_filter(...)` must surface missing collection as `io::ErrorKind::NotFound`

This slice only transports those semantics over HTTP; it does not define a second validation model.

## Error Semantics

Return codes should follow existing daemon conventions:

- `200 OK` for success, including `deleted: 0`
- `400 BAD_REQUEST` for invalid filter expressions
- `404 NOT_FOUND` when the collection does not exist
- `500 INTERNAL_SERVER_ERROR` for other I/O or internal failures

Malformed JSON and missing required `filter` field should also return `400 BAD_REQUEST`.

An empty or whitespace-only `filter` string is treated as invalid input and should therefore also return `400 BAD_REQUEST`. The daemon does not need its own filter parser for this; it may rely on the same core validation path as other invalid filter strings.

All error responses continue using:

```json
{ "error": "..." }
```

Unlike the existing plain `Json<T>` mutation routes, this slice should explicitly wrap JSON extractor failures into the daemon `ErrorResponse` envelope instead of leaving axum's default rejection body in place. That keeps malformed request handling aligned with the existing custom `search_records` route, which already maps `JsonRejection` into `ErrorResponse`.

## Correctness Requirements

- The daemon route must preserve core latest-live delete semantics exactly.
- The daemon must not silently treat invalid filters as zero matches.
- A valid filter with no matches must return `200` with `deleted: 0`.
- The route must be backward compatible with the existing `/records` delete-by-id endpoint.
- The route must not introduce a second delete path with different semantics.

## Validation

Add `http_smoke` coverage for:

1. positive delete-by-filter on a real collection
2. repeated delete-by-filter returning `deleted: 0`
3. invalid filter returning `400` with daemon error envelope
4. empty or whitespace-only filter returning `400` with daemon error envelope
5. malformed JSON or missing `filter` field returning `400` with daemon error envelope
6. missing collection returning `404`
7. a latest-live / shadowing scenario on multi-version data, to prove the daemon transport is delegating to the same core semantics rather than applying a flat-layout delete
8. explicit regression coverage that the existing `DELETE /collections/:collection/records` delete-by-id route still behaves unchanged

## Risks

- This route is still scan-based because core `delete_by_filter` is scan-based; acceptable for this slice.
- The dedicated sub-route slightly expands the HTTP surface, but avoids ambiguity on the existing `/records` delete endpoint.
