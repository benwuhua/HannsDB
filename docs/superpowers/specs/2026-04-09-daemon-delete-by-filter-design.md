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

## Error Semantics

Return codes should follow existing daemon conventions:

- `200 OK` for success, including `deleted: 0`
- `400 BAD_REQUEST` for invalid filter expressions
- `404 NOT_FOUND` when the collection does not exist
- `500 INTERNAL_SERVER_ERROR` for other I/O or internal failures

All error responses continue using:

```json
{ "error": "..." }
```

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
4. missing collection returning `404`
5. a latest-live / shadowing scenario on multi-version data, to prove the daemon transport is delegating to the same core semantics rather than applying a flat-layout delete

## Risks

- This route is still scan-based because core `delete_by_filter` is scan-based; acceptable for this slice.
- The dedicated sub-route slightly expands the HTTP surface, but avoids ambiguity on the existing `/records` delete endpoint.
