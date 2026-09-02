# VRhino Native API v1 alpha

Status: canonical Public integration contract for Native API v1 alpha.

This is the normative client contract for Native API v1 alpha. The API is a
local application surface of the primary `vrhino` executable; it is not a
Shared Runtime or Backend interface.

## Compatibility and versioning

The HTTP breaking-major boundary is `/api/v1/`.

- Clients must ignore unknown additive response fields.
- Existing required fields and documented route semantics must not silently
  change within v1.
- A breaking request, response, routing, lifecycle, or replay change requires
  a future `/api/v2/`; VRhino does not use `/api/v1.0` or `/api/v1.1`.
- `vrhino.product.input-schema.v1` versions Product inputs independently of
  the HTTP route version.
- `schema_version` in version, model-list, and model-info documents versions
  those envelopes independently of the HTTP route version.

## Process, command, and local security

The server is part of the primary executable:

```text
vrhino [--cache-root PATH] serve [--host HOST] [--port PORT]
```

The defaults are `127.0.0.1:11435`. The process runs in the foreground, does
not daemonize, initializes no Backend and loads no model until a run executes.
Ctrl+C performs orderly shutdown and normally exits zero. There is no separate
`vrhino-server`, Python service, or Node service.

Native API v1 alpha is a **local API**. It has no authentication, TLS, or CORS
policy and is not intended for direct exposure to an untrusted network. A
non-loopback host is accepted only when explicitly selected and produces a
warning that authentication and TLS are absent. Browser-origin direct access
is not a v1-alpha target; desktop applications, native integrations, and local
trusted proxies can use the API directly.

## Route set

The complete v1 alpha route set is exactly:

```text
GET    /api/v1/version
GET    /api/v1/models
GET    /api/v1/models/{model}
POST   /api/v1/runs
GET    /api/v1/runs/{id}
DELETE /api/v1/runs/{id}
GET    /api/v1/runs/{id}/events
```

No other route is implied. JSON responses use
`application/json; charset=utf-8`. Errors never fall back to HTML.

## Version

`GET /api/v1/version` returns the canonical application/build projection
shared with `vrhino --version`:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | integer | Version response envelope, currently `1` |
| `version` | string | VRhino product version |
| `git_head` | string | Build source identity |
| `runtime_contract` | string | Runtime/driver contract identity |
| `model_package_schema` | integer | Supported package schema |
| `vrm.format_major` | integer | VRM format major |
| `vrm.format_minor` | integer | VRM format minor |
| `vrm.metadata_schema` | integer | VRM metadata schema |

## Installed models

`GET /api/v1/models` lists local installed packages only. It performs no
Registry lookup, pull, source acquisition, conversion, repair, or network
discovery. The response is deterministically sorted by canonical reference:

```json
{"models":[],"schema_version":1}
```

Each entry has exactly the current stable projection fields `reference`,
`namespace`, `name`, `version`, `architecture`, `product_family`, and
`installed`. Clients must still ignore future additive fields. Local cache,
manifest, CAS, checkout, and credential data are not projected.

`GET /api/v1/models/{model}` returns the same canonical projection as
`vrhino info MODEL --json`; HTTP maintains no second model schema. Successor
packages expose their ProductInputSchema and frozen profile. A legacy package
is readable with `product.input_schema=null` and
`product.frozen_profile=null`.

`{model}` is one percent-encoded path component. Encode a canonical
`namespace/name:version` exactly once: `/` as `%2F` and `:` as `%3A`.
Literal separators, invalid escapes, NUL, residual percent encoding,
query/fragment ambiguity, traversal-like identities, and non-canonical package
references fail closed. The canonical package parser remains authoritative
after exactly one transport decode.

## Run request

`POST /api/v1/runs` accepts one bounded JSON object:

```json
{
  "model": "namespace/name:version",
  "inputs": {},
  "parameters": {},
  "outputs": {}
}
```

`model` is required. `inputs`, `parameters`, and `outputs` are optional and
default to empty objects. Unknown top-level or group fields are rejected.

The resolved package's exact `vrhino.product.input-schema.v1` declaration is
the sole source for admitted names, types, required fields, validation, and
defaults. The HTTP layer has no model-name, architecture-name, seed-default,
or frozen-profile parameter table. Clients must inspect the model detail and
construct a request from its `product.input_schema`; they must not infer
undocumented parameters. Legacy packages with `input_schema=null` remain
readable but are not executable through Native API v1.

JSON integers are exact signed 64-bit values and never pass through `double`.
A Product `uint64` value above `INT64_MAX` uses a canonical unsigned decimal
string, including `"18446744073709551615"` for `UINT64_MAX`. Signs, leading
zeroes, non-digits, and overflow are rejected.

### Local media and outputs

`media.video`, `media.audio`, and explicit `media.mp4` values are absolute
local filesystem paths interpreted with the authority of the server process.
Video/audio inputs must already be regular files. Remote URLs, `data:` URLs,
base64, multipart upload, relative paths, and source acquisition are absent.
Existing Product preflight remains authoritative for decodability, FPS,
duration, and output publication.

An explicit output must be absolute and must not already exist, have an
existing `.partial`, or be reserved by another admitted Job. If output is
omitted, VRhino assigns:

```text
<VRHINO_HOME>/runs/<job-id>/output.mp4
```

The managed destination is separate from packages and CAS. Output publication
never silently overwrites. Reservations are process-memory admission state and
are released at terminal completion; a successful media artifact persists.

### Acceptance

Successful admission returns HTTP 202, a
`Location: /api/v1/runs/{id}` header, and the initial Job snapshot. Admission
is synchronous and bounded; Product execution is asynchronous and is never
awaited by the POST handler.

## Job status and lifecycle

The stable states are:

```text
queued
running
succeeded
failed
cancelled
```

`GET /api/v1/runs/{id}` returns a deterministic snapshot:

| Field | Presence | Meaning |
|---|---|---|
| `id` | required | Opaque process-local Job ID |
| `model` | required | Canonical installed model reference |
| `status` | required | One stable lifecycle state |
| `cancel_requested` | required | Cooperative cancellation requested |
| `output.path` | required | Explicit or managed local artifact path |
| `output.managed` | required | Whether VRhino generated the destination |
| `stage` | optional | Current/latest generic RunEvent stage |
| `progress` | optional | `completed`, `total`, and `unit` |
| `error` | failed only | Safe bounded `code` and `message` |

Progress units are `step`, `frame`, and `chunk`. Messages are human diagnostic
text and must never be parsed for semantics.

Job IDs are opaque. Their current encoding is not client semantics. The Job
registry, FIFO, snapshots, and event history exist only in server memory. The
registry retains at most 64 current/recent records and prunes terminal records;
a server restart invalidates every Job ID and all replay history. Successful
output files exist independently of this registry.

Exactly one Product worker executes the existing in-process Product path. Up
to four Jobs can currently wait in the bounded FIFO. Parallel Product/GPU
execution, priority, retry scheduling, model residency, and multi-GPU
scheduling are absent.

## Cancellation and shutdown

`DELETE /api/v1/runs/{id}` is idempotent for a known Job:

- queued: remove from FIFO, never invoke Product, transition to `cancelled`,
  return HTTP 200;
- running: request cooperative cancellation for only that Job and return HTTP
  202 while unwind is pending;
- terminal: preserve the immutable terminal state and return HTTP 200;
- unknown or expired ID: return HTTP 404 `job_not_found`.

DELETE never signals or terminates `vrhino serve`. On server shutdown, new
admission stops, queued Jobs cancel without execution, the active RunSession is
cooperatively cancelled, stream waiters wake, and workers join.

## NDJSON RunEvents

`GET /api/v1/runs/{id}/events` and
`GET /api/v1/runs/{id}/events?after=N` return:

```text
Content-Type: application/x-ndjson; charset=utf-8
Cache-Control: no-store
Transfer-Encoding: chunked
```

Each chunk contains one complete UTF-8 JSON object followed by `\n`. There is
no array wrapper, comma, heartbeat, ping, timestamp, or synthetic `done`
record. Every line contains `sequence` and `kind`, with optional `state`,
`stage`, `completed`, `total`, `unit`, and `message` copied from the canonical
RunEvent. HTTP does not assign another sequence.

RunEvent sequences are per Job `uint64` values beginning at 1 and
strictly increasing. Kinds are `state`, `stage`, `progress`, and `message`.
States, generic stages, and progress units use the canonical RunEvent names.
Optional message text is omitted when it is diagnostic-only, invalid UTF-8,
over 4 KiB, or would make the complete line exceed 16 KiB. Machine-semantic
fields are never truncated.

### Replay cursor

`after=N` is a strict base-10 unsigned sequence and means the client has
processed N. Zero is allowed. Signs, whitespace, hex, fractions, duplicates,
empty values, overflow, and trailing data are rejected.

- omitted `after`: start at the earliest event currently retained; no
  continuity before it is asserted;
- explicit `after=N`: emit only sequence values greater than N and assert
  continuity through N;
- N newer than the latest sequence: HTTP 400 `invalid_request`;
- N older than retained continuity: HTTP 409
  `event_history_truncated` before streaming, with numeric
  `details.first_available_sequence` and `details.latest_sequence`;
- terminal Job and N equal to latest: HTTP 200 with an empty NDJSON body.

The only event history is the bounded 256-event drop-oldest ring for that Job.
A stream that falls behind that ring closes rather than silently skipping or
growing a client queue. Reconnect with the last processed sequence; on 409,
read current Job status and attach without `after` if retained best-effort
events are still useful. A disconnected subscriber does not cancel or alter
the Job. After the existing terminal state event is written, the stream closes
normally.

## Error contract

Pre-stream errors use one envelope:

```json
{"error":{"code":"invalid_request","message":"request is invalid"}}
```

Only `event_history_truncated` currently adds a `details` object. Clients must
ignore unknown additive error fields and must not parse `message` for
semantics.

| HTTP | `error.code` | Route/condition | Retry guidance |
|---:|---|---|---|
| 400 | `invalid_request` | malformed target, model reference, Job ID, JSON, Product value, method body, content type/encoding, or future event cursor | change request |
| 400 | `model_unavailable` | installed legacy package has no ProductInputSchema v1 | not executable through v1 |
| 404 | `model_not_found` | exact local model is not installed | retry only after local installation changes |
| 404 | `job_not_found` | unknown, expired, or post-restart Job ID | resubmit if appropriate |
| 404 | `not_found` | unknown route | do not retry unchanged |
| 405 | `method_not_allowed` | known resource with unsupported method | change method |
| 409 | `conflict` | output exists, `.partial` exists, or destination is reserved | choose/free destination |
| 409 | `event_history_truncated` | explicit replay continuity is no longer retained | resynchronize; do not retry same cursor |
| 413 | `invalid_request` | request body exceeds the transport bound | reduce request |
| 414 | `invalid_request` | request target exceeds the transport bound | reduce target |
| 429 | `overloaded` | Product FIFO/registry or event-subscriber capacity reached | bounded backoff/retry |
| 500 | `internal` | unexpected sanitized server failure | retry only by policy |
| 503 | `model_unavailable` | Product execution is unavailable in this build or admission is shutting down | retry after server availability changes |

Execution failures after HTTP 202 are terminal Job state, not later HTTP
errors. A failed snapshot may contain `invalid_input`, `backend_unavailable`,
`out_of_memory`, `output_failed`, or `execution_failed`. Cancellation is the
terminal `cancelled` state. Messages are safe summaries, not exception text.

## Bounds

### V1 client interoperability limits

These limits are externally observable and clients must respect them. A future
v1 implementation may relax them additively, but must not silently reduce them
without compatibility review.

| Input/output | Current limit |
|---|---:|
| decoded canonical model reference | 192 bytes |
| HTTP request body | 65,536 bytes |
| NDJSON event line, including newline | 16,384 bytes |
| optional exposed RunEvent message | 4,096 bytes |

### Internal alpha implementation limits

These values bound resource use and are documented for operational planning;
clients must handle overload, eviction, and truncation rather than assume the
numbers are permanent v1 semantics.

| Resource | Current value |
|---|---:|
| request target | 2,048 bytes |
| header/line | 4,096 bytes |
| header count | 64 |
| JSON nesting depth | 32 |
| entries per JSON object/array | 1,024 |
| total JSON values | 4,096 |
| decoded bytes per JSON string | 32,768 |
| HTTP workers | 8 |
| queued HTTP tasks | 16 |
| simultaneous event subscribers | 4 |
| Product workers | 1 |
| queued Product Jobs | 4 |
| retained Job records | 64 |
| retained RunEvents per RunSession | 256 |
| read/write timeout | 5 seconds each |
| keep-alive idle/max requests | 2 seconds / 10 |

## Normative client rules

Clients must:

- ignore unknown additive response and error fields;
- use ProductInputSchema declarations rather than infer Product parameters;
- treat `input_schema=null` as non-executable through Native API v1;
- preserve canonical decimal-string `uint64` values without floating-point
  conversion;
- treat Job IDs as opaque and invalid after server restart;
- use RunEvent sequence only within one Job;
- handle `event_history_truncated` by resynchronizing through current status
  and the documented retained-stream policy;
- treat `message` as human text and never parse it for state or progress;
- tolerate asynchronous failure after a successful 202 admission;
- handle 429 through bounded backoff rather than assuming capacity.

## Explicit non-goals

Native API v1 alpha does not provide remote/cloud operation, multi-user
authentication, TLS termination, browser CORS policy, multipart upload, remote
media URLs, a persistent Job database, persistent event cursors, model
residency, a model scheduler, parallel GPU runs, multi-GPU scheduling, OpenAI
compatibility, pull routes, health/metrics/debug/OpenAPI routes, SDKs, or an
official WebUI. An OpenAPI artifact is intentionally deferred until the alpha
contract has completed its first publication review; the normative document
and checked-in examples are authoritative in the meantime.

## Examples

Privacy-safe machine-readable examples live under
`docs/api/examples/native-api-v1/`. They illustrate deterministic response
shapes but do not replace canonical projections or ProductInputSchema. Values
such as Git heads, Job IDs, model facts, and local paths are illustrative.
