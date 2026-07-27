# Dashboard (`mlx dashboard`)

A local web dashboard for browsing models, agent sessions, inference metrics, and
the paged-attention cold cache. It ships inside `@mlx-node/dashboard` and starts
with `mlx dashboard`.

The dashboard is a **separate viewer process**. It never links the native addon
(no Metal init, instant start). All data comes from disk under `~/.mlx-node`;
live-ish behavior is polling + SSE. Requires Node.js ≥ 22.19.

```bash
mlx dashboard                       # start on 127.0.0.1:6590, open a browser
mlx dashboard --port 8080           # pick a port
mlx dashboard --no-open             # do not launch a browser
mlx dashboard --host 0.0.0.0        # bind elsewhere (LOUD warning; no auth)
mlx dashboard --db ./scratch.db     # override the SQLite index path
mlx dashboard --models-dir ./models # read local models from a specific dir
```

| Flag           | Default                    | Purpose                                                               |
| -------------- | -------------------------- | --------------------------------------------------------------------- |
| `--port`       | `6590`                     | Port to listen on (`0` = ephemeral)                                   |
| `--host`       | `127.0.0.1`                | Host to bind; any non-loopback host prints a no-auth exposure warning |
| `--no-open`    | (opens a browser)          | Do not open the dashboard in a browser                                |
| `--db`         | `~/.mlx-node/dashboard.db` | Override the disposable SQLite index path                             |
| `--models-dir` | `~/.mlx-node/models`       | Directory to read local models from                                   |
| `-h`, `--help` | —                          | Print help                                                            |

## Pages

| Page           | Content                                                                                                 | Actions                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Overview       | Stat tiles (models/disk, sessions, tokens 7d, cache size + hit rate), recent sessions, active downloads | —                                                                |
| Models         | Local table: name, family, quant, size, ctx window                                                      | delete; install from the recommended catalog (live SSE progress) |
| Sessions       | Table with search + filters (cwd, model, date)                                                          | open, rename (idle only, see below), delete, copy resume command |
| Session detail | Transcript (collapsible tool calls) + per-turn tokens / tok-s chips + charts                            | —                                                                |
| Metrics        | Tokens/day (in/out/cached), tok/s + TTFT per model, MTP acceptance, model share                         | date range                                                       |
| Cache          | Cold-tier disk usage vs quota, entry count + age histogram, hit/miss trend                              | clear all, evict older-than-N-days                               |

## Data sources

All state lives on disk; SQLite is only an index.

| Source                          | Location                                                                                            | Role                                                             |
| ------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| pi session JSONL                | `~/.mlx-node/agent/sessions/--<cwd>--/*.jsonl`                                                      | Transcripts + per-turn token usage; the source of truth          |
| `MetricsTrace` JSONL            | `~/.mlx-node/metrics/traces/<date>-<pid>.jsonl`                                                     | Per-turn throughput / cold-cache deltas (see Phase B below)      |
| Disposable SQLite index         | `~/.mlx-node/dashboard.db`                                                                          | Index over the JSONL above; **rebuilt on next start if deleted** |
| Local models                    | `--models-dir` → `MLX_MODELS_DIR` → `modelsDir` in `~/.mlx-node/config.json` → `~/.mlx-node/models` | Model list, size on disk, family/quant/ctx from `config.json`    |
| Cold tier (paged prefix blocks) | `~/.mlx-node/cache/paged/v1/` (`MLX_COLD_CACHE_DIR` override)                                       | Cache page scan + management                                     |

The SQLite index is **disposable**: deleting `dashboard.db` loses nothing — it is
rebuilt from the JSONL on next start. Ingest runs on boot, then incrementally by
file mtime every 30 s (plus a manual `POST /api/ingest`). Trace files older than
30 days are pruned after ingest.

Trace records are metadata-only: every field is a number, a small enumerated
string, or an identifier — never prompt content, tool arguments, or model output.
Records join to session entries via `mlxTraceId`, a UUID the agent provider mints
per turn and stamps on both the trace record and the pi assistant message.

## Persist paged cache (`persistPagedCache`)

Phase A wires the existing `ColdCacheManager` SSD cold tier into paged inference.
When enabled, full paged prefix blocks are captured to the cold tier on publish and
restored on a hot-cache miss before falling back to a normal prefill.

- Library default is **off** (`persistPagedCache` per-model config field, TS
  `persistPagedCache`). `mlx agent` turns it **on by default for every allowlisted
  family** — `qwen3`, `qwen3_5`, `qwen3_5_moe`, `gemma4` — via a temporary config
  overlay, so a warm prefix survives a restart without opting in. Disable with
  `mlx agent --no-persist-cache`. The overlay writes an EXPLICIT value, so the
  flag beats whatever the checkpoint's `config.json` hard-codes; a family off the
  allowlist is handed no policy at all and the overlay never touches its field.
- **Restore is gated to an allowlist: `qwen3`, `qwen3_5`, `qwen3_5_moe`, `gemma4`.**
  Dense qwen3 sizes its pool over every layer, so paged blocks fully determine its
  layer state. The other three are hybrid — their pools cover the full-attention
  layers only — but each now persists its out-of-pool state as a cold-tier
  **sidecar**: sliding-window `RotatingKVCache` state for gemma4, GDN recurrent
  state for qwen3.5/3.6 dense and MoE. A `ColdSidecarPolicy` makes the restore
  reconcile DOWN to a boundary a validated sidecar actually backs (vLLM's per-group
  rule), and the `aux_prefix_unbacked` latch fails closed if an in-process hot hit
  would resume a K/V prefix whose out-of-pool half is missing. `lfm2` / `lfm2_moe`
  keep short-conv state outside the pool with no serialization path, so they are
  **not** restore-eligible. A family joins the allowlist only after its restart-parity
  gate passes on real weights with `hits > 0` and `corruptions == 0` — see
  `docs/paged-cache.md`. The allowlist is enforced natively in
  `cold_tier::resolve_persist_cold`, so a family that is off it never persists or
  restores even under an explicit `persistPagedCache` or `MLX_PERSIST_PAGED_CACHE=1`.
- The tier is fail-open: any validation or I/O failure falls through silently to a
  fresh prefill. Quota is 10 % of FS capacity capped at 100 GiB, LRU by rebuilt
  mtime recency. All destructive I/O is descriptor-relative and no-follow, contained
  to a managed `mlx-paged-v1` child.
- **Cold-tier eviction is per-block global-LRU and NOT chain-aware.** Prefix chains
  are written head-to-tail, so under sustained quota pressure the oldest blocks —
  the root/head of a chain — are evicted first. Because restore stops at the first
  missing block, an evicted head makes the rest of that chain unrestorable while its
  suffix blocks still occupy quota; the tier then falls open to a normal prefill
  (correct, just no speedup). The common case — a working set within quota — is
  unaffected. Chain-aware eviction (evict leaves/tails first, or cascade a head's
  descendants) is a future improvement.
- The quota is a **per-process best-effort cap**, not a strict cross-process limit:
  each `mlx agent` process enforces it against its own startup scan, so several
  processes sharing one root before either writes can transiently exceed it (up to
  ~N×quota for N concurrent writers). It self-corrects — the next process to scan the
  root evicts LRU back down to quota on its first write — and the free-space floor is
  enforced against a live `statvfs` re-sampled per eviction. Strict cross-process
  quota enforcement (interprocess locking) is out of scope for v1.
- `MLX_COLD_CACHE_DIR=<parent>` relocates the tier; the cache operates in a
  `mlx-paged-v1` child of that parent (never the parent verbatim). Default root is
  `~/.mlx-node/cache/paged/v1`. `coldCacheStats()` (NAPI) exposes cumulative
  counters, root path, and quota.

## Environment variables

| Var                   | Effect                                                                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MLX_AGENT_METRICS=0` | Kill switch for the always-on metrics sink (`0` / `false` / `off`). Without it, the agent writes a trace per turn — no metrics page data means no traces to ingest. |
| `MLX_COLD_CACHE_DIR`  | Parent dir for the cold tier (operates in its `mlx-paged-v1` child). Default `~/.mlx-node/cache/paged/v1`.                                                          |
| `MLX_COLD_CAPTURE_BLOCKS_PER_TURN` | Blocks one turn's cold-tier capture walk may persist. Default `128` (2048 tokens at block size 16). Raising it covers a long prompt in fewer turns at the cost of turn tail. |
| `MLX_COLD_CAPTURE_BUDGET_MS` | Wall-clock ceiling on one capture walk. Default `250`. A walk that hits it warns and ratchets less than configured. |
| `MLX_MODELS_DIR`      | Local models directory when `--models-dir` is omitted.                                                                                                              |

## Security model

- Binds `127.0.0.1` by default. Binding any other host prints a loud warning: the
  dashboard has **no authentication** and exposes local models, sessions, and cache
  to anyone who can reach the address.
- **No auth.** Mutating (non-GET/HEAD) requests are guarded by a local-origin check:
  under the default loopback bind the `Host` header must name a loopback host
  (optionally with the server port), and any present `Origin` must be an `http`
  loopback origin. An explicit non-loopback `--host` bind intentionally widens the
  accepted `Host` set to that bound interface's address(es) — the no-auth exposure
  noted above — while still rejecting a rebound foreign host. This blocks drive-by
  CSRF / DNS-rebinding against the loopback port. GET/HEAD (static assets + read
  endpoints) are unguarded.
- Destructive ops (delete model/session, clear/evict cache) require UI confirmation
  and resolve paths strictly inside their managed roots before any removal.
- The per-model **size on disk** shown in the Models table (and the delete-confirmation
  "(X GB)" hint) is a **best-effort, display-only estimate**, not a security boundary:
  the directory walk no-follows the final component but traverses lexical paths, so a
  concurrent symlink swap of an ancestor directory in the user's own local store could
  redirect sizing onto an external tree. The worst outcome is a wrong size number —
  deletion never uses the size, only the path-checked model name — and a fully robust
  defense would need descriptor-relative traversal that `node:fs` does not expose.

## Known limitation

The Metrics page's "trend per model" charts render as **per-model comparison bars,
not time-series lines**, because `/api/metrics/overview` returns per-model aggregates
rather than per-day-per-model rows. A future follow-up would add a daily-bucketed
per-model query to plot true trends over time.

## HTTP API

Read-only `GET` endpoints plus a few guarded mutations. `GET /health` returns
`{ status, modelsDir, sessionsRoot, tracesDir }`.

| Route                       | Method       | Purpose                                                       |
| --------------------------- | ------------ | ------------------------------------------------------------- |
| `/api/models`               | GET          | Local models: name, path, family, quant, size, ctx window     |
| `/api/models/:name`         | DELETE       | Delete a model dir (path-checked)                             |
| `/api/catalog`              | GET          | Recommended catalog + installed/installable state             |
| `/api/downloads`            | GET / POST   | List active jobs / start a catalog download                   |
| `/api/downloads/:id/events` | GET (SSE)    | Per-file + byte progress, resume-aware                        |
| `/api/sessions`             | GET          | Indexed session list; search + filters                        |
| `/api/sessions/:id`         | GET          | Transcript (active branch) + per-turn usage                   |
| `/api/sessions/:id`         | PATCH/DELETE | Rename (refused while active, see below) / delete file + rows |
| `/api/sessions/:id/metrics` | GET          | Joined trace metrics for the session                          |
| `/api/metrics/overview`     | GET          | Aggregates: tokens/day, tok/s + TTFT, MTP acceptance, share   |
| `/api/cache`                | GET / DELETE | Cold-tier scan / clear all or evict older-than-N-days         |
| `/api/ingest`               | POST         | Trigger an incremental rescan                                 |

**Rename is durable but idle-only.** A rename appends a `session_info` entry to the
pi session JSONL (the source of truth) — it is _not_ stored index-only, because the
SQLite index is disposable and would lose the name on the next rebuild. pi has no
cross-process lock, so if a live agent (a separate process) appends a turn between
the dashboard's snapshot of the file and its append, one of the two writes is
orphaned on the next resume. To avoid that race the rename is **refused with 409
while the session file appears active** (modified within the last ~30 s); rename an
idle session instead. This is a best-effort product rule, not a hard guarantee: a
session that goes idle and is then written concurrently, or one that goes live
inside the brief check→append window, can still race — the pre-check removes the
realistic reachability, it does not eliminate the theoretical race.

## Design

The full design and phasing (Phase A cold-cache wiring, Phase B metrics sink,
Phase C dashboard package) live in
`docs/superpowers/specs/2026-07-20-mlx-dashboard-design.md`.
