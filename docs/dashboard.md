# Dashboard (Control Panel window)

Browse local models, `mlx agent` sessions, inference metrics, and the
paged-attention cold cache. It ships inside `@mlx-node/dashboard` and is the Control Panel
window of the **mlx-node desktop app** — open it from the tray.

**There is no `mlx dashboard` command and no HTTP server.** Both were removed once
the app replaced them: the SPA is served from a custom `app://` scheme and speaks to
the runtime over a MessagePort. That deletes a whole class of exposure rather than
guarding it — there is no port to bind, nothing to firewall, and no `--host` that can
put an unauthenticated control panel API on a LAN.

The dashboard still never links the native addon (no Metal init, instant start), and
all data still comes from disk under `~/.mlx-node`.

The [macOS desktop app](../packages/desktop/README.md) downloads app updates in the
background. Use **Restart to Update…** from the tray when ready, or **Check for Updates…**
to check manually.

## Where it runs

```
CONTROL PANEL  utilityProcess               no native addon
  createDashboardRuntime()
    main thread   DownloadManager + the RPC transport      async only
    worker thread DashboardDb + every synchronous FS walk
  ⟵ MessagePort ⟶ Control Panel renderer   (MAIN brokers the port once, then leaves)
```

`DashboardDb` and the synchronous filesystem walks own a `node:worker_threads`
worker, because `node:sqlite` and drizzle are synchronous: a trivial call issued
during a 1.5 s query used to wait 1449 ms. With the split, a download-progress call
made during a heavy query returns in 0.2 ms instead of 296 ms.

Route ownership is data on the route (`mainRoute` / `workerRoute`), so a new route
cannot silently land on the wrong thread. Only the four download routes and the
catalog update check stay on the transport thread, because network I/O and progress
events are what must never stall — and for the same reason none of them touches the
filesystem: the update check answers with upstream shas alone.

## Using it as a library

```ts
import { createDashboardRuntime } from '@mlx-node/dashboard';
import { serveRuntimeOverPort } from '@mlx-node/dashboard';

const runtime = createDashboardRuntime({ modelsDir, dbPath });
// `call({method, path})` is the whole API surface; it never rejects for an API
// failure, it returns a `{ok:false, status, code, message}` envelope.
```

## Pages

| Page     | Content                                                                                                   | Actions                                                                  |
| -------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Overview | Stat tiles (models/disk, cache size + hit rate, active downloads)                                         | —                                                                        |
| Models   | Local table: name, family, quant, size, ctx window                                                        | delete; install from the recommended catalog (live port-pushed progress) |
| Sessions | `mlx agent` sessions: list (search, directory filter), detail with transcript, per-turn charts, subagents | copy resume command, rename, delete (refused while live)                 |
| Metrics  | Tokens/day (in/out/cached), tok/s + TTFT per model, MTP acceptance, model share                           | date range                                                               |
| Cache    | Cold-tier disk usage vs quota, entry count + age histogram, hit/miss trend                                | clear all, evict older-than-N-days                                       |

Every catalog entry installs its Unsloth UD-Q4_K_XL GGUF (one variant out of
the repo's dozens) plus the base-model tokenizer sidecars fetched from the
entry's `assetsRepo`; the download runner glob-filters the manifest and stages
the sidecars before publishing. No current entry carries a `draft` companion —
Qwen3.8-27B ships its MTP speculative weights inside the same repo and the
native loader auto-detects them. The card UI still renders a draft companion
(with divider, own install button, progress, cancel, and update controls)
whenever a catalog entry carries one, for example a manually installed
**DFlash2** draft: the roughly 3.85 GB checkpoint lives at
`qwen3.8-27b-dflash2` in the configured models directory, and compatible
Qwen3.8-27B targets find it automatically on their next load. It is not another
chat model. Draft-only checkpoints are excluded from the local-model table and
model counts. Their bytes still contribute to disk usage on Models and
Overview. Expand
**Companion weights** in the local storage panel to inspect or delete them;
deleting a separate companion leaves the target model in place.

## Data sources

All state lives on disk; SQLite is only an index.

| Source                          | Location                                                                                            | Role                                                             |
| ------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `MetricsTrace` JSONL            | `~/.mlx-node/metrics/traces/<date>-<pid>.jsonl`                                                     | Per-turn throughput / cold-cache deltas (see Phase B below)      |
| Disposable SQLite index         | `~/.mlx-node/dashboard.db`                                                                          | Index over the JSONL above; **rebuilt on next start if deleted** |
| Local models                    | `--models-dir` → `MLX_MODELS_DIR` → `modelsDir` in `~/.mlx-node/config.json` → `~/.mlx-node/models` | Model list, size on disk, family/quant/ctx from `config.json`    |
| Cold tier (paged prefix blocks) | `~/.mlx-node/cache/paged/v1/` (`MLX_COLD_CACHE_DIR` override)                                       | Cache page scan + management                                     |
| `mlx agent` sessions            | `<agentHome>/sessions/<encoded cwd>/<id>/` (`agentHome` = `GROK_HOME` or `~/.mlx-agent`)            | Sessions page; read from files on each call, never indexed       |

The SQLite index is **disposable**: deleting `dashboard.db` loses nothing — it is
rebuilt from the JSONL on next start. Ingest runs on boot, then incrementally by
file mtime every 30 s (plus a manual `POST /api/ingest`). Trace files older than
30 days are pruned after ingest.

Trace records are metadata-only: every field is a number, a small enumerated
string, or an identifier — never prompt content, tool arguments, or model output.

## Sessions

Read straight from the grok-build session folders on the database worker. No
SQLite table: each call re-lists the folders and reuses parsed `summary.json`,
`meta.json` and `updates.jsonl` while their (mtime, size) is unchanged.

```
<agentHome>/
  active_sessions.json            TUI registry {session_id, pid}; read only, no lock
  sessions/<encoded cwd>/<id>/
    summary.json                  title, cwd, model, kind; rename rewrites it
    updates.jsonl                 transcript + turn_completed metrics (rewinds applied)
    subagents/<child>/meta.json   the ONLY parent -> child link
```

- List: top-level sessions only. Subagent children, `headless` runs and empty
  husks are left out; a child is reachable from its parent's detail page.
- Rename does what grok's `/rename` does to a dormant session: `generated_title`,
  `session_summary` only while it is empty, `title_is_manual: true`,
  `updated_at`, plus `title_refresh_idx = 2`. Title: 1..100 characters after
  removing control and bidi characters and trimming. Written as temp file +
  fsync + rename, pretty JSON like serde.
- Delete removes the session folder and its subagent children (followed through
  `meta.json` whose `parent_session_id` matches, looked up by id in every cwd
  folder). Like grok it leaves forks, `prompt_history.jsonl`,
  `session_search.sqlite` and the cwd folder alone.
- Delete first renames each folder to `<cwd>/.deleting-<uuid>`, deepest child
  first and the requested session last, then removes those folders. Dot names are
  skipped by our scan and by grok, so the rename is the delete:

  ```
  rename fails part way   -> requested session still listed; retry deletes the rest
  crash between renames   -> same; the renamed children are just gone
  removal fails (EACCES)  -> 200; .deleting-* stays, invisible, retried on the next delete
  ```

Write guards (all answer 409 and change nothing; checked again right before the write):

| Guard        | Rule                                                                                                                                                                         |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| registered   | the id, or any subagent ancestor, is in `active_sessions.json` with a live pid (`kill(pid, 0)`, EPERM = alive)                                                               |
| recent write | any regular file in the session folder changed in the last 30 s (backup for mlx-agent < v0.3.0, where ACP and `-p` runs never register); our own rename does not count       |
| open turn    | the transcript ends inside a turn and was written in the last 10 min                                                                                                         |
| unreadable   | `active_sessions.json` cannot be parsed                                                                                                                                      |
| containment  | the folder must be a real `<sessions>/<cwd>/<id>` directory (no symlink at either level, realpath inside the sessions root); ids with `/`, `\`, NUL or a leading `.` are 400 |

grok's `*.lock` files are never opened: grok blocks on `LOCK_EX`, so a lock held
from Node would stall the agent. From mlx-agent v0.3.0 every resident session
(TUI, `-p`, ACP, subagents) has a row in `active_sessions.json`, so an idle holder
is detected. With an older binary, an idle ACP/headless holder with no recent
writes cannot be detected.

## Persist paged cache (`persistPagedCache`)

Phase A wires the existing `ColdCacheManager` SSD cold tier into paged inference.
When enabled, full paged prefix blocks are captured to the cold tier on publish and
restored on a hot-cache miss before falling back to a normal prefill.

- Library default is **off** (`persistPagedCache` per-model config field, TS
  `persistPagedCache`). `mlx agent` turns it **on by default for every allowlisted
  family** — `qwen3`, `qwen3_5`, `qwen3_5_moe`, `gemma4`, `lfm2`, and
  `lfm2_moe` — via a temporary config
  overlay, so a warm prefix survives a restart without opting in. Disable with
  `mlx agent --no-persist-cache`. The overlay writes an EXPLICIT value, so the
  flag beats whatever the checkpoint's `config.json` hard-codes; a family off the
  allowlist is handed no policy at all and the overlay never touches its field.
- **Restore is gated to an allowlist: `qwen3`, `qwen3_5`, `qwen3_5_moe`, `gemma4`, `lfm2`, and `lfm2_moe`.**
  Dense qwen3 sizes its pool over every layer, so paged blocks fully determine its
  layer state. The other three are hybrid — their pools cover the full-attention
  layers only — but each now persists its out-of-pool state as a cold-tier
  **sidecar**: grouped sliding-window K/V for gemma4, GDN recurrent state for
  qwen3.5/3.6 dense and MoE, and ShortConv recurrent state for LFM2/2.5. A
  `ColdSidecarPolicy` makes the restore
  reconcile DOWN to a boundary a validated sidecar actually backs (vLLM's per-group
  rule), and the `aux_prefix_unbacked` latch fails closed if an in-process hot hit
  would resume a K/V prefix whose out-of-pool half is missing. Gemma4 instead
  installs every paged group at one validated boundary or resets them all to zero.
  A family joins the allowlist only after its restart-parity
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

| Var                                | Effect                                                                                                                                                                       |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MLX_AGENT_METRICS=0`              | Kill switch for the always-on metrics sink (`0` / `false` / `off`). Without it, the agent writes a trace per turn — no metrics page data means no traces to ingest.          |
| `MLX_COLD_CACHE_DIR`               | Parent dir for the cold tier (operates in its `mlx-paged-v1` child). Default `~/.mlx-node/cache/paged/v1`.                                                                   |
| `MLX_COLD_CAPTURE_BLOCKS_PER_TURN` | Blocks one turn's cold-tier capture walk may persist. Default `128` (2048 tokens at block size 16). Raising it covers a long prompt in fewer turns at the cost of turn tail. |
| `MLX_COLD_CAPTURE_BUDGET_MS`       | Wall-clock ceiling on one capture walk. Default `250`. A walk that hits it warns and ratchets less than configured.                                                          |
| `MLX_MODELS_DIR`                   | Local models directory when `--models-dir` is omitted.                                                                                                                       |
| `GROK_HOME`                        | `mlx agent` config home read by the Sessions page (`agentHome` runtime option overrides). Default `~/.mlx-agent`.                                                            |

## Security model

The HTTP attack surface no longer exists. There is no listening socket, so there is
nothing to bind, nothing to firewall, and no host/origin guard to get wrong — the
class of bug is deleted rather than mitigated. What remains:

- **The renderer is sandboxed** (`contextIsolation: true`, `nodeIntegration: false`,
  `sandbox: true`) and reaches the runtime only through a transferred MessagePort.
- **A strict CSP ships as a response header** on `app://`. `script-src` is `'self'`
  with no `unsafe-inline` and no `unsafe-eval`; `connect-src` is `'self'` only,
  because the SPA speaks over a port and has no reason to reach the network at all.
  The scheme is registered with `bypassCSP: false`, or the header would be decorative.
- **`app://` serves only inside the bundle.** A miss under `/assets/` returns 404
  rather than falling back to `index.html`, so a stale or mis-copied bundle fails
  loudly instead of returning HTML labelled `text/javascript`.
- **Mutations are still the dangerous part** — model delete, cache clear, session rename/delete —
  and they are reachable by anything that gets the port. The port is minted
  by MAIN, transferred once to the Control Panel window it created, and consumed on receipt.

## Known limitation

The Metrics page's "trend per model" charts render as **per-model comparison bars,
not time-series lines**, because `/api/metrics/overview` returns per-model aggregates
rather than per-day-per-model rows. A future follow-up would add a daily-bucketed
per-model query to plot true trends over time.

## API

The same route table, addressed the same way, over a MessagePort instead of HTTP.
`runtime.call({ method, path })` is the whole surface; it **never rejects for an API
failure**, it returns `{ok: true, status, body}` or `{ok: false, status, code, message}`.
That envelope is structured-clonable by design — it is what makes the port transport
cheap, and it gives code written against the in-process runtime an identical signature
over a port.

`GET /health` returns `{ status, modelsDir, tracesDir }`.

| Route                       | Method       | Purpose                                                                                |
| --------------------------- | ------------ | -------------------------------------------------------------------------------------- |
| `/api/models`               | GET          | Local models: name, path, family, quant, size, ctx window                              |
| `/api/models/:name`         | DELETE       | Delete a model dir (path-checked)                                                      |
| `/api/catalog`              | GET          | Recommended catalog + installed/installable state                                      |
| `/api/catalog/updates`      | GET          | Upstream sha per catalog repo; the page joins it against `/api/catalog`                |
| `/api/downloads`            | GET / POST   | List active jobs / start a catalog download                                            |
| `/api/downloads/:id/events` | GET          | Not served over the port — use `runtime.subscribe(jobId, fn)`; answers 503             |
| `/api/metrics/overview`     | GET          | Aggregates: tokens/day, tok/s + TTFT, MTP acceptance, share                            |
| `/api/cache`                | GET / DELETE | Cold-tier scan / clear all or evict older-than-N-days                                  |
| `/api/ingest`               | POST         | Trigger an incremental rescan                                                          |
| `/api/sessions`             | GET          | `?q=&cwd=&limit=&offset=` → `{sessions, total, cwds}`; limit 1..500, default 100       |
| `/api/sessions/:id`         | GET          | `{session, transcript, turns, transcriptError?, resumeCommand}`; subagent ids work too |
| `/api/sessions/:id`         | PATCH        | Body `{title}` → `{id, title}`; 400 bad title, 404, 409 live/recent/changed            |
| `/api/sessions/:id`         | DELETE       | `{deleted: [id, ...children]}`; 404, 409 when it or any child is live                  |
