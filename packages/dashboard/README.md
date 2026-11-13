# @mlx-node/dashboard

The mlx-node Control Panel UI: browse local models, agent sessions, inference metrics, and
the paged-attention cold cache. Rendered inside the mlx-node desktop app — there is
no server and no port.

## Requirements

- Node.js ≥ 22.19.0
- Data written by `mlx agent` under `~/.mlx-node` (sessions, metric traces,
  cold-tier cache). Works with an empty home too — pages degrade to empty states.

## Usage

Not started on its own. The desktop app creates the runtime, serves the built SPA
over its `app://` scheme, and hands the page one end of a `MessageChannel`; the SPA
calls `connectDashboardApi(port)` and every API call rides it.

See [`docs/dashboard.md`](../../docs/dashboard.md) for pages, data sources,
`persistPagedCache`, and the `MLX_AGENT_METRICS` / `MLX_COLD_CACHE_DIR` env vars.

## Design

- **Separate viewer process.** It never links the native addon — no Metal init,
  instant start. All data comes from disk; the JSONL under `~/.mlx-node` is the
  source of truth, and the SQLite index (`~/.mlx-node/dashboard.db`) is disposable
  and rebuilt on demand.
- **No HTTP.** The API is a transport-independent route table (`src/api/`) driven
  through `src/runtime.ts`. `src/rpc/` bridges it onto a MessagePort. Nothing
  listens on a socket, so there is nothing to firewall and no origin to guard.
- **Two threads.** The runtime thread owns downloads and the port; a
  `node:worker_threads` worker owns the synchronous SQLite index, so a heavy query
  cannot stall download progress. Ownership is declared per route in
  `src/api/routes.ts`.
- **`node:sqlite` + Drizzle ORM** index over pi session JSONL and metric traces.
- **React + Vite + Tailwind + shadcn/ui** SPA in `ui/`, built to `web/`.

## Layout

```
packages/dashboard/
├── src/
│   ├── runtime.ts    the transport-independent runtime: call / subscribe / drain
│   ├── rpc/          MessagePort bridge (port adapters, wire protocol, client, host)
│   ├── api/          route table + handlers + the typed error model
│   ├── worker/       the SQLite worker thread and its message contract
│   ├── db/           Drizzle schema + node:sqlite driver
│   ├── ingest/       pi JSONL + metric traces → sqlite
│   ├── models.ts     discover / size / quant / delete
│   ├── download.ts   @huggingface/hub runner with structured progress
│   ├── catalog.ts    installed-state overlay on the agent's catalog
│   └── cache.ts      cold-tier dir scan + management
└── ui/               Vite app (React + Tailwind + shadcn/ui) → built to web/
```

## Build

```bash
yarn workspace @mlx-node/dashboard build       # tsc -b + vite build (ui → web)
yarn workspace @mlx-node/dashboard build:ui    # ui only
```

The package publishes `dist` and `web` (the built SPA).
