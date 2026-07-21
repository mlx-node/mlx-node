# @mlx-node/dashboard

Local web dashboard for mlx-node: browse local models, agent sessions, inference
metrics, and the paged-attention cold cache. Started with `mlx dashboard`.

## Requirements

- Node.js ≥ 22.19.0
- Data written by `mlx agent` under `~/.mlx-node` (sessions, metric traces,
  cold-tier cache). Works with an empty home too — pages degrade to empty states.

## Usage

```bash
mlx dashboard                 # start on 127.0.0.1:6590 and open a browser
mlx dashboard --no-open       # start without a browser
mlx dashboard --port 8080     # pick a port
```

See [`docs/dashboard.md`](../../docs/dashboard.md) for flags, pages, data sources,
`persistPagedCache`, the `MLX_AGENT_METRICS` / `MLX_COLD_CACHE_DIR` env vars, and the
security model.

## Design

- **Separate viewer process.** It never links the native addon — no Metal init,
  instant start. All data comes from disk; the JSONL under `~/.mlx-node` is the
  source of truth, and the SQLite index (`~/.mlx-node/dashboard.db`) is disposable
  and rebuilt on demand.
- **`node:http`** server: static SPA + JSON `/api` + SSE. Binds `127.0.0.1` by
  default; mutating requests are guarded by a local-origin check (no auth).
- **`node:sqlite` + Drizzle ORM** index over pi session JSONL and metric traces.
- **React + Vite + Tailwind + shadcn/ui** SPA in `ui/`, built to `web/`.

## Layout

```
packages/dashboard/
├── src/
│   ├── server.ts     node:http; static + /api + SSE; local-origin guard
│   ├── api.ts        route handlers
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
