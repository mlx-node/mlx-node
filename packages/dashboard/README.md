# @mlx-node/dashboard

The mlx-node Control Panel UI: Chat and Work with a local model, local models, agent sessions,
inference metrics, web search providers, and the paged-attention cold cache. Rendered inside the mlx-node desktop app — there is
no server and no port.

## Requirements

- Node.js ≥ 22.19.0
- Data written by `mlx agent` under `~/.mlx-node` (sessions, metric traces,
  cold-tier cache). Works with an empty home too — pages degrade to empty states.

## Usage

Not started on its own. The desktop app creates the runtime, serves the built SPA
over its `app://` scheme, and hands the page one end of a `MessageChannel`; the SPA
calls `connectDashboardApi(port)` and every API call rides it.

See [`docs/dashboard.md`](../../docs/dashboard.md) for the eleven routes, Chat and Work, web search providers, data sources, the live channel,
the design system, the dev harness, `persistPagedCache`, and the env vars.

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
- **`node:sqlite` + Drizzle ORM** index over metric traces and the runtime series.
- **Live push.** A `watch` request over the port gets `live` events: heartbeat snapshots and "new data" after a
  finished turn. No page polls for live state; the one timer re-reads the Coding Agents status every 30 s
  while the window is visible.
- **Chat and Work.** `src/chat/` starts the bundled `mlx-agent` on demand, one process per mode, over ACP on stdio.
  Chat has no shell, an empty `HOME` and the app's `mlx_web` search / fetch tools; Work runs in a folder the user
  chooses, with the real `HOME` and `~/.mlx-agent`, and asks through a card in the page before a command or an edit that no saved grant allows.
  Provider keys live in the macOS Keychain; the provider skills ship in `skills/`.
- **React + Vite + Arc** SPA in `ui/`, built to `web/`: vendored Arc free components in `ui/src/components/arc/`,
  own components in `ui/src/components/arc-x/`, a locked Tailwind layer for page layout, fonts bundled.

## Layout

```
packages/dashboard/
├── src/
│   ├── runtime.ts    the transport-independent runtime: call / subscribe / drain
│   ├── rpc/          MessagePort bridge (port adapters, wire protocol, client, host)
│   ├── api/          route table + handlers + the typed error model
│   ├── worker/       the SQLite worker thread and its message contract
│   ├── db/           Drizzle schema + node:sqlite driver
│   ├── ingest/       metric traces + runtime series → sqlite (tail reads)
│   ├── live.ts       live watcher: heartbeat files + supervisor.json → live events
│   ├── models.ts     discover / size / quant / delete
│   ├── download.ts   @huggingface/hub runner with structured progress
│   ├── catalog.ts    installed-state overlay on the agent's catalog
│   ├── chat/         Chat and Work: agent processes, ACP, permissions, providers, the mlx_web server
│   └── cache.ts      cold-tier dir scan + management
├── skills/           provider skill folders, pinned, with manifest.json
├── scripts/          dev harness: dev, dev:fixture, dev:shots, the /__gallery page (never shipped)
└── ui/               Vite app (React + Arc + locked Tailwind) → built to web/
```

## Build

```bash
yarn workspace @mlx-node/dashboard build       # tsc -b + vite build (ui → web)
yarn workspace @mlx-node/dashboard build:ui    # ui only
yarn workspace @mlx-node/dashboard dev         # dev harness on a fixture home: http://127.0.0.1:5179/
```

The package publishes `dist` and `web` (the built SPA).
