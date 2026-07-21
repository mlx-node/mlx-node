/**
 * Dev helper: serve the built dashboard SPA (`packages/dashboard/web`) through
 * the C5 `node:http` server, so a production UI build can be previewed without
 * the Vite dev server. Run with `oxnode packages/dashboard/scripts/dev-dashboard.ts`.
 *
 * Honors `PORT` (default 6590). All data roots use the server's real defaults
 * under `$HOME/.mlx-node`; point `HOME` elsewhere to sandbox a throwaway run.
 * Not published — the package ships only `dist` and `web`.
 */

import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { startDashboardServer } from '../src/index.ts';

const here = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(here, '..', 'web');
const port = process.env.PORT !== undefined ? Number(process.env.PORT) : 6590;

const server = await startDashboardServer({ port, webRoot });
console.log(`[mlx-dashboard] serving ${webRoot}`);
console.log(`[mlx-dashboard] listening on ${server.url}`);

function shutdown(): void {
  void server.close().then(() => process.exit(0));
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
