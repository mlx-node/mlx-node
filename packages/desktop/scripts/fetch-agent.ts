/**
 * Download the prebuilt `mlx-agent` release binary into `out/agent/mlx-agent`,
 * where package.ts stages it into `Contents/Resources/native/`.
 *
 *   yarn oxnode packages/desktop/scripts/fetch-agent.ts
 *
 * The release workflow runs this BEFORE `package.ts --sign` so a fetch failure
 * costs seconds, not a signed-then-broken bundle. The downloader is the same
 * one `mlx agent` uses at runtime (R2 mirror first, GitHub release fallback,
 * latest-tag resolution + SHA256 verification), so the bundled binary is
 * bit-identical to what a standalone CLI would fetch.
 */

import { chmodSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, '..', '..', '..');
// `dist` — this script runs after `yarn build`, and the source condition does
// not apply to the release scripts' oxnode invocation.
const { downloadAgentBinary } = await import(
  join(REPO_ROOT, 'packages', 'cli', 'dist', 'commands', 'agent', 'grok-build.js')
);

const dest = join(HERE, '..', 'out', 'agent', 'mlx-agent');
mkdirSync(dirname(dest), { recursive: true });
const path = await downloadAgentBinary(dest);
chmodSync(path, 0o755);
console.log(`mlx-agent staged at ${path}`);
