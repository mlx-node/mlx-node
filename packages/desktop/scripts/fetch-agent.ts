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

import { chmodSync, mkdirSync, renameSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, '..', '..', '..');
// `dist` — this script runs after `yarn build`, and the source condition does
// not apply to the release scripts' oxnode invocation.
const { downloadAgentBinary, agentTagFromBinPath } = await import(
  join(REPO_ROOT, 'packages', 'cli', 'dist', 'commands', 'agent', 'grok-build.js')
);

const dest = join(HERE, '..', 'out', 'agent', 'mlx-agent');
mkdirSync(dirname(dest), { recursive: true });
// dest is the managed base path: the binary lands at the sibling
// `mlx-agent-v<tag>` file the `.tag` pointer names. No explicit tag — the
// downloader walks its verified-candidate list, so a latest that verifies
// but cannot be fetched still yields an older complete release instead of
// failing the packaging run. The tag is read back from the installed path
// rather than assumed: the pin (or a fallback) may name another release.
const path = await downloadAgentBinary(dest);
const tag = agentTagFromBinPath(path);
if (tag === undefined) throw new Error(`cannot determine the release tag for ${path}`);
renameSync(path, dest);
writeFileSync(`${dest}.tag`, tag, { mode: 0o644 });
chmodSync(dest, 0o755);
console.log(`mlx-agent staged at ${dest} (${tag})`);
