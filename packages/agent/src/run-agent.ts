/**
 * `runAgent` — the boot shell that hands control to pi's `main()` with
 * the mlx provider and permission gate installed.
 *
 * Spike-proven boot contract:
 * - The env vars below must be set BEFORE any runtime import of
 *   `@earendil-works/pi-coding-agent` (pi reads its config env at
 *   import/call time) — hence the dynamic import and the type-only
 *   top-level pi import here.
 * - pi's `main()` RETURNS on the happy path but `process.exit()`s on
 *   help/error/package-command paths, so nothing critical may run after
 *   `await main()`; any cleanup belongs in `process.on('exit')`.
 * - In print/json mode pi takes over stdout (our writes are rerouted to
 *   stderr) and reads non-TTY stdin to EOF as prompt input — this
 *   wrapper must never consume or hold stdin.
 */

import { homedir } from 'node:os';
import { join } from 'node:path';

import type { InlineExtension } from '@earendil-works/pi-coding-agent';

import { createPermissionGateExtension } from './extensions/permission-gate.js';
import { createMlxProviderExtension } from './provider/index.js';
import type { MlxModelInfo } from './provider/models.js';

/** Shape of pi's `main(argv, { extensionFactories })` — also the test seam. */
export type RunAgentMain = (args: string[], opts: { extensionFactories: InlineExtension[] }) => Promise<void>;

export interface RunAgentOptions {
  /** Resolved models directory (context for callers/diagnostics — discovery already ran). */
  modelsDir: string;
  /** Discovered models to serve through the in-process `mlx` provider. */
  models: MlxModelInfo[];
  /** Passthrough args handed to pi's `main()` verbatim. */
  argv: string[];
  /** Test seam; when set, the pi dynamic import is skipped entirely. */
  mainImpl?: RunAgentMain;
}

/**
 * Seed the pi/mlx environment (never clobbering user-set values) and run
 * pi's `main()` with the two mlx inline extensions. May not return: pi
 * `process.exit()`s on help/error paths.
 */
export async function runAgent(opts: RunAgentOptions): Promise<void> {
  process.env.PI_CODING_AGENT_DIR ??= join(homedir(), '.mlx-node', 'agent');
  process.env.PI_SKIP_VERSION_CHECK ??= '1';
  // Mirrors `mlx launch claude`: chunked paged prefill keeps long-prompt
  // TTFT bounded on the default paged path.
  process.env.MLX_PAGED_PREFILL_CHUNK_SIZE ??= '2048';

  // The `??` short-circuit keeps the pi import strictly behind the seam:
  // with `mainImpl` injected, pi is never imported at all.
  const main: RunAgentMain = opts.mainImpl ?? (await import('@earendil-works/pi-coding-agent')).main;
  await main(opts.argv, {
    extensionFactories: [createMlxProviderExtension(opts.models), createPermissionGateExtension()],
  });
}
