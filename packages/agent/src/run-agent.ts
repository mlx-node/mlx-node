/**
 * `runAgent` — the boot shell that hands control to pi's `main()` with
 * the mlx provider, permission gate, local subagents, and terminal branding installed.
 *
 * Spike-proven boot contract:
 * - The env vars below must be set BEFORE any runtime import of
 *   `@earendil-works/pi-coding-agent` (pi reads its config env at
 *   import/call time) — hence the dynamic import and the type-only
 *   top-level pi import here.
 * - pi's `main()` RETURNS on the happy path but `process.exit()`s on
 *   help/error/package-command paths, so nothing critical may run after
 *   `await main()`.
 * - In print/json mode pi takes over stdout (our writes are rerouted to
 *   stderr) and reads non-TTY stdin to EOF as prompt input — this
 *   wrapper must never consume or hold stdin. Paged-config temp cleanup in
 *   `finally` is best-effort for normal returns/rejections, never correctness-
 *   critical on pi's hard `process.exit()` paths.
 */

import { homedir } from 'node:os';
import { join } from 'node:path';

import type { InlineExtension } from '@earendil-works/pi-coding-agent';
import { PagedConfigOverrideManager } from '@mlx-node/lm';

import { createPermissionGateExtension } from './extensions/permission-gate.js';
import { createSubagentExtension } from './extensions/subagent.js';
import { createTerminalTitleExtension } from './extensions/terminal-title.js';
import { createTraceNoticeExtension } from './extensions/trace-notice.js';
import { createMlxProviderExtension } from './provider/index.js';
import { MlxModelHost } from './provider/model-host.js';
import {
  type FilterableModelRegistryConstructor,
  installMlxOnlyModelRegistryFilter,
} from './provider/model-registry-filter.js';
import type { MlxModelInfo } from './provider/models.js';

/** Shape of pi's `main(argv, { extensionFactories })` — also the test seam. */
export type RunAgentMain = (args: string[], opts: { extensionFactories: InlineExtension[] }) => Promise<void>;

export interface RunAgentPi {
  main: RunAgentMain;
  ModelRegistry: FilterableModelRegistryConstructor;
}

/** @internal Narrow lifecycle seam for the agent's paged config overlays. */
export interface AgentPagedConfigOverrides {
  resolve(modelPath: string, modelType?: string): Promise<string>;
  cleanup(): Promise<void>;
}

export interface RunAgentOptions {
  /** Resolved models directory (context for callers/diagnostics — discovery already ran). */
  modelsDir: string;
  /** Discovered models to serve through the in-process `mlx` provider. */
  models: MlxModelInfo[];
  /** Passthrough args handed to pi's `main()` verbatim. */
  argv: string[];
  /** Native inference-log path to surface after Pi takes over the TUI. */
  traceLogFile?: string;
  /** @internal Test seam; when set, the pi dynamic import is skipped entirely. */
  piImpl?: RunAgentPi;
  /** @internal Test seam for paged model-path resolution and cleanup. */
  pagedConfigOverrides?: AgentPagedConfigOverrides;
}

/** @internal Exact opt-in parser kept separate so non-`1` values stay disabled. */
export function agentGemmaDraftEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return env.MLX_AGENT_ENABLE_GEMMA_DRAFT === '1';
}

/**
 * Seed the pi/mlx environment (never clobbering user-set values) and run
 * pi's `main()` with the mlx inline extensions. May not return: pi
 * `process.exit()`s on help/error paths.
 *
 * This is a dedicated-process entrypoint: pi owns stdin/stdout, signal
 * handlers, and process exit while `main()` runs. Do not run it concurrently
 * with another pi SDK/runtime in the same process. The temporary registry
 * policy is restored when `main()` returns or rejects.
 */
export async function runAgent(opts: RunAgentOptions): Promise<void> {
  process.env.PI_CODING_AGENT_DIR ??= join(homedir(), '.mlx-node', 'agent');
  process.env.PI_SKIP_VERSION_CHECK ??= '1';
  // Mirrors `mlx launch claude`: chunked paged prefill keeps long-prompt
  // TTFT bounded on the default paged path.
  process.env.MLX_PAGED_PREFILL_CHUNK_SIZE ??= '2048';

  // Force every paged-capable agent family through an isolated config clone.
  // This includes quantized LFM2 (whose standalone default is deliberately
  // flat) and Qwen3.5 dense/MoE (whose text-only defaults are flat). Gemma4's
  // paged overlay intentionally hides an embedded draft/ directory: the
  // current speculative executor is flat-cache-only and can regress quantized
  // agent workloads. Users may explicitly opt back into that native behavior.
  const preserveEmbeddedGemmaDraft = agentGemmaDraftEnabled();
  const pagedConfigOverrides =
    opts.pagedConfigOverrides ?? new PagedConfigOverrideManager({ preserveEmbeddedGemmaDraft });
  const modelHost = new MlxModelHost(
    opts.models.map((model) => model.discovered),
    {
      resolveModelPathFn: (model) => pagedConfigOverrides.resolve(model.path, model.modelType),
      requirePagedCache: true,
    },
  );

  // Keep the pi import strictly behind the seam. The seam carries BOTH main
  // and its registry class, so tests and production exercise the same policy
  // installation/lifecycle instead of being able to bypass it accidentally.
  const pi: RunAgentPi = opts.piImpl ?? (await import('@earendil-works/pi-coding-agent'));
  const restoreModelRegistry = installMlxOnlyModelRegistryFilter(
    pi.ModelRegistry,
    opts.models.map((model) => model.discovered.name),
  );
  const subagentsEnabled =
    opts.models.length > 0 && !opts.argv.includes('--no-extensions') && !opts.argv.includes('-ne');
  try {
    await pi.main(opts.argv, {
      extensionFactories: [
        createMlxProviderExtension(opts.models, modelHost),
        createPermissionGateExtension(),
        ...(subagentsEnabled ? [createSubagentExtension()] : []),
        ...(opts.traceLogFile !== undefined ? [createTraceNoticeExtension(opts.traceLogFile)] : []),
        createTerminalTitleExtension(),
      ],
    });
  } finally {
    try {
      restoreModelRegistry();
    } finally {
      await pagedConfigOverrides.cleanup();
    }
  }
}
