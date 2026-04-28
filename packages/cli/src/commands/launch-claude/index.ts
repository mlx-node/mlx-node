/** `mlx launch claude` — start a local Anthropic-compatible server and spawn Claude Code pointed at it. */

import { spawn } from 'node:child_process';
import type { ChildProcess } from 'node:child_process';
import { accessSync, constants as fsConstants } from 'node:fs';
import { createServer as netCreateServer } from 'node:net';
import { constants as osConstants } from 'node:os';
import { delimiter, join } from 'node:path';
import { parseArgs } from 'node:util';

import { loadModel } from '@mlx-node/lm';
import { createServer } from '@mlx-node/server';

import { resolveMlxNodeHome, resolveModelsDir } from '../../config.js';
import { discoverModels } from './discover.js';
import { attachLogger, resolveLogDir, type Logger } from './logger.js';
import { cleanupPagedOverrides, resolvePagedAwareModelPath } from './paged-config-override.js';
import { makeSwapController } from './swap.js';

function printHelp(): void {
  console.log(`
Launch Claude Code pointed at a local mlx-node server

Usage:
  mlx launch claude [options]

Options:
  --port <n>         Port for the local server (default: auto-pick a free port)
  --host <h>         Host to bind (default: 127.0.0.1)
  --models-dir <dir> Directory to discover models from
                     (default: ~/.mlx-node/models; overridable via
                     MLX_MODELS_DIR env or ~/.mlx-node/config.json)
  --model <name>     Which discovered model is bound to Claude Code's
                     "Custom model" slot (/model menu entry 5). Must
                     match a directory name under --models-dir.
                     Defaults to ANTHROPIC_MODEL env, else the first
                     discovered model (alphabetical).
  -v, --verbose      Write every HTTP request/response to a log dir for
                     post-hoc analysis (cache hits, tool calls, SSE chunks)
  --log-dir <dir>    Override the verbose log directory (implies --verbose).
                     Default: ~/.mlx-node/logs/<ISO-timestamp>/.
                     Also honors MLX_LOG_DIR env.
  -h, --help         Show this help message

  --                 Everything after this separator is forwarded to the
                     spawned \`claude\` binary verbatim.

Environment variables:
  MLX_PAGED_PREFILL_CHUNK_SIZE  Tokens per paged-prefill chunk. Defaults to
                                4096 under \`mlx launch claude\` to bound
                                cold-prefill memory peaks; set to 0 to
                                disable chunking, or to a smaller value
                                (e.g. 1024 / 512) if 4096 still peaks.

Examples:
  mlx launch claude
  mlx launch claude --verbose
  mlx launch claude --log-dir /tmp/mlx-debug
  mlx launch claude --verbose -- --resume
  mlx launch claude -- "write a haiku about kv caches"
`);
}

async function pickFreePort(): Promise<number> {
  // Bind to port 0, read the assigned port, close immediately. Racy in theory
  // (another process could grab the port before we listen), but acceptable
  // for a developer-local CLI.
  return await new Promise<number>((resolve, reject) => {
    const probe = netCreateServer();
    probe.unref();
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const addr = probe.address();
      if (addr && typeof addr === 'object') {
        const port = addr.port;
        probe.close(() => resolve(port));
      } else {
        probe.close(() => reject(new Error('could not determine free port')));
      }
    });
  });
}

function findClaudeOnPath(): string | null {
  const pathEnv = process.env.PATH ?? '';
  for (const dir of pathEnv.split(delimiter)) {
    if (!dir) continue;
    const candidate = join(dir, 'claude');
    try {
      accessSync(candidate, fsConstants.X_OK);
      return candidate;
    } catch {
      /* not here */
    }
  }
  return null;
}

export async function run(argv: string[]): Promise<void> {
  const parsed = parseArgs({
    args: argv,
    options: {
      port: { type: 'string' },
      host: { type: 'string' },
      'models-dir': { type: 'string' },
      model: { type: 'string' },
      verbose: { type: 'boolean', short: 'v', default: false },
      'log-dir': { type: 'string' },
      help: { type: 'boolean', short: 'h', default: false },
    },
    allowPositionals: true,
  });
  const args = parsed.values;
  const claudeArgs = parsed.positionals;

  if (args.help) {
    printHelp();
    return;
  }

  // Bound paged-prefill memory peak by chunking the prompt: default of
  // 4096 tokens/chunk caps per-chunk SDPA + MoE intermediates to
  // chunk-sized tiles instead of full-prompt tiles. Empirically 4096
  // keeps Qwen3.6-35b-a3b 28-43K-token cold prefills at ~50 GB wired
  // memory with zero swap (vs. the unchunked 117 GB / 39 GB swap), and
  // halves the per-chunk synchronize_and_clear_cache + host-roundtrip
  // K/V replay overhead vs. 1024. Respect any user-provided value (set
  // in shell) so power users can tune down (e.g. 1024 / 512) if a
  // larger context still peaks. The MLX env var is read via OnceLock on
  // first paged-prefill call, so setting `process.env` here — before any
  // model loads — is sufficient to apply the default.
  if (process.env.MLX_PAGED_PREFILL_CHUNK_SIZE == null) {
    process.env.MLX_PAGED_PREFILL_CHUNK_SIZE = '4096';
  }

  const modelsDir = resolveModelsDir(args['models-dir']);
  const discovered = await discoverModels(modelsDir);
  if (discovered.length === 0) {
    console.error(`No models discovered under ${modelsDir}.`);
    console.error('Run: mlx download model --model Qwen/Qwen3.5-9B');
    process.exit(1);
  }

  // Which discovered model becomes Claude Code's "Custom model" slot.
  // Precedence: --model flag > ANTHROPIC_MODEL env > discovered[0] (alphabetical).
  const requestedModel = args.model ?? process.env.ANTHROPIC_MODEL;
  const defaultModel = requestedModel != null ? discovered.find((m) => m.name === requestedModel) : undefined;
  if (requestedModel != null && !defaultModel) {
    console.error(`Model "${requestedModel}" not found under ${modelsDir}.`);
    console.error('Discovered models:');
    for (const m of discovered) console.error(`  - ${m.name}`);
    process.exit(1);
  }
  const boundModel = defaultModel ?? discovered[0];

  const portArg = args.port != null ? Number(args.port) : undefined;
  if (portArg !== undefined && (!Number.isInteger(portArg) || portArg <= 0)) {
    console.error(`Invalid --port: ${String(args.port)}`);
    process.exit(1);
  }
  const port = portArg ?? (await pickFreePort());
  const host = args.host ?? '127.0.0.1';

  const claudeBin = findClaudeOnPath();
  if (!claudeBin) {
    console.error('Could not find `claude` on PATH.');
    console.error('Install Claude Code: https://docs.claude.com/en/docs/claude-code/quickstart');
    process.exit(1);
  }

  // The swap controller needs the registry from the server instance, but the
  // server needs the controller's callbacks at construction. Bridge via a
  // late-bound holder: the callbacks capture `ctrlRef.current` by closure.
  const ctrlRef: { current: ReturnType<typeof makeSwapController> | null } = { current: null };
  const server = await createServer({
    port,
    host,
    resolveModel: (name) => ctrlRef.current!.resolveModel(name),
    listModels: () => ctrlRef.current!.listModels(),
  });
  // Wrap loadModel so Qwen3.5 dense / MoE checkpoints get
  // `use_block_paged_cache: true` injected via a temp-dir clone with
  // patched config.json (see ./paged-config-override.ts for why this
  // command turns paged ON despite the upstream Qwen3.5 default being
  // OFF). Other model types pass through unmodified.
  const loadModelPagedAware = async (path: string) => loadModel(await resolvePagedAwareModelPath(path));
  ctrlRef.current = makeSwapController(discovered, server.registry, loadModelPagedAware, boundModel.name);

  // Verbose logging: attach AFTER `createServer` so we wrap every
  // incoming request (including `GET /v1/models` which claude fires
  // on startup). `--log-dir` implies `--verbose`.
  const verbose = args.verbose || args['log-dir'] != null || process.env.MLX_LOG_DIR != null;
  let logger: Logger | null = null;
  if (verbose) {
    const logDir = resolveLogDir(args['log-dir'], resolveMlxNodeHome());
    logger = attachLogger(server.server, logDir);
  }

  console.log(
    `[mlx] models dir: ${modelsDir} | listening on http://${host}:${port} | discovered ${discovered.length} model(s) | default: ${boundModel.name}`,
  );
  // Mirror the Rust-side parser in crates/mlx-core/src/array/memory.rs::
  // paged_prefill_chunk_size: trim whitespace, parse i32, reject negative
  // and unparseable; fall back to 0 (disabled). Logging the EFFECTIVE value
  // — not the raw env string — prevents misleading "abc tokens" output when
  // the user typo'd a non-numeric value (which the parser silently maps to
  // 0/disabled).
  const rawChunkSize = process.env.MLX_PAGED_PREFILL_CHUNK_SIZE;
  let effectiveChunkSize = 0;
  if (rawChunkSize != null) {
    const trimmed = rawChunkSize.trim();
    // Mirror Rust's `s.trim().parse::<i32>()`: integer-only, signed,
    // i32 range [-2^31, 2^31). The TS parser must reject anything Rust
    // would, otherwise the log shows "N tokens" while chunking is
    // actually disabled.
    const I32_MAX = 0x7fff_ffff;
    // Rust's parse::<i32>() accepts an optional leading '+' or '-'.
    const parsed = /^[+-]?\d+$/.test(trimmed) ? Number.parseInt(trimmed, 10) : Number.NaN;
    if (Number.isSafeInteger(parsed) && parsed >= 0 && parsed <= I32_MAX) {
      effectiveChunkSize = parsed;
    } else {
      console.warn(
        `[mlx] warning: MLX_PAGED_PREFILL_CHUNK_SIZE=${JSON.stringify(rawChunkSize)} is not a non-negative i32 integer; treating as 0 (disabled)`,
      );
    }
  }
  const chunkLabel = effectiveChunkSize === 0 ? '0 (disabled)' : String(effectiveChunkSize);
  console.log(
    `[mlx] paged-prefill chunk size: ${chunkLabel} tokens (set MLX_PAGED_PREFILL_CHUNK_SIZE=N to override; 0 disables)`,
  );
  if (logger) {
    console.log(`[mlx] verbose logging → ${logger.logDir}`);
    console.log(`[mlx] tail -f "${join(logger.logDir, 'session.log')}"`);
  }

  const child = spawn(claudeBin, claudeArgs, {
    stdio: 'inherit',
    env: {
      ...process.env,
      ANTHROPIC_BASE_URL: `http://${host}:${port}`,
      ANTHROPIC_AUTH_TOKEN: 'mlx-node-local',
      ANTHROPIC_MODEL: boundModel.name,
      // NOTE: intentionally NOT setting ANTHROPIC_SMALL_FAST_MODEL /
      // ANTHROPIC_DEFAULT_HAIKU_MODEL. Claude Code falls back to
      // `claude-haiku-*` for subagents + title generation; the swap
      // controller aliases any unknown name to the current resident
      // so those calls always follow whatever model the user picked
      // via `/model`.
    },
  });

  let shuttingDown = false;
  let childExited = false;
  const shutdown = async (exitCode: number): Promise<void> => {
    if (shuttingDown) return;
    shuttingDown = true;
    if (logger) {
      try {
        await logger.close();
      } catch {
        /* ignore */
      }
    }
    try {
      await server.close();
    } catch {
      /* ignore */
    }
    try {
      await cleanupPagedOverrides();
    } catch {
      /* ignore */
    }
    process.exit(exitCode);
  };

  child.on('exit', (code, signal) => {
    childExited = true;
    void shutdown(computeExitCode(code, signal));
  });
  child.on('error', (err) => {
    console.error(`[mlx] failed to spawn claude: ${err.message}`);
    void shutdown(1);
  });

  const forwardSignal = makeChildKillEscalation({
    child,
    isShuttingDown: () => shuttingDown,
    hasChildExited: () => childExited,
  });
  process.on('SIGINT', () => forwardSignal('SIGINT'));
  process.on('SIGTERM', () => forwardSignal('SIGTERM'));
}

/**
 * Map Node's `child.on('exit', (code, signal))` callback args onto a single
 * shell-style exit code.
 *
 * Background: Node delivers `code === null && signal !== null` whenever the
 * child terminated due to a signal. The previous handler ignored `signal`
 * and coerced `null → 0`, so a SIGINT/SIGTERM/SIGKILL'd `claude` would
 * report success — CI jobs treating exit code as "did the run pass" got a
 * false green even when the process was killed. POSIX convention is
 * `128 + signal_number` for signal-killed processes; we look the number up
 * in `os.constants.signals` (Node exposes the standard signals there).
 *
 * Falls back to `1` for the genuinely-unknown cases:
 *   - `(null, null)` — should never happen per Node's docs.
 *   - `(null, <signal not in os.constants.signals>)` — defensive, in case a
 *     non-standard or platform-specific signal name reaches us.
 *
 * Exported purely for unit testing — the real `child.on('exit', ...)` callback
 * delegates straight to this helper.
 */
export function computeExitCode(code: number | null, signal: NodeJS.Signals | null): number {
  if (code != null) return code;
  if (signal != null) {
    const signals = osConstants.signals as Record<string, number | undefined>;
    const num = signals[signal];
    if (typeof num === 'number') return 128 + num;
    return 1;
  }
  return 1;
}

/**
 * Build a signal forwarder that escalates to SIGKILL if the child hasn't
 * exited within `escalateAfterMs`. Factored out (and exported) so the
 * escalation logic is unit-testable without spawning a real process.
 *
 * Tracks termination via a caller-supplied `hasChildExited` predicate
 * because `subprocess.killed` flips to true the moment the *signal* is
 * sent, not when the child terminates — making it useless as an
 * "is the child gone yet?" check.
 */
export function makeChildKillEscalation(opts: {
  child: Pick<ChildProcess, 'kill'>;
  isShuttingDown: () => boolean;
  hasChildExited: () => boolean;
  escalateAfterMs?: number;
}): (sig: NodeJS.Signals) => void {
  const escalateAfterMs = opts.escalateAfterMs ?? 5000;
  return (sig: NodeJS.Signals): void => {
    if (opts.isShuttingDown()) return;
    opts.child.kill(sig);
    const timer = setTimeout(() => {
      if (!opts.hasChildExited()) opts.child.kill('SIGKILL');
    }, escalateAfterMs);
    timer.unref();
  };
}
