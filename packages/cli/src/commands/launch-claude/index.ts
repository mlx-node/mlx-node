/** `mlx launch claude` — start a local inference host and spawn Claude Code pointed at it. */

import { spawn } from 'node:child_process';
import type { ChildProcess } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { accessSync, constants as fsConstants } from 'node:fs';
import { constants as osConstants } from 'node:os';
import { delimiter, join } from 'node:path';
import { parseArgs } from 'node:util';

import {
  createInferenceHost,
  LAUNCHER_ENGINE_POLICY,
  ModelNotFoundError,
  NoModelsDiscoveredError,
  resolveLogDir,
  resolveMlxNodeHome,
  type InferenceHostOptions,
} from '@mlx-node/server/host';

function printHelp(): void {
  console.log(`
Launch Claude Code pointed at a local mlx-node server

Usage:
  mlx launch claude [options]

Options:
  --port <n>         Port for the local server (default: auto-pick a free port)
  --host <h>         Host to bind (default: 127.0.0.1). The server requires a
                     per-launch token that only the spawned \`claude\` is given,
                     so a non-loopback bind is reachable but not usable by
                     anyone else on the network.
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
                                  2048 under \`mlx launch claude\` to bound
                                  cold-prefill memory peaks; set to 0 to
                                  disable chunking, or tune explicitly for
                                  your workload.
    MLX_PAGED_PREFILL_EVAL_INTERVAL
                                  Layer cadence for eval+clear during paged
                                  prefill. Defaults to 8.
    MLX_PAGED_DECODE_CACHE_CLEAR_INTERVAL
                                  Token cadence for paged decode cache clear.
                                  Defaults to 1024 (mirrors DFlash's
                                  _DECODE_CLEAR_CACHE_INTERVAL_TOKENS).
    MLX_PAGED_CACHE_MEMORY_MB      Paged KV cache memory budget override for
                                  paged-aware Qwen3.5 launch.
    MLX_GEMMA4_NATIVE_KV_WRITE    Set to 0/false/off to disable graph-native
                                  Gemma4 global KV writes.
    MLX_GEMMA4_MAX_SLIDING_RESTORE_TOKENS
                                  Optional emergency cap on cached Gemma4
                                  prefix tokens replayed for sliding-cache
                                  restore. Unset by default.
    MLX_GEMMA4_SLIDING_CHECKPOINT_LIMIT
                                  Override retained Gemma4 sliding-cache
                                  checkpoints. Defaults dynamically from the
                                  sliding window and paged block size.
    MLX_INFERENCE_TRACE           Set to 1/true/on to write native inference
                                  phase traces to a file.
    MLX_INFERENCE_TRACE_FILE      Override the native trace file path.
                                  Defaults to <log-dir>/inference-trace.log.

Examples:
  mlx launch claude
  mlx launch claude --verbose
  mlx launch claude --log-dir /tmp/mlx-debug
  mlx launch claude --verbose -- --resume
  mlx launch claude -- "write a haiku about kv caches"
`);
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

function envFlagEnabled(value: string | undefined): boolean {
  if (value == null) return false;
  switch (value.trim().toLowerCase()) {
    case '1':
    case 'true':
    case 'yes':
    case 'on':
      return true;
    default:
      return false;
  }
}

/** The flag surface `mlx launch claude` maps onto {@link InferenceHostOptions}. */
export interface LaunchClaudeFlags {
  port?: number;
  host?: string;
  modelsDir?: string;
  model?: string;
  logDir?: string;
  /**
   * The per-launch secret. Not a flag — {@link run} generates it and hands the
   * same value to the host and to the spawned `claude`. Optional only so the
   * pure mapping stays callable from a test without one.
   */
  authToken?: string;
}

/**
 * A fresh secret for one `mlx launch claude` run, shared with exactly one
 * child process and never written down.
 *
 * The pre-auth launcher passed the constant `mlx-node-local` to Claude Code and
 * configured no token on the host at all, so the string was decorative: every
 * route was open to any local process, and to any web page the user's browser
 * was showing (`Access-Control-Allow-Origin: *` was the default with no token,
 * which made the replies readable cross-origin as well).
 */
function newLaunchAuthToken(): string {
  return randomBytes(32).toString('base64url');
}

/**
 * Map parsed flags onto host options, applying this launcher's policy.
 *
 * Exported and kept pure so the one thing the shared-host extraction could
 * silently drop — the 2048-token paged-prefill chunk `mlx launch claude` has
 * always applied to bound the cold-prefill memory peak — is assertable without
 * spawning `claude` or loading a model.
 */
export function launchClaudeHostOptions(flags: LaunchClaudeFlags): InferenceHostOptions {
  return {
    port: flags.port,
    host: flags.host,
    modelsDir: flags.modelsDir,
    model: flags.model,
    logDir: flags.logDir,
    authToken: flags.authToken,
    // Launcher policy, not an engine default: the shared native var still
    // reads 0 as "disable chunking", and a value already set in the user's
    // shell wins (see `applyEnginePolicy`).
    enginePolicy: LAUNCHER_ENGINE_POLICY,
  };
}

/**
 * Build the environment the spawned `claude` runs under.
 *
 * Exported and pure because the one rule here that is not obvious was measured
 * rather than assumed, and a regression would look like "every request 401s"
 * with nothing in the code to point at.
 *
 * Measured against Claude Code 2.1.220, sending both variables at a local
 * base URL:
 *
 *   ANTHROPIC_AUTH_TOKEN only  → `authorization: Bearer <token>`
 *   + ANTHROPIC_API_KEY        → `authorization: Bearer <token>`
 *                                `x-api-key: <the user's own key>`
 *
 * The gate reads `x-api-key` FIRST (Anthropic clients send it, and checking it
 * first stops an injected `authorization` shadowing a caller's real key), so an
 * inherited `ANTHROPIC_API_KEY` would beat our bearer and 401 the whole
 * session. It is therefore dropped, not overwritten: the child is pointed at
 * loopback and has no reason to hold a key for api.anthropic.com — one that
 * `--verbose` would then write into the request log on disk.
 */
export function claudeChildEnv(
  parent: NodeJS.ProcessEnv,
  opts: { baseUrl: string; model: string; authToken: string },
): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {
    ...parent,
    ANTHROPIC_BASE_URL: opts.baseUrl,
    ANTHROPIC_AUTH_TOKEN: opts.authToken,
    ANTHROPIC_MODEL: opts.model,
    // NOTE: intentionally NOT setting ANTHROPIC_SMALL_FAST_MODEL /
    // ANTHROPIC_DEFAULT_HAIKU_MODEL. Claude Code falls back to
    // `claude-haiku-*` for subagents + title generation; the swap
    // controller aliases any unknown name to the current resident
    // so those calls always follow whatever model the user picked
    // via `/model`.
  };
  // `delete`, not `= undefined`: `spawn` skipping undefined values is an
  // implementation detail of `child_process`, and this must not depend on it.
  delete env.ANTHROPIC_API_KEY;
  return env;
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

  const portArg = args.port != null ? Number(args.port) : undefined;
  if (portArg !== undefined && (!Number.isInteger(portArg) || portArg <= 0)) {
    console.error(`Invalid --port: ${String(args.port)}`);
    process.exit(1);
  }

  // Resolve the verbose log directory BEFORE the host starts, because the
  // native trace file path has to be in the environment before the first
  // model load. `--log-dir` implies `--verbose`.
  const traceRequested = envFlagEnabled(process.env.MLX_INFERENCE_TRACE);
  const verbose = args.verbose || args['log-dir'] != null || process.env.MLX_LOG_DIR != null || traceRequested;
  let logDir: string | undefined;
  if (verbose) {
    logDir = resolveLogDir(args['log-dir'], resolveMlxNodeHome());
    if (
      traceRequested &&
      (process.env.MLX_INFERENCE_TRACE_FILE == null || process.env.MLX_INFERENCE_TRACE_FILE.trim() === '')
    ) {
      process.env.MLX_INFERENCE_TRACE_FILE = join(logDir, 'inference-trace.log');
    }
  }

  // Checked BEFORE the host starts: a missing `claude` is the single most
  // likely failure here, and standing a server up just to tear it down would
  // load the paged-override machinery for nothing.
  const claudeBin = findClaudeOnPath();
  if (!claudeBin) {
    console.error('Could not find `claude` on PATH.');
    console.error('Install Claude Code: https://docs.claude.com/en/docs/claude-code/quickstart');
    process.exit(1);
  }

  const authToken = newLaunchAuthToken();

  const host = await createInferenceHost(
    launchClaudeHostOptions({
      port: portArg,
      host: args.host,
      modelsDir: args['models-dir'],
      model: args.model,
      logDir,
      authToken,
    }),
  ).catch((err: unknown) => {
    if (err instanceof NoModelsDiscoveredError) {
      console.error(err.message);
      console.error('Run: mlx download model --model Qwen/Qwen3.5-9B');
      process.exit(1);
    }
    if (err instanceof ModelNotFoundError) {
      console.error(err.message);
      console.error('Discovered models:');
      for (const name of err.available) console.error(`  - ${name}`);
      process.exit(1);
    }
    throw err;
  });

  console.log(
    `[mlx] models dir: ${host.modelsDir} | listening on ${host.url} | discovered ${host.models.length} model(s) | default: ${host.boundModel}`,
  );
  if (host.logDir !== null) {
    console.log(`[mlx] verbose logging → ${host.logDir}`);
    console.log(`[mlx] tail -f "${join(host.logDir, 'session.log')}"`);
  }

  const child = spawn(claudeBin, claudeArgs, {
    stdio: 'inherit',
    env: claudeChildEnv(process.env, { baseUrl: host.url, model: host.boundModel, authToken }),
  });

  let shuttingDown = false;
  let childExited = false;
  const shutdown = async (exitCode: number): Promise<void> => {
    if (shuttingDown) return;
    shuttingDown = true;
    // `close()` guards each disposer internally, but a rejection here would
    // otherwise skip `process.exit` and hang the launcher on a live handle.
    try {
      await host.close();
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
 * sent, not when the child terminated — making it useless as an
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
