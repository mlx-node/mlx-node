/** `mlx launch claude` — start a local inference host and spawn Claude Code pointed at it. */

import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { join } from 'node:path';
import { parseArgs } from 'node:util';

import {
  createInferenceHost,
  ModelNotFoundError,
  NoModelsDiscoveredError,
  resolveLogDir,
  resolveMlxNodeHome,
  type InferenceHost,
} from '@mlx-node/server/host';

import {
  claudeChildEnv,
  computeExitCode,
  launchClaudeHostOptions,
  makeChildKillEscalation,
  resolveClaudeLaunch,
} from './helpers.js';

export {
  claudeChildEnv,
  claudeSpawnSpec,
  computeExitCode,
  findClaudeOnPath,
  launchClaudeHostOptions,
  makeChildKillEscalation,
  resolveClaudeLaunch,
  resolveWindowsClaudeShim,
  type LaunchClaudeFlags,
} from './helpers.js';

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
                                  cold-prefill memory peaks. Qwen3.5 models
                                  cap it at 2048 (the memory plan prices that
                                  chunk); set it lower to shrink the chunks.
    MLX_PAGED_PREFILL_EVAL_INTERVAL
                                  Layer cadence for eval+clear during paged
                                  prefill. Defaults to 8.
    MLX_PAGED_DECODE_CACHE_CLEAR_INTERVAL
                                  Token cadence for paged decode cache clear.
                                  Defaults to 1024 (mirrors DFlash's
                                  _DECODE_CLEAR_CACHE_INTERVAL_TOKENS).
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

/**
 * A fresh secret for one `mlx launch claude` run, shared with exactly one
 * child process and never written down.
 */
function newLaunchAuthToken(): string {
  return randomBytes(32).toString('base64url');
}

/**
 * Load the bound model before Claude Code starts and return the window its
 * default text turn gets; `undefined` (after printing why) when the load
 * fails or the model publishes no window.
 */
export async function loadServedWindow(
  host: Pick<InferenceHost, 'boundModel' | 'loadModel' | 'server'>,
): Promise<number | undefined> {
  console.log(`[mlx] loading ${host.boundModel} …`);
  await host.loadModel(host.boundModel).catch((err: unknown) => {
    console.error(`[mlx] load of "${host.boundModel}" failed: ${err instanceof Error ? err.message : String(err)}`);
  });
  return host.server.registry.servedContext(host.boundModel)?.windowTokens;
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
  let launch: { claudeBin: string; spawnSpec: { file: string; args: string[] } } | null;
  try {
    launch = resolveClaudeLaunch(claudeArgs);
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
  }
  if (!launch) {
    console.error('Could not find `claude` on PATH.');
    console.error('Install Claude Code: https://docs.claude.com/en/docs/claude-code/quickstart');
    process.exit(1);
  }
  const { spawnSpec } = launch;

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

  const contextTokens = await loadServedWindow(host);

  const childEnv = claudeChildEnv(process.env, {
    baseUrl: host.url,
    model: host.boundModel,
    authToken,
    ...(contextTokens !== undefined ? { contextTokens } : {}),
  });
  const child = spawn(spawnSpec.file, spawnSpec.args, {
    stdio: 'inherit',
    env: childEnv,
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
