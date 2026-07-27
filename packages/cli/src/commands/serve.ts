/**
 * `mlx serve` — run the inference host on its own, in the foreground.
 *
 * The same `createInferenceHost` the desktop app runs inside an Electron
 * `utilityProcess`, with nothing supervising it and stdout attached. That is
 * the point: when the sidecar wedges, this is how you reproduce it in a
 * terminal and watch it happen.
 */

import { parseArgs } from 'node:util';

import {
  createInferenceHost,
  LAUNCHER_ENGINE_POLICY,
  ModelNotFoundError,
  NoModelsDiscoveredError,
  resolveLogDir,
  resolveMlxNodeHome,
  type InferenceHost,
  type InferenceHostOptions,
} from '@mlx-node/server/host';

/** Grace period handed to `close()` on SIGINT/SIGTERM. Matches the server default. */
const SHUTDOWN_TIMEOUT_MS = 5000;

function printHelp(): void {
  console.log(`
Serve local models over an Anthropic/OpenAI-compatible HTTP API

Usage:
  mlx serve [options]

Options:
  --port <n>          Port to listen on (default: auto-pick a free port;
                      0 binds an ephemeral port and prints the real one)
  --host <h>          Host to bind (default: 127.0.0.1). A non-loopback bind
                      exposes your local models to the network — pair it
                      with --auth-token.
  --models-dir <dir>  Directory to discover models from
                      (default: ~/.mlx-node/models; overridable via
                      MLX_MODELS_DIR env or ~/.mlx-node/config.json)
  --model <name>      Which discovered model is the default. Must match a
                      directory name under --models-dir. Defaults to
                      ANTHROPIC_MODEL env, else the first discovered model.
  --auth-token <tok>  Require this secret on every route except /health,
                      as \`x-api-key\` or \`authorization: Bearer\`. Also
                      readable from MLX_SERVER_AUTH_TOKEN.
  -v, --verbose       Write every HTTP request/response to a log dir
  --log-dir <dir>     Override the verbose log directory (implies --verbose).
                      Default: ~/.mlx-node/logs/<ISO-timestamp>/.
  -h, --help          Show this help message

Models load lazily: the first request naming a model loads it, and only one
model stays resident at a time.

Examples:
  mlx serve
  mlx serve --port 8080
  mlx serve --port 8080 --model qwen3.5-9b
  mlx serve --host 0.0.0.0 --auth-token "$(openssl rand -hex 16)"
`);
}

export interface ServeArgs {
  help: boolean;
  port?: number;
  host?: string;
  modelsDir?: string;
  model?: string;
  authToken?: string;
  logDir?: string;
}

/**
 * Parse `mlx serve` flags. Pure and side-effect free so the flag surface is
 * unit-testable without binding a port. Throws on an invalid `--port`; the
 * caller maps that to a message plus a non-zero exit.
 *
 * `--port 0` is legal here (unlike `mlx launch claude`, which rejects it):
 * an operator asking for an ephemeral port gets one, and the real port is
 * printed. `undefined` still means "pick one for me".
 */
export function parseServeArgs(argv: string[]): ServeArgs {
  const parsed = parseArgs({
    args: argv,
    options: {
      port: { type: 'string' },
      host: { type: 'string' },
      'models-dir': { type: 'string' },
      model: { type: 'string' },
      'auth-token': { type: 'string' },
      verbose: { type: 'boolean', short: 'v', default: false },
      'log-dir': { type: 'string' },
      help: { type: 'boolean', short: 'h', default: false },
    },
    allowPositionals: false,
  });
  const values = parsed.values;

  let port: number | undefined;
  if (values.port !== undefined) {
    const parsedPort = Number(values.port);
    if (!Number.isInteger(parsedPort) || parsedPort < 0 || parsedPort > 65535) {
      throw new Error(`Invalid --port "${values.port}": expected an integer between 0 and 65535.`);
    }
    port = parsedPort;
  }

  const verbose = (values.verbose ?? false) || values['log-dir'] != null;

  return {
    help: values.help ?? false,
    port,
    host: values.host,
    modelsDir: values['models-dir'],
    model: values.model,
    authToken: values['auth-token'],
    logDir: verbose ? resolveLogDir(values['log-dir'], resolveMlxNodeHome()) : undefined,
  };
}

/** Seams injected only by tests; production callers pass nothing. */
export interface ServeRunOptions {
  /** Merged over the flag-derived host options (store path, stub loader, …). */
  hostOverrides?: Partial<InferenceHostOptions>;
  /**
   * Invoked once the host is listening. When supplied, `run` resolves after
   * it settles instead of parking on signals — the signal handlers are still
   * installed first, so a test can grab and invoke the real one.
   */
  onReady?: (host: InferenceHost) => void | Promise<void>;
  /** Stands in for `process.exit` so a test can observe the code. */
  exit?: (code: number) => void;
}

export async function run(argv: string[], deps: ServeRunOptions = {}): Promise<void> {
  let args: ServeArgs;
  try {
    args = parseServeArgs(argv);
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    printHelp();
    process.exit(1);
  }

  if (args.help) {
    printHelp();
    return;
  }

  let host: InferenceHost;
  try {
    host = await createInferenceHost({
      port: args.port,
      host: args.host,
      modelsDir: args.modelsDir,
      model: args.model,
      authToken: args.authToken,
      logDir: args.logDir,
      // Same paged-prefill chunking `mlx launch claude` applies. A value
      // already in the environment wins; see `env-policy.ts`.
      enginePolicy: LAUNCHER_ENGINE_POLICY,
      ...deps.hostOverrides,
    });
  } catch (err) {
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
  }

  console.log(
    `[mlx] models dir: ${host.modelsDir} | listening on ${host.url} | discovered ${host.models.length} model(s) | default: ${host.boundModel}`,
  );
  if (host.logDir !== null) console.log(`[mlx] verbose logging → ${host.logDir}`);
  console.log('Press Ctrl+C to stop.');

  const exit = deps.exit ?? ((code: number): void => process.exit(code));

  let shuttingDown = false;
  const shutdown = async (signal: NodeJS.Signals): Promise<void> => {
    if (shuttingDown) return;
    shuttingDown = true;
    // Detach first: a second Ctrl+C during the grace period should reach the
    // default handler and kill us outright rather than queue another close.
    process.removeListener('SIGINT', onSigint);
    process.removeListener('SIGTERM', onSigterm);
    console.log(`\nReceived ${signal}, shutting down mlx serve…`);
    try {
      await host.close({ timeoutMs: SHUTDOWN_TIMEOUT_MS });
    } catch {
      /* `close()` guards each disposer; never let shutdown itself throw */
    }
    exit(0);
  };
  const onSigint = (): void => void shutdown('SIGINT');
  const onSigterm = (): void => void shutdown('SIGTERM');
  process.on('SIGINT', onSigint);
  process.on('SIGTERM', onSigterm);

  if (deps.onReady !== undefined) await deps.onReady(host);
}
