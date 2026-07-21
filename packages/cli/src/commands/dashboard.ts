/** `mlx dashboard` — start the local web dashboard and open it in a browser. */

import { spawn } from 'node:child_process';
import { parseArgs } from 'node:util';

import { startDashboardServer } from '@mlx-node/dashboard';

import { expandPiAgentDir } from './agent/index.js';

const DEFAULT_PORT = 6590;
const DEFAULT_HOST = '127.0.0.1';
const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '::1']);

export interface DashboardArgs {
  help: boolean;
  port: number;
  host: string;
  open: boolean;
  db?: string;
  modelsDir?: string;
  sessionDir?: string;
}

/** Dependencies injected only by tests; production callers pass nothing. */
export interface DashboardRunOptions {
  openBrowser?: (url: string) => void;
}

function printHelp(): void {
  console.log(`
Start the local mlx-node web dashboard

Usage:
  mlx dashboard [options]

Options:
  --port <n>          Port to listen on (default: ${DEFAULT_PORT})
  --host <h>          Host to bind (default: ${DEFAULT_HOST}). Binding a
                      non-loopback host exposes local models, sessions, and
                      cache to your network; the dashboard has no auth.
  --no-open           Do not open the dashboard in a browser
  --db <path>         Override the dashboard SQLite index path
                      (default: ~/.mlx-node/dashboard.db)
  --models-dir <path> Directory to read local models from
                      (default: ~/.mlx-node/models)
  --session-dir <dir> Agent sessions root to list (default: honors
                      PI_CODING_AGENT_DIR, else ~/.mlx-node/agent/sessions).
                      Pass the same dir the agent uses to see custom sessions.
  -h, --help          Show this help message

Examples:
  mlx dashboard
  mlx dashboard --port 8080
  mlx dashboard --no-open
`);
}

/**
 * Parse `mlx dashboard` flags into a validated shape. Pure and side-effect free
 * so the flag surface is unit-testable without starting a server. Throws on an
 * invalid `--port` (non-integer or out of range); the caller maps that to a
 * clear error + non-zero exit.
 */
export function parseDashboardArgs(argv: string[]): DashboardArgs {
  const parsed = parseArgs({
    args: argv,
    options: {
      port: { type: 'string' },
      host: { type: 'string' },
      'no-open': { type: 'boolean', default: false },
      db: { type: 'string' },
      'models-dir': { type: 'string' },
      'session-dir': { type: 'string' },
      help: { type: 'boolean', short: 'h', default: false },
    },
    allowPositionals: false,
  });
  const values = parsed.values;

  let port = DEFAULT_PORT;
  if (values.port !== undefined) {
    const parsedPort = Number(values.port);
    if (!Number.isInteger(parsedPort) || parsedPort < 0 || parsedPort > 65535) {
      throw new Error(`Invalid --port "${values.port}": expected an integer between 0 and 65535.`);
    }
    port = parsedPort;
  }

  return {
    help: values.help ?? false,
    port,
    host: values.host ?? DEFAULT_HOST,
    open: !(values['no-open'] ?? false),
    db: values.db,
    modelsDir: values['models-dir'],
    sessionDir: values['session-dir'],
  };
}

function isLoopbackHost(host: string): boolean {
  return LOOPBACK_HOSTS.has(host);
}

/**
 * Best-effort open of `url` in the platform browser without adding a
 * dependency. Detached + unref'd so it never keeps the dashboard process
 * alive, and spawn failures (headless box, no opener) are swallowed.
 */
function openInBrowser(url: string): void {
  let command: string;
  let args: string[];
  switch (process.platform) {
    case 'darwin':
      command = 'open';
      args = [url];
      break;
    case 'win32':
      command = 'cmd';
      args = ['/c', 'start', '', url];
      break;
    default:
      command = 'xdg-open';
      args = [url];
      break;
  }
  try {
    const child = spawn(command, args, { stdio: 'ignore', detached: true });
    child.on('error', () => {
      /* no browser opener available; ignore */
    });
    child.unref();
  } catch {
    /* opening a browser is optional; ignore failures */
  }
}

export async function run(argv: string[], deps: DashboardRunOptions = {}): Promise<void> {
  let args: DashboardArgs;
  try {
    args = parseDashboardArgs(argv);
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    printHelp();
    process.exit(1);
  }

  if (args.help) {
    printHelp();
    return;
  }

  if (!isLoopbackHost(args.host)) {
    console.warn(
      `WARNING: binding to non-loopback host "${args.host}" exposes the dashboard (local models, sessions, and cache) to your network. The dashboard has no authentication.`,
    );
  }

  const openBrowser = deps.openBrowser ?? openInBrowser;

  // Explicit `--session-dir` wins; expanded via the SAME tilde/file-URL rule the
  // agent applies to `PI_CODING_AGENT_DIR`. Left undefined, the server falls back
  // to `agentSessionsRoot()`, which honors `PI_CODING_AGENT_DIR` then the default.
  const sessionsRoot = args.sessionDir === undefined ? undefined : expandPiAgentDir(args.sessionDir);

  const server = await startDashboardServer({
    port: args.port,
    host: args.host,
    dbPath: args.db,
    modelsDir: args.modelsDir,
    sessionsRoot,
  });

  console.log(`mlx dashboard listening on ${server.url}`);
  console.log('Press Ctrl+C to stop.');

  if (args.open) {
    openBrowser(server.url);
  }

  let shuttingDown = false;
  const shutdown = async (signal: NodeJS.Signals): Promise<void> => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log(`\nReceived ${signal}, shutting down mlx dashboard…`);
    try {
      await server.close();
    } catch {
      /* ignore close errors during shutdown */
    }
    process.exit(0);
  };
  process.on('SIGINT', () => void shutdown('SIGINT'));
  process.on('SIGTERM', () => void shutdown('SIGTERM'));
}
