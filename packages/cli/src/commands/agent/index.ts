/**
 * `mlx agent` — boot the Grok Build coding agent over the local inference
 * host (fully offline).
 *
 * The agent binary (`mlx-agent`, built from the `mlx` branch of
 * github.com/mlx-node/grok-build) is a standalone Rust process that speaks
 * HTTP to `mlx serve`. This command's whole job:
 *
 *   1. lift mlx-owned flags out of argv (`--models-dir`, tracing, help,
 *      the mlx-owned `update` positional),
 *   2. make sure at least one local model exists (first-run wizard),
 *   3. spawn `mlx serve --port 0 --auth-token <tok>` + the agent binary,
 *      pointed at the real port via `GROK_*_BASE_URL`/`XAI_API_KEY`
 *      (see ./grok-build.ts for the env contract).
 *
 * The binary owns every other flag, so this command never `parseArgs`es the
 * full argv: {@link scanAgentArgs} lifts out only what mlx handles and
 * forwards the rest verbatim.
 */

import { chmodSync, closeSync, mkdirSync, openSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';

// Native-free subpaths: the help path must print without loading the addon, and
// the family list must have exactly ONE definition (the drift guard's).
import { CHAT_FAMILY_IDS, coldTierRestoreFamilyList } from '@mlx-node/agent/catalog';
import { readAgentDefaultModel } from '@mlx-node/agent/paths';

import {
  agentUpdateStatus,
  applyAgentUpdate,
  launchGrokAgent,
  type AgentUpdateDeps,
  type AgentUpdateStatus,
  type GrokAgentLaunchOptions,
} from './grok-build.js';

export { expandAgentDir } from '@mlx-node/agent/paths';

export interface AgentArgScan {
  /** Value of `--models-dir` (the flag pair is removed from `passthrough`). */
  modelsDir?: string;
  /** `--models-dir` was present without a value — usage error. */
  modelsDirMissingValue: boolean;
  /** Enable bounded native inference diagnostics (`--trace-dir` implies this). */
  trace: boolean;
  /** Directory selected by `--trace-dir` (the flag pair is removed from `passthrough`). */
  traceDir?: string;
  /** `--trace-dir` was present without a value — usage error. */
  traceDirMissingValue: boolean;
  /**
   * Whether to persist the SSD paged cold tier (on by default; the mlx-owned
   * `--no-persist-cache` flag turns it off). A SINGLE process-wide boolean
   * applied to every family in `COLD_TIER_RESTORE_FAMILIES` — not to qwen3
   * alone. Lifted out of the argv like the other mlx flags — never forwarded
   * to the agent binary.
   */
  persistPagedCache: boolean;
  /**
   * `-h`/`--help` seen and this is NOT a binary-managed subcommand
   * (`models`/`export`/`doctor`/… print their own help inside the binary, so
   * those pass through untouched).
   */
  help: boolean;
  /** Leading `update` positional — intercepted by `run()` for the mlx-owned release download. */
  update: boolean;
  /**
   * The binary answers these BEFORE model resolution and exits — they need no
   * inference server: `--version`/`-v`, or any leading subcommand other than
   * the interactive default (see {@link GROK_SUBCOMMANDS}), plus `agent`
   * itself when `-h`/`--help` follows (its help also exits early). The
   * launcher skips spawning `mlx serve` for them.
   */
  grokOneShot: boolean;
  /** Args forwarded to the agent binary in their original order. */
  passthrough: string[];
}

/**
 * Leading positionals that are real subcommands of the agent binary. Every one
 * except `agent` (ACP server) is a local metadata operation that never needs a
 * running model — the launcher skips the inference host for them. `update`
 * also lands here so the leading-positional rule is one table; `run()`
 * intercepts it for the managed download before the handoff.
 */
const GROK_SUBCOMMANDS: ReadonlySet<string> = new Set([
  'agent',
  'config',
  'doctor',
  'leader',
  'logout',
  'login',
  'mcp',
  'plugin',
  'memory',
  'models',
  'sessions',
  'usage',
  'setup',
  'share',
  'wrap',
  'export',
  'trace',
  'update',
]);

/**
 * Flags in option-NAME position that unconditionally consume the NEXT token as
 * their value (grok CLI, crates/codegen/xai-grok-pager/src/app/cli.rs).
 * Re-verify on an upstream sync. Shared by BOTH argv walks — the
 * {@link scanAgentArgs} option-name lift and the {@link withDefaultModel}
 * model scan — so a value token after any of these is never re-interpreted:
 * `--system-prompt-override --models-dir` sets the prompt to "--models-dir",
 * so mlx must forward that value verbatim rather than strip it as its own
 * flag. The inline `--opt=value` form is not modeled.
 */
const VALUE_CONSUMING_ARGS: ReadonlySet<string> = new Set([
  '--model',
  '-m',
  '--agent',
  '--agents',
  '--agent-profile',
  '--cwd',
  '--leader-socket',
  '--session-id',
  '-s',
  '--resume',
  '-r',
  '--load',
  '--rules',
  '--append-system-prompt',
  '--system-prompt-override',
  // Alias of --system-prompt-override in the fork's clap definition.
  '--system-prompt',
  '--tools',
  '--disallowed-tools',
  '--max-turns',
  '--reasoning-effort',
  '--effort',
  '--permission-mode',
  '--allow',
  '--deny',
  '--prompt-file',
  '--prompt-json',
  '--output-format',
  '--json-schema',
  '--sandbox',
  '--worktree',
  '--ref',
  '--worktree-ref',
  '--cli-chat-proxy-base-url',
  '--xai-api-base-url',
  '--grok-ws-origin',
  '--grok-ws-url',
  '--plugin-dir',
  '--client-identifier',
  '--debug-file',
  '--storage-mode',
  '--compaction-mode',
  '--hunk-tracker-mode',
  '--background-wait-timeout',
  '--installer',
  '--trust-folder',
  '--terminal',
  '--fs-read',
  '--fs-write',
  '--memory-flush',
  '--todo-gate',
  '--local-workspace',
  '--local-workspace-cwd',
  '--restore-code',
]);

/**
 * Grok flag→value mapping; shared with delegate so option values stay opaque
 * to the mlx flag scan. `-p`/`--single` conditionally consumes the next token
 * (the prompt text), matching the binary's parser.
 */
export function agentOptionConsumesNext(argv: readonly string[], index: number): boolean {
  const arg = argv[index];
  const next = argv[i_next(index, argv)];
  if (next === undefined) return false;
  if (VALUE_CONSUMING_ARGS.has(arg!)) return true;
  if (arg === '--single' || arg === '-p') {
    return !next.startsWith('@') && (!next.startsWith('-') || next.startsWith('---'));
  }
  return false;
}

function i_next(index: number, argv: readonly string[]): number {
  return index + 1 < argv.length ? index + 1 : argv.length;
}

/**
 * Pure manual scan of `mlx agent`'s argv — see {@link AgentArgScan}.
 *
 * ONE value-aware walk (shared with {@link withDefaultModel}'s model scan via
 * {@link VALUE_CONSUMING_ARGS}): mlx's own options are recognized ONLY in an
 * option-NAME position. A token sitting in a value-consumer's value slot is
 * forwarded verbatim, never hijacked as mlx's flag. Routing (help/update/
 * subcommand) reads the value-aware passthrough head, so a stripped
 * `--models-dir` pair cannot mask what the binary will see at argv[0].
 */
export function scanAgentArgs(argv: string[]): AgentArgScan {
  const passthrough: string[] = [];
  let modelsDir: string | undefined;
  let modelsDirMissingValue = false;
  let trace = false;
  let traceDir: string | undefined;
  let traceDirMissingValue = false;
  let persistPagedCache = true;
  let helpSeen = false;
  let versionSeen = false;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;

    if (arg === '--') {
      passthrough.push(...argv.slice(i));
      break;
    }

    // Value-consumer in an option-NAME position: the following token is its
    // VALUE, never an option name. Forward BOTH verbatim and skip the value so
    // it is never interpreted as mlx's --models-dir/help.
    if (agentOptionConsumesNext(argv, i)) {
      passthrough.push(arg, argv[i + 1]!);
      i++;
      continue;
    }

    // mlx-only options, recognized ONLY here (an option-NAME position).
    if (arg === '--models-dir') {
      const next = argv[i + 1];
      // The SPACE-form value must be a real path token: absent, empty, or
      // option-looking (`-…`) values are usage errors. Consuming an option
      // here would swallow the next flag. Dash-leading dirs must use the
      // `--models-dir=<dir>` form.
      if (next === undefined || next.startsWith('-')) {
        modelsDirMissingValue = true;
      } else if (next.length === 0) {
        modelsDirMissingValue = true;
        i++; // the empty token was the (unusable) value — consume it
      } else {
        modelsDir = next;
        i++;
      }
      continue;
    }
    if (arg.startsWith('--models-dir=')) {
      const value = arg.slice('--models-dir='.length);
      if (value.length === 0) {
        modelsDirMissingValue = true;
      } else {
        modelsDir = value;
      }
      continue;
    }
    if (arg === '--trace') {
      trace = true;
      continue;
    }
    if (arg === '--no-persist-cache') {
      persistPagedCache = false;
      continue;
    }
    if (arg === '--trace-dir') {
      trace = true;
      const next = argv[i + 1];
      if (next === undefined || next.startsWith('-')) {
        traceDirMissingValue = true;
      } else if (next.length === 0) {
        traceDirMissingValue = true;
        i++;
      } else {
        traceDir = next;
        i++;
      }
      continue;
    }
    if (arg.startsWith('--trace-dir=')) {
      trace = true;
      const value = arg.slice('--trace-dir='.length);
      if (value.length === 0) {
        traceDirMissingValue = true;
      } else {
        traceDir = value;
      }
      continue;
    }
    if (arg === '-h' || arg === '--help') {
      helpSeen = true;
    }
    if (arg === '--version' || arg === '-v' || arg === '-V') {
      versionSeen = true;
    }
    passthrough.push(arg);
  }

  // Route on what the binary will actually see at argv[0] — the value-aware
  // passthrough head — so a stripped `--models-dir` pair or a consumed value
  // cannot mask a subcommand or the blocked `update`.
  const leadingCommand = passthrough[0] ?? '';
  const isSubcommand = GROK_SUBCOMMANDS.has(leadingCommand);
  return {
    modelsDir,
    modelsDirMissingValue,
    trace,
    traceDir,
    traceDirMissingValue,
    persistPagedCache,
    help: helpSeen && !isSubcommand,
    update: leadingCommand === 'update',
    grokOneShot: versionSeen || (isSubcommand && (leadingCommand !== 'agent' || helpSeen)),
    passthrough,
  };
}

const DEFAULT_AGENT_LOG_FILTER = 'mlx_core::inference=info,mlx_core::decode=info';

export interface AgentTracingSetupOptions {
  /** @internal Hermetic environment seam for tests. */
  env?: NodeJS.ProcessEnv;
  /** @internal Hermetic clock seam for tests. */
  now?: Date;
  /** @internal Hermetic home seam for tests. */
  homeDir?: string;
  /** @internal Hermetic pid seam for tests. */
  pid?: number;
  /** @internal Where the "inference log" announcement goes (stderr default). */
  announce?: (line: string) => void;
}

/**
 * Configure the native Rust `tracing` subscriber before the spawned `mlx
 * serve` loads the addon. Env is exported into THIS process so the serve
 * subprocess inherits it.
 */
export function configureAgentTracing(
  scan: Pick<AgentArgScan, 'trace' | 'traceDir'>,
  options: AgentTracingSetupOptions = {},
): string | undefined {
  const env = options.env ?? process.env;
  const requested = scan.trace || scan.traceDir !== undefined;
  if (!requested) return undefined;

  env.MLX_NODE_LOG ??= DEFAULT_AGENT_LOG_FILTER;

  const explicitFile = env.MLX_NODE_LOG_FILE;
  let traceDir: string | undefined;
  let logFile: string;
  if (explicitFile !== undefined && explicitFile.trim() !== '') {
    logFile = explicitFile.trim();
  } else {
    traceDir = scan.traceDir
      ? resolve(scan.traceDir)
      : join(
          options.homeDir ?? homedir(),
          '.mlx-node',
          'logs',
          'agent',
          `${(options.now ?? new Date()).toISOString().replace(/[:.]/g, '-')}-pid-${options.pid ?? process.pid}`,
        );
    logFile = join(traceDir, 'inference.log');
    env.MLX_NODE_LOG_FILE = logFile;
  }

  // Diagnostics are never a launch prerequisite. Pre-create private paths when
  // possible; the subscriber in the serve subprocess opens the real writer.
  try {
    const parent = dirname(logFile);
    mkdirSync(parent, { recursive: true, mode: 0o700 });
    if (traceDir !== undefined) chmodSync(traceDir, 0o700);
    const fd = openSync(logFile, 'a', 0o600);
    closeSync(fd);
    chmodSync(logFile, 0o600);
  } catch {
    // Best effort by contract: diagnostics I/O must never prevent startup.
  }

  (options.announce ?? ((line: string) => console.error(line)))(`mlx agent: inference log ${logFile}`);
  return logFile;
}

/**
 * Agent-bin flags whose presence means the user already picked a model or a
 * prior session, so {@link withDefaultModel} must not inject `-m`.
 * `-m`/`--model` is an explicit pick; `-c`/`--continue`, `-r`/`--resume`,
 * `-s`/`--session-id`, `--fork-session`, and `--load` restore a session whose
 * own saved model should win.
 */
const MODEL_OR_SESSION_CARRIER_ARGS: ReadonlySet<string> = new Set([
  '-m',
  '--model',
  '-c',
  '--continue',
  '-r',
  '--resume',
  '-s',
  '--session-id',
  '--fork-session',
  '--load',
]);

/** Collect real option names, respecting value-consumers and `--`. */
function collectAgentOptionNames(argv: readonly string[]): Set<string> {
  const optionNames = new Set<string>();
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i]!;
    if (token === '--') break;
    optionNames.add(token);
    if (agentOptionConsumesNext(argv, i)) {
      i++;
    }
  }
  return optionNames;
}

/**
 * Inject `-m <default>` on a fresh run so the session lands on the locally
 * chosen default. Suppressed entirely when the user picked a model or named a
 * session to resume — the binary's catalog comes from `/v1/models`, so every
 * listed model is already local; no provider-scope juggling remains.
 */
export function withDefaultModel(passthrough: string[], defaultModelId: string): string[] {
  const optionNames = collectAgentOptionNames(passthrough);
  if (Array.from(MODEL_OR_SESSION_CARRIER_ARGS).some((arg) => optionNames.has(arg))) {
    return passthrough;
  }
  return ['-m', defaultModelId, ...passthrough];
}

/** The default model id grok's `[models]` table persisted. */
export interface PersistedGrokDefault {
  modelId: string;
}

/**
 * Read the persisted `[models] default = "…"` the binary wrote to
 * `~/.mlx-agent/config.toml` (its `/model` picker persists there) via the
 * shared reader in `@mlx-node/agent/paths`.
 */
export function readPersistedDefaultModel(agentDir?: string): PersistedGrokDefault | undefined {
  const modelId = readAgentDefaultModel(agentDir);
  return modelId === undefined ? undefined : { modelId };
}

/**
 * Pick the `-m` value {@link withDefaultModel} injects on a fresh run:
 * a still-discovered persisted pick, else the first discovered model.
 */
export function chooseDefaultModel(
  models: readonly string[],
  persisted: PersistedGrokDefault | undefined,
): { modelId: string; notice?: string } {
  const fallback = models[0]!;
  if (persisted !== undefined && models.includes(persisted.modelId)) {
    return { modelId: persisted.modelId };
  }
  return { modelId: fallback };
}

/**
 * mlx-side help text; the binary's own flag list prints via a forwarded
 * `--help` (it exits before serving). Exported so tests can assert the
 * `--no-persist-cache` copy matches the allowlist it applies to.
 */
export function agentPreambleText(): string {
  return `
mlx agent — Grok Build coding agent running fully offline on MLX

Usage:
  mlx agent [options]

mlx options (handled before the agent binary sees the args):
  --models-dir <dir>        Local models directory (default: ~/.mlx-node/models;
                            also via MLX_MODELS_DIR or ~/.mlx-node/config.json).
                            Dash-leading paths need the --models-dir=<dir> form.
  --trace                   Enable bounded native inference diagnostics.
  --trace-dir <dir>         Write inference.log in this directory (implies
                            --trace; dash-leading paths need --trace-dir=<dir>).
  --no-persist-cache        Disable the on-by-default SSD cold tier for persisted
                            paged prefix blocks. One switch for ALL restore-eligible
                            families (${coldTierRestoreFamilyList().join(', ')});
                            every other family never persists, flag or not.

First run: when no local model exists, an interactive wizard offers a curated
download. Agent config home: ~/.mlx-agent (override: GROK_HOME).

Environment:
  MLX_AGENT_BIN             Path to a locally built agent binary (dev override;
                            default: <repo>/grok-build/target/release/mlx-agent,
                            then the prebuilt download under ~/.mlx-node/bin).
  MLX_AGENT_TAG             Pin the release tag 'mlx agent update' installs and
                            downloads resolve (default: newest published).
  MLX_CLI_ENTRY             Path to this CLI's entry point, used to spawn the
                            inference host (default: the running 'mlx' bin).
  MLX_NODE_LOG              Override the Rust tracing target filter used by --trace.
  MLX_NODE_LOG_FILE         Override the Rust tracing log file used by --trace.

Notes:
  The agent binary is the forked Grok Build TUI. It discovers models from the
  spawned inference host's /v1/models and speaks the Anthropic Messages wire
  protocol (/v1/messages) — the same path 'mlx launch claude' uses.
  'mlx agent update' downloads the newest mlx-agent release into
  ~/.mlx-node/bin (shadowed by MLX_AGENT_BIN or a vendored build; --check
  reports status as JSON). Subcommands (models, sessions, export, doctor, mcp, plugin,
  memory, config, leader, login, logout, setup, share, usage, wrap, trace)
  pass through to the binary and run without the inference host.

Binary options:
`;
}

/** @internal Print {@link agentPreambleText} ahead of the binary's flag list. */
function printAgentPreamble(): void {
  console.log(agentPreambleText());
}

/**
 * Injectable seams for {@link run}'s argv-routing tests. Production leaves
 * them unset and fills each via the deferred imports; types are `typeof
 * import(...)` lookups (erased at compile time) so the module stays importable
 * without the native addon.
 */
export interface AgentRunDeps {
  resolveModelsDir?: (typeof import('@mlx-node/server/host/paths'))['resolveModelsDir'];
  /** Discover local chat models (native-free path via @mlx-node/lm). */
  discoverModels?: (dir: string) => Promise<{ name: string }[]>;
  /** Spawn the inference host + agent binary. */
  launch?: (opts: GrokAgentLaunchOptions) => Promise<number>;
  /** Whole first-run wizard step (imports + IO wiring included). */
  wizard?: (modelsDir: string) => Promise<void>;
  /** Persisted-default reader; production = {@link readPersistedDefaultModel}. */
  readPersistedDefault?: typeof readPersistedDefaultModel;
  /** `mlx agent update --check`: report-only probe (no download). */
  updateStatus?: (deps: AgentUpdateDeps) => Promise<AgentUpdateStatus>;
  /** `mlx agent update`: download + install into the managed cache. */
  applyUpdate?: (deps: AgentUpdateDeps) => Promise<{ status: AgentUpdateStatus; downloaded: boolean }>;
  /** Update-flow seams forwarded to both update functions (test-only). */
  updateDeps?: AgentUpdateDeps;
}

/** Production wizard step: interactive catalog pick + download. */
async function runProductionWizard(modelsDir: string): Promise<void> {
  const { runFirstRunWizard } = await import('./wizard.js');
  const { select } = await import('@inquirer/prompts');
  const { run: downloadModel } = await import('../download-model.js');
  await runFirstRunWizard({
    io: {
      select: (opts) => select(opts),
      isTTY: Boolean(process.stdin.isTTY && process.stdout.isTTY),
      log: (line) => console.log(line),
    },
    download: (downloadArgv) => downloadModel(downloadArgv),
    modelsDir,
  });
}

/** Discover model names only (native-free); production = @mlx-node/lm discovery. */
async function discoverModelNames(modelsDir: string): Promise<{ name: string }[]> {
  const { discoverLocalChatModels } = await import('@mlx-node/lm/model-discovery');
  const models = await discoverLocalChatModels(modelsDir);
  return models.map((m) => ({ name: m.name }));
}

const PINNED_NOTICE: Readonly<Partial<Record<AgentUpdateStatus['pinned'], string>>> = {
  env: 'the installed release stays unused while MLX_AGENT_BIN points elsewhere',
  vendored: 'the installed release stays unused while grok-build/target/release/mlx-agent exists',
};

/**
 * `mlx agent update [--check]`: mlx-owned self-update. The fork binary's own
 * `update` subcommand was built for npm-style installs and can only corrupt a
 * download-managed binary, so the positional is intercepted here instead of
 * forwarded. `--check` emits the machine-readable status the desktop tray
 * consumes.
 */
async function runAgentUpdate(scan: AgentArgScan, deps: AgentRunDeps): Promise<void> {
  // scan.help stays false for subcommand argv, so read the passthrough directly.
  if (scan.passthrough.includes('--help') || scan.passthrough.includes('-h')) {
    console.log('Usage: mlx agent update [--check]\n\n  --check   print update status as JSON and exit');
    return;
  }
  const updateDeps = deps.updateDeps ?? {};
  const checkOnly = scan.passthrough.includes('--check');
  try {
    if (checkOnly) {
      const status = await (deps.updateStatus ?? agentUpdateStatus)(updateDeps);
      console.log(JSON.stringify(status));
      return;
    }
    const { status, downloaded } = await (deps.applyUpdate ?? applyAgentUpdate)(updateDeps);
    if (downloaded) {
      console.log(`mlx agent: updated to ${status.latest} → ${status.managedPath}`);
    } else {
      console.log(`mlx agent: already up to date (${status.latest})`);
    }
    const pin = PINNED_NOTICE[status.pinned];
    if (pin !== undefined) console.warn(`mlx agent: warning — ${pin}`);
  } catch (error) {
    console.error(`mlx agent: update failed — ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  }
}

export async function run(argv: string[], deps: AgentRunDeps = {}): Promise<void> {
  const scan = scanAgentArgs(argv);

  if (scan.update) {
    await runAgentUpdate(scan, deps);
    return;
  }

  if (scan.modelsDirMissingValue) {
    console.error('Missing value for --models-dir (a dash-leading path needs the --models-dir=<dir> form)');
    process.exitCode = 1;
    return;
  }

  if (scan.traceDirMissingValue) {
    console.error('Missing value for --trace-dir (a dash-leading path needs the --trace-dir=<dir> form)');
    process.exitCode = 1;
    return;
  }

  // Must run before the spawned `mlx serve` starts so its native tracing
  // subscriber picks up the env we export here.
  configureAgentTracing(scan);

  const launch = deps.launch ?? launchGrokAgent;

  if (scan.help) {
    printAgentPreamble();
    // The binary prints its full flag list and exits; no server needed.
    process.exitCode = await launch({ argv: ['--help'], needsServer: false });
    return;
  }

  // One-shots and binary subcommands (models/sessions/export/…) resolve and
  // exit inside the binary — forward verbatim with no server and no wizard.
  if (scan.grokOneShot) {
    process.exitCode = await launch({ argv: scan.passthrough, needsServer: false });
    return;
  }

  const resolveModelsDir = deps.resolveModelsDir ?? (await import('@mlx-node/server/host/paths')).resolveModelsDir;
  const discover = deps.discoverModels ?? discoverModelNames;
  const modelsDir = resolveModelsDir(scan.modelsDir);

  let models = await discover(modelsDir);

  if (models.length === 0) {
    try {
      await (deps.wizard ?? runProductionWizard)(modelsDir);
    } catch (error) {
      console.error(error instanceof Error ? error.message : String(error));
      process.exitCode = 1;
      return;
    }

    models = await discover(modelsDir);
    if (models.length === 0) {
      console.error(`No usable model found in ${modelsDir} after the download.`);
      console.error(
        `Expected a subdirectory with a config.json for a supported family (${CHAT_FAMILY_IDS.join('/')}).`,
      );
      console.error('Check the download output above, or point --models-dir at an existing models directory.');
      process.exitCode = 1;
      return;
    }
  }

  const persisted = (deps.readPersistedDefault ?? readPersistedDefaultModel)();
  const { modelId } = chooseDefaultModel(
    models.map((m) => m.name),
    persisted,
  );
  const agentArgv = withDefaultModel(scan.passthrough, modelId);

  process.exitCode = await launch({
    argv: agentArgv,
    modelsDir,
    needsServer: true,
  });
}
