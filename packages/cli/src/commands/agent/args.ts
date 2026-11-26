import { chmodSync, closeSync, mkdirSync, openSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';

// Native-free subpaths: the help path must print without loading the addon, and
// the family list must have exactly ONE definition (the drift guard's).
import { coldTierRestoreFamilyList } from '@mlx-node/agent/catalog';
import { readAgentDefaultModel } from '@mlx-node/agent/paths';
import { parseCacheDir, parseMaxCacheDisk, resolveCacheDisk } from '@mlx-node/server/host/cache-disk';
import { parseDecodeShare } from '@mlx-node/server/host/decode-share';
import { parseIdleRelease } from '@mlx-node/server/host/idle-release';
import { parseMaxContext } from '@mlx-node/server/host/max-context';
import { parseMaxMemory } from '@mlx-node/server/host/max-memory';
import {
  PERSIST_PAGED_CACHE_DEFAULT,
  resolvePersistPagedCache,
  WHOLE_PROMPT_CAPTURE_BLOCKS,
  WHOLE_PROMPT_CAPTURE_BUDGET_MS,
  type PersistPagedCache,
} from '@mlx-node/server/host/persist-cache';

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
  /** Native text of `--max-cache-disk` (lifted out of the argv, forwarded to `mlx serve`). */
  cacheDisk?: string;
  /**
   * The paged KV disk cache for the spawned `mlx serve`: on by default,
   * `--no-persist-cache` sets the flag off.
   */
  persistPagedCache: PersistPagedCache;
  /** Absolute `--cache-dir` (lifted out of the argv, forwarded to `mlx serve`). */
  cacheDir?: string;
  /** A cache flag had a missing or bad value, or the cache flags conflict: the usage error. */
  cacheDiskError?: string;
  /** Value of `--kv-format` (lifted out of the argv, forwarded to `mlx serve`). */
  kvFormat?: 'int8' | 'bf16';
  /** `--kv-format` had a missing or unknown value (the raw value, `''` when missing) — usage error. */
  kvFormatInvalid?: string;
  /** Bytes of `--max-memory` (lifted out of the argv, forwarded to `mlx serve`); unset for `auto`. */
  maxMemoryBytes?: number;
  /** `--max-memory` had a missing or unparseable value (the raw value, `''` when missing) — usage error. */
  maxMemoryInvalid?: string;
  /** Tokens of `--max-context` (lifted out of the argv, forwarded to `mlx serve`); unset for `auto`. */
  maxContextTokens?: number;
  /** `--max-context` had a missing or unparseable value (the raw value, `''` when missing) — usage error. */
  maxContextInvalid?: string;
  /** Value of `--decode-share` (lifted out of the argv, forwarded to `mlx serve`). */
  decodeShare?: number;
  /** `--decode-share` had a missing or unparseable value (the raw value, `''` when missing) — usage error. */
  decodeShareInvalid?: string;
  /** Seconds of `--idle-release` (lifted out of the argv, forwarded to `mlx serve`); `Infinity` for `off`. */
  idleReleaseSeconds?: number;
  /** `--idle-release` had a missing or unparseable value (the raw value, `''` when missing) — usage error. */
  idleReleaseInvalid?: string;
  /** `-h`/`--help` seen and this is NOT a binary-managed subcommand. */
  help: boolean;
  /** Leading `update` positional — intercepted by `run()` for the mlx-owned release download. */
  update: boolean;
  /** The binary answers these BEFORE model resolution and exits — they need no inference server. */
  grokOneShot: boolean;
  /** Args forwarded to the agent binary in their original order. */
  passthrough: string[];
}

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

function i_next(index: number, argv: readonly string[]): number {
  return index + 1 < argv.length ? index + 1 : argv.length;
}

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

/** Pure manual scan of `mlx agent`'s argv — see {@link AgentArgScan}. */
export function scanAgentArgs(
  argv: string[],
  env: NodeJS.ProcessEnv | Readonly<Record<string, string | undefined>> = process.env,
): AgentArgScan {
  const passthrough: string[] = [];
  let modelsDir: string | undefined;
  let modelsDirMissingValue = false;
  let trace = false;
  let traceDir: string | undefined;
  let traceDirMissingValue = false;
  let cacheDisk: string | undefined;
  let persistCacheOff = false;
  let cacheDir: string | undefined;
  let cacheDiskError: string | undefined;
  const takeCacheValue = (parse: (value: string) => string, value: string, assign: (parsed: string) => void) => {
    try {
      assign(parse(value));
    } catch (error) {
      cacheDiskError = error instanceof Error ? error.message : String(error);
    }
  };
  const takeCacheDisk = (value: string): void =>
    takeCacheValue(parseMaxCacheDisk, value, (parsed) => (cacheDisk = parsed));
  const takeCacheDir = (value: string): void => takeCacheValue(parseCacheDir, value, (parsed) => (cacheDir = parsed));
  let kvFormat: 'int8' | 'bf16' | undefined;
  let kvFormatInvalid: string | undefined;
  const takeKvFormat = (value: string): void => {
    if (value === 'int8' || value === 'bf16') {
      kvFormat = value;
      kvFormatInvalid = undefined;
    } else {
      kvFormatInvalid = value;
    }
  };
  let maxMemoryBytes: number | undefined;
  let maxMemoryInvalid: string | undefined;
  const takeMaxMemory = (value: string): void => {
    try {
      maxMemoryBytes = parseMaxMemory(value);
      maxMemoryInvalid = undefined;
    } catch {
      maxMemoryInvalid = value;
    }
  };
  let maxContextTokens: number | undefined;
  let maxContextInvalid: string | undefined;
  const takeMaxContext = (value: string): void => {
    try {
      maxContextTokens = parseMaxContext(value);
      maxContextInvalid = undefined;
    } catch {
      maxContextInvalid = value;
    }
  };
  let decodeShare: number | undefined;
  let decodeShareInvalid: string | undefined;
  const takeDecodeShare = (value: string): void => {
    try {
      decodeShare = parseDecodeShare(value);
      decodeShareInvalid = undefined;
    } catch {
      decodeShareInvalid = value;
    }
  };
  let idleReleaseSeconds: number | undefined;
  let idleReleaseInvalid: string | undefined;
  const takeIdleRelease = (value: string): void => {
    try {
      idleReleaseSeconds = parseIdleRelease(value);
      idleReleaseInvalid = undefined;
    } catch {
      idleReleaseInvalid = value;
    }
  };
  let helpSeen = false;
  let versionSeen = false;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;

    if (arg === '--') {
      passthrough.push(...argv.slice(i));
      break;
    }

    if (agentOptionConsumesNext(argv, i)) {
      passthrough.push(arg, argv[i + 1]!);
      i++;
      continue;
    }

    if (arg === '--models-dir') {
      const next = argv[i + 1];
      if (next === undefined || next.startsWith('-')) {
        modelsDirMissingValue = true;
      } else if (next.length === 0) {
        modelsDirMissingValue = true;
        i++;
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
      persistCacheOff = true;
      continue;
    }
    if (arg === '--max-cache-disk' || arg === '--cache-dir') {
      const next = argv[i + 1];
      if (next === undefined || next.startsWith('-')) {
        cacheDiskError = `Missing value for ${arg}`;
      } else {
        (arg === '--max-cache-disk' ? takeCacheDisk : takeCacheDir)(next);
        i++;
      }
      continue;
    }
    if (arg.startsWith('--max-cache-disk=')) {
      takeCacheDisk(arg.slice('--max-cache-disk='.length));
      continue;
    }
    if (arg.startsWith('--cache-dir=')) {
      takeCacheDir(arg.slice('--cache-dir='.length));
      continue;
    }
    if (arg === '--kv-format') {
      const next = argv[i + 1];
      if (next === undefined || next.startsWith('-')) {
        kvFormatInvalid = '';
      } else {
        takeKvFormat(next);
        i++;
      }
      continue;
    }
    if (arg.startsWith('--kv-format=')) {
      takeKvFormat(arg.slice('--kv-format='.length));
      continue;
    }
    if (arg === '--max-memory') {
      const next = argv[i + 1];
      if (next === undefined || next.startsWith('-')) {
        maxMemoryInvalid = '';
      } else {
        takeMaxMemory(next);
        i++;
      }
      continue;
    }
    if (arg.startsWith('--max-memory=')) {
      takeMaxMemory(arg.slice('--max-memory='.length));
      continue;
    }
    if (arg === '--max-context') {
      const next = argv[i + 1];
      if (next === undefined || next.startsWith('-')) {
        maxContextInvalid = '';
      } else {
        takeMaxContext(next);
        i++;
      }
      continue;
    }
    if (arg.startsWith('--max-context=')) {
      takeMaxContext(arg.slice('--max-context='.length));
      continue;
    }
    if (arg === '--decode-share') {
      const next = argv[i + 1];
      if (next === undefined || next.startsWith('-')) {
        decodeShareInvalid = '';
      } else {
        takeDecodeShare(next);
        i++;
      }
      continue;
    }
    if (arg.startsWith('--decode-share=')) {
      takeDecodeShare(arg.slice('--decode-share='.length));
      continue;
    }
    if (arg === '--idle-release') {
      const next = argv[i + 1];
      if (next === undefined || next.startsWith('-')) {
        idleReleaseInvalid = '';
      } else {
        takeIdleRelease(next);
        i++;
      }
      continue;
    }
    if (arg.startsWith('--idle-release=')) {
      takeIdleRelease(arg.slice('--idle-release='.length));
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

  const persistPagedCache: PersistPagedCache = persistCacheOff
    ? { flag: false, byDefault: PERSIST_PAGED_CACHE_DEFAULT.byDefault }
    : PERSIST_PAGED_CACHE_DEFAULT;
  if (cacheDiskError === undefined) {
    try {
      resolveCacheDisk({
        cacheDisk: cacheDisk ?? 'auto',
        persistPagedCache: resolvePersistPagedCache(persistPagedCache, env),
        offByFlag: persistCacheOff,
        ...(cacheDir !== undefined ? { cacheDir } : {}),
      });
    } catch (error) {
      cacheDiskError = error instanceof Error ? error.message : String(error);
    }
  }

  const leadingCommand = passthrough[0] ?? '';
  const isSubcommand = GROK_SUBCOMMANDS.has(leadingCommand);
  return {
    modelsDir,
    modelsDirMissingValue,
    trace,
    traceDir,
    traceDirMissingValue,
    persistPagedCache,
    ...(cacheDisk !== undefined ? { cacheDisk } : {}),
    ...(cacheDir !== undefined ? { cacheDir } : {}),
    ...(cacheDiskError !== undefined ? { cacheDiskError } : {}),
    ...(kvFormat !== undefined ? { kvFormat } : {}),
    ...(kvFormatInvalid !== undefined ? { kvFormatInvalid } : {}),
    ...(maxMemoryBytes !== undefined ? { maxMemoryBytes } : {}),
    ...(maxMemoryInvalid !== undefined ? { maxMemoryInvalid } : {}),
    ...(maxContextTokens !== undefined ? { maxContextTokens } : {}),
    ...(maxContextInvalid !== undefined ? { maxContextInvalid } : {}),
    ...(decodeShare !== undefined ? { decodeShare } : {}),
    ...(decodeShareInvalid !== undefined ? { decodeShareInvalid } : {}),
    ...(idleReleaseSeconds !== undefined ? { idleReleaseSeconds } : {}),
    ...(idleReleaseInvalid !== undefined ? { idleReleaseInvalid } : {}),
    help: helpSeen && !isSubcommand,
    update: leadingCommand === 'update',
    grokOneShot: versionSeen || (isSubcommand && (leadingCommand !== 'agent' || helpSeen)),
    passthrough,
  };
}

const DEFAULT_AGENT_LOG_FILTER = 'mlx_core=warn,mlx_paged_attn=warn,mlx_core::inference=info,mlx_core::decode=info';

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

export function withDefaultModel(passthrough: string[], defaultModelId: string): string[] {
  const optionNames = collectAgentOptionNames(passthrough);
  if (Array.from(MODEL_OR_SESSION_CARRIER_ARGS).some((arg) => optionNames.has(arg))) {
    return passthrough;
  }
  return ['-m', defaultModelId, ...passthrough];
}

const DISABLE_WEB_SEARCH_FLAG = '--disable-web-search';

export function withWebSearchDisabled(argv: string[]): string[] {
  return collectAgentOptionNames(argv).has(DISABLE_WEB_SEARCH_FLAG) ? argv : [DISABLE_WEB_SEARCH_FLAG, ...argv];
}

/** The `-m`/`--model` value in option-name position, if any. */
export function agentModelArg(argv: readonly string[]): string | undefined {
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i]!;
    if (token === '--') return undefined;
    if (token.startsWith('--model=')) return token.slice('--model='.length) || undefined;
    if ((token === '-m' || token === '--model') && i + 1 < argv.length) return argv[i + 1];
    if (agentOptionConsumesNext(argv, i)) i++;
  }
  return undefined;
}

/** The default model id grok's `[models]` table persisted. */
export interface PersistedGrokDefault {
  modelId: string;
}

export function readPersistedDefaultModel(agentDir?: string): PersistedGrokDefault | undefined {
  const modelId = readAgentDefaultModel(agentDir);
  return modelId === undefined ? undefined : { modelId };
}

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
  --kv-format <fmt>         Target K/V cache format: int8 or bf16 (default: int8
                            where the int8 kernels serve the model, else bf16).
                            Sets the block-paged pool (qwen3_5, qwen3_5_moe,
                            qwen4_exp).
                            bf16 uses more memory.
  --max-memory <size>       Metal memory budget, e.g. 28G; it can only lower the
                            automatic budget (default: auto).
  --max-context <tokens>    Context token limit, e.g. 100K or 1M (K = 1024 tokens,
                            M = 1000000 tokens), up to the model's limit, at most
                            1048576; past its native window a model with YaRN
                            turns it on, which is static and changes answers at
                            every length, so pass more than the native window
                            only when a request needs it (default: auto, the
                            native window when memory holds it).
  --decode-share <share>    Decode time owed per unit of prefill time while
                            other requests generate (default: 0.5). Higher
                            keeps their output faster during a long prompt and
                            slows that prompt; 0 alternates one command each.
  --idle-release <duration> Time without a request before the engine unwires its
                            memory and frees the weights: seconds, or with an s,
                            m or h suffix, e.g. 30m (default: 10m); off keeps
                            both, though POST /v1/memory/release still frees
                            the weights.
  --no-persist-cache        Turn off the paged KV disk cache, which is on by
                            default (a conversation resumes after an engine
                            restart without prefilling again).
                            MLX_PERSIST_PAGED_CACHE=0 also turns it off when this
                            flag is absent. One switch for ALL restore-eligible
                            families (${coldTierRestoreFamilyList().join(', ')})
                            and the Qwen3.5 dense and MoE state tier and the
                            Flash Next state tier, whose files are then
                            temporary; every other family never persists, flag
                            or not. A turn of those
                            restore-eligible families writes the whole prompt
                            (up to ${WHOLE_PROMPT_CAPTURE_BLOCKS} blocks within
                            ${WHOLE_PROMPT_CAPTURE_BUDGET_MS} ms) before it ends.
  --max-cache-disk <size>   SSD quota for the states of Qwen3.5 dense and MoE
                            and Flash Next loads: auto (default; a quarter of
                            the volume's free space beyond its floor, at most
                            32G), a size such as 16G, or 0 to disable.
  --cache-dir <dir>         Where the persistent Qwen3.5 and Flash Next state
                            tiers keep their files (default:
                            ~/.mlx-node/cache/prefix-cache).

First run: when no local model exists, an interactive wizard offers a curated
download. Agent config home: ~/.mlx-agent (override: MLX_AGENT_HOME).

Environment:
  MLX_AGENT_BIN             Path to a locally built agent binary (dev override;
                            default: <repo>/grok-build/target/release/mlx-agent,
                            then the prebuilt download under ~/.mlx-node/bin).
  MLX_AGENT_TAG             Pin the release tag 'mlx agent update' installs and
                            downloads resolve (default: newest published).
  MLX_CLI_ENTRY             Path to this CLI's entry point, used to spawn the
                            inference host (default: the running 'mlx' bin).
  MLX_COLD_CAPTURE_BLOCKS_PER_TURN, MLX_COLD_CAPTURE_BUDGET_MS
                            Cap one turn's disk cache write of the
                            restore-eligible families (blocks, ms); a set
                            value beats the whole-prompt default.
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
