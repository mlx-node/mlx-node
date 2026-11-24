/** `mlx launch claude` — start a local inference host and spawn Claude Code pointed at it. */

import { spawn } from 'node:child_process';
import type { ChildProcess } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { accessSync, closeSync, constants as fsConstants, fstatSync, openSync, readSync, statSync } from 'node:fs';
import { constants as osConstants } from 'node:os';
import { dirname, isAbsolute, join, resolve } from 'node:path';
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

import { withLoopbackNoProxy } from '../loopback-env.js';

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

const DEFAULT_WIN_PATHEXT = '.COM;.EXE;.BAT;.CMD';
const DIRECT_WIN_EXEC_EXT_RE = /^\.(?:exe|com)$/i;
const BATCH_SHIM_EXT_RE = /^\.(?:cmd|bat)$/i;
const BATCH_SHIM_FILE_RE = /\.(?:cmd|bat)$/i;
const DIRECT_EXEC_TARGET_RE = /\.(?:exe|com)$/i;
const JS_SCRIPT_TARGET_RE = /\.(?:js|mjs|cjs)$/i;
const MAX_SHIM_BYTES = 64 * 1024;
const CMD_SHIM_ENDLOCAL_PREFIX_RE =
  /^endlocal\s*&\s*goto\s+[^\s&|]+\s+2>nul\s*\|\|\s*title\s+%comspec%\s*&\s*(?:set\s+pathext=%pathext:;\.js;=;%\s*&\s*)?/i;

function readEnvCaseInsensitive(env: NodeJS.ProcessEnv, name: string): string | undefined {
  if (env[name] !== undefined) return env[name];
  const target = name.toLowerCase();
  for (const [key, value] of Object.entries(env)) {
    if (key.toLowerCase() === target && value !== undefined) return value;
  }
  return undefined;
}

function isRegularFile(filePath: string): boolean {
  try {
    return statSync(filePath).isFile();
  } catch {
    return false;
  }
}

function listClaudeCandidatesOnPath(
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
): string[] {
  const pathEnv = readEnvCaseInsensitive(env, 'PATH') ?? '';
  const found: string[] = [];
  if (platform === 'win32') {
    const rawExt = readEnvCaseInsensitive(env, 'PATHEXT') ?? DEFAULT_WIN_PATHEXT;
    const exts = rawExt
      .split(';')
      .map((entry) => entry.trim())
      .filter((entry) => entry !== '')
      .map((entry) => (entry.startsWith('.') ? entry : `.${entry}`));
    // Prefer direct PE executables (.exe, .com) over batch shims (.cmd, .bat)
    // within each PATH directory, preserving relative order inside each group
    // so native claude.exe wins even when PATHEXT lists .CMD before .EXE.
    const directExts = exts.filter((ext) => DIRECT_WIN_EXEC_EXT_RE.test(ext));
    const batchExts = exts.filter((ext) => BATCH_SHIM_EXT_RE.test(ext));
    const otherExts = exts.filter((ext) => !DIRECT_WIN_EXEC_EXT_RE.test(ext) && !BATCH_SHIM_EXT_RE.test(ext));
    const orderedExts = [...directExts, ...batchExts, ...otherExts];
    const names: string[] = [];
    const push = (candidate: string): void => {
      if (!names.includes(candidate)) names.push(candidate);
    };
    if (orderedExts.length === 0) {
      push('claude');
    } else {
      for (const ext of orderedExts) {
        const lower = ext.toLowerCase();
        push(`claude${lower}`);
        if (ext !== lower) push(`claude${ext}`);
      }
    }
    const seen = new Set<string>();
    for (const rawDir of pathEnv.split(';')) {
      let dir = rawDir.trim();
      if (dir.length >= 2 && dir.startsWith('"') && dir.endsWith('"')) dir = dir.slice(1, -1);
      if (!dir) continue;
      for (const name of names) {
        const candidate = join(dir, name);
        const key = candidate.toLowerCase();
        if (seen.has(key)) continue;
        try {
          if (!statSync(candidate).isFile()) continue;
          accessSync(candidate, fsConstants.F_OK);
          seen.add(key);
          found.push(candidate);
        } catch {
          /* not here */
        }
      }
    }
    return found;
  }
  const seen = new Set<string>();
  for (const dir of pathEnv.split(':')) {
    if (!dir) continue;
    const candidate = join(dir, 'claude');
    if (seen.has(candidate)) continue;
    try {
      if (!statSync(candidate).isFile()) continue;
      accessSync(candidate, fsConstants.X_OK);
      seen.add(candidate);
      found.push(candidate);
    } catch {
      /* not here */
    }
  }
  return found;
}

export function findClaudeOnPath(
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
): string | null {
  return listClaudeCandidatesOnPath(env, platform)[0] ?? null;
}

/**
 * Walk `PATH` and return the first launchable `claude` installation. On Windows,
 * stale or unrecognized `.cmd`/`.bat` wrappers are skipped so a valid later
 * installation (e.g. native `claude.exe`) is not shadowed; if candidates exist
 * and none can be launched safely, throws with the rejection details.
 */
export function resolveClaudeLaunch(
  claudeArgs: readonly string[],
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
  nodeExecPath: string = process.execPath,
): { claudeBin: string; spawnSpec: { file: string; args: string[] } } | null {
  const candidates = listClaudeCandidatesOnPath(env, platform);
  if (candidates.length === 0) return null;
  const rejected: string[] = [];
  for (const candidate of candidates) {
    try {
      const spawnSpec = claudeSpawnSpec(candidate, claudeArgs, platform, env, nodeExecPath);
      return { claudeBin: candidate, spawnSpec };
    } catch (err) {
      rejected.push(err instanceof Error ? err.message : String(err));
    }
  }
  throw new Error(rejected.join('\n'));
}

function unrecognizedWindowsShimError(shimPath: string, reason?: string): Error {
  const detail = reason ? ` (${reason})` : '';
  return new Error(
    `Refusing to forward arguments through cmd.exe for unrecognized Windows .cmd/.bat wrapper at ${shimPath}${detail}. ` +
      'Install native claude.exe or standard npm @anthropic-ai/claude-code so mlx launch claude can invoke it directly without shell interpolation.',
  );
}

function readBoundedShimFile(shimPath: string): string {
  const fd = openSync(shimPath, 'r');
  try {
    const stat = fstatSync(fd);
    if (!stat.isFile()) {
      throw new Error('path is not a regular file');
    }
    if (stat.size > MAX_SHIM_BYTES) {
      throw new Error(`file exceeds ${MAX_SHIM_BYTES} bytes`);
    }
    const buf = Buffer.alloc(stat.size + 1);
    let offset = 0;
    while (offset < buf.length) {
      const bytesRead = readSync(fd, buf, offset, buf.length - offset, offset);
      if (bytesRead === 0) break;
      offset += bytesRead;
    }
    if (offset > MAX_SHIM_BYTES) {
      throw new Error(`file exceeds ${MAX_SHIM_BYTES} bytes`);
    }
    return buf.subarray(0, offset).toString('utf8');
  } finally {
    closeSync(fd);
  }
}

function tokenizeBatchCommand(input: string): string[] | null {
  const tokens: string[] = [];
  let i = 0;
  while (i < input.length) {
    while (i < input.length && (input[i] === ' ' || input[i] === '\t')) i++;
    if (i >= input.length) break;
    let token = '';
    let inQuotes = false;
    while (i < input.length) {
      const ch = input[i];
      if (ch === '"') {
        inQuotes = !inQuotes;
        i++;
        continue;
      }
      if (!inQuotes && (ch === ' ' || ch === '\t')) break;
      if (!inQuotes && (ch === '&' || ch === '|' || ch === '<' || ch === '>' || ch === '^')) {
        return null;
      }
      token += ch;
      i++;
    }
    if (inQuotes || token === '') return null;
    tokens.push(token);
  }
  return tokens;
}

function isConcreteNodeRunner(token: string): boolean {
  const stripped = token.replace(/^(?:%dp0%|%~dp0)[\\/]*/i, '');
  const segments = stripped.split(/[\\/]/);
  const base = segments[segments.length - 1] ?? '';
  return /^node(?:\.exe)?$/i.test(base);
}

function isNodeRunner(token: string, progValues: readonly string[]): boolean {
  if (/^%_prog%$/i.test(token)) {
    return progValues.every((value) => isConcreteNodeRunner(value));
  }
  return isConcreteNodeRunner(token);
}

function expandShimTargetPath(rawTarget: string, shimDir: string): string | null {
  const dp0Match = rawTarget.match(/^(?:%dp0%|%~dp0)(.*)$/i);
  if (dp0Match) {
    const segments = dp0Match[1].split(/[\\/]+/).filter((seg) => seg !== '');
    if (segments.length === 0) return null;
    return resolve(shimDir, ...segments);
  }
  if (isAbsolute(rawTarget)) {
    return resolve(rawTarget);
  }
  return null;
}

/**
 * Resolve a standard Windows `.cmd` / `.bat` CLI shim (npm, pnpm, yarn, or a
 * direct `.exe`/`.com` wrapper) into a direct `spawn` target so arguments are
 * forwarded literally via `CreateProcessW` without ever invoking `cmd.exe`.
 *
 * Throws if the shim is unrecognized or its target does not exist on disk —
 * never falls back to `cmd.exe /c`.
 */
export function resolveWindowsClaudeShim(shimPath: string, nodeExecPath?: string): { file: string; args: string[] } {
  let content: string;
  try {
    content = readBoundedShimFile(shimPath);
  } catch (err) {
    throw unrecognizedWindowsShimError(shimPath, err instanceof Error ? err.message : String(err));
  }

  const shimDir = dirname(shimPath);
  const lines = content.replace(/^\uFEFF/, '').split(/\r?\n/);
  const progValues: string[] = [];
  let resolvedCandidate: { kind: 'node' | 'exe'; target: string } | null = null;

  for (const rawLine of lines) {
    let line = rawLine.trim();
    if (!line || line === ')') continue;
    if (line.startsWith('@')) line = line.slice(1).trimStart();
    if (!line) continue;

    // Strip standard npm/pnpm/yarn `cmd-shim` endLocal boilerplate before checking
    // boilerplate keywords so the trailing `"%_prog%" "%dp0%\...\cli.js" %*` is parsed.
    if (CMD_SHIM_ENDLOCAL_PREFIX_RE.test(line)) {
      line = line.replace(CMD_SHIM_ENDLOCAL_PREFIX_RE, '').trimStart();
      if (line.startsWith('@')) line = line.slice(1).trimStart();
      if (!line) continue;
    }

    // Allow only standard `cmd-shim` `SET` assignments (`_prog`, `dp0`, and
    // `PATHEXT` JS-stripping); reject any other `SET` (e.g. `CLAUDE_CONFIG_DIR`)
    // so a custom wrapper's environment or profile is never silently discarded.
    if (/^set(?:\s|$)/i.test(line)) {
      const assign = line.slice(3).trim();
      const unquoted =
        assign.length >= 2 && assign.startsWith('"') && assign.endsWith('"') ? assign.slice(1, -1) : assign;
      const eq = unquoted.indexOf('=');
      if (eq === -1) {
        throw unrecognizedWindowsShimError(shimPath, 'unsupported SET statement');
      }
      const varName = unquoted.slice(0, eq).trim().toLowerCase();
      const rawVal = unquoted.slice(eq + 1).trim();
      const varVal =
        rawVal.length >= 2 && rawVal.startsWith('"') && rawVal.endsWith('"') ? rawVal.slice(1, -1) : rawVal;
      if (varName === '_prog') {
        progValues.push(varVal);
      } else if (varName === 'dp0') {
        if (!/^%~dp0[\\/]?$/i.test(varVal)) {
          throw unrecognizedWindowsShimError(shimPath, `unsupported dp0 assignment: ${varVal}`);
        }
      } else if (varName === 'pathext') {
        if (!/^%pathext:;\.js;=;%$/i.test(varVal)) {
          throw unrecognizedWindowsShimError(shimPath, `unsupported PATHEXT assignment: ${varVal}`);
        }
      } else {
        throw unrecognizedWindowsShimError(shimPath, `unsupported environment assignment: ${varName}`);
      }
      continue;
    }

    // Allow only recognized `cmd-shim` boilerplate; reject custom branching or
    // side-effect commands rather than silently dropping wrapper logic.
    if (
      /^:(?:find_dp0|start)$/i.test(line) ||
      line.startsWith('::') ||
      /^rem(?:\s|$)/i.test(line) ||
      /^echo\s+off$/i.test(line) ||
      /^(?:setlocal|endlocal)$/i.test(line) ||
      /^goto\s+(?:start|:eof)$/i.test(line) ||
      /^call\s+:find_dp0$/i.test(line) ||
      /^exit\s+\/b(?:\s+%errorlevel%)?$/i.test(line) ||
      /^if\s+exist\s+"%(?:dp0%|~dp0)[\\/]?node\.exe"\s*\($/i.test(line) ||
      /^\)\s*else\s*\($/i.test(line)
    ) {
      continue;
    }

    // Every remaining executable command line must be a recognized `%*` forwarding call.
    const argsForwardMatch = line.match(/^(.*\S)\s+%\*$/);
    if (!argsForwardMatch) {
      throw unrecognizedWindowsShimError(shimPath, 'contains unsupported batch commands');
    }

    const tokens = tokenizeBatchCommand(argsForwardMatch[1]);
    if (!tokens || tokens.length === 0) {
      throw unrecognizedWindowsShimError(shimPath, 'contains unsupported shell syntax');
    }
    if (tokens[0].toLowerCase() === 'call') {
      tokens.shift();
    }

    let candidate: { kind: 'node' | 'exe'; target: string } | null = null;
    if (tokens.length === 1 && DIRECT_EXEC_TARGET_RE.test(tokens[0])) {
      const target = expandShimTargetPath(tokens[0], shimDir);
      if (target) candidate = { kind: 'exe', target };
    } else if (tokens.length === 2 && isNodeRunner(tokens[0], progValues) && JS_SCRIPT_TARGET_RE.test(tokens[1])) {
      const target = expandShimTargetPath(tokens[1], shimDir);
      if (target) candidate = { kind: 'node', target };
    }

    if (!candidate) {
      throw unrecognizedWindowsShimError(shimPath, 'unrecognized command invocation');
    }

    if (
      resolvedCandidate &&
      (resolvedCandidate.kind !== candidate.kind || resolvedCandidate.target !== candidate.target)
    ) {
      throw unrecognizedWindowsShimError(shimPath, 'shim dispatches to multiple distinct targets');
    }
    resolvedCandidate = candidate;
  }

  if (!resolvedCandidate) {
    throw unrecognizedWindowsShimError(shimPath, 'no supported executable or JS entry script found');
  }

  if (!isRegularFile(resolvedCandidate.target)) {
    throw unrecognizedWindowsShimError(shimPath, `resolved target is not a regular file: ${resolvedCandidate.target}`);
  }

  if (resolvedCandidate.kind === 'node') {
    const siblingNode = join(shimDir, 'node.exe');
    const nodeExe = isRegularFile(siblingNode) ? siblingNode : (nodeExecPath ?? process.execPath);
    return { file: nodeExe, args: [resolvedCandidate.target] };
  }

  return { file: resolvedCandidate.target, args: [] };
}

export function claudeSpawnSpec(
  claudeBin: string,
  claudeArgs: readonly string[],
  platform: NodeJS.Platform = process.platform,
  _env: NodeJS.ProcessEnv = process.env,
  nodeExecPath: string = process.execPath,
): { file: string; args: string[] } {
  if (platform === 'win32' && BATCH_SHIM_FILE_RE.test(claudeBin)) {
    const resolved = resolveWindowsClaudeShim(claudeBin, nodeExecPath);
    return { file: resolved.file, args: [...resolved.args, ...claudeArgs] };
  }
  return { file: claudeBin, args: [...claudeArgs] };
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
    hostKind: 'launch',
    // Launcher policy, not an engine default: the shared native var reads 0
    // as the family default, and a value already set in the user's shell
    // wins (see `applyEnginePolicy`).
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
const CLAUDE_ENV_OVERRIDES = new Set([
  'anthropic_api_key',
  'anthropic_base_url',
  'anthropic_auth_token',
  'anthropic_model',
]);
const CLAUDE_CONTEXT_ENV_OVERRIDES = new Set(['claude_code_max_context_tokens', 'claude_code_auto_compact_window']);

export function claudeChildEnv(
  parent: NodeJS.ProcessEnv,
  opts: { baseUrl: string; model: string; authToken: string; contextTokens?: number },
): NodeJS.ProcessEnv {
  // Strip case variants before writing canonical keys: on Windows, spreading
  // `process.env` into a plain object leaves mixed-case spellings (such as
  // `Anthropic_Api_Key`) untouched by `delete env.ANTHROPIC_API_KEY`.
  const env: NodeJS.ProcessEnv = {};
  for (const [key, value] of Object.entries(parent)) {
    const lower = key.toLowerCase();
    if (CLAUDE_ENV_OVERRIDES.has(lower)) continue;
    if (opts.contextTokens !== undefined && CLAUDE_CONTEXT_ENV_OVERRIDES.has(lower)) continue;
    env[key] = value;
  }
  env.ANTHROPIC_BASE_URL = opts.baseUrl;
  env.ANTHROPIC_AUTH_TOKEN = opts.authToken;
  env.ANTHROPIC_MODEL = opts.model;
  // NOTE: intentionally NOT setting ANTHROPIC_SMALL_FAST_MODEL /
  // ANTHROPIC_DEFAULT_HAIKU_MODEL. Claude Code falls back to
  // `claude-haiku-*` for subagents + title generation; the swap
  // controller aliases any unknown name to the current resident
  // so those calls always follow whatever model the user picked
  // via `/model`.
  // The served window, so Claude Code compacts before the server rejects
  // (Splash `install/clients.py` sets the same two; an inherited value loses).
  if (opts.contextTokens !== undefined) {
    env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(opts.contextTokens);
    env.CLAUDE_CODE_AUTO_COMPACT_WINDOW = String(opts.contextTokens);
  }
  return withLoopbackNoProxy(env);
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
