/**
 * Pure CLI flag parser for `mlx serve`. Kept in an addon-free module so unit
 * tests and callers can validate the flag surface without loading `@mlx-node/core`.
 */

import { parseArgs } from 'node:util';

import { parseThinkingEndText, THINKING_END_TEXT_MAX } from '@mlx-node/server/host/agent-thinking';
import { parseCacheDir, parseMaxCacheDisk, resolveCacheDisk } from '@mlx-node/server/host/cache-disk';
import { parseDecodeShare } from '@mlx-node/server/host/decode-share';
import { parseIdleRelease } from '@mlx-node/server/host/idle-release';
import { parseMaxContext } from '@mlx-node/server/host/max-context';
import { parseMaxMemory } from '@mlx-node/server/host/max-memory';
import { resolveLogDir, resolveMlxNodeHome } from '@mlx-node/server/host/paths';
import {
  PERSIST_CACHE_CONFLICT_ERROR,
  PERSIST_PAGED_CACHE_DEFAULT,
  resolvePersistPagedCache,
  type PersistPagedCache,
} from '@mlx-node/server/host/persist-cache';

export interface ServeArgs {
  help: boolean;
  port?: number;
  host?: string;
  modelsDir?: string;
  model?: string;
  preload?: string;
  authToken?: string;
  modelResidency?: 'capability' | 'single';
  kvFormat?: 'int8' | 'bf16';
  maxMemoryBytes?: number;
  maxContextTokens?: number;
  decodeShare?: number;
  activations?: 'auto' | 'bf16';
  /** `--idle-release` in seconds; `Infinity` for `off`. */
  idleReleaseSeconds?: number;
  /** `--max-cache-disk` as the native load option's text: `'auto'`, `'0'` or a byte count. */
  cacheDisk: string;
  /** `--persist-cache` (true) or `--no-persist-cache` (false); unset when neither was given. */
  persistPagedCache?: boolean;
  cacheDir?: string;
  /** `--persist-whole-prompt`; unset when not given. */
  persistWholePrompt?: true;
  thinkingTokenBudget?: number;
  /** `--thinking-cycle-guard`; unset when not given. */
  thinkingCycleGuard?: true;
  thinkingEndText?: string;
  /** `--thinking-levels`; unset when not given. */
  thinkingLevels?: true;
  rerankConfig?: string;
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
 *
 * `env` is read only to check `--cache-dir` against the resolved switch.
 */
export function parseServeArgs(
  argv: string[],
  env: NodeJS.ProcessEnv | Readonly<Record<string, string | undefined>> = process.env,
): ServeArgs {
  const parsed = parseArgs({
    args: argv,
    options: {
      port: { type: 'string' },
      host: { type: 'string' },
      'models-dir': { type: 'string' },
      model: { type: 'string' },
      preload: { type: 'string' },
      'model-residency': { type: 'string' },
      'kv-format': { type: 'string' },
      'max-memory': { type: 'string' },
      'max-context': { type: 'string' },
      'decode-share': { type: 'string' },
      activations: { type: 'string' },
      'idle-release': { type: 'string' },
      'max-cache-disk': { type: 'string' },
      'persist-cache': { type: 'boolean' },
      'no-persist-cache': { type: 'boolean' },
      'cache-dir': { type: 'string' },
      'persist-whole-prompt': { type: 'boolean' },
      'thinking-token-budget': { type: 'string' },
      'thinking-cycle-guard': { type: 'boolean' },
      'thinking-end-text': { type: 'string' },
      'thinking-levels': { type: 'boolean' },
      'rerank-config': { type: 'string' },
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

  const modelResidency = values['model-residency'];
  if (modelResidency !== undefined && modelResidency !== 'single' && modelResidency !== 'capability')
    throw new Error('--model-residency must be capability or single');
  const kvFormat = values['kv-format'];
  if (kvFormat !== undefined && kvFormat !== 'int8' && kvFormat !== 'bf16')
    throw new Error('--kv-format must be int8 or bf16');
  const maxMemoryText = values['max-memory'];
  const maxMemoryBytes = maxMemoryText === undefined ? undefined : parseMaxMemory(maxMemoryText);
  const maxContextText = values['max-context'];
  const maxContextTokens = maxContextText === undefined ? undefined : parseMaxContext(maxContextText);
  const decodeShareText = values['decode-share'];
  const decodeShare = decodeShareText === undefined ? undefined : parseDecodeShare(decodeShareText);
  const activations = values.activations;
  if (activations !== undefined && activations !== 'auto' && activations !== 'bf16')
    throw new Error('--activations must be auto or bf16');
  const idleReleaseText = values['idle-release'];
  const idleReleaseSeconds = idleReleaseText === undefined ? undefined : parseIdleRelease(idleReleaseText);
  const cacheDiskText = values['max-cache-disk'];
  const cacheDisk = cacheDiskText === undefined ? 'auto' : parseMaxCacheDisk(cacheDiskText);
  if (values['persist-cache'] === true && values['no-persist-cache'] === true) {
    throw new Error(PERSIST_CACHE_CONFLICT_ERROR);
  }
  const persistPagedCache =
    values['persist-cache'] === true ? true : values['no-persist-cache'] === true ? false : undefined;
  const cacheDirText = values['cache-dir'];
  const cacheDirArg = cacheDirText === undefined ? undefined : parseCacheDir(cacheDirText);
  const { cacheDir } = resolveCacheDisk({
    cacheDisk,
    persistPagedCache: resolvePersistPagedCache(persistOption(persistPagedCache), env),
    offByFlag: persistPagedCache === false,
    ...(cacheDirArg !== undefined ? { cacheDir: cacheDirArg } : {}),
  });
  const budgetText = values['thinking-token-budget'];
  const thinkingTokenBudget = budgetText === undefined ? undefined : parseThinkingTokenBudget(budgetText);
  const endText = values['thinking-end-text'];
  const thinkingEndText = endText === undefined ? undefined : parseServeEndText(endText);
  const verbose = (values.verbose ?? false) || values['log-dir'] != null;

  return {
    help: values.help ?? false,
    port,
    host: values.host,
    modelsDir: values['models-dir'],
    model: values.model,
    ...(values.preload !== undefined ? { preload: values.preload } : {}),
    ...(modelResidency !== undefined ? { modelResidency } : {}),
    ...(kvFormat !== undefined ? { kvFormat } : {}),
    ...(maxMemoryBytes !== undefined ? { maxMemoryBytes } : {}),
    ...(maxContextTokens !== undefined ? { maxContextTokens } : {}),
    ...(decodeShare !== undefined ? { decodeShare } : {}),
    ...(activations !== undefined ? { activations } : {}),
    ...(idleReleaseSeconds !== undefined ? { idleReleaseSeconds } : {}),
    cacheDisk,
    ...(persistPagedCache !== undefined ? { persistPagedCache } : {}),
    ...(cacheDir !== undefined ? { cacheDir } : {}),
    ...(values['persist-whole-prompt'] === true ? { persistWholePrompt: true as const } : {}),
    ...(thinkingTokenBudget !== undefined ? { thinkingTokenBudget } : {}),
    ...(values['thinking-cycle-guard'] === true ? { thinkingCycleGuard: true as const } : {}),
    ...(thinkingEndText !== undefined ? { thinkingEndText } : {}),
    ...(values['thinking-levels'] === true ? { thinkingLevels: true as const } : {}),
    ...(values['rerank-config'] !== undefined ? { rerankConfig: values['rerank-config'] } : {}),
    authToken: values['auth-token'],
    logDir: verbose ? resolveLogDir(values['log-dir'], resolveMlxNodeHome()) : undefined,
  };
}

/** The host's switch: the flag over `MLX_PERSIST_PAGED_CACHE` over on. */
export function persistOption(flag: boolean | undefined): PersistPagedCache {
  return flag === undefined ? PERSIST_PAGED_CACHE_DEFAULT : { flag, byDefault: PERSIST_PAGED_CACHE_DEFAULT.byDefault };
}

/** A nonnegative decimal integer that fits the native i32 budget. */
function parseThinkingTokenBudget(text: string): number {
  let value = 0;
  for (let index = 0; index < text.length; index++) {
    const digit = text.charCodeAt(index) - 48;
    if (digit < 0 || digit > 9) {
      value = Number.NaN;
      break;
    }
    value = value * 10 + digit;
  }
  if (text.length === 0 || !(value <= 2_147_483_647)) {
    throw new Error(`Invalid --thinking-token-budget "${text}": expected a nonnegative integer.`);
  }
  return value;
}

function parseServeEndText(text: string): string {
  try {
    return parseThinkingEndText(text);
  } catch {
    throw new Error(`Invalid --thinking-end-text: expected 1 to ${THINKING_END_TEXT_MAX} characters.`);
  }
}
