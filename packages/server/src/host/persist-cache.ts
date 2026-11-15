/**
 * The paged KV disk cache switch each host entry hands to the inference host,
 * one for every family: the cold tier of the families in
 * `COLD_TIER_RESTORE_FAMILIES`, the SSD state tier of `qwen3_5` /
 * `qwen3_5_moe` (persistent on, temporary off) and that of `qwen4_exp`
 * (persistent on, none off). The native loads read
 * `MLX_PERSIST_PAGED_CACHE` at every load when nothing more explicit decides;
 * a GGUF load reads no config TS can write, so that env is the one input that
 * reaches every family and format. The same module resolves the per-turn
 * capture limits (`MLX_COLD_CAPTURE_*`) of the cold tier's capture walk.
 *
 * No addon import: `mlx agent` resolves the switch before it spawns `mlx serve`.
 */

import { MAX_YARN_CONTEXT_TOKENS } from './max-context.js';

export const PERSIST_PAGED_CACHE_ENV = 'MLX_PERSIST_PAGED_CACHE';

/** Read ONCE per process by `cold_capture_budget` (`cold_tier.rs`), at the first paged load. */
export const COLD_CAPTURE_BLOCKS_ENV = 'MLX_COLD_CAPTURE_BLOCKS_PER_TURN';
export const COLD_CAPTURE_BUDGET_MS_ENV = 'MLX_COLD_CAPTURE_BUDGET_MS';

/** `DEFAULT_CAPTURE_BLOCKS_PER_TURN` / `DEFAULT_CAPTURE_BUDGET_MS` in `cold_tier.rs`. */
export const NATIVE_CAPTURE_BLOCKS_PER_TURN = 128;
export const NATIVE_CAPTURE_BUDGET_MS = 250;

/** Tokens per paged KV block: the `paged_block_size` every family falls back to. */
export const PAGED_BLOCK_TOKENS = 16;

/**
 * The block cap when an entry asks for the whole prompt: the largest window a
 * load serves, so the cap never ends a walk before the prompt does.
 */
export const WHOLE_PROMPT_CAPTURE_BLOCKS = Math.ceil(MAX_YARN_CONTEXT_TOKENS / PAGED_BLOCK_TOKENS);

/**
 * The walk's time cap when an entry asks for the whole prompt. The turn's
 * result waits for the walk, and a hybrid family's state sidecar may then wait
 * up to this long again for a writer slot. Blocks a walk leaves behind are
 * captured by the next turn's walk.
 */
export const WHOLE_PROMPT_CAPTURE_BUDGET_MS = 10_000;

/** The `mlx serve` flag that asks for the whole-prompt capture limits. */
export const PERSIST_WHOLE_PROMPT_FLAG = '--persist-whole-prompt';

/**
 * What a host entry asks for. `flag` is the user's explicit choice on the
 * command line (`--persist-cache` / `--no-persist-cache`); `byDefault` is the
 * entry's own default. Precedence: `flag` > `MLX_PERSIST_PAGED_CACHE` >
 * `byDefault`.
 */
export interface PersistPagedCache {
  flag?: boolean;
  byDefault: boolean;
}

/** `mlx serve`, `mlx agent`, `mlx launch claude` and the desktop engine: on. */
export const PERSIST_PAGED_CACHE_DEFAULT: PersistPagedCache = Object.freeze({ byDefault: true });

/** The native env parser, `parse_bool_env` in `cold_tier.rs`; anything else is no answer. */
export function parsePersistEnv(value: string | undefined): boolean | undefined {
  if (value === undefined) return undefined;
  const text = value.trim().toLowerCase();
  if (text === '1' || text === 'true' || text === 'on' || text === 'yes') return true;
  if (text === '0' || text === 'false' || text === 'off' || text === 'no') return false;
  return undefined;
}

/** Whether the disk cache is on: `flag` > `MLX_PERSIST_PAGED_CACHE` > `byDefault`. */
export function resolvePersistPagedCache(
  option: PersistPagedCache,
  env: NodeJS.ProcessEnv | Readonly<Record<string, string | undefined>> = process.env,
): boolean {
  return option.flag ?? parsePersistEnv(env[PERSIST_PAGED_CACHE_ENV]) ?? option.byDefault;
}

/** The `mlx serve` flag that carries a resolved switch to a spawned host. */
export function persistPagedCacheFlag(on: boolean): '--persist-cache' | '--no-persist-cache' {
  return on ? '--persist-cache' : '--no-persist-cache';
}

/** Both `--persist-cache` and `--no-persist-cache` given. */
export const PERSIST_CACHE_CONFLICT_ERROR = '--persist-cache and --no-persist-cache cannot be used together';

/** The capture limits every paged load of this process reads. */
export interface ColdCaptureBudget {
  blocksPerTurn: number;
  budgetMs: number;
}

/** Rust `char::is_whitespace`, the set `str::trim` strips. */
function isRustWhitespace(code: number): boolean {
  return (
    (code >= 0x09 && code <= 0x0d) ||
    code === 0x20 ||
    code === 0x85 ||
    code === 0xa0 ||
    code === 0x1680 ||
    (code >= 0x2000 && code <= 0x200a) ||
    code === 0x2028 ||
    code === 0x2029 ||
    code === 0x202f ||
    code === 0x205f ||
    code === 0x3000
  );
}

const U64_MAX_DIGITS = '18446744073709551615';

/**
 * The native capture env parser, `raw.trim().parse::<u64>()` (`usize` for the
 * block count, the same 64 bits on every supported target): the number the
 * loads read, or `undefined` where native falls back to its default.
 */
export function parseCaptureEnv(value: string | undefined): number | undefined {
  if (value === undefined) return undefined;
  let start = 0;
  let end = value.length;
  while (start < end && isRustWhitespace(value.charCodeAt(start))) start++;
  while (end > start && isRustWhitespace(value.charCodeAt(end - 1))) end--;
  if (start < end && value.charCodeAt(start) === 0x2b) start++;
  if (start === end) return undefined;
  for (let index = start; index < end; index++) {
    const code = value.charCodeAt(index);
    if (code < 0x30 || code > 0x39) return undefined;
  }
  while (start < end - 1 && value.charCodeAt(start) === 0x30) start++;
  const digits = value.slice(start, end);
  if (digits.length > U64_MAX_DIGITS.length) return undefined;
  if (digits.length === U64_MAX_DIGITS.length && digits > U64_MAX_DIGITS) return undefined;
  return Number(digits);
}

/**
 * The capture limits the loads will read, and the env writes that make them
 * so. `null` when the disk cache is off. Per variable: a value native parses >
 * the whole-prompt value when `wholePrompt` > the native default (not
 * written).
 */
export function resolveColdCapture(
  wholePrompt: boolean,
  persistOn: boolean,
  env: NodeJS.ProcessEnv | Readonly<Record<string, string | undefined>> = process.env,
): { budget: ColdCaptureBudget | null; writes: Record<string, string> } {
  const writes: Record<string, string> = {};
  if (!persistOn) return { budget: null, writes };
  const pick = (name: string, whole: number, native: number): number => {
    const set = parseCaptureEnv(env[name]);
    if (set !== undefined) return set;
    if (!wholePrompt) return native;
    writes[name] = String(whole);
    return whole;
  };
  return {
    budget: {
      blocksPerTurn: pick(COLD_CAPTURE_BLOCKS_ENV, WHOLE_PROMPT_CAPTURE_BLOCKS, NATIVE_CAPTURE_BLOCKS_PER_TURN),
      budgetMs: pick(COLD_CAPTURE_BUDGET_MS_ENV, WHOLE_PROMPT_CAPTURE_BUDGET_MS, NATIVE_CAPTURE_BUDGET_MS),
    },
    writes,
  };
}
