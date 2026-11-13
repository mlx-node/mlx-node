/**
 * The paged KV disk cache (cold tier) switch each host entry hands to the
 * inference host. The native loads read `MLX_PERSIST_PAGED_CACHE` at every load
 * when the model config does not decide; a GGUF load reads no config TS can
 * write, so that env is the one input that reaches every family and format.
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

/**
 * How long `InferenceHost.close()` waits for the disk cache's queued writes.
 * A hybrid (qwen3_5) restore needs the turn's state sidecar, which is still in
 * the queue when a one-shot `mlx agent -p` stops its serve right after the
 * turn. The desktop supervisor's SIGKILL window must cover the HTTP close
 * budget plus this.
 */
export const COLD_CACHE_DRAIN_TIMEOUT_MS = 2_000;

/**
 * What a host entry asks for. `flag` is the user's explicit choice on the
 * command line (`--persist-cache` / `--no-persist-cache`); `byDefault` is the
 * entry's own default. Precedence: `flag` > `MLX_PERSIST_PAGED_CACHE` >
 * `byDefault`. `wholePrompt` asks for {@link WHOLE_PROMPT_CAPTURE_BLOCKS} and
 * {@link WHOLE_PROMPT_CAPTURE_BUDGET_MS} where the env sets no value; false
 * keeps the native capture defaults.
 */
export interface PersistPagedCache {
  flag?: boolean;
  byDefault: boolean;
  wholePrompt: boolean;
}

/** The `mlx serve` flag that carries `wholePrompt` to a spawned host. */
export const PERSIST_WHOLE_PROMPT_FLAG = '--persist-whole-prompt';

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
 * the whole-prompt value when `option.wholePrompt` > the native default (not
 * written).
 */
export function resolveColdCapture(
  option: PersistPagedCache,
  persistOn: boolean,
  env: NodeJS.ProcessEnv | Readonly<Record<string, string | undefined>> = process.env,
): { budget: ColdCaptureBudget | null; writes: Record<string, string> } {
  const writes: Record<string, string> = {};
  if (!persistOn) return { budget: null, writes };
  const pick = (name: string, wholePrompt: number, native: number): number => {
    const set = parseCaptureEnv(env[name]);
    if (set !== undefined) return set;
    if (!option.wholePrompt) return native;
    writes[name] = String(wholePrompt);
    return wholePrompt;
  };
  return {
    budget: {
      blocksPerTurn: pick(COLD_CAPTURE_BLOCKS_ENV, WHOLE_PROMPT_CAPTURE_BLOCKS, NATIVE_CAPTURE_BLOCKS_PER_TURN),
      budgetMs: pick(COLD_CAPTURE_BUDGET_MS_ENV, WHOLE_PROMPT_CAPTURE_BUDGET_MS, NATIVE_CAPTURE_BUDGET_MS),
    },
    writes,
  };
}
