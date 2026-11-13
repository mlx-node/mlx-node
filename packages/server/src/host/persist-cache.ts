/**
 * The paged KV disk cache switch each host entry hands to the inference host,
 * one for every family: the cold tier of the families in
 * `COLD_TIER_RESTORE_FAMILIES`, and the SSD state tier of `qwen3_5` /
 * `qwen3_5_moe` (persistent on, temporary off). The native loads read
 * `MLX_PERSIST_PAGED_CACHE` at every load when nothing more explicit decides;
 * a GGUF load reads no config TS can write, so that env is the one input that
 * reaches every family and format.
 *
 * No addon import: `mlx agent` resolves the switch before it spawns `mlx serve`.
 */

export const PERSIST_PAGED_CACHE_ENV = 'MLX_PERSIST_PAGED_CACHE';

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
