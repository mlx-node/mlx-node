/**
 * Engine tuning that is LAUNCHER POLICY, not an engine default.
 *
 * The native engine reads these knobs from the process environment through a
 * `OnceLock`: the FIRST read latches the value for the life of the process, so
 * every variable here must be in place before any model is loaded and must
 * never be mutated afterwards. That one-shot latch is why the policy is a
 * plain data object rather than something a caller can toggle at runtime.
 *
 * Two application shapes, deliberately different:
 *
 *   - In-process host (`mlx serve`, `mlx launch claude`): call
 *     {@link applyEnginePolicy} to write `process.env` BEFORE the first load.
 *     A value the user already set in their shell always wins — this is a
 *     default, not an override.
 *   - Out-of-process host (Electron `utilityProcess.fork`): pass
 *     {@link engineEnvFor} into `fork({ env })` and never touch
 *     `process.env` in the child. The returned map is unconditional: it
 *     describes the child's starting environment, and the parent is expected
 *     to spread the user's own env under it if it wants shell values to win.
 *
 * Keeping the two apart matters. `MLX_PAGED_PREFILL_CHUNK_SIZE` reads as
 * `0` ("no chunking") in Rust when unset; 2048 is the value
 * `mlx launch claude` has historically applied to bound the cold-prefill
 * memory peak, and baking it into the engine would silently change every
 * other embedder's behaviour.
 */

/** Env var names this module owns. Exported so tests can assert the exact set. */
export const ENGINE_POLICY_ENV_VARS = ['MLX_PAGED_PREFILL_CHUNK_SIZE'] as const;

export interface EnginePolicy {
  /**
   * `MLX_PAGED_PREFILL_CHUNK_SIZE` — tokens per paged-prefill chunk.
   * `0` disables chunking (the Rust default when the var is unset).
   */
  pagedPrefillChunkSize?: number;
}

/**
 * The policy `mlx launch claude` has applied since the flag existed, and the
 * one `mlx serve` and the desktop sidecar inherit.
 *
 * 2048 matches the mlx-lm / mlx-vlm default and reduces per-chunk overhead
 * versus the older 1024 for long Qwen dense contexts.
 */
export const LAUNCHER_ENGINE_POLICY: EnginePolicy = Object.freeze({ pagedPrefillChunkSize: 2048 });

/**
 * Render a policy as the environment map a forked child should start with.
 *
 * Unconditional by construction: there is no "already set" to respect in a
 * child that does not exist yet.
 */
export function engineEnvFor(policy: EnginePolicy): Record<string, string> {
  const env: Record<string, string> = {};
  if (policy.pagedPrefillChunkSize !== undefined) {
    env.MLX_PAGED_PREFILL_CHUNK_SIZE = String(policy.pagedPrefillChunkSize);
  }
  return env;
}

/**
 * Apply a policy to an in-process environment as a DEFAULT.
 *
 * Only writes vars that are currently unset — matching the historical
 * `if (process.env.X == null)` guard, so an explicit `X=` (empty string) in
 * the user's shell is treated as "the user has an opinion" and left alone.
 *
 * Returns the names actually written, so a caller can log or assert on them.
 */
export function applyEnginePolicy(policy: EnginePolicy, env: NodeJS.ProcessEnv = process.env): string[] {
  const applied: string[] = [];
  for (const [name, value] of Object.entries(engineEnvFor(policy))) {
    if (env[name] == null) {
      env[name] = value;
      applied.push(name);
    }
  }
  return applied;
}
