/**
 * The sidecar's environment, built once and handed to `fork({ env })`.
 *
 * Not a convenience wrapper. The engine reads its knobs through a Rust
 * `OnceLock`: the first read latches the value for the life of the process and
 * every later read returns the latched one. A variable that is not present at
 * fork time can never be supplied afterwards, and — worse — writing it later
 * *succeeds silently* and changes nothing. That is why the supervisor never
 * touches `process.env` and why this returns a complete environment rather than
 * a patch.
 *
 * The fork-time-only set, as of this writing: `MLX_PAGED_PREFILL_CHUNK_SIZE`,
 * `MLX_PAGED_PREFILL_EVAL_INTERVAL`, `MLX_PAGED_DECODE_CACHE_CLEAR_INTERVAL`,
 * `MLX_PAGED_GROUPED_{QWEN35,GEMMA4}`, `MLX_GEMMA4_PAGED_{DECODE,PREFILL}_ROUTE`,
 * `MLX_{QWEN35_,}NATIVE_KV_WRITE`, `MLX_INFERENCE_TRACE[_FILE]`, and roughly
 * twenty MTP/sampling knobs. They are fork parameters, not settings.
 */

/**
 * `MLX_INFERENCE_TRACE` and `MLX_INFERENCE_TRACE_FILE` together are the ONLY
 * way a class-(b) failure is observable.
 *
 * `mlx_trace_native_error` (crates/mlx-sys/src/mlx_common.h) is a no-op unless
 * both are set: it checks the flag, checks the path, and returns. With them
 * unset, `mlx_array_eval` / `mlx_synchronize` / `mlx_clear_cache` catch the C++
 * exception, write nothing, and hand back arrays whose contents are wrong while
 * `/health` reports `ok`.
 */
const TRACE_FLAG = 'MLX_INFERENCE_TRACE';
const TRACE_FILE = 'MLX_INFERENCE_TRACE_FILE';

export interface ChildEnvInput {
  /**
   * The environment to inherit. Defaults to `process.env` at the call site;
   * it is READ and copied, never written.
   */
  baseEnv: NodeJS.ProcessEnv;
  /**
   * Engine policy for this fork — normally `engineEnvFor(LAUNCHER_ENGINE_POLICY)`
   * from `@mlx-node/server/host`.
   *
   * Passed in rather than imported: `@mlx-node/server/host` statically imports
   * `@mlx-node/lm`, which maps the 233 MB native addon into whatever process
   * evaluates it (measured: +29 MB RSS and `mlx-core.darwin-arm64.node` in
   * `process.report().sharedObjects` on the import alone). MAIN must never do
   * that — it is the reason INFERENCE is a separate process at all.
   */
  enginePolicyEnv: Readonly<Record<string, string>>;
  /** Supervisor-chosen settings, e.g. the models directory picked in the UI. */
  overrides?: Readonly<Record<string, string>>;
  /** Absolute path this generation's child must write its trace to. */
  traceFile: string;
}

/**
 * Layer order, lowest precedence first:
 *
 *   1. engine policy — a DEFAULT, matching `applyEnginePolicy`'s in-process
 *      semantics where a value the user already set in their shell wins;
 *   2. the inherited environment — the user's shell overrides the policy;
 *   3. supervisor overrides — chosen in the UI, so they beat the shell;
 *   4. the trace variables — FORCED, above everything.
 *
 * The trace layer is last and unconditional because it is not a preference. A
 * user with `MLX_INFERENCE_TRACE_FILE` already exported would otherwise point
 * the child's trace at a file the supervisor is not watching, and the `lying`
 * state would silently stop working — the failure mode being that everything
 * looks fine, which is precisely what class (b) already does.
 */
export function buildChildEnv(input: ChildEnvInput): Record<string, string> {
  const env: Record<string, string> = { ...input.enginePolicyEnv };
  for (const [name, value] of Object.entries(input.baseEnv)) {
    if (value !== undefined) env[name] = value;
  }
  Object.assign(env, input.overrides ?? {});
  env[TRACE_FLAG] = '1';
  env[TRACE_FILE] = input.traceFile;
  return env;
}
