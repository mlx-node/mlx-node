/**
 * Identity of the single in-process `mlx` provider, shared between the provider
 * registration ({@link ../provider/index.ts}) and the mlx-only policy adapter
 * ({@link ./model-registry-filter.ts}) so the two can never drift. In particular
 * the adapter pins mlx auth to {@link MLX_API_KEY}/{@link MLX_BASE_URL}, which
 * MUST match what `registerProvider('mlx', …)` sets.
 */

/** Reserved provider id for the local mlx provider. */
export const MLX_PROVIDER_ID = 'mlx';
/** Provider `api` tag for local mlx models. */
export const MLX_API = 'mlx';
/** Local (non-network) base URL for the mlx provider. */
export const MLX_BASE_URL = 'mlx://local';
/** Literal marker apiKey — flags the provider configured; never a real key. */
export const MLX_API_KEY = 'mlx-local';
