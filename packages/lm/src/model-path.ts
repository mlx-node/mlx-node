/**
 * Weak-map registry mapping loaded streaming model instances to their
 * checkpoint / prepared-assets directory.
 *
 * Kept in a zero-addon leaf module so `@mlx-node/lm/chat-session` and
 * `@mlx-node/server` can inspect loaded model paths without eagerly importing
 * `stream.ts` (which subclasses native classes from `@mlx-node/core`).
 */
export const modelPathsForTokenizers = new WeakMap<object, string>();

/**
 * Return the recorded checkpoint or prepared-assets directory for a loaded
 * streaming model instance, if available.
 */
export function getLoadedModelPath(model: object): string | undefined {
  const recorded = modelPathsForTokenizers.get(model);
  if (recorded !== undefined) return recorded;
  const nativeAssets = (model as { modelAssetsPath?: () => unknown }).modelAssetsPath;
  if (typeof nativeAssets === 'function') {
    const value = nativeAssets.call(model);
    if (typeof value === 'string' && value.length > 0) return value;
  }
  return undefined;
}
