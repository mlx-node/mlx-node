/**
 * Process-local policy adapter for pi's canonical `ModelRuntime`.
 *
 * `mlx agent` is an offline/local product, but pi's runtime also composes every
 * built-in cloud provider. CLI `--models mlx/*` only sets the initial selector
 * scope: Tab, `/models`, RPC enumeration, explicit model resolution, and
 * restored sessions all read the runtime's unscoped catalog/availability
 * directly (the `ModelRegistry` facade handed to extensions delegates to the
 * same runtime). Filter those reads at their shared boundary — the runtime
 * prototype — so every path sees only the exact local models this process
 * serves. Patching the runtime (not the extension-only facade) is what keeps
 * the mlx-only guarantee across the selector / listing / resolution paths.
 *
 * Keep this adapter isolated: once pi exposes a first-class provider allowlist
 * in `MainOptions`, this file can be replaced by that option without touching
 * the provider or CLI layers.
 */

interface RuntimeModel {
  provider: string;
  id: string;
  api: string;
  baseUrl: string;
}

export interface FilterableModelRuntime<TModel extends RuntimeModel = RuntimeModel> {
  getModels(providerId?: string): readonly TModel[];
  getAvailableSnapshot(): readonly TModel[];
  getAvailable(providerId?: string): Promise<readonly TModel[]>;
  getModel(provider: string, modelId: string): TModel | undefined;
  hasConfiguredAuth(providerId: string): boolean;
}

export interface FilterableModelRuntimeConstructor<TModel extends RuntimeModel = RuntimeModel> {
  prototype: FilterableModelRuntime<TModel>;
}

const activePrototypes = new WeakSet<object>();

type RuntimeMethodName = keyof FilterableModelRuntime<RuntimeModel>;

function requireMethodDescriptor(prototype: object, name: RuntimeMethodName): PropertyDescriptor {
  const descriptor = Object.getOwnPropertyDescriptor(prototype, name);
  if (!descriptor || typeof descriptor.value !== 'function' || descriptor.writable !== true) {
    throw new Error(`mlx agent: incompatible pi ModelRuntime.${name}; expected a writable prototype method`);
  }
  return descriptor;
}

/**
 * Install an exact local-model read policy for one `runAgent()` lifetime.
 * Returns an idempotent restore callback.
 */
export function installMlxOnlyModelRegistryFilter<TModel extends RuntimeModel>(
  Runtime: FilterableModelRuntimeConstructor<TModel>,
  modelIds: Iterable<string>,
): () => void {
  const prototype = Runtime.prototype;
  if (activePrototypes.has(prototype)) {
    throw new Error('mlx agent: concurrent ModelRuntime filtering in one process is not supported');
  }

  const allowedIds = new Set(modelIds);
  const isAllowed = (model: TModel): boolean =>
    model.provider === 'mlx' && allowedIds.has(model.id) && model.api === 'mlx' && model.baseUrl === 'mlx://local';

  const descriptors = {
    getModels: requireMethodDescriptor(prototype, 'getModels'),
    getAvailableSnapshot: requireMethodDescriptor(prototype, 'getAvailableSnapshot'),
    getAvailable: requireMethodDescriptor(prototype, 'getAvailable'),
    getModel: requireMethodDescriptor(prototype, 'getModel'),
    hasConfiguredAuth: requireMethodDescriptor(prototype, 'hasConfiguredAuth'),
  };
  const getModels = descriptors.getModels.value as FilterableModelRuntime<TModel>['getModels'];
  const getAvailableSnapshot = descriptors.getAvailableSnapshot.value as FilterableModelRuntime<TModel>['getAvailableSnapshot'];
  const getAvailable = descriptors.getAvailable.value as FilterableModelRuntime<TModel>['getAvailable'];
  const getModel = descriptors.getModel.value as FilterableModelRuntime<TModel>['getModel'];
  const hasConfiguredAuth = descriptors.hasConfiguredAuth.value as FilterableModelRuntime<TModel>['hasConfiguredAuth'];

  Object.defineProperties(prototype, {
    getModels: {
      ...descriptors.getModels,
      value(this: FilterableModelRuntime<TModel>, providerId?: string): TModel[] {
        return getModels.call(this, providerId).filter(isAllowed);
      },
    },
    getAvailableSnapshot: {
      ...descriptors.getAvailableSnapshot,
      value(this: FilterableModelRuntime<TModel>): TModel[] {
        return getAvailableSnapshot.call(this).filter(isAllowed);
      },
    },
    getAvailable: {
      ...descriptors.getAvailable,
      // The runtime read is async, so filter the resolved snapshot. Preserve the
      // Promise contract (never turn a rejection into a filtered success).
      value(this: FilterableModelRuntime<TModel>, providerId?: string): Promise<TModel[]> {
        return getAvailable.call(this, providerId).then((models) => models.filter(isAllowed));
      },
    },
    getModel: {
      ...descriptors.getModel,
      value(this: FilterableModelRuntime<TModel>, provider: string, modelId: string): TModel | undefined {
        if (provider !== 'mlx' || !allowedIds.has(modelId)) return undefined;
        const model = getModel.call(this, provider, modelId);
        return model && isAllowed(model) ? model : undefined;
      },
    },
    hasConfiguredAuth: {
      ...descriptors.hasConfiguredAuth,
      // The runtime signature takes a providerId string (not a model), so gate on
      // the provider id alone: only 'mlx' may ever report configured auth.
      value(this: FilterableModelRuntime<TModel>, providerId: string): boolean {
        return providerId === 'mlx' && hasConfiguredAuth.call(this, providerId);
      },
    },
  });

  activePrototypes.add(prototype);
  let restored = false;
  return () => {
    if (restored) return;
    Object.defineProperties(prototype, descriptors);
    activePrototypes.delete(prototype);
    restored = true;
  };
}
