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
 * The guarantee is an ALLOWLIST across three surfaces, all keyed on the `mlx`
 * provider id:
 *  1. Model reads (`getModels`/`getAvailable*`/`getModel`) — exact local-model
 *     identity (`api === 'mlx' && baseUrl === 'mlx://local'`).
 *  2. Provider/auth reads (`getProviders`/`getProvider`/`checkAuth`/`getAuth`/
 *     `isUsingOAuth`/`hasConfiguredAuth`/`listCredentials`/`getProviderAuthStatus`)
 *     — never surface, report configured, resolve auth for, or enumerate a
 *     credential of any non-`mlx` provider. `getAuth` is the pivotal one: pi's
 *     `/login`, `/logout`, and the built-in `/llama` command all resolve auth
 *     through it, and its OAuth/`LLAMA_BASE_URL` `fetch` consults neither
 *     `PI_OFFLINE` nor any allow-network flag; returning `undefined` for non-mlx
 *     makes those commands fail before any network or second-model-host load.
 *  3. Auth mutation / network (`login` rejected for non-mlx; `refresh` forced to
 *     `allowNetwork: false` so an explicit `allowNetwork: true` — e.g. pi's
 *     `update --models` package command — can never override offline mode).
 *
 * The `mlx` provider is registered with a literal apiKey and never needs
 * `/login`; streaming still works because `prepareRequest` calls `getAuth(model)`
 * with `model.provider === 'mlx'`, which passes through.
 *
 * Surfaces 1-3 patch the runtime's PUBLIC facade, but pi's own internals read the
 * composed provider map `this.models` (pi-ai `ModelsImpl`) directly — a boundary
 * the facade patches cannot reach (e.g. `refresh` resolving a command-backed
 * cloud credential). So there is a 4th, structural surface: the `recomposeProvider`
 * choke composes ONLY `mlx` into `this.models`, making every internal read
 * mlx-only by construction. It requires installation before the runtime is
 * constructed (which `runAgent` guarantees).
 *
 * Keep this adapter isolated: once pi exposes a first-class provider allowlist
 * in `MainOptions`, this file can be replaced by that option without touching
 * the provider or CLI layers.
 */

import { MLX_API, MLX_API_KEY, MLX_BASE_URL, MLX_PROVIDER_ID } from './mlx-identity.js';

interface RuntimeModel {
  provider: string;
  id: string;
  api: string;
  baseUrl: string;
}

/** Minimal shape of a pi `Provider` — only the id is needed to gate on provider. */
interface RuntimeProvider {
  id: string;
}

/** Minimal shape of a pi `CredentialInfo` — only the provider id is needed. */
interface RuntimeCredential {
  providerId: string;
}

/** Minimal shape of a pi `AuthStatus` — only the configured flag is asserted. */
interface RuntimeAuthStatus {
  configured: boolean;
}

export interface FilterableModelRuntime<TModel extends RuntimeModel = RuntimeModel> {
  getModels(providerId?: string): readonly TModel[];
  getAvailableSnapshot(): readonly TModel[];
  getAvailable(providerId?: string): Promise<readonly TModel[]>;
  getModel(provider: string, modelId: string): TModel | undefined;
  getProvider(providerId: string): RuntimeProvider | undefined;
  hasConfiguredAuth(providerId: string): boolean;
  checkAuth(providerId: string): Promise<unknown>;
  isUsingOAuth(providerId: string): boolean;
  getProviders(): readonly RuntimeProvider[];
  listCredentials(): Promise<readonly RuntimeCredential[]>;
  getProviderAuthStatus(providerId: string): RuntimeAuthStatus;
  login(providerId: string, type: unknown, interaction: unknown): Promise<unknown>;
  refresh(options?: { allowNetwork?: boolean; force?: boolean; signal?: unknown }): Promise<unknown>;
}

export interface FilterableModelRuntimeConstructor<TModel extends RuntimeModel = RuntimeModel> {
  prototype: FilterableModelRuntime<TModel>;
}

const activePrototypes = new WeakSet<object>();

/**
 * `getAuth` is intentionally NOT part of {@link FilterableModelRuntime}: pi types
 * it with two overloads (`getAuth(providerId)` and `getAuth(model)`), which a
 * single structural signature cannot capture without breaking the runtime→
 * interface assignment. It is wrapped by name (mlx pins a fixed resolution; the
 * original is never delegated to).
 */

/** Method names wrapped through the structural interface, plus the loose `getAuth`. */
const STRUCTURAL_METHODS = [
  'getModels',
  'getAvailableSnapshot',
  'getAvailable',
  'getModel',
  'getProvider',
  'hasConfiguredAuth',
  'checkAuth',
  'isUsingOAuth',
  'getProviders',
  'listCredentials',
  'getProviderAuthStatus',
  'login',
  'refresh',
] as const;

function requireMethodDescriptor(prototype: object, name: string): PropertyDescriptor {
  const descriptor = Object.getOwnPropertyDescriptor(prototype, name);
  if (!descriptor || typeof descriptor.value !== 'function' || descriptor.writable !== true) {
    throw new Error(`mlx agent: incompatible pi ModelRuntime.${name}; expected a writable prototype method`);
  }
  return descriptor;
}

/**
 * Install an exact local-model / mlx-only-provider policy for one `runAgent()`
 * lifetime. Returns an idempotent restore callback.
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
    model.provider === MLX_PROVIDER_ID &&
    allowedIds.has(model.id) &&
    model.api === MLX_API &&
    model.baseUrl === MLX_BASE_URL;

  // Capture every original descriptor up front (fail-closed if pi's method shape
  // changed), so restore can put them all back verbatim.
  const originals: Record<string, PropertyDescriptor> = {};
  for (const name of STRUCTURAL_METHODS) originals[name] = requireMethodDescriptor(prototype, name);
  originals.getAuth = requireMethodDescriptor(prototype, 'getAuth');
  // `recomposeProvider` is a private funnel method; capture it by name (it is not
  // part of the structural interface — see the choke wrapper below).
  originals.recomposeProvider = requireMethodDescriptor(prototype, 'recomposeProvider');

  const iface = originals as unknown as {
    [K in (typeof STRUCTURAL_METHODS)[number]]: { value: FilterableModelRuntime<TModel>[K] };
  };
  const getModels = iface.getModels.value;
  const getAvailableSnapshot = iface.getAvailableSnapshot.value;
  const getAvailable = iface.getAvailable.value;
  const getModel = iface.getModel.value;
  const getProvider = iface.getProvider.value;
  const hasConfiguredAuth = iface.hasConfiguredAuth.value;
  const checkAuth = iface.checkAuth.value;
  const isUsingOAuth = iface.isUsingOAuth.value;
  const getProviders = iface.getProviders.value;
  const listCredentials = iface.listCredentials.value;
  const getProviderAuthStatus = iface.getProviderAuthStatus.value;
  const login = iface.login.value;
  const refresh = iface.refresh.value;
  const recomposeProvider = originals.recomposeProvider.value as (this: object, providerId: string) => void;

  Object.defineProperties(prototype, {
    getModels: {
      ...originals.getModels,
      value(this: FilterableModelRuntime<TModel>, providerId?: string): TModel[] {
        return getModels.call(this, providerId).filter(isAllowed);
      },
    },
    getAvailableSnapshot: {
      ...originals.getAvailableSnapshot,
      value(this: FilterableModelRuntime<TModel>): TModel[] {
        return getAvailableSnapshot.call(this).filter(isAllowed);
      },
    },
    getAvailable: {
      ...originals.getAvailable,
      // The runtime read is async, so filter the resolved snapshot. Preserve the
      // Promise contract (never turn a rejection into a filtered success).
      value(this: FilterableModelRuntime<TModel>, providerId?: string): Promise<TModel[]> {
        return getAvailable.call(this, providerId).then((models) => models.filter(isAllowed));
      },
    },
    getModel: {
      ...originals.getModel,
      value(this: FilterableModelRuntime<TModel>, provider: string, modelId: string): TModel | undefined {
        if (provider !== MLX_PROVIDER_ID || !allowedIds.has(modelId)) return undefined;
        const model = getModel.call(this, provider, modelId);
        return model && isAllowed(model) ? model : undefined;
      },
    },
    getProvider: {
      ...originals.getProvider,
      // Streaming uses `this.models.getProvider` (a different object); this only
      // gates external reads (e.g. `/logout`). Non-mlx must never surface.
      value(this: FilterableModelRuntime<TModel>, providerId: string): RuntimeProvider | undefined {
        return providerId === MLX_PROVIDER_ID ? getProvider.call(this, providerId) : undefined;
      },
    },
    hasConfiguredAuth: {
      ...originals.hasConfiguredAuth,
      // The runtime signature takes a providerId string (not a model), so gate on
      // the provider id alone: only 'mlx' may ever report configured auth.
      value(this: FilterableModelRuntime<TModel>, providerId: string): boolean {
        return providerId === MLX_PROVIDER_ID && hasConfiguredAuth.call(this, providerId);
      },
    },
    checkAuth: {
      ...originals.checkAuth,
      value(this: FilterableModelRuntime<TModel>, providerId: string): Promise<unknown> {
        return providerId === MLX_PROVIDER_ID ? checkAuth.call(this, providerId) : Promise.resolve(undefined);
      },
    },
    isUsingOAuth: {
      ...originals.isUsingOAuth,
      value(this: FilterableModelRuntime<TModel>, providerId: string): boolean {
        return providerId === MLX_PROVIDER_ID && isUsingOAuth.call(this, providerId);
      },
    },
    getProviders: {
      ...originals.getProviders,
      // `/login` enumerates from here; hide every cloud provider so only mlx is
      // ever offered for sign-in.
      value(this: FilterableModelRuntime<TModel>): RuntimeProvider[] {
        return getProviders.call(this).filter((provider) => provider.id === MLX_PROVIDER_ID);
      },
    },
    listCredentials: {
      ...originals.listCredentials,
      // `/logout` enumerates stored credentials from here (bypassing the composed
      // model map); never reveal a non-mlx credential.
      value(this: FilterableModelRuntime<TModel>): Promise<RuntimeCredential[]> {
        return listCredentials.call(this).then((creds) => creds.filter((cred) => cred.providerId === MLX_PROVIDER_ID));
      },
    },
    getProviderAuthStatus: {
      ...originals.getProviderAuthStatus,
      // Reads the raw credential/config layer (not the model map); only mlx may
      // report configured.
      value(this: FilterableModelRuntime<TModel>, providerId: string): RuntimeAuthStatus {
        return providerId === MLX_PROVIDER_ID ? getProviderAuthStatus.call(this, providerId) : { configured: false };
      },
    },
    getAuth: {
      ...originals.getAuth,
      // Pivotal offline gate + reserved-id invariant. `/login`, `/logout`, and the
      // built-in `/llama` command resolve auth through the runtime's `getAuth`,
      // whose OAuth / `LLAMA_BASE_URL` fetch ignores PI_OFFLINE — so non-mlx must
      // resolve to nothing (string form `getAuth(id)` or model form via
      // `model.provider`). For mlx we do NOT delegate to the composed provider:
      // a persisted `models.json` `{ oauth:'radius', baseUrl }` overlay under the
      // `mlx` id promotes a Radius builtin that merges a radius OAuth method into
      // the composed `mlx` auth; with a stored/expired `mlx` oauth credential the
      // real resolution would trigger an offline OAuth refresh throw ("OAuth
      // refresh failed for mlx") and never reach our local stream. Pin mlx to the
      // fixed local credential instead — identical to the no-overlay case, but
      // immune to the overlay. prepareRequest reads only auth.apiKey/baseUrl(/headers)
      // and dispatches streamSimple on the (api-matched) local closure.
      value(this: object, providerOrModel: unknown, _overrides?: unknown): Promise<unknown> {
        const providerId =
          typeof providerOrModel === 'string'
            ? providerOrModel
            : (providerOrModel as { provider?: unknown } | null | undefined)?.provider;
        if (providerId !== MLX_PROVIDER_ID) return Promise.resolve(undefined);
        return Promise.resolve({ auth: { apiKey: MLX_API_KEY, baseUrl: MLX_BASE_URL } });
      },
    },
    login: {
      ...originals.login,
      // Authoritative offline gate: reject any non-mlx login BEFORE dispatch, so
      // the cloud OAuth `fetch` (e.g. radius.pi.dev) can never fire — even if some
      // path passes a provider id directly, bypassing the filtered enumeration.
      value(
        this: FilterableModelRuntime<TModel>,
        providerId: string,
        type: unknown,
        interaction: unknown,
      ): Promise<unknown> {
        if (providerId !== MLX_PROVIDER_ID) {
          return Promise.reject(new Error('mlx agent is offline: provider login is disabled'));
        }
        return login.call(this, providerId, type, interaction);
      },
    },
    refresh: {
      ...originals.refresh,
      // Force offline: pi's `refresh({ allowNetwork })` honours an explicit
      // `true` (e.g. the `update --models` package command) even under
      // PI_OFFLINE=1. Pin it off so no catalog fetch can escape the boundary.
      value(
        this: FilterableModelRuntime<TModel>,
        options: { allowNetwork?: boolean; force?: boolean; signal?: unknown } = {},
      ): Promise<unknown> {
        return refresh.call(this, { ...options, allowNetwork: false });
      },
    },
    recomposeProvider: {
      ...originals.recomposeProvider,
      // CHOKE POINT (structural backbone). `recomposeProvider` is the single
      // funnel that writes a provider into the runtime's model map `this.models`
      // (pi's `rebuildProviders` sweep of builtins/config + the extension
      // `registerProvider`/`registerNativeProvider`/`unregisterProvider` paths).
      // Compose ONLY `mlx`, so `this.models` is mlx-only by construction. That
      // closes the whole class of INTERNAL `this.models.*` reads the public gates
      // above cannot reach — most importantly `refresh`'s availability pass, which
      // resolves a configured cloud provider's credential (executing a
      // command-backed apiKey) with no allowNetwork gate — plus the `/llama`
      // getAuth and catalog-refresh paths. Requires install before the runtime is
      // created (runAgent installs before pi.main constructs it); the mlx provider
      // registers later via its own `recomposeProvider('mlx')`, allowed through.
      value(this: object, providerId: string): void {
        if (providerId !== MLX_PROVIDER_ID) return;
        recomposeProvider.call(this, providerId);
      },
    },
  });

  activePrototypes.add(prototype);
  let restored = false;
  return () => {
    if (restored) return;
    Object.defineProperties(prototype, originals);
    activePrototypes.delete(prototype);
    restored = true;
  };
}
