import { access, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { ModelRuntime } from '@earendil-works/pi-coding-agent';
import { describe, expect, it } from 'vite-plus/test';

import {
  installMlxOnlyModelRegistryFilter,
  type FilterableModelRuntime,
} from '../src/provider/model-registry-filter.js';

interface FakeModel {
  provider: string;
  id: string;
  api: string;
  baseUrl: string;
}

const model = (provider: string, id: string): FakeModel => ({
  provider,
  id,
  api: provider === 'mlx' ? 'mlx' : 'openai-completions',
  baseUrl: provider === 'mlx' ? 'mlx://local' : `https://${provider}.example`,
});

function makeRuntimeClass() {
  return class FakeModelRuntime implements FilterableModelRuntime<FakeModel> {
    /** Records the allowNetwork the last refresh saw — proves the offline override. */
    lastRefreshAllowNetwork: boolean | undefined;

    constructor(private models: FakeModel[]) {}

    private providerIds(): string[] {
      return [...new Set(this.models.map((entry) => entry.provider))];
    }

    getModels(providerId?: string): FakeModel[] {
      return providerId ? this.models.filter((entry) => entry.provider === providerId) : this.models;
    }

    getAvailableSnapshot(): FakeModel[] {
      return this.models;
    }

    async getAvailable(providerId?: string): Promise<FakeModel[]> {
      return providerId ? this.models.filter((entry) => entry.provider === providerId) : this.models;
    }

    getModel(provider: string, modelId: string): FakeModel | undefined {
      return this.models.find((entry) => entry.provider === provider && entry.id === modelId);
    }

    getProvider(providerId: string): { id: string } | undefined {
      return this.providerIds().includes(providerId) ? { id: providerId } : undefined;
    }

    hasConfiguredAuth(providerId: string): boolean {
      return this.providerIds().includes(providerId);
    }

    async checkAuth(providerId: string): Promise<{ providerId: string } | undefined> {
      return this.providerIds().includes(providerId) ? { providerId } : undefined;
    }

    isUsingOAuth(providerId: string): boolean {
      // Model a cloud provider as OAuth-configured; mlx uses a literal apiKey.
      return providerId !== 'mlx' && this.providerIds().includes(providerId);
    }

    getProviders(): { id: string }[] {
      return this.providerIds().map((id) => ({ id }));
    }

    async listCredentials(): Promise<{ providerId: string }[]> {
      return this.providerIds().map((providerId) => ({ providerId }));
    }

    getProviderAuthStatus(providerId: string): { configured: boolean } {
      return { configured: this.providerIds().includes(providerId) };
    }

    async getAuth(providerOrModel: unknown, _overrides?: unknown): Promise<{ provider: string } | undefined> {
      const id =
        typeof providerOrModel === 'string' ? providerOrModel : (providerOrModel as FakeModel | undefined)?.provider;
      return id !== undefined && this.providerIds().includes(id) ? { provider: id } : undefined;
    }

    async login(providerId: string, _type?: unknown, _interaction?: unknown): Promise<{ providerId: string }> {
      return { providerId };
    }

    async refresh(options?: { allowNetwork?: boolean }): Promise<void> {
      this.lastRefreshAllowNetwork = options?.allowNetwork;
    }

    // Present so the filter's fail-closed descriptor check finds it. The fake has
    // no real composed map, so the choke is exercised against the real runtime.
    recomposeProvider(_providerId: string): void {}

    /** Test helper (not part of the pi surface) to mutate the served model set. */
    setModels(models: FakeModel[]): void {
      this.models = models;
    }
  };
}

describe('installMlxOnlyModelRegistryFilter', () => {
  it('filters every runtime read/auth path to mlx-only, then restores', async () => {
    const Runtime = makeRuntimeClass();
    const restore = installMlxOnlyModelRegistryFilter(Runtime, ['local-a', 'local-b', 'wrong-api', 'wrong-url']);
    const runtime = new Runtime([
      model('groq', 'llama'),
      model('mlx', 'local-a'),
      model('anthropic', 'claude'),
      model('mlx', 'local-b'),
      model('mlx', 'undiscovered'),
      { ...model('mlx', 'wrong-api'), api: 'openai-completions' },
      { ...model('mlx', 'wrong-url'), baseUrl: 'https://remote.example' },
    ]);

    expect(runtime.getModels().map((entry) => entry.id)).toEqual(['local-a', 'local-b']);
    // A provider-scoped catalog read is still filtered to mlx-only.
    expect(runtime.getModels('groq').map((entry) => entry.id)).toEqual([]);
    expect(runtime.getAvailableSnapshot().map((entry) => entry.id)).toEqual(['local-a', 'local-b']);
    expect((await runtime.getAvailable()).map((entry) => entry.id)).toEqual(['local-a', 'local-b']);
    expect((await runtime.getAvailable('mlx')).map((entry) => entry.id)).toEqual(['local-a', 'local-b']);
    expect(runtime.getModel('mlx', 'local-b')).toEqual(model('mlx', 'local-b'));
    expect(runtime.getModel('mlx', 'undiscovered')).toBeUndefined();
    expect(runtime.getModel('mlx', 'wrong-api')).toBeUndefined();
    expect(runtime.getModel('mlx', 'wrong-url')).toBeUndefined();
    expect(runtime.getModel('groq', 'llama')).toBeUndefined();
    expect(runtime.hasConfiguredAuth('mlx')).toBe(true);
    expect(runtime.hasConfiguredAuth('groq')).toBe(false);
    expect(runtime.hasConfiguredAuth('anthropic')).toBe(false);

    // Provider/auth reads are mlx-only: no cloud provider surfaces, resolves auth,
    // reports configured, or lists a credential.
    expect(runtime.getProvider('mlx')).toEqual({ id: 'mlx' });
    expect(runtime.getProvider('groq')).toBeUndefined();
    expect(await runtime.checkAuth('mlx')).toEqual({ providerId: 'mlx' });
    expect(await runtime.checkAuth('groq')).toBeUndefined();
    expect(runtime.isUsingOAuth('groq')).toBe(false);
    // mlx auth is PINNED to the fixed local credential (never delegated), so a
    // tampered overlay on the reserved id can't redirect it; cloud resolves nothing.
    const pinnedAuth = { auth: { apiKey: 'mlx-local', baseUrl: 'mlx://local' } };
    expect(await runtime.getAuth('mlx')).toEqual(pinnedAuth);
    expect(await runtime.getAuth('groq')).toBeUndefined();
    // Model-object form (the streaming path) is pinned for mlx, blocks cloud.
    expect(await runtime.getAuth(model('mlx', 'local-a'))).toEqual(pinnedAuth);
    expect(await runtime.getAuth(model('groq', 'llama'))).toBeUndefined();
    expect((await runtime.listCredentials()).map((cred) => cred.providerId)).toEqual(['mlx']);
    expect(runtime.getProviderAuthStatus('mlx')).toEqual({ configured: true });
    expect(runtime.getProviderAuthStatus('groq')).toEqual({ configured: false });

    // `/login` enumeration and dispatch are mlx-only too.
    expect(runtime.getProviders().map((entry) => entry.id)).toEqual(['mlx']);
    await expect(runtime.login('groq', undefined, undefined)).rejects.toThrow(/offline/);
    await expect(runtime.login('mlx', undefined, undefined)).resolves.toEqual({ providerId: 'mlx' });

    // An explicit allowNetwork:true refresh is forced offline.
    await runtime.refresh({ allowNetwork: true });
    expect(runtime.lastRefreshAllowNetwork).toBe(false);

    restore();
    restore();
    expect(runtime.getModel('groq', 'llama')).toEqual(model('groq', 'llama'));
    expect(runtime.getModel('mlx', 'undiscovered')).toEqual(model('mlx', 'undiscovered'));
    expect(runtime.hasConfiguredAuth('groq')).toBe(true);
    expect(runtime.getProvider('groq')).toEqual({ id: 'groq' });
    expect(await runtime.getAuth('groq')).toEqual({ provider: 'groq' });
    expect((await runtime.listCredentials()).map((cred) => cred.providerId)).toEqual(['groq', 'mlx', 'anthropic']);
    expect(runtime.getProviderAuthStatus('groq')).toEqual({ configured: true });
    expect(runtime.getProviders().map((entry) => entry.id)).toEqual(['groq', 'mlx', 'anthropic']);
    await expect(runtime.login('groq', undefined, undefined)).resolves.toEqual({ providerId: 'groq' });
    await runtime.refresh({ allowNetwork: true });
    expect(runtime.lastRefreshAllowNetwork).toBe(true);
  });

  it('survives model-set changes, rejects concurrent installation, and can be reinstalled after restore', () => {
    const Runtime = makeRuntimeClass();
    const restore = installMlxOnlyModelRegistryFilter(Runtime, ['before', 'after-a', 'after-b']);
    expect(() => installMlxOnlyModelRegistryFilter(Runtime, ['other'])).toThrow(/concurrent/);
    const runtime = new Runtime([model('mlx', 'before')]);

    runtime.setModels([model('groq', 'late-cloud'), model('mlx', 'after-a'), model('mlx', 'after-b')]);

    expect(runtime.getModels().map((entry) => entry.id)).toEqual(['after-a', 'after-b']);
    expect(runtime.getAvailableSnapshot().map((entry) => entry.id)).toEqual(['after-a', 'after-b']);

    restore();
    const restoreAgain = installMlxOnlyModelRegistryFilter(Runtime, ['after-b']);
    expect(runtime.getModels().map((entry) => entry.id)).toEqual(['after-b']);
    restoreAgain();
  });

  it('fails fast when the pinned pi runtime method shape changes', () => {
    class IncompatibleRuntime {}
    expect(() => installMlxOnlyModelRegistryFilter(IncompatibleRuntime as never, ['local'])).toThrow(
      /incompatible pi ModelRuntime\.getModels/,
    );
  });

  it('filters the pinned pi ModelRuntime catalog + auth to mlx-only and restores it', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'mlx-runtime-filter-'));
    try {
      const runtime = await ModelRuntime.create({ authPath: join(dir, 'auth.json'), modelsPath: null });
      // `anthropic` is a builtin cloud provider present offline (bundled catalog).
      const cloud = runtime.getModels().find((entry) => entry.provider === 'anthropic');
      expect(cloud).toBeDefined();

      runtime.registerProvider('mlx', {
        api: 'mlx',
        baseUrl: 'mlx://local',
        apiKey: 'mlx-local',
        models: [
          {
            id: 'local',
            name: 'local',
            reasoning: true,
            input: ['text'],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 4096,
            maxTokens: 1024,
          },
        ],
      });

      // Deterministically configure a REAL cloud provider's auth with a fake key,
      // fully offline (`allowNetwork:false` writes the credential in-memory and
      // refreshes without any fetch). This makes anthropic genuinely authenticated
      // AND available BEFORE the filter, so the availability/auth assertions below
      // actually exercise the wrappers instead of passing on an empty auth store.
      await runtime.setRuntimeApiKey('anthropic', 'sk-ant-fake-key', { allowNetwork: false });
      expect(runtime.hasConfiguredAuth('anthropic')).toBe(true);
      expect(runtime.getAvailableSnapshot().some((entry) => entry.provider === 'anthropic')).toBe(true);
      expect((await runtime.getAvailable()).some((entry) => entry.provider === 'anthropic')).toBe(true);
      expect(runtime.getModel('anthropic', cloud!.id)).toBeDefined();
      // Pre-filter, anthropic really is a resolvable, configured, enumerable provider.
      expect(runtime.getProvider('anthropic')).toBeDefined();
      expect(await runtime.getAuth('anthropic')).toBeDefined();
      expect(await runtime.checkAuth('anthropic')).toBeDefined();
      expect(runtime.getProviderAuthStatus('anthropic').configured).toBe(true);

      const restore = installMlxOnlyModelRegistryFilter(ModelRuntime, ['local']);
      try {
        expect(runtime.getModels().map((entry) => `${entry.provider}/${entry.id}`)).toEqual(['mlx/local']);
        // Availability reads are mlx-only despite the live, configured cloud auth.
        expect(runtime.getAvailableSnapshot().every((entry) => entry.provider === 'mlx')).toBe(true);
        expect(runtime.getAvailableSnapshot().map((entry) => `${entry.provider}/${entry.id}`)).toEqual(['mlx/local']);
        const available = await runtime.getAvailable();
        expect(available.every((entry) => entry.provider === 'mlx')).toBe(true);
        expect(available.map((entry) => `${entry.provider}/${entry.id}`)).toEqual(['mlx/local']);
        expect(runtime.getModel('mlx', 'local')).toBeDefined();
        expect(runtime.getModel(cloud!.provider, cloud!.id)).toBeUndefined();
        expect(runtime.hasConfiguredAuth('mlx')).toBe(true);
        // Suppressed even though anthropic auth is really configured.
        expect(runtime.hasConfiguredAuth('anthropic')).toBe(false);
        // Provider/auth reads suppress the configured cloud provider entirely.
        expect(runtime.getProvider('anthropic')).toBeUndefined();
        expect(await runtime.getAuth('anthropic')).toBeUndefined();
        expect(await runtime.checkAuth('anthropic')).toBeUndefined();
        expect(runtime.isUsingOAuth('anthropic')).toBe(false);
        expect(runtime.getProviderAuthStatus('anthropic').configured).toBe(false);
        expect((await runtime.listCredentials()).every((cred) => cred.providerId === 'mlx')).toBe(true);
      } finally {
        restore();
      }

      // Restored: the real cloud auth/availability is visible again.
      expect(runtime.getModel(cloud!.provider, cloud!.id)).toBeDefined();
      expect(runtime.hasConfiguredAuth('anthropic')).toBe(true);
      expect(runtime.getProvider('anthropic')).toBeDefined();
      expect(await runtime.getAuth('anthropic')).toBeDefined();
      expect(runtime.getProviderAuthStatus('anthropic').configured).toBe(true);
      expect(runtime.getAvailableSnapshot().some((entry) => entry.provider === 'anthropic')).toBe(true);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('blocks /login, provider-auth, and explicit-network refresh with no outbound fetch', async () => {
    const savedOffline = process.env.PI_OFFLINE;
    // Hermetic: force offline at create() time, and fail any outbound fetch so a
    // leaked OAuth/catalog request is impossible to miss.
    process.env.PI_OFFLINE = '1';
    const realFetch = globalThis.fetch;
    const fetchTargets: string[] = [];
    globalThis.fetch = (async (input: unknown): Promise<never> => {
      fetchTargets.push(String(input));
      throw new Error('network blocked in test');
    }) as unknown as typeof fetch;

    const dir = await mkdtemp(join(tmpdir(), 'mlx-runtime-login-'));
    try {
      const runtime = await ModelRuntime.create({ authPath: join(dir, 'auth.json'), modelsPath: null });
      // `radius` is a builtin OAuth (cloud) provider; before the filter it is
      // enumerable by `/login`.
      expect(runtime.getProviders().some((entry) => entry.id === 'radius')).toBe(true);

      runtime.registerProvider('mlx', {
        api: 'mlx',
        baseUrl: 'mlx://local',
        apiKey: 'mlx-local',
        models: [
          {
            id: 'local',
            name: 'local',
            reasoning: true,
            input: ['text'],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 4096,
            maxTokens: 1024,
          },
        ],
      });

      const restore = installMlxOnlyModelRegistryFilter(ModelRuntime, ['local']);
      try {
        // `/login` sees only mlx — every cloud provider is hidden.
        expect(runtime.getProviders().map((entry) => entry.id)).toEqual(['mlx']);
        expect(runtime.getProvider('radius')).toBeUndefined();
        // An explicit `/login radius` is rejected BEFORE any OAuth fetch fires.
        // The interaction arg is never consulted (login rejects first), so a cast
        // stub is fine.
        await expect(runtime.login('radius', 'oauth', {} as Parameters<typeof runtime.login>[2])).rejects.toThrow(
          /offline/,
        );
        // Provider auth resolution for a cloud provider returns nothing, no network.
        expect(await runtime.getAuth('radius')).toBeUndefined();
        // An explicit allowNetwork:true refresh (pi's `update --models` package
        // command) is forced offline: no pi.dev catalog fetch despite the composed
        // builtin cloud providers.
        await runtime.refresh({ allowNetwork: true, force: true });
        expect(fetchTargets).toEqual([]);
      } finally {
        restore();
      }

      // Restored: radius is enumerable again.
      expect(runtime.getProviders().some((entry) => entry.id === 'radius')).toBe(true);
    } finally {
      globalThis.fetch = realFetch;
      await rm(dir, { recursive: true, force: true });
      if (savedOffline === undefined) delete process.env.PI_OFFLINE;
      else process.env.PI_OFFLINE = savedOffline;
    }
  });

  it('choke: cloud providers never compose, so their command-backed credentials never resolve', async () => {
    const savedOffline = process.env.PI_OFFLINE;
    process.env.PI_OFFLINE = '1';
    const dir = await mkdtemp(join(tmpdir(), 'mlx-runtime-choke-'));
    // A cloud provider configured with a command-backed apiKey (`!<cmd>`); pi's
    // refresh resolves it by running the command (execSync), NOT gated by
    // allowNetwork. The command touches a sentinel file if it ever runs.
    const controlSentinel = join(dir, 'control-ran');
    const chokeSentinel = join(dir, 'choke-ran');
    const writeModels = (file: string, sentinel: string): Promise<void> =>
      writeFile(file, JSON.stringify({ providers: { anthropic: { apiKey: `!touch '${sentinel}'` } } }));

    try {
      // Control: WITHOUT the filter, anthropic composes into this.models and its
      // command apiKey resolves during refresh → the sentinel is created. This
      // proves the mechanism is real (guards the choke assertion from vacuity).
      const controlModels = join(dir, 'control-models.json');
      await writeModels(controlModels, controlSentinel);
      const control = await ModelRuntime.create({
        authPath: join(dir, 'control-auth.json'),
        modelsPath: controlModels,
      });
      await control.refresh({ allowNetwork: false, force: true });
      await access(controlSentinel); // rejects (fails the test) if the command never ran

      // Choke: WITH the filter installed BEFORE create, anthropic never composes,
      // so its command apiKey is never resolved → the sentinel is NOT created.
      const chokeModels = join(dir, 'choke-models.json');
      await writeModels(chokeModels, chokeSentinel);
      const restore = installMlxOnlyModelRegistryFilter(ModelRuntime, ['local']);
      try {
        const runtime = await ModelRuntime.create({
          authPath: join(dir, 'choke-auth.json'),
          modelsPath: chokeModels,
        });
        runtime.registerProvider('mlx', {
          api: 'mlx',
          baseUrl: 'mlx://local',
          apiKey: 'mlx-local',
          models: [
            {
              id: 'local',
              name: 'local',
              reasoning: true,
              input: ['text'],
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
              contextWindow: 4096,
              maxTokens: 1024,
            },
          ],
        });
        await runtime.refresh({ allowNetwork: true, force: true });

        await expect(access(chokeSentinel)).rejects.toThrow(); // never created
        // mlx still composes and is the only provider.
        expect(runtime.getProviders().map((entry) => entry.id)).toEqual(['mlx']);
        expect(runtime.getModel('mlx', 'local')).toBeDefined();
      } finally {
        restore();
      }
    } finally {
      await rm(dir, { recursive: true, force: true });
      if (savedOffline === undefined) delete process.env.PI_OFFLINE;
      else process.env.PI_OFFLINE = savedOffline;
    }
  });

  it('reserved id: mlx auth is pinned to the local credential under a radius mlx overlay, no network', async () => {
    const savedOffline = process.env.PI_OFFLINE;
    process.env.PI_OFFLINE = '1';
    const realFetch = globalThis.fetch;
    const fetchTargets: string[] = [];
    globalThis.fetch = (async (input: unknown): Promise<never> => {
      fetchTargets.push(String(input));
      throw new Error('network blocked in test');
    }) as unknown as typeof fetch;

    const dir = await mkdtemp(join(tmpdir(), 'mlx-runtime-reserved-'));
    try {
      // A crafted models.json hijacks the reserved `mlx` id with a radius OAuth
      // overlay; pi promotes it into a Radius builtin and merges a radius oauth
      // method into the composed `mlx` provider's auth.
      const models = join(dir, 'models.json');
      await writeFile(
        models,
        JSON.stringify({ providers: { mlx: { oauth: 'radius', baseUrl: 'https://radius.example/v1' } } }),
      );

      const restore = installMlxOnlyModelRegistryFilter(ModelRuntime, ['local']);
      try {
        const runtime = await ModelRuntime.create({ authPath: join(dir, 'auth.json'), modelsPath: models });
        runtime.registerProvider('mlx', {
          api: 'mlx',
          baseUrl: 'mlx://local',
          apiKey: 'mlx-local',
          models: [
            {
              id: 'local',
              name: 'local',
              reasoning: true,
              input: ['text'],
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
              contextWindow: 4096,
              maxTokens: 1024,
            },
          ],
        });

        // getAuth resolves to OUR fixed local credential (never the merged radius
        // oauth), for both the string form and the model form (the streaming path).
        // So streaming uses local auth and no radius refresh can fire.
        const fixed = { auth: { apiKey: 'mlx-local', baseUrl: 'mlx://local' } };
        expect(await runtime.getAuth('mlx')).toEqual(fixed);
        const mlxModel = runtime.getModel('mlx', 'local');
        expect(mlxModel).toBeDefined();
        expect(await runtime.getAuth(mlxModel!)).toEqual(fixed);
        expect(fetchTargets).toEqual([]);
      } finally {
        restore();
      }
    } finally {
      globalThis.fetch = realFetch;
      await rm(dir, { recursive: true, force: true });
      if (savedOffline === undefined) delete process.env.PI_OFFLINE;
      else process.env.PI_OFFLINE = savedOffline;
    }
  });
});
