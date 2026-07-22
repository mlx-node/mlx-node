import { mkdtemp, rm } from 'node:fs/promises';
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
    constructor(private models: FakeModel[]) {}

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

    hasConfiguredAuth(providerId: string): boolean {
      return this.models.some((entry) => entry.provider === providerId);
    }

    getProviders(): { id: string }[] {
      return [...new Set(this.models.map((entry) => entry.provider))].map((id) => ({ id }));
    }

    async login(providerId: string, _type?: unknown, _interaction?: unknown): Promise<{ providerId: string }> {
      return { providerId };
    }

    refresh(models: FakeModel[]): void {
      this.models = models;
    }
  };
}

describe('installMlxOnlyModelRegistryFilter', () => {
  it('filters every runtime read path to exact discovered local models, then restores', async () => {
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
    // `/login` enumeration and dispatch are mlx-only too: cloud providers are
    // hidden from the login list and non-mlx sign-in is rejected before dispatch.
    expect(runtime.getProviders().map((entry) => entry.id)).toEqual(['mlx']);
    await expect(runtime.login('groq', undefined, undefined)).rejects.toThrow(/offline/);
    await expect(runtime.login('mlx', undefined, undefined)).resolves.toEqual({ providerId: 'mlx' });

    restore();
    restore();
    expect(runtime.getModel('groq', 'llama')).toEqual(model('groq', 'llama'));
    expect(runtime.getModel('mlx', 'undiscovered')).toEqual(model('mlx', 'undiscovered'));
    expect(runtime.hasConfiguredAuth('groq')).toBe(true);
    expect(runtime.getProviders().map((entry) => entry.id)).toEqual(['groq', 'mlx', 'anthropic']);
    await expect(runtime.login('groq', undefined, undefined)).resolves.toEqual({ providerId: 'groq' });
  });

  it('survives refreshes, rejects concurrent installation, and can be reinstalled after restore', () => {
    const Runtime = makeRuntimeClass();
    const restore = installMlxOnlyModelRegistryFilter(Runtime, ['before', 'after-a', 'after-b']);
    expect(() => installMlxOnlyModelRegistryFilter(Runtime, ['other'])).toThrow(/concurrent/);
    const runtime = new Runtime([model('mlx', 'before')]);

    runtime.refresh([model('groq', 'late-cloud'), model('mlx', 'after-a'), model('mlx', 'after-b')]);

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

  it('filters the pinned pi ModelRuntime catalog to mlx-only and restores it', async () => {
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
      } finally {
        restore();
      }

      // Restored: the real cloud auth/availability is visible again.
      expect(runtime.getModel(cloud!.provider, cloud!.id)).toBeDefined();
      expect(runtime.hasConfiguredAuth('anthropic')).toBe(true);
      expect(runtime.getAvailableSnapshot().some((entry) => entry.provider === 'anthropic')).toBe(true);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('hides cloud providers from /login and rejects non-mlx sign-in with no network', async () => {
    const savedOffline = process.env.PI_OFFLINE;
    // Hermetic: force offline at create() time, and fail any outbound fetch so a
    // leaked OAuth request is impossible to miss.
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
        // An explicit `/login radius` is rejected BEFORE any OAuth fetch fires.
        // The interaction arg is never consulted (login rejects first), so a cast
        // stub is fine.
        await expect(
          runtime.login('radius', 'oauth', {} as Parameters<typeof runtime.login>[2]),
        ).rejects.toThrow(/offline/);
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
});
