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

    restore();
    restore();
    expect(runtime.getModel('groq', 'llama')).toEqual(model('groq', 'llama'));
    expect(runtime.getModel('mlx', 'undiscovered')).toEqual(model('mlx', 'undiscovered'));
    expect(runtime.hasConfiguredAuth('groq')).toBe(true);
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
      // The full builtin catalog contains cloud providers regardless of auth.
      const cloud = runtime.getModels().find((entry) => entry.provider === 'groq' || entry.provider === 'anthropic');
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

      const restore = installMlxOnlyModelRegistryFilter(ModelRuntime, ['local']);
      try {
        expect(runtime.getModels().map((entry) => `${entry.provider}/${entry.id}`)).toEqual(['mlx/local']);
        // Availability reads are mlx-only regardless of any ambient cloud auth.
        expect(runtime.getAvailableSnapshot().every((entry) => entry.provider === 'mlx')).toBe(true);
        expect(runtime.getAvailableSnapshot().map((entry) => `${entry.provider}/${entry.id}`)).toEqual(['mlx/local']);
        const available = await runtime.getAvailable();
        expect(available.every((entry) => entry.provider === 'mlx')).toBe(true);
        expect(available.map((entry) => `${entry.provider}/${entry.id}`)).toEqual(['mlx/local']);
        expect(runtime.getModel('mlx', 'local')).toBeDefined();
        expect(runtime.getModel(cloud!.provider, cloud!.id)).toBeUndefined();
        expect(runtime.hasConfiguredAuth('mlx')).toBe(true);
        expect(runtime.hasConfiguredAuth('anthropic')).toBe(false);
      } finally {
        restore();
      }

      expect(runtime.getModel(cloud!.provider, cloud!.id)).toBeDefined();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
