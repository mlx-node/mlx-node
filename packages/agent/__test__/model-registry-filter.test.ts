import { AuthStorage, ModelRegistry } from '@earendil-works/pi-coding-agent';
import { describe, expect, it } from 'vite-plus/test';

import {
  installMlxOnlyModelRegistryFilter,
  type FilterableModelRegistry,
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

function makeRegistryClass() {
  return class FakeModelRegistry implements FilterableModelRegistry<FakeModel> {
    constructor(private models: FakeModel[]) {}

    getAll(): FakeModel[] {
      return this.models;
    }

    getAvailable(): FakeModel[] {
      return this.models;
    }

    find(provider: string, modelId: string): FakeModel | undefined {
      return this.models.find((entry) => entry.provider === provider && entry.id === modelId);
    }

    hasConfiguredAuth(_model: FakeModel): boolean {
      return true;
    }

    refresh(models: FakeModel[]): void {
      this.models = models;
    }
  };
}

describe('installMlxOnlyModelRegistryFilter', () => {
  it('filters every read path to exact discovered local models, then restores the registry', () => {
    const Registry = makeRegistryClass();
    const restore = installMlxOnlyModelRegistryFilter(Registry, ['local-a', 'local-b', 'wrong-api', 'wrong-url']);
    const registry = new Registry([
      model('groq', 'llama'),
      model('mlx', 'local-a'),
      model('anthropic', 'claude'),
      model('mlx', 'local-b'),
      model('mlx', 'undiscovered'),
      { ...model('mlx', 'wrong-api'), api: 'openai-completions' },
      { ...model('mlx', 'wrong-url'), baseUrl: 'https://remote.example' },
    ]);

    expect(registry.getAll().map((entry) => entry.id)).toEqual(['local-a', 'local-b']);
    expect(registry.getAvailable().map((entry) => entry.id)).toEqual(['local-a', 'local-b']);
    expect(registry.find('mlx', 'local-b')).toEqual(model('mlx', 'local-b'));
    expect(registry.find('mlx', 'undiscovered')).toBeUndefined();
    expect(registry.find('mlx', 'wrong-api')).toBeUndefined();
    expect(registry.find('mlx', 'wrong-url')).toBeUndefined();
    expect(registry.find('groq', 'llama')).toBeUndefined();
    expect(registry.hasConfiguredAuth(model('mlx', 'local-a'))).toBe(true);
    expect(registry.hasConfiguredAuth(model('groq', 'llama'))).toBe(false);
    expect(registry.hasConfiguredAuth({ ...model('mlx', 'wrong-api'), api: 'openai-completions' })).toBe(false);
    expect(registry.hasConfiguredAuth({ ...model('mlx', 'wrong-url'), baseUrl: 'https://remote.example' })).toBe(false);

    restore();
    restore();
    expect(registry.find('groq', 'llama')).toEqual(model('groq', 'llama'));
    expect(registry.find('mlx', 'undiscovered')).toEqual(model('mlx', 'undiscovered'));
  });

  it('survives refreshes, rejects concurrent installation, and can be reinstalled after restore', () => {
    const Registry = makeRegistryClass();
    const restore = installMlxOnlyModelRegistryFilter(Registry, ['before', 'after-a', 'after-b']);
    expect(() => installMlxOnlyModelRegistryFilter(Registry, ['other'])).toThrow(/concurrent/);
    const registry = new Registry([model('mlx', 'before')]);

    registry.refresh([model('groq', 'late-cloud'), model('mlx', 'after-a'), model('mlx', 'after-b')]);

    expect(registry.getAll().map((entry) => entry.id)).toEqual(['after-a', 'after-b']);
    expect(registry.getAvailable().map((entry) => entry.id)).toEqual(['after-a', 'after-b']);

    restore();
    const restoreAgain = installMlxOnlyModelRegistryFilter(Registry, ['after-b']);
    expect(registry.getAll().map((entry) => entry.id)).toEqual(['after-b']);
    restoreAgain();
  });

  it('fails fast when the pinned pi registry method shape changes', () => {
    class IncompatibleRegistry {}
    expect(() => installMlxOnlyModelRegistryFilter(IncompatibleRegistry as never, ['local'])).toThrow(
      /incompatible pi ModelRegistry\.getAll/,
    );
  });

  it('matches the pinned pi ModelRegistry and remains effective after refresh', () => {
    const authStorage = AuthStorage.inMemory({ groq: { type: 'api_key', key: 'test-groq-key' } });
    const registry = ModelRegistry.inMemory(authStorage);
    const groq = registry.getAll().find((entry) => entry.provider === 'groq');
    expect(groq).toBeDefined();

    const restore = installMlxOnlyModelRegistryFilter(ModelRegistry, ['local']);
    try {
      registry.registerProvider('mlx', {
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

      expect(registry.getAll().map((entry) => `${entry.provider}/${entry.id}`)).toEqual(['mlx/local']);
      expect(registry.getAvailable().map((entry) => `${entry.provider}/${entry.id}`)).toEqual(['mlx/local']);
      expect(registry.find('groq', groq!.id)).toBeUndefined();
      expect(registry.hasConfiguredAuth(groq!)).toBe(false);

      registry.refresh();
      expect(registry.getAll().map((entry) => `${entry.provider}/${entry.id}`)).toEqual(['mlx/local']);
    } finally {
      restore();
    }

    expect(registry.find('groq', groq!.id)).toBeDefined();
  });
});
