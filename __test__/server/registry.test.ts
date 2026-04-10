import { ModelRegistry } from '@mlx-node/server';
import { describe, expect, it, vi } from 'vite-plus/test';

describe('ModelRegistry', () => {
  it('registers and retrieves a model', () => {
    const registry = new ModelRegistry();
    const mockModel = { chat: vi.fn() };

    registry.register('test-model', mockModel);

    expect(registry.get('test-model')).toBe(mockModel);
  });

  it('returns undefined for unknown model', () => {
    const registry = new ModelRegistry();

    expect(registry.get('nonexistent')).toBeUndefined();
  });

  it('replaces a model when registering with the same name', () => {
    const registry = new ModelRegistry();
    const model1 = { chat: vi.fn() };
    const model2 = { chat: vi.fn() };

    registry.register('test-model', model1);
    registry.register('test-model', model2);

    expect(registry.get('test-model')).toBe(model2);
  });

  it('lists all registered models in OpenAI format', () => {
    const registry = new ModelRegistry();
    registry.register('model-a', { chat: vi.fn() });
    registry.register('model-b', { chat: vi.fn() });

    const models = registry.list();

    expect(models).toHaveLength(2);
    expect(models[0].id).toBe('model-a');
    expect(models[0].object).toBe('model');
    expect(models[0].owned_by).toBe('mlx-node');
    expect(typeof models[0].created).toBe('number');
    expect(models[1].id).toBe('model-b');
  });

  it('returns empty list when no models registered', () => {
    const registry = new ModelRegistry();
    expect(registry.list()).toEqual([]);
  });

  it('unregisters a model and returns true', () => {
    const registry = new ModelRegistry();
    registry.register('model-a', { chat: vi.fn() });

    expect(registry.unregister('model-a')).toBe(true);
    expect(registry.get('model-a')).toBeUndefined();
  });

  it('returns false when unregistering a non-existent model', () => {
    const registry = new ModelRegistry();
    expect(registry.unregister('nonexistent')).toBe(false);
  });

  it('hasStreamSupport returns true for objects with chatStream method', () => {
    const registry = new ModelRegistry();
    const streamModel = {
      chat: vi.fn(),
      chatStream: vi.fn(),
    };

    expect(registry.hasStreamSupport(streamModel)).toBe(true);
  });

  it('hasStreamSupport returns false for objects without chatStream method', () => {
    const registry = new ModelRegistry();
    const noStreamModel = {
      chat: vi.fn(),
    };

    expect(registry.hasStreamSupport(noStreamModel)).toBe(false);
  });

  it('hasStreamSupport returns false when chatStream is not a function', () => {
    const registry = new ModelRegistry();
    const badStreamModel = {
      chat: vi.fn(),
      chatStream: 'not-a-function',
    } as any;

    expect(registry.hasStreamSupport(badStreamModel)).toBe(false);
  });
});
