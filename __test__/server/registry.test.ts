import type { SessionCapableModel } from '@mlx-node/lm';
import { ModelRegistry } from '@mlx-node/server';
import { describe, expect, it, vi } from 'vite-plus/test';

/**
 * Build a minimal session-capable model mock. Every method is a vi.fn() so
 * tests can spy or stub per-method when needed.
 */
function createMockSessionModel(): SessionCapableModel {
  const emptyResult = {
    text: '',
    toolCalls: [],
    thinking: null,
    numTokens: 0,
    promptTokens: 0,
    reasoningTokens: 0,
    finishReason: 'stop',
    rawText: '',
    performance: undefined,
  };
  // eslint-disable-next-line @typescript-eslint/require-await
  async function* emptyStream(): AsyncGenerator<Record<string, unknown>> {
    yield { done: true, text: '', finishReason: 'stop', toolCalls: [], numTokens: 0, promptTokens: 0 };
  }
  return {
    chatSessionStart: vi.fn().mockResolvedValue(emptyResult),
    chatSessionContinue: vi.fn().mockResolvedValue(emptyResult),
    chatSessionContinueTool: vi.fn().mockResolvedValue(emptyResult),
    chatStreamSessionStart: vi.fn(() => emptyStream()),
    chatStreamSessionContinue: vi.fn(() => emptyStream()),
    chatStreamSessionContinueTool: vi.fn(() => emptyStream()),
    resetCaches: vi.fn(),
  } as unknown as SessionCapableModel;
}

describe('ModelRegistry', () => {
  it('registers and retrieves a model', () => {
    const registry = new ModelRegistry();
    const mockModel = createMockSessionModel();

    registry.register('test-model', mockModel);

    expect(registry.get('test-model')).toBe(mockModel);
  });

  it('returns undefined for unknown model', () => {
    const registry = new ModelRegistry();

    expect(registry.get('nonexistent')).toBeUndefined();
  });

  it('replaces a model when registering with the same name', () => {
    const registry = new ModelRegistry();
    const model1 = createMockSessionModel();
    const model2 = createMockSessionModel();

    registry.register('test-model', model1);
    registry.register('test-model', model2);

    expect(registry.get('test-model')).toBe(model2);
  });

  it('lists all registered models in OpenAI format', () => {
    const registry = new ModelRegistry();
    registry.register('model-a', createMockSessionModel());
    registry.register('model-b', createMockSessionModel());

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
    registry.register('model-a', createMockSessionModel());

    expect(registry.unregister('model-a')).toBe(true);
    expect(registry.get('model-a')).toBeUndefined();
  });

  it('returns false when unregistering a non-existent model', () => {
    const registry = new ModelRegistry();
    expect(registry.unregister('nonexistent')).toBe(false);
  });

  it('hasStreamSupport returns true for session-capable models', () => {
    const registry = new ModelRegistry();
    const streamModel = createMockSessionModel();

    expect(registry.hasStreamSupport(streamModel)).toBe(true);
  });

  it('hasStreamSupport returns false for objects without chatStreamSessionStart method', () => {
    const registry = new ModelRegistry();
    const noStreamModel = {
      chatSessionStart: vi.fn(),
      chatSessionContinue: vi.fn(),
      chatSessionContinueTool: vi.fn(),
      chatStreamSessionContinue: vi.fn(),
      chatStreamSessionContinueTool: vi.fn(),
      resetCaches: vi.fn(),
    } as unknown as SessionCapableModel;

    expect(registry.hasStreamSupport(noStreamModel)).toBe(false);
  });

  it('hasStreamSupport returns false when chatStreamSessionStart is not a function', () => {
    const registry = new ModelRegistry();
    const badStreamModel = {
      ...createMockSessionModel(),
      chatStreamSessionStart: 'not-a-function',
    } as unknown as SessionCapableModel;

    expect(registry.hasStreamSupport(badStreamModel)).toBe(false);
  });

  it('provisions a SessionRegistry alongside every registered model', () => {
    const registry = new ModelRegistry();
    registry.register('sess-model', createMockSessionModel());

    const sessReg = registry.getSessionRegistry('sess-model');
    expect(sessReg).toBeDefined();
    expect(sessReg!.size).toBe(0);
  });

  it('getSessionRegistry returns undefined for unknown model', () => {
    const registry = new ModelRegistry();
    expect(registry.getSessionRegistry('nonexistent')).toBeUndefined();
  });

  it('replaces the SessionRegistry when re-registering a model', () => {
    const registry = new ModelRegistry();
    registry.register('m', createMockSessionModel());
    const firstReg = registry.getSessionRegistry('m');
    registry.register('m', createMockSessionModel());
    const secondReg = registry.getSessionRegistry('m');

    expect(firstReg).toBeDefined();
    expect(secondReg).toBeDefined();
    expect(secondReg).not.toBe(firstReg);
  });

  it('listSessionRegistries returns one registry per registered model', () => {
    const registry = new ModelRegistry();
    registry.register('a', createMockSessionModel());
    registry.register('b', createMockSessionModel());

    const regs = registry.listSessionRegistries();
    expect(regs).toHaveLength(2);
  });

  it('listSessionRegistries returns empty array when no models are registered', () => {
    const registry = new ModelRegistry();
    expect(registry.listSessionRegistries()).toEqual([]);
  });
});
