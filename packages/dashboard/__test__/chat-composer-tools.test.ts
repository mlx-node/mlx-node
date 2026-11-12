/**
 * @vitest-environment happy-dom
 */

import { describe, expect, it } from 'vite-plus/test';

import type { ChatHostState } from '../ui/src/lib/chat-types.js';
import { modelOptions, startModel } from '../ui/src/pages/chat/composer-tools.js';
import { READY_CHAT_HOST } from './render.js';

const MODELS = ['alpha', 'bravo', 'charlie', 'delta'].map((id) => ({ id, contextTokens: 32768 }));

function host(over: Partial<ChatHostState>): ChatHostState {
  return { ...READY_CHAT_HOST, models: MODELS, resident: [], loading: null, defaultModelId: 'alpha', ...over };
}

describe('startModel', () => {
  it('starts a new conversation on the model in memory, before the stored choice', () => {
    const state = host({ resident: [{ modelId: 'charlie', sizeBytes: null }] });
    expect(startModel(state, 'bravo')).toBe('charlie');
    expect(startModel(state, null)).toBe('charlie');
  });

  it('with no model in memory: the model in load, then the stored choice, then the default, then the first', () => {
    expect(startModel(host({ loading: { modelId: 'delta', startedAt: 1 } }), 'bravo')).toBe('delta');
    expect(startModel(host({}), 'bravo')).toBe('bravo');
    expect(startModel(host({ defaultModelId: 'charlie' }), 'gone')).toBe('charlie');
    expect(startModel(host({ defaultModelId: null }), null)).toBe('alpha');
  });

  it('ignores a model in memory that the engine does not serve to the app', () => {
    const state = host({ resident: [{ modelId: 'other', sizeBytes: null }] });
    expect(startModel(state, 'bravo')).toBe('bravo');
    expect(startModel(null, 'bravo')).toBeNull();
    expect(startModel(host({ models: [] }), 'bravo')).toBeNull();
  });
});

describe('modelOptions', () => {
  it('lists the model in memory first, then the model in load, then the others in their order', () => {
    const state = host({ loading: { modelId: 'delta', startedAt: 1 } });
    const options = modelOptions(state, new Set(['charlie']), undefined, 'chat');
    expect(options.map((option) => [option.id, option.state])).toEqual([
      ['charlie', 'loaded'],
      ['delta', 'loading'],
      ['alpha', 'idle'],
      ['bravo', 'idle'],
    ]);
  });

  it('keeps the order of the engine when no model is in memory', () => {
    const options = modelOptions(host({}), new Set(), undefined, 'chat');
    expect(options.map((option) => option.id)).toEqual(['alpha', 'bravo', 'charlie', 'delta']);
  });
});
