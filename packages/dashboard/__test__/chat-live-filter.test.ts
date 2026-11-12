import { describe, expect, it } from 'vite-plus/test';

import { liveFilter } from '../src/chat/agent/live-filter.js';

const model = (id: string, meta?: Record<string, unknown>) => ({
  xai: true,
  update: { sessionUpdate: 'model_changed', model_id: id },
  meta,
});

describe('liveFilter', () => {
  it('drops a model_changed while no turn runs (the one of every session/new and session/load)', () => {
    expect(liveFilter(model('q'), { turnRunning: false, modelId: 'q' })).toBe('drop');
    expect(liveFilter(model('other'), { turnRunning: false, modelId: 'q' })).toBe('drop');
  });

  it('during a turn drops a model_changed to the current model and passes a real switch', () => {
    expect(liveFilter(model('q'), { turnRunning: true, modelId: 'q' })).toBe('drop');
    expect(liveFilter(model('r'), { turnRunning: true, modelId: 'q' })).toBe('pass');
  });

  it('drops replayed lines, wherever the flag sits', () => {
    const chunk = { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'x' } };
    expect(
      liveFilter({ xai: false, update: chunk, meta: { isReplay: true } }, { turnRunning: true, modelId: 'q' }),
    ).toBe('drop');
    expect(
      liveFilter(
        { xai: false, update: { ...chunk, _meta: { isReplay: true } }, meta: undefined },
        { turnRunning: true, modelId: 'q' },
      ),
    ).toBe('drop');
  });

  it('passes everything else, also an ACP update named model_changed', () => {
    const chunk = { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'x' } };
    expect(liveFilter({ xai: false, update: chunk, meta: undefined }, { turnRunning: false, modelId: 'q' })).toBe(
      'pass',
    );
    expect(
      liveFilter(
        { xai: false, update: { sessionUpdate: 'model_changed', model_id: 'r' }, meta: undefined },
        { turnRunning: false, modelId: 'q' },
      ),
    ).toBe('pass');
  });
});
