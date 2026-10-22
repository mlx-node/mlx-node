import { describe, expect, it, vi } from 'vite-plus/test';

import { matchFamily } from '../../packages/lm/src/family-data.js';
import {
  normalizeRequest,
  parseRerankBody,
  rerankWireResponse,
  validateRerankerPolicy,
} from '../../packages/rerank/src/protocol.js';
import { wrapReranker } from '../../packages/rerank/src/runtime.js';
import type { RerankRequest, RerankResponse } from '../../packages/rerank/src/types.js';
const response: RerankResponse = {
  results: [
    { index: 1, relevanceScore: 0.8 },
    { index: 0, relevanceScore: 0.2 },
  ],
  usage: { inputTokens: 5, promptTokens: 30 },
  meta: {
    profile: 'jina-v3',
    blocks: 1,
    processedDocuments: 2,
    maxBlockTokens: 32768,
    maxDocumentsPerBlock: 125,
    truncatedQuery: false,
    truncatedDocumentIndices: [],
  },
};
describe('rerank SDK contract', () => {
  it('rejects misspelled and invalid policy limits before loading', () => {
    expect(validateRerankerPolicy({ maxDocuments: 20 })).toEqual({ maxDocuments: 20 });
    for (const value of [null, [], { maxTokens: 100 }, { maxDocuments: 0 }, { prefillChunkSize: 2.5 }])
      expect(() => validateRerankerPolicy(value)).toThrow();
  });
  it('recognizes the architecture independently of directory names', () => {
    expect(matchFamily('/arbitrary/path', { model_type: 'qwen3', architectures: ['JinaForRanking'] })).toBe(
      'jina_reranker',
    );
    expect(matchFamily('/jina-reranker', { model_type: 'qwen3', architectures: ['Qwen3ForCausalLM'] })).toBe('qwen3');
  });
  it('normalizes text objects without reordering empty or duplicate documents', () => {
    expect(normalizeRequest({ query: 'q', documents: ['', { text: 'a' }, 'a'] })).toMatchObject({
      documents: ['', 'a', 'a'],
      truncate: false,
    });
    expect(() => parseRerankBody({ model: 'm', query: ' ', documents: [] })).toThrow();
    expect(() => parseRerankBody({ model: 'm', query: 'q', documents: [], top_n: 0 })).toThrow();
    expect(() => parseRerankBody({ model: 'm', query: 'q', documents: [42] })).toThrow();
    expect(() => parseRerankBody({ model: 'm', query: 'q', documents: [], max_block_tokens: 999999 })).toThrow();
  });
  it('returns original text only when requested and disposes once', async () => {
    const native = {
      infoJson: JSON.stringify({ family: 'jina_reranker', policy: { maxDocumentsPerBlock: 17 }, profile: 'jina-v3' }),
      startRerank: vi.fn(() => ({ cancel: vi.fn(), result: async () => JSON.stringify(response) })),
      dispose: vi.fn(async () => {}),
    };
    const model = wrapReranker(native);
    const result = await model.rerank({ query: 'q', documents: ['original a', 'original b'], returnDocuments: true });
    expect(rerankWireResponse('m', result).meta).toHaveProperty('max_documents_per_block', 17);
    expect(rerankWireResponse('m', result).results[0]).toEqual({
      index: 1,
      relevance_score: 0.8,
      document: { text: 'original b' },
    });
    expect((await model.rerank({ query: 'q', documents: ['a', 'b'] })).results[0].document).toBeUndefined();
    await Promise.all([model.dispose(), model.dispose()]);
    expect(native.dispose).toHaveBeenCalledTimes(1);
    await expect(model.rerank({ query: 'q', documents: [] })).rejects.toThrow('disposed');
  });
  it('does not settle cancellation until native work completes', async () => {
    let finish!: (s: string) => void;
    const done = new Promise<string>((r) => {
      finish = r;
    });
    const cancel = vi.fn();
    const model = wrapReranker({
      infoJson: JSON.stringify({ policy: {} }),
      startRerank: () => ({ cancel, result: () => done }),
      dispose: async () => {},
    });
    const controller = new AbortController();
    let settled = false;
    const call = model.rerank({ query: 'q', documents: ['a', 'b'], signal: controller.signal }).finally(() => {
      settled = true;
    });
    const rejected = expect(call).rejects.toThrow('cancelled');
    controller.abort(new Error('cancelled'));
    await Promise.resolve();
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(settled).toBe(false);
    finish(JSON.stringify(response));
    await rejected;
  });
  it.each([false, true])(
    'keeps the original signal and cleans it up when request options are reused (abort=%s)',
    async (abort) => {
      const original = new AbortController();
      const replacement = new AbortController();
      const remove = vi.spyOn(original.signal, 'removeEventListener');
      let finish!: (value: string) => void;
      const done = new Promise<string>((resolve) => {
        finish = resolve;
      });
      const model = wrapReranker({
        infoJson: JSON.stringify({ policy: {} }),
        startRerank: () => ({ cancel: vi.fn(), result: () => done }),
        dispose: async () => {},
      });
      const request: RerankRequest = {
        query: 'q',
        documents: ['a', 'b'],
        signal: original.signal,
        returnDocuments: true,
      };
      const pending = model.rerank(request);
      request.signal = replacement.signal;
      request.returnDocuments = false;
      if (abort) original.abort(new Error('original cancellation'));
      else replacement.abort(new Error('unrelated cancellation'));
      finish(JSON.stringify(response));
      if (abort) await expect(pending).rejects.toThrow('original cancellation');
      else expect((await pending).results[0].document).toEqual({ text: 'b' });
      expect(remove).toHaveBeenCalledWith('abort', expect.any(Function));
    },
  );
});
