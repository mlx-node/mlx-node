import { expect, it, vi } from 'vite-plus/test';

import { detectConvertModelType } from '../../packages/cli/src/commands/convert-detect.js';
import { matchFamily } from '../../packages/lm/src/family-data.js';
import { rerankerFamilies, rerankerFamily } from '../../packages/rerank/src/families.js';
import {
  familyLoadOptions,
  rerankWireResponse,
  validateRerankerFamilies,
  validateRerankerPolicy,
} from '../../packages/rerank/src/protocol.js';
import { wrapReranker } from '../../packages/rerank/src/runtime.js';
import type { RerankerInfo, RerankResponse } from '../../packages/rerank/src/types.js';

it('shares structural family dispatch and keeps plain XLM-R outside rerank', () => {
  for (const descriptor of rerankerFamilies) {
    const config = { architectures: [descriptor.architecture], model_type: 'unrelated-name' };
    expect(rerankerFamily(config)).toBe(descriptor.family);
    expect(matchFamily('/unknown', config)).toBe(descriptor.family);
    expect(detectConvertModelType(config)).toBe(descriptor.family);
  }
  expect(rerankerFamily({ model_type: 'xlm-roberta', architectures: ['XLMRobertaModel'] })).toBeUndefined();
  expect(rerankerFamily({ architectures: rerankerFamilies.map((f) => f.architecture) })).toBeUndefined();
});
it('validates family-specific policy without changing the Jina compatibility entry', () => {
  expect(validateRerankerPolicy({ maxPairTokens: 8192 }, 'xlm_roberta_reranker')).toEqual({ maxPairTokens: 8192 });
  expect(() => validateRerankerPolicy({ maxPairTokens: 8192 })).toThrow('Unknown');
  expect(() => validateRerankerPolicy({ prefillChunkSize: 512 }, 'xlm_roberta_reranker')).toThrow('Unknown');
  expect(() => validateRerankerFamilies({ policy: {}, families: { jina_reranker: {} } })).toThrow('twice');
  expect(() => validateRerankerFamilies({ families: { bogus: {} } } as never)).toThrow('Invalid');
  expect(() =>
    validateRerankerFamilies({ families: { xlm_roberta_reranker: { policy: { maxBlockTokens: 10 } } } } as never),
  ).toThrow('Unknown');
  expect(familyLoadOptions({ policy: { maxDocuments: 10 } }, 'xlm_roberta_reranker')).toEqual({});
  expect(familyLoadOptions({ policy: { maxDocuments: 10 } }, 'jina_reranker')).toEqual({
    policy: { maxDocuments: 10 },
  });
});
it.each([
  'maxPairTokens',
  'maxQueryTokens',
  'maxDocumentTokens',
  'maxBatchSize',
  'maxBatchTokens',
  'maxDocuments',
  'maxInputTokens',
])('rejects an out-of-range BGE %s during native-free startup validation', (key) => {
  expect(() =>
    validateRerankerFamilies({
      families: {
        xlm_roberta_reranker: { policy: { [key]: 2 ** 31 } },
      },
    }),
  ).toThrow('32-bit');
  expect(validateRerankerPolicy({ [key]: 2 ** 31 - 1 }, 'xlm_roberta_reranker')).toEqual({ [key]: 2 ** 31 - 1 });
});
it('passes BGE scores exactly once through SDK and HTTP without Jina block metadata', async () => {
  const info: RerankerInfo = {
    family: 'xlm_roberta_reranker',
    profile: 'xlmr-cross-encoder-v1',
    execution: 'cross-encoder',
    scoreType: 'sigmoid',
    contextWindow: 8192,
    quantized: false,
    weightFormats: ['dense'],
    policy: {
      maxPairTokens: 8192,
      maxQueryTokens: 512,
      maxDocumentTokens: 8192,
      maxBatchSize: 8,
      maxBatchTokens: 8192,
      maxDocuments: 1024,
      maxInputTokens: 1000000,
    },
  };
  const response: RerankResponse = {
    results: [
      { index: 0, relevanceScore: 0.5 },
      { index: 1, relevanceScore: 0 },
    ],
    usage: { inputTokens: 2, promptTokens: 10 },
    meta: {
      profile: 'xlmr-cross-encoder-v1',
      processedDocuments: 2,
      batches: 1,
      maxPairTokens: 8192,
      maxBatchSize: 8,
      maxBatchTokens: 8192,
      paddedTokens: 10,
      paddingTokens: 0,
      truncatedQuery: false,
      truncatedDocumentIndices: [],
    },
  };
  const native = {
    infoJson: JSON.stringify(info),
    startRerank: vi.fn(() => ({ cancel: vi.fn(), result: async () => JSON.stringify(response) })),
    dispose: vi.fn(async () => {}),
  };
  const model = wrapReranker(native);
  const result = await model.rerank({ query: 'q', documents: ['', 'a'], returnDocuments: true });
  expect(result.meta).toEqual(response.meta);
  const wire = rerankWireResponse('bge', result);
  expect(wire.results).toEqual([
    { index: 0, relevance_score: 0.5, document: { text: '' } },
    { index: 1, relevance_score: 0, document: { text: 'a' } },
  ]);
  expect(wire.meta).not.toHaveProperty('blocks');
  expect(wire.meta).not.toHaveProperty('max_block_tokens');
  expect(JSON.stringify(wire)).not.toMatch(/logit/i);
  await model.dispose();
});
