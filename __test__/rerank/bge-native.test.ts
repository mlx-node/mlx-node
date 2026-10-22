import { mkdtemp, mkdir, writeFile, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { XlmRobertaRerankerModel, convertModel } from '@mlx-node/core';
import { beforeAll, afterAll, expect, it } from 'vite-plus/test';

import { loadSession } from '../../packages/lm/src/models/model-loader.js';
import { loadReranker } from '../../packages/rerank/src/index.js';

type Tensor = {
  shape: number[];
  data: Float32Array | Int32Array | Int16Array | Uint8Array;
  dtype?: 'F32' | 'I32' | 'I16' | 'BOOL';
};
let root: string;
const config = {
  architectures: ['XLMRobertaForSequenceClassification'],
  model_type: 'xlm-roberta',
  hidden_size: 64,
  intermediate_size: 128,
  num_hidden_layers: 2,
  num_attention_heads: 2,
  vocab_size: 64,
  layer_norm_eps: 1e-5,
  max_position_embeddings: 66,
  type_vocab_size: 1,
  pad_token_id: 1,
  bos_token_id: 0,
  eos_token_id: 2,
  hidden_act: 'gelu',
  id2label: { '0': 'LABEL_0' },
};
function weights(): Record<string, Tensor> {
  let seed = 1;
  const tensor = (shape: number[], norm = false): Tensor => ({
    shape,
    data: Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, () =>
      norm ? 1 : Math.sin(seed++ * 0.73) * 0.1,
    ),
  });
  const result: Record<string, Tensor> = {};
  const linear = (base: string, rows: number, cols: number) => {
    result[base + '.weight'] = tensor([rows, cols]);
    result[base + '.bias'] = tensor([rows]);
  };
  const norm = (base: string) => {
    result[base + '.weight'] = tensor([64], true);
    result[base + '.bias'] = tensor([64]);
  };
  result['roberta.embeddings.word_embeddings.weight'] = tensor([64, 64]);
  result['roberta.embeddings.position_embeddings.weight'] = tensor([66, 64]);
  result['roberta.embeddings.token_type_embeddings.weight'] = tensor([1, 64]);
  norm('roberta.embeddings.LayerNorm');
  for (let i = 0; i < 2; i++) {
    const base = `roberta.encoder.layer.${i}.`;
    for (const part of ['query', 'key', 'value']) linear(base + 'attention.self.' + part, 64, 64);
    linear(base + 'attention.output.dense', 64, 64);
    norm(base + 'attention.output.LayerNorm');
    linear(base + 'intermediate.dense', 128, 64);
    linear(base + 'output.dense', 64, 128);
    norm(base + 'output.LayerNorm');
  }
  linear('classifier.dense', 64, 64);
  linear('classifier.out_proj', 1, 64);
  return result;
}
async function save(path: string, tensors: Record<string, Tensor>) {
  let offset = 0;
  const header: Record<string, unknown> = {};
  const buffers = Object.entries(tensors).map(([key, t]) => {
    header[key] = { dtype: t.dtype ?? 'F32', shape: t.shape, data_offsets: [offset, offset + t.data.byteLength] };
    offset += t.data.byteLength;
    return Buffer.from(t.data.buffer);
  });
  const json = JSON.stringify(header);
  const bytes = Buffer.from(json.padEnd(Math.ceil(Buffer.byteLength(json) / 8) * 8, ' '));
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(bytes.length));
  await writeFile(path, Buffer.concat([length, bytes, ...buffers]));
}
async function fixture(
  name: string,
  edit?: (t: Record<string, Tensor>, c: Record<string, unknown>) => void,
  shard = false,
) {
  const path = join(root, name);
  await mkdir(path);
  const t = weights();
  const c: Record<string, unknown> = { ...config };
  edit?.(t, c);
  const vocab = { '<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, query: 4, alpha: 5, beta: 6, gamma: 7 };
  const tokenizer = {
    version: '1.0',
    truncation: null,
    padding: null,
    normalizer: null,
    pre_tokenizer: { type: 'WhitespaceSplit' },
    post_processor: {
      type: 'RobertaProcessing',
      sep: ['</s>', 2],
      cls: ['<s>', 0],
      trim_offsets: false,
      add_prefix_space: false,
    },
    decoder: null,
    model: { type: 'WordLevel', vocab, unk_token: '<unk>' },
    added_tokens: ['<s>', '</s>', '<pad>'].map((content) => ({
      id: vocab[content as keyof typeof vocab],
      content,
      special: true,
      single_word: false,
      lstrip: false,
      rstrip: false,
      normalized: false,
    })),
  };
  await Promise.all([
    writeFile(join(path, 'config.json'), JSON.stringify(c)),
    writeFile(join(path, 'tokenizer.json'), JSON.stringify(tokenizer)),
    writeFile(join(path, 'tokenizer_config.json'), JSON.stringify({ model_max_length: 64 })),
    writeFile(join(path, 'LICENSE'), 'fixture attribution'),
  ]);
  if (shard) {
    const shards: Record<string, Tensor>[] = [{}, {}];
    const weight_map: Record<string, string> = {};
    Object.entries(t).forEach(([key, value], i) => {
      shards[i % 2][key] = value;
      weight_map[key] = `model-${i % 2}.safetensors`;
    });
    await Promise.all(shards.map((values, i) => save(join(path, `model-${i}.safetensors`), values)));
    await writeFile(join(path, 'model.safetensors.index.json'), JSON.stringify({ weight_map }));
  } else await save(join(path, 'model.safetensors'), t);
  return path;
}
beforeAll(async () => {
  root = await mkdtemp(join(tmpdir(), 'mlx-bge-'));
});
afterAll(async () => {
  await rm(root, { recursive: true, force: true });
});
it('uses pair postprocessing, token-level truncation and padded batch budgets', async () => {
  const path = await fixture('pairs', undefined, true);
  const model = await loadReranker(path, {
    policy: { maxPairTokens: 12, maxQueryTokens: 3, maxDocumentTokens: 8, maxBatchTokens: 24 },
  });
  try {
    expect(model.info).toMatchObject({
      family: 'xlm_roberta_reranker',
      scoreType: 'sigmoid',
      execution: 'cross-encoder',
    });
    await expect(model.rerank({ query: 'query alpha beta gamma', documents: [] })).rejects.toMatchObject({
      code: 'input_too_large',
    });
    const docs = ['alpha '.repeat(10), '', 'beta', '<pad>'];
    const response = await model.rerank({
      query: 'query alpha beta gamma',
      documents: docs,
      truncate: true,
      returnDocuments: true,
      topN: 99,
    });
    expect(response.results).toHaveLength(4);
    expect(response.meta).toMatchObject({
      profile: 'xlmr-cross-encoder-v1',
      batches: 2,
      processedDocuments: 4,
      paddedTokens: 40,
      paddingTokens: 5,
      truncatedQuery: true,
      truncatedDocumentIndices: [0],
    });
    expect(response.usage).toEqual({ inputTokens: 16, promptTokens: 35 });
    for (const item of response.results) {
      expect(item.document?.text).toBe(docs[item.index]);
      expect(item.relevanceScore).toBeGreaterThanOrEqual(0);
      expect(item.relevanceScore).toBeLessThanOrEqual(1);
    }
    expect((await model.rerank({ query: 'query', documents: [] })).results).toEqual([]);
  } finally {
    await Promise.all([model.dispose(), model.dispose()]);
  }
  await expect(loadSession(path)).rejects.toThrow('rerank');
  await expect(loadReranker(path, { policy: { prefillChunkSize: 2 } })).rejects.toMatchObject({
    code: 'invalid_request',
  });
});
it('keeps single/batched/padded/reordered inference consistent', async () => {
  const path = await fixture('batching');
  const docs = ['alpha beta', '', ...Array<string>(9).fill('gamma'), 'alpha '.repeat(50), '<pad>'];
  const scoreSets: number[][] = [];
  for (const batch of [1, 8]) {
    const model = await loadReranker(path, { policy: { maxBatchSize: batch, maxQueryTokens: 8, maxBatchTokens: 128 } });
    try {
      const response = await model.rerank({ query: 'query', documents: docs });
      scoreSets.push(response.results.toSorted((a, b) => a.index - b.index).map((x) => x.relevanceScore));
      const reversed = await model.rerank({ query: 'query', documents: docs.toReversed() });
      const byIndex = reversed.results.toSorted((a, b) => b.index - a.index);
      response.results.forEach((row) => expect(row.relevanceScore).toBeCloseTo(byIndex[row.index].relevanceScore, 5));
    } finally {
      await model.dispose();
    }
  }
  scoreSets[0].forEach((s, i) => expect(s).toBeCloseTo(scoreSets[1][i], 5));
});
it.each([0, 1000, -1000])('applies stable sigmoid once and sorts saturation ties at logit %s', async (logit) => {
  const path = await fixture(`score-${logit}`, (t) => {
    t['classifier.out_proj.weight'].data.fill(0);
    t['classifier.out_proj.bias'].data.fill(logit);
  });
  const model = await loadReranker(path);
  try {
    const r = await model.rerank({ query: 'query', documents: ['beta', '', 'alpha'], topN: 2 });
    expect(r.results).toEqual([
      { index: 0, relevanceScore: logit === 0 ? 0.5 : logit > 0 ? 1 : 0 },
      { index: 1, relevanceScore: logit === 0 ? 0.5 : logit > 0 ? 1 : 0 },
    ]);
  } finally {
    await model.dispose();
  }
});
it.each([NaN, Infinity, -Infinity])('rejects nonfinite logits before sigmoid: %s', async (logit) => {
  const path = await fixture(`nonfinite-${logit}`, (t) => {
    t['classifier.out_proj.bias'].data.fill(logit);
  });
  const model = await loadReranker(path);
  try {
    await expect(model.rerank({ query: 'query', documents: ['alpha'] })).rejects.toMatchObject({
      code: 'numerical_error',
    });
  } finally {
    await model.dispose();
  }
});
it.each([
  'missing-bias',
  'wrong-position',
  'wrong-head',
  'orphan-sidecar',
  'wrong-profile',
  'decoder',
  'activation',
  'unsupported-dense-quantization',
  'unsupported-dense-override',
  'empty-head-override',
  'scalar-head-override',
  'empty-embedding-override',
  'scalar-attention-override',
])('fails closed on invalid %s', async (kind) => {
  const path = await fixture(kind, (t, c) => {
    if (kind === 'missing-bias') delete t['roberta.encoder.layer.0.attention.self.query.bias'];
    if (kind === 'wrong-position') c.max_position_embeddings = 65;
    if (kind === 'wrong-head') t['classifier.out_proj.weight'] = { shape: [2, 64], data: new Float32Array(128) };
    if (kind === 'orphan-sidecar') t['classifier.dense.biases'] = { shape: [64, 1], data: new Float32Array(64) };
    if (kind === 'wrong-profile') c.reranker = { version: 1, profile: 'jina-v3' };
    if (kind === 'decoder') c.is_decoder = true;
    if (kind === 'activation') c.hidden_act = 'gelu_new';
    if (kind === 'unsupported-dense-quantization') c.quantization = { mode: 'mxfp8', bits: 8, group_size: 32 };
    if (kind === 'unsupported-dense-override')
      c.quantization = { 'classifier.dense': { mode: 'mxfp8', bits: 8, group_size: 32 } };
    if (kind === 'empty-head-override') c.quantization = { 'classifier.dense': {} };
    if (kind === 'scalar-head-override') c.quantization = { 'classifier.dense': false };
    if (kind === 'empty-embedding-override') c.quantization = { 'roberta.embeddings.word_embeddings': {} };
    if (kind === 'scalar-attention-override')
      c.quantization = { 'roberta.encoder.layer.0.attention.self.query': false };
  });
  await expect(loadReranker(path)).rejects.toThrow();
});
it.each([
  ['I32', 'classifier.dense.weight'],
  ['BOOL', 'classifier.out_proj.bias'],
  ['I16', 'roberta.embeddings.position_embeddings.weight'],
] as const)('rejects %s source storage for %s before conversion casts', async (dtype, key) => {
  const inputDir = await fixture(`invalid-source-dtype-${dtype}`, (t) => {
    const original = t[key];
    const count = original.shape.reduce((a, b) => a * b, 1);
    t[key] = {
      shape: original.shape,
      dtype,
      data: dtype === 'I32' ? new Int32Array(count) : dtype === 'I16' ? new Int16Array(count) : new Uint8Array(count),
    };
  });
  await expect(loadReranker(inputDir)).rejects.toThrow(/dtype|floating/);
  await expect(
    convertModel({ inputDir, outputDir: join(root, `invalid-source-converted-${dtype}`), dtype: 'bfloat16' }),
  ).rejects.toThrow(/dtype|floating/);
});
it.each([
  ['f16', 'float16'],
  ['bf16', 'bfloat16'],
  ['f32', 'float32'],
])('converts dtype alias %s into loadable canonical metadata', async (dtype, canonical) => {
  const inputDir = await fixture(`source-${dtype}`);
  const sourceRevision = '1'.repeat(40);
  await writeFile(
    join(inputDir, '.mlx-download-complete.json'),
    JSON.stringify({
      repo: 'example/bge',
      revision: sourceRevision,
      files: ['config.json', 'model.safetensors'],
      completedAt: '2026-10-21T00:00:00Z',
      scope: 'full',
    }),
  );
  const outputDir = join(root, dtype);
  await convertModel({ inputDir, outputDir, dtype });
  expect(JSON.parse(await readFile(join(outputDir, 'config.json'), 'utf8')).reranker.computeDtype).toBe(canonical);
  expect(JSON.parse(await readFile(join(outputDir, 'config.json'), 'utf8')).reranker.sourceRevision).toBe(
    sourceRevision,
  );
  const model = await loadReranker(outputDir);
  try {
    const response = await model.rerank({ query: 'query', documents: ['alpha'] });
    expect(response.results[0].relevanceScore).toBeGreaterThanOrEqual(0);
    expect(response.results[0].relevanceScore).toBeLessThanOrEqual(1);
  } finally {
    await model.dispose();
  }
});
it('does not claim a source revision from a partial download', async () => {
  const inputDir = await fixture('source-partial-download', (_, config) => {
    config._commit_hash = 'old-config-revision';
  });
  const metadataDir = join(inputDir, '.cache/huggingface/download');
  await mkdir(metadataDir, { recursive: true });
  await writeFile(join(metadataDir, 'config.json.metadata'), 'old-hub-revision\netag\n');
  await writeFile(
    join(inputDir, '.mlx-download-complete.json'),
    JSON.stringify({
      repo: 'example/bge',
      revision: '1'.repeat(40),
      files: ['config.json'],
      completedAt: '2026-10-21T00:00:00Z',
      scope: 'partial',
    }),
  );
  const outputDir = join(root, 'partial-download');
  await convertModel({ inputDir, outputDir, dtype: 'float16' });
  expect(JSON.parse(await readFile(join(outputDir, 'config.json'), 'utf8')).reranker.sourceRevision).toBeNull();
});
it.each(['affine', 'mxfp4'])('converts %s with packed embedding and intact linear bias', async (quantMode) => {
  const inputDir = await fixture(`source-${quantMode}`);
  const outputDir = join(root, quantMode);
  await convertModel({
    inputDir,
    outputDir,
    dtype: 'float16',
    quantize: true,
    quantBits: 4,
    quantMode,
    quantGroupSize: quantMode === 'affine' ? 64 : 32,
  });
  const buf = await readFile(join(outputDir, 'model.safetensors'));
  const n = Number(buf.readBigUInt64LE());
  const header = JSON.parse(buf.subarray(8, 8 + n).toString());
  expect(header['roberta.embeddings.word_embeddings.weight'].dtype).toBe('U32');
  expect(header['classifier.dense.weight'].dtype).toBe('F16');
  expect(header['roberta.encoder.layer.0.attention.self.query.bias'].dtype).toBe('F16');
  expect(await readFile(join(outputDir, 'LICENSE'), 'utf8')).toBe('fixture attribution');
  const model = await loadReranker(outputDir);
  try {
    expect(model.info.quantized).toBe(true);
    expect((await model.rerank({ query: 'query', documents: ['alpha', 'beta'] })).results).toHaveLength(2);
  } finally {
    await model.dispose();
  }
});
it('cancels running and queued native tasks and joins before disposal resolves', async () => {
  const path = await fixture('cancel');
  const model = await XlmRobertaRerankerModel.load(path);
  const request = JSON.stringify({ query: 'query', documents: Array<string>(1024).fill('alpha '.repeat(50)) });
  const active = model.startRerank(request);
  const queued = model.startRerank(request);
  const a = expect(active.result()).rejects.toThrow('cancelled');
  const b = expect(queued.result()).rejects.toThrow('cancelled');
  active.cancel();
  queued.cancel();
  await Promise.all([model.dispose(), model.dispose(), a, b]);
  expect(() => model.startRerank(request)).toThrow('closed');
});
