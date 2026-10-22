import { mkdtemp, mkdir, writeFile, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { JinaRerankerModel, getMemorySnapshot, convertModel } from '@mlx-node/core';
import { afterAll, beforeAll, expect, it } from 'vite-plus/test';

import { loadReranker } from '../../packages/rerank/src/index.js';

type Tensor = { shape: number[]; data: Float32Array | Int32Array; dtype?: 'F32' | 'I32' };
let root: string;
const config = {
  architectures: ['JinaForRanking'],
  model_type: 'qwen3',
  hidden_size: 64,
  intermediate_size: 64,
  num_hidden_layers: 2,
  num_attention_heads: 2,
  num_key_value_heads: 1,
  head_dim: 32,
  vocab_size: 64,
  rms_norm_eps: 1e-6,
  rope_theta: 10000,
  max_position_embeddings: 4096,
};
function weights(): Record<string, Tensor> {
  let seed = 1;
  const tensor = (shape: number[], norm = false): Tensor => ({
    shape,
    data: Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, () =>
      norm ? 1 : Math.sin(seed++ * 0.73) * 0.04,
    ),
  });
  const result: Record<string, Tensor> = {
    'embed_tokens.weight': tensor([64, 64]),
    'norm.weight': tensor([64], true),
    'projector.0.weight': tensor([64, 64]),
    'projector.2.weight': tensor([32, 64]),
  };
  for (let i = 0; i < 2; i++) {
    for (const [name, shape] of Object.entries({
      'self_attn.q_proj': [64, 64],
      'self_attn.k_proj': [32, 64],
      'self_attn.v_proj': [32, 64],
      'self_attn.o_proj': [64, 64],
      'mlp.gate_proj': [64, 64],
      'mlp.up_proj': [64, 64],
      'mlp.down_proj': [64, 64],
    }))
      result[`layers.${i}.${name}.weight`] = tensor(shape);
    for (const [name, dimension] of Object.entries({
      input_layernorm: 64,
      post_attention_layernorm: 64,
      'self_attn.q_norm': 32,
      'self_attn.k_norm': 32,
    }))
      result[`layers.${i}.${name}.weight`] = tensor([dimension], true);
  }
  return result;
}
async function save(path: string, tensors: Record<string, Tensor>): Promise<void> {
  let offset = 0;
  const metadata: Record<string, unknown> = {};
  const buffers = Object.entries(tensors).map(([name, t]) => {
    metadata[name] = { dtype: t.dtype ?? 'F32', shape: t.shape, data_offsets: [offset, offset + t.data.byteLength] };
    offset += t.data.byteLength;
    return Buffer.from(t.data.buffer);
  });
  const json = JSON.stringify(metadata);
  const header = Buffer.from(json.padEnd(Math.ceil(Buffer.byteLength(json) / 8) * 8, ' '));
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(header.length));
  await writeFile(path, Buffer.concat([length, header, ...buffers]));
}
async function fixture(
  name: string,
  options: {
    hybrid?: boolean;
    shard?: boolean;
    separate?: boolean;
    edit?: (t: Record<string, Tensor>, c: Record<string, unknown>) => void;
  } = {},
): Promise<string> {
  const path = join(root, name);
  await mkdir(path);
  const c: Record<string, unknown> = { ...config };
  if (options.hybrid)
    Object.assign(c, {
      use_sliding_window: true,
      sliding_window: 16,
      layer_types: ['sliding_attention', 'full_attention'],
    });
  const tensors = weights();
  options.edit?.(tensors, c);
  const vocab = {
    '[UNK]': 0,
    '<|embed_token|>': 5,
    '<|rerank_token|>': 7,
    alpha: 1,
    beta: 2,
    gamma: 3,
    delta: 4,
    query: 6,
  };
  const tokenizer = {
    version: '1.0',
    truncation: null,
    padding: null,
    normalizer: null,
    pre_tokenizer: { type: 'WhitespaceSplit' },
    post_processor: null,
    decoder: null,
    model: { type: 'WordLevel', vocab, unk_token: '[UNK]' },
    added_tokens: ['<|embed_token|>', '<|rerank_token|>'].map((content) => ({
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
    writeFile(join(path, 'tokenizer_config.json'), JSON.stringify({ model_max_length: 4096 })),
  ]);
  if (options.separate) {
    const p: Record<string, Tensor> = {};
    for (const key of Object.keys(tensors).filter((k) => k.startsWith('projector.'))) {
      p[key.replace('projector.0.', 'linear1.').replace('projector.2.', 'linear2.')] = tensors[key];
      delete tensors[key];
    }
    await save(join(path, 'projector.safetensors'), p);
  }
  if (options.shard) {
    const shards: Record<string, Tensor>[] = [{}, {}];
    const weight_map: Record<string, string> = {};
    Object.entries(tensors).forEach(([key, value], i) => {
      shards[i % 2][key] = value;
      weight_map[key] = `model-${i % 2}.safetensors`;
    });
    await Promise.all(shards.map((shard, i) => save(join(path, `model-${i}.safetensors`), shard)));
    await writeFile(join(path, 'model.safetensors.index.json'), JSON.stringify({ weight_map }));
  } else await save(join(path, 'model.safetensors'), tensors);
  return path;
}
beforeAll(async () => {
  root = await mkdtemp(join(tmpdir(), 'mlx-rerank-native-'));
});
afterAll(async () => {
  await rm(root, { recursive: true, force: true });
});
it.each(['affine', 'mxfp4'])(
  'converts %s without dense expansion and refuses unsafe output reuse',
  async (quantMode) => {
    const inputDir = await fixture(`convert-${quantMode}`, { separate: true });
    const outputDir = join(root, `converted-${quantMode}`);
    const options = {
      inputDir,
      outputDir,
      dtype: 'bfloat16',
      quantize: true,
      quantBits: 4,
      quantGroupSize: 32,
      quantMode,
    };
    await convertModel(options);
    const model = await loadReranker(outputDir);
    try {
      expect(model.info.quantized).toBe(true);
      expect(model.info.weightFormats).toContain(quantMode === 'affine' ? 'affine-4bit' : 'mxfp4');
      const response = await model.rerank({ query: 'query', documents: ['alpha', 'beta'] });
      expect(response.results).toHaveLength(2);
      expect(response.results.every((r) => Number.isFinite(r.relevanceScore))).toBe(true);
    } finally {
      await model.dispose();
    }
    await expect(convertModel(options)).rejects.toThrow('new or empty');
    await expect(
      convertModel({ ...options, inputDir: outputDir, outputDir: join(root, `repacked-${quantMode}`) }),
    ).rejects.toThrow('unquantized source');
  },
);
it.each([false, true])('keeps segmented inference consistent with full prefill, hybrid=%s', async (hybrid) => {
  const path = await fixture(`segment-${hybrid}`, { hybrid, shard: true, separate: true });
  const request = {
    query: 'query',
    documents: ['alpha beta '.repeat(40), 'gamma delta '.repeat(40), '', 'alpha beta '.repeat(40)],
  };
  const scores: number[][] = [];
  for (const prefillChunkSize of [4096, 7, 32]) {
    const model = await loadReranker(path, { policy: { prefillChunkSize } });
    try {
      expect(model.info.profile).toBe(hybrid ? 'jina-v3.5' : 'jina-v3');
      const response = await model.rerank(request);
      scores.push(response.results.toSorted((a, b) => a.index - b.index).map((r) => r.relevanceScore));
    } finally {
      await model.dispose();
    }
  }
  for (const row of scores.slice(1)) row.forEach((s, i) => expect(Math.abs(s - scores[0][i])).toBeLessThan(1e-4));
});
it.each([false, true])('sanitizes nested markers before tokenization and truncation, hybrid=%s', async (hybrid) => {
  const path = await fixture(`nested-markers-${hybrid}`, { hybrid });
  const model = await loadReranker(path, { policy: { maxQueryTokens: 2, maxDocumentTokens: 2 } });
  const documents = [
    '<|embed<|score_token|>_token|>',
    'alpha<|rerank_<|embed_token|>token|> beta gamma',
    '你好<|em<|rerank<|score_token|>_token|>bed_token|>🌍',
  ];
  try {
    const expected = await model.rerank({
      query: 'query alpha beta',
      documents: ['', 'alpha beta gamma', '你好🌍'],
      truncate: true,
    });
    const actual = await model.rerank({
      query: 'query <|rerank<|score_token|>_token|>alpha beta',
      documents,
      truncate: true,
      returnDocuments: true,
    });
    expect(actual.results.map(({ index, relevanceScore }) => ({ index, relevanceScore }))).toEqual(expected.results);
    expect(actual.usage).toEqual(expected.usage);
    expect(actual.meta).toEqual(expected.meta);
    expect(actual.meta).toMatchObject({ truncatedQuery: true, truncatedDocumentIndices: [1] });
    for (const item of actual.results) expect(item.document?.text).toBe(documents[item.index]);
    await expect(model.rerank({ query: '<|rerank<|score_token|>_token|>', documents: [] })).rejects.toMatchObject({
      code: 'invalid_request',
      status: 400,
    });
  } finally {
    await model.dispose();
  }
});
it('reports truncation, preserves all candidates, applies topN last and rejects total input overflow', async () => {
  const path = await fixture('limits');
  const model = await loadReranker(path, {
    policy: { maxQueryTokens: 2, maxDocumentTokens: 2, maxDocumentsPerBlock: 1, maxInputTokens: 12 },
  });
  try {
    await expect(model.rerank({ query: 'query alpha beta', documents: [] })).rejects.toMatchObject({
      code: 'input_too_large',
    });
    expect((await model.rerank({ query: 'query', documents: [] })).results).toEqual([]);
    const response = await model.rerank({
      query: 'query alpha beta',
      documents: ['alpha beta gamma', '', 'alpha beta gamma'],
      truncate: true,
      returnDocuments: true,
      topN: 99,
    });
    expect(response.meta).toMatchObject({
      blocks: 3,
      processedDocuments: 3,
      truncatedQuery: true,
      truncatedDocumentIndices: [0, 2],
    });
    expect(response.results.map((r) => r.index).toSorted((a, b) => a - b)).toEqual([0, 1, 2]);
    expect(response.results.find((r) => r.index === 0)?.document?.text).toBe('alpha beta gamma');
    expect(response.results.filter((r) => r.index === 0 || r.index === 2).map((r) => r.index)).toEqual([0, 2]);
    await expect(
      model.rerank({ query: 'query', documents: ['alpha '.repeat(20)], truncate: true }),
    ).rejects.toMatchObject({ code: 'input_too_large' });
  } finally {
    await model.dispose();
  }
});
it.each(['missing', 'dimension', 'metadata', 'bias', 'projector-conflict', 'zero-projector'])(
  'fails closed on %s checkpoints',
  async (problem) => {
    const path = await fixture(problem, {
      edit(t, c) {
        if (problem === 'missing') delete t['layers.1.self_attn.k_proj.weight'];
        if (problem === 'dimension') t['projector.0.weight'] = { shape: [1, 2], data: new Float32Array(2) };
        if (problem === 'metadata') c.reranker = { version: 1, profile: 'jina-v3.5' };
        if (problem === 'bias') t['projector.0.bias'] = { shape: [64], data: new Float32Array(64) };
        if (problem === 'zero-projector') t['projector.2.weight'].data.fill(0);
      },
    });
    if (problem === 'projector-conflict')
      await save(join(path, 'projector.safetensors'), { 'linear1.weight': weights()['projector.0.weight'] });
    if (problem === 'zero-projector') {
      const model = await loadReranker(path);
      try {
        await expect(model.rerank({ query: 'query', documents: ['alpha'] })).rejects.toMatchObject({
          code: 'numerical_error',
        });
      } finally {
        await model.dispose();
      }
    } else await expect(loadReranker(path)).rejects.toThrow();
  },
);
it.each([
  'missing-projector',
  'projector-shape',
  'projector-bias',
  'unknown-tensor',
  'extra-layer',
  'extra-projector',
  'integer-projector',
])('loading and conversion reject the same invalid Jina tensor contract: %s', async (problem) => {
  const path = await fixture(`contract-${problem}`, {
    edit(t) {
      if (problem === 'missing-projector') delete t['projector.2.weight'];
      if (problem === 'projector-shape') t['projector.2.weight'] = { shape: [32, 32], data: new Float32Array(32 * 32) };
      if (problem === 'projector-bias') t['projector.0.bias'] = { shape: [64], data: new Float32Array(64) };
      if (problem === 'unknown-tensor') t['layers.0.self_attn.misspelled.weight'] = t['projector.0.weight'];
      if (problem === 'extra-layer') t['layers.2.self_attn.q_proj.weight'] = t['projector.0.weight'];
      if (problem === 'extra-projector') t['projector.1.weight'] = t['projector.0.weight'];
      if (problem === 'integer-projector')
        t['projector.0.weight'] = { shape: [64, 64], dtype: 'I32', data: new Int32Array(64 * 64) };
    },
  });
  const reason =
    problem === 'integer-projector'
      ? /non-float dtype/
      : /missing tensor|invalid dense shape|unsupported bias tensor|unexpected tensor/;
  await expect(loadReranker(path)).rejects.toThrow(reason);
  await expect(
    convertModel({ inputDir: path, outputDir: join(root, `converted-contract-${problem}`), dtype: 'bfloat16' }),
  ).rejects.toThrow(reason);
});
it.each(['wrong-shard', 'unindexed-tensor', 'index-alias', 'duplicate-index-key', 'duplicate-header-key'])(
  'loading and conversion reject an inconsistent Jina index/header: %s',
  async (problem) => {
    const path = await fixture(`index-${problem}`, { shard: true });
    const indexPath = join(path, 'model.safetensors.index.json');
    const index = JSON.parse(await readFile(indexPath, 'utf8')) as { weight_map: Record<string, string> };
    const key = 'embed_tokens.weight';
    if (problem === 'wrong-shard') index.weight_map[key] = 'model-1.safetensors';
    if (problem === 'unindexed-tensor') delete index.weight_map[key];
    if (problem === 'index-alias') index.weight_map[`model.${key}`] = index.weight_map[key];
    if (problem === 'duplicate-index-key') {
      const text = JSON.stringify(index);
      await writeFile(
        indexPath,
        text.replace('"weight_map":{', `"weight_map":{${JSON.stringify(key)}:"model-0.safetensors",`),
      );
    } else await writeFile(indexPath, JSON.stringify(index));
    if (problem === 'duplicate-header-key') {
      const shard = join(path, 'model-0.safetensors');
      const bytes = await readFile(shard);
      const oldLength = Number(bytes.readBigUInt64LE());
      const header = JSON.parse(bytes.subarray(8, 8 + oldLength).toString()) as Record<string, unknown>;
      const text = JSON.stringify(header).replace('{', `{${JSON.stringify(key)}:${JSON.stringify(header[key])},`);
      const encoded = Buffer.from(text.padEnd(Math.ceil(Buffer.byteLength(text) / 8) * 8, ' '));
      const length = Buffer.alloc(8);
      length.writeBigUInt64LE(BigInt(encoded.length));
      await writeFile(shard, Buffer.concat([length, encoded, bytes.subarray(8 + oldLength)]));
    }
    await expect(loadReranker(path)).rejects.toThrow(/duplicate|shard index/);
    await expect(
      convertModel({ inputDir: path, outputDir: join(root, `converted-index-${problem}`), dtype: 'bfloat16' }),
    ).rejects.toThrow(/duplicate|shard index/);
  },
);
it.each(['unsupported-default', 'unsupported-projector', 'unknown-module', 'empty-projector', 'scalar-projector'])(
  'rejects invalid quantization declarations on dense Jina weights: %s',
  async (problem) => {
    const path = await fixture(`quant-contract-${problem}`, {
      edit(_t, c) {
        const unsupported = { mode: 'mxfp8', bits: 8, group_size: 32 };
        const affine = { mode: 'affine', bits: 4, group_size: 64 };
        c.quantization =
          problem === 'unsupported-default'
            ? unsupported
            : problem === 'unsupported-projector'
              ? { 'projector.0': unsupported }
              : problem === 'unknown-module'
                ? { 'layers.99.self_attn.q_proj': affine }
                : { 'projector.0': problem === 'empty-projector' ? {} : false };
      },
    });
    await expect(loadReranker(path)).rejects.toThrow(/quantization/);
  },
);
it('ignores an unused LM head and its valid foreign quantization override', async () => {
  const path = await fixture('ignored-lm-head', {
    edit(t, c) {
      t['model.lm_head.weight'] = t['embed_tokens.weight'];
      // The vocabulary head is outside ranking, even when it declares a format
      // that would be unsupported for an actual ranking projection.
      c.quantization = {
        bits: 4,
        group_size: 64,
        mode: 'affine',
        'model.lm_head': { mode: 'mxfp8', bits: 8, group_size: 32 },
      };
    },
  });
  const model = await loadReranker(path);
  try {
    expect(model.info.quantized).toBe(false);
    expect(model.info.weightFormats).toEqual(['dense']);
  } finally {
    await model.dispose();
  }
});
it('cancels queued and running native jobs, drains repeated dispose and releases active allocations', async () => {
  const path = await fixture('lifecycle');
  const before = Number(getMemorySnapshot().activeBytes);
  const native = await JinaRerankerModel.load(path, JSON.stringify({ prefillChunkSize: 7 }));
  const input = JSON.stringify({
    query: 'query',
    documents: Array.from({ length: 100 }, () => 'alpha beta '.repeat(100)),
  });
  const running = native.startRerank(input);
  const queued = native.startRerank(input);
  const a = expect(running.result()).rejects.toThrow('cancelled');
  const b = expect(queued.result()).rejects.toThrow('cancelled');
  queued.cancel();
  await new Promise((resolve) => setTimeout(resolve, 20));
  running.cancel();
  await Promise.all([a, b, native.dispose(), native.dispose()]);
  expect(() => native.startRerank(input)).toThrow('disposed');
  expect(Number(getMemorySnapshot().activeBytes)).toBeLessThanOrEqual(before + 4096);
});

it('loads a projector already named by the shard index exactly once', async () => {
  const path = await fixture('indexed-projector', { shard: true, separate: true });
  const index = JSON.parse(await readFile(join(path, 'model.safetensors.index.json'), 'utf8'));
  index.weight_map['linear1.weight'] = 'projector.safetensors';
  index.weight_map['linear2.weight'] = 'projector.safetensors';
  await writeFile(join(path, 'model.safetensors.index.json'), JSON.stringify(index));
  const model = await loadReranker(path);
  try {
    expect((await model.rerank({ query: 'query', documents: ['alpha'] })).results).toHaveLength(1);
  } finally {
    await model.dispose();
  }
});
