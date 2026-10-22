import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { LoadableModel } from '@mlx-node/lm';
import type { RerankerModel } from '@mlx-node/rerank';
import type { DiscoveredReranker } from '@mlx-node/rerank/catalog';
import type { TtsModel } from '@mlx-node/tts';
import { afterEach, expect, it, vi } from 'vite-plus/test';

import type { AudioModel } from '../../../packages/server/src/audio-registry.js';
import { discoverAudioModels } from '../../../packages/server/src/host/audio-models.js';
import { createInferenceHost, type InferenceHost } from '../../../packages/server/src/host/index.js';

let host: InferenceHost | undefined;
let root: string;
afterEach(async () => {
  await host?.close({ timeoutMs: 100 });
  host = undefined;
  if (root) await rm(root, { recursive: true, force: true });
});
async function setup(residency: 'capability' | 'single' = 'capability', chat = false, rerank = false) {
  root = await mkdtemp(join(tmpdir(), 'mlx-audio-models-'));
  for (const [name, kind] of [
    ['asr', 'asr'],
    ['tts', 'tts'],
    ['tts2', 'tts'],
  ] as const) {
    const path = join(root, name);
    await mkdir(join(path, 'speech_tokenizer'), { recursive: true });
    await writeFile(
      join(path, 'config.json'),
      JSON.stringify({
        model_type: `qwen3_${kind}`,
        audio_config: {},
        text_config: {},
        tts_model_type: 'custom_voice',
        tts_model_size: '0b6',
        talker_config: { spk_id: { v: 1 }, codec_language_id: { english: 2 } },
      }),
    );
    await writeFile(join(path, 'speech_tokenizer/config.json'), JSON.stringify({ output_sample_rate: 44100 }));
    for (const name of [
      'model.safetensors',
      'tokenizer.json',
      'generation_config.json',
      'speech_tokenizer/model.safetensors',
    ])
      await writeFile(join(path, name), 'test fixture');
  }
  if (chat) {
    await mkdir(join(root, 'chat'));
    await writeFile(join(root, 'chat/config.json'), JSON.stringify({ model_type: 'qwen3_5' }));
    await writeFile(join(root, 'chat/model.safetensors'), 'test fixture');
  }
  const rankModels: RerankerModel[] = [];
  const rankFamilies = [
    ['jina', 'jina_reranker', 'JinaForRanking', 'jina-v3'],
    ['bge', 'xlm_roberta_reranker', 'XLMRobertaForSequenceClassification', 'xlmr-cross-encoder-v1'],
  ] as const;
  if (rerank)
    for (const [name, , architecture] of rankFamilies) {
      const path = join(root, name);
      await mkdir(path);
      await writeFile(join(path, 'config.json'), JSON.stringify({ architectures: [architecture] }));
      for (const file of ['model.safetensors', 'tokenizer.json', 'tokenizer_config.json'])
        await writeFile(join(path, file), 'fixture');
    }
  const loadRerankModel = vi.fn(async (entry: DiscoveredReranker) => {
    const descriptor = rankFamilies.find(([name]) => entry.name === name);
    if (!descriptor) throw Error('unexpected rerank load');
    const model = {
      info: { family: descriptor[1], profile: descriptor[3] },
      dispose: vi.fn(async () => {}),
      rerank: vi.fn(),
    } as unknown as RerankerModel;
    rankModels.push(model);
    return model;
  });
  const loaded: AudioModel[] = [];
  const loadAudioModel = vi.fn(async (entry) => {
    const model: AudioModel =
      entry.kind === 'asr'
        ? { kind: 'asr', model: { dispose: vi.fn(async () => {}), createStream: vi.fn(), startTranscription: vi.fn() } }
        : {
            kind: 'tts',
            model: {
              dispose: vi.fn(async () => {}),
              capabilities: {
                family: 'qwen3_tts',
                variant: 'custom_voice',
                sampleRate: 44100,
                channels: 1,
                voices: ['v'],
                languages: ['english'],
                conditioning: [{ voice: 'preset', instruct: 'unsupported' }],
                voiceCloning: false,
                textStreaming: 'segmented',
                audioStreaming: true,
              },
            } as unknown as TtsModel,
          };
    loaded.push(model);
    return model;
  });
  host = await createInferenceHost({
    modelsDir: root,
    model: 'asr',
    modelResidency: residency,
    port: 0,
    disableStore: true,
    sweepOrphanTempRoots: false,
    loadAudioModel,
    loadRerankModel,
    reranker: {
      families: {
        jina_reranker: { policy: { maxBlockTokens: 1000 } },
        xlm_roberta_reranker: { policy: { maxPairTokens: 256, maxQueryTokens: 128 } },
      },
    },
    loadModel: async () => ({}) as LoadableModel,
  });
  return { host, loaded, loadAudioModel, rankModels, loadRerankModel };
}
it('starts an audio-only host and lists checkpoint metadata without loading weights or chat choices', async () => {
  const { host, loadAudioModel } = await setup();
  expect(host.models).toEqual([]);
  expect(host.audioModels).toHaveLength(3);
  const listing = await (await fetch(`${host.url}/v1/models`)).json();
  expect(listing.data).toHaveLength(3);
  expect(loadAudioModel).not.toHaveBeenCalled();
  const voices = await (await fetch(`${host.url}/v1/audio/voices?model=tts`)).json();
  expect(voices.data[0]).toMatchObject({ sample_rate: 44100, http_speech: true });
  const wrong = await fetch(`${host.url}/v1/responses`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'asr', input: 'hello' }),
  });
  expect(wrong.status).toBe(400);
  expect((await wrong.json()).error.code).toBe('model_capability_mismatch');
  expect(loadAudioModel).not.toHaveBeenCalled();
});

it('keeps chat and audio catalog creation times stable across requests and lazy loads', async () => {
  const { host, loadAudioModel } = await setup('capability', true);
  const list = async () => (await (await fetch(`${host.url}/v1/models`)).json()).data;
  const before = await list();
  expect(before).toHaveLength(4);
  const now = vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 60_000);
  try {
    expect(await list()).toEqual(before);
    expect(loadAudioModel).not.toHaveBeenCalled();
    await host.loadModel('tts');
    expect(await list()).toEqual(before);
  } finally {
    now.mockRestore();
  }
});

it.each([
  ['asr', 'model.safetensors'],
  ['asr', 'tokenizer.json'],
  ['tts', 'model.safetensors'],
  ['tts', 'tokenizer.json'],
  ['tts', 'generation_config.json'],
  ['tts', 'speech_tokenizer/model.safetensors'],
])('does not advertise incomplete %s checkpoints missing %s', async (name, file) => {
  const { loadAudioModel } = await setup();
  await rm(join(root, name, file));
  expect((await discoverAudioModels(root)).map((m) => m.name)).not.toContain(name);
  expect(loadAudioModel).not.toHaveBeenCalled();
});

it('accepts complete indexed shards but rejects missing or empty shard files', async () => {
  await setup();
  const path = join(root, 'asr');
  await rm(join(path, 'model.safetensors'));
  const first = 'model-00001-of-00002.safetensors';
  const second = 'model-00002-of-00002.safetensors';
  await writeFile(join(path, 'model.safetensors.index.json'), JSON.stringify({ weight_map: { a: first, b: second } }));
  await writeFile(join(path, first), 'test fixture');
  expect((await discoverAudioModels(root)).map((m) => m.name)).not.toContain('asr');
  await writeFile(join(path, second), '');
  expect((await discoverAudioModels(root)).map((m) => m.name)).not.toContain('asr');
  await writeFile(join(path, second), 'test fixture');
  expect((await discoverAudioModels(root)).map((m) => m.name)).toContain('asr');
});

it('keeps the native TTS tokenizer fallback discoverable', async () => {
  await setup();
  const path = join(root, 'tts');
  await rm(join(path, 'tokenizer.json'));
  for (const name of ['vocab.json', 'merges.txt', 'tokenizer_config.json'])
    await writeFile(join(path, name), 'test fixture');
  expect((await discoverAudioModels(root)).map((m) => m.name)).toContain('tts');
});

it.each(['weights.safetensors', 'model.safetensors'])(
  'rejects an empty preferred %s instead of advertising a fallback',
  async (name) => {
    await setup();
    const path = join(root, 'tts');
    await writeFile(join(path, 'model-00001-of-00001.safetensors'), 'test fixture');
    await writeFile(join(path, name), '');
    expect((await discoverAudioModels(root)).map((m) => m.name)).not.toContain('tts');
    await rm(join(path, name));
    expect((await discoverAudioModels(root)).map((m) => m.name)).toContain('tts');
  },
);

it('rejects unknown VoiceDesign instruction profiles during discovery', async () => {
  await setup();
  const path = join(root, 'tts/config.json');
  const config = JSON.parse(await readFile(path, 'utf8'));
  config.tts_model_type = 'voice_design';
  config.tts_model_size = 'unknown';
  await writeFile(path, JSON.stringify(config));
  const failure = vi.fn();
  expect((await discoverAudioModels(root, failure)).map((m) => m.name)).not.toContain('tts');
  expect(failure).toHaveBeenCalledWith(expect.objectContaining({ message: 'Unknown VoiceDesign instruction profile' }));
});

it('rejects and disposes a loaded model when its HTTP streaming capability differs from discovery', async () => {
  const { host, loaded, loadAudioModel } = await setup();
  const load = loadAudioModel.getMockImplementation()!;
  loadAudioModel.mockImplementationOnce(async (entry) => {
    const binding = await load(entry);
    if (binding.kind !== 'tts') return binding;
    return {
      kind: 'tts',
      model: {
        ...binding.model,
        capabilities: { ...binding.model.capabilities, audioStreaming: false },
      },
    };
  });
  await expect(host.loadModel('tts')).rejects.toThrow(
    'TTS catalogue does not match loaded checkpoint: audio_streaming',
  );
  expect(loaded[0].model.dispose).toHaveBeenCalledOnce();
  expect(host.server.registry.audio.get('tts')).toBeUndefined();
});
it('coalesces cold loads, retains another capability and releases a replaced TTS instance', async () => {
  const { host, loaded, loadAudioModel } = await setup();
  await Promise.all([host.loadModel('asr'), host.loadModel('asr')]);
  expect(loadAudioModel).toHaveBeenCalledTimes(1);
  await host.loadModel('tts');
  await host.loadModel('tts2');
  expect(host.server.registry.audio.list().map((e) => e.id)).toEqual(['asr', 'tts2']);
  expect(loaded[0].model.dispose).not.toHaveBeenCalled();
  expect(loaded[1].model.dispose).toHaveBeenCalledOnce();
  await host.close();
  expect(loaded[0].model.dispose).toHaveBeenCalledOnce();
  expect(loaded[2].model.dispose).toHaveBeenCalledOnce();
});
it('single residency disposes ASR before loading TTS', async () => {
  const { host, loaded } = await setup('single');
  await host.loadModel('asr');
  await host.loadModel('tts');
  expect(loaded[0].model.dispose).toHaveBeenCalledOnce();
  expect(host.server.registry.audio.list().map((e) => e.id)).toEqual(['tts']);
});

it.each(['single', 'capability'] as const)('keeps chat aliases inside chat when residency is %s', async (policy) => {
  const { host } = await setup(policy, true);
  await host.loadModel('chat');
  await host.loadModel('asr');
  expect(!!host.server.registry.getAny('chat')).toBe(policy === 'capability');
  // Host.loadModel intentionally requires a discovered name. HTTP owns aliases.
  await host.loadModel('chat');
  expect(!!host.server.registry.audio.get('asr')).toBe(policy === 'capability');
  expect(host.server.registry.getAny('chat')).toBeDefined();
  expect(host.server.registry.getAny('asr')).toBeUndefined();
});
it.each(['single', 'capability'] as const)(
  'switches BGE/Jina alongside chat, ASR and TTS with %s residency',
  async (policy) => {
    const { host, rankModels, loadRerankModel } = await setup(policy, true, true);
    await host.loadModel('chat');
    await host.loadModel('asr');
    await host.loadModel('tts');
    await host.loadModel('jina');
    expect(loadRerankModel).toHaveBeenLastCalledWith(
      expect.objectContaining({ name: 'jina', modelType: 'jina_reranker' }),
    );
    await host.loadModel('bge');
    expect(loadRerankModel).toHaveBeenLastCalledWith(
      expect.objectContaining({ name: 'bge', modelType: 'xlm_roberta_reranker' }),
    );
    expect(vi.spyOn(rankModels[0], 'dispose')).toHaveBeenCalledOnce();
    expect(host.server.registry.rerank.get('jina')).toBeUndefined();
    expect(host.server.registry.rerank.get('bge')).toBeDefined();
    for (const name of ['asr', 'tts']) expect(!!host.server.registry.audio.get(name)).toBe(policy === 'capability');
    expect(!!host.server.registry.getAny('chat')).toBe(policy === 'capability');
    for (const name of ['chat', 'asr', 'tts']) {
      await host.loadModel(name);
      expect(!!host.server.registry.rerank.get('bge')).toBe(policy === 'capability');
      await host.loadModel('bge');
    }
    await host.close();
    for (const model of rankModels) expect(vi.spyOn(model, 'dispose')).toHaveBeenCalledOnce();
  },
);
