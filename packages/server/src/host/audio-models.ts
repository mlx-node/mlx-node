/** Audio inventory is config-only: listing models never imports the addon. */
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { inspectTtsConfig } from '@mlx-node/tts/catalog';

import { AudioError } from '../audio-io.js';
import { asrCapabilities, ttsCapabilities, type AudioModel } from '../audio-registry.js';
import type { ModelRegistry } from '../registry.js';
import { chatCapability, withCapabilityModels, type CapabilityAdapter } from './capability-models.js';
import { hasAudioCheckpointFiles } from './checkpoint-files.js';
import type { SwapController } from './swap.js';

export interface DiscoveredAudioModel {
  name: string;
  path: string;
  modelType: 'qwen3_asr' | 'qwen3_tts';
  kind: 'asr' | 'tts';
  capabilities: Record<string, unknown>;
}
export async function discoverAudioModels(
  root: string,
  onFailure?: (error: unknown) => void,
): Promise<DiscoveredAudioModel[]> {
  let entries;
  try {
    entries = await readdir(root, { withFileTypes: true });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') onFailure?.(error);
    return [];
  }
  const result: DiscoveredAudioModel[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory() && !entry.isSymbolicLink()) continue;
    const path = join(root, entry.name);
    try {
      const config = JSON.parse(await readFile(join(path, 'config.json'), 'utf8'));
      if (config.model_type === 'qwen3_asr') {
        if (!(await hasAudioCheckpointFiles(path, 'asr'))) continue;
        if (!config.audio_config || !config.text_config)
          throw new Error(`Unsupported ASR checkpoint layout at ${path}; convert Qwen/Qwen3-ASR-1.7B-hf first`);
        result.push({ name: entry.name, path, modelType: 'qwen3_asr', kind: 'asr', capabilities: asrCapabilities });
      } else if (config.model_type === 'qwen3_tts') {
        if (!(await hasAudioCheckpointFiles(path, 'tts'))) continue;
        const codec = JSON.parse(await readFile(join(path, 'speech_tokenizer/config.json'), 'utf8'));
        const caps = inspectTtsConfig(config, codec);
        result.push({
          name: entry.name,
          path,
          modelType: 'qwen3_tts',
          kind: 'tts',
          capabilities: ttsCapabilities(caps),
        });
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') onFailure?.(error);
    }
  }
  return result.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}
export async function loadAudioModel(entry: DiscoveredAudioModel): Promise<AudioModel> {
  if (entry.kind === 'asr') {
    const { Qwen3AsrModel } = await import('@mlx-node/asr');
    return { kind: 'asr', model: await Qwen3AsrModel.load(entry.path) };
  }
  const { loadTtsModel } = await import('@mlx-node/tts');
  return { kind: 'tts', model: await loadTtsModel(entry.path) };
}

/** Audio adapters own audio loading; the host owns cross-capability residency. */
export function audioCapabilities(
  discovered: DiscoveredAudioModel[],
  registry: ModelRegistry,
  loader: (entry: DiscoveredAudioModel) => Promise<AudioModel>,
): CapabilityAdapter[] {
  const created = Math.floor(Date.now() / 1000);
  return (['asr', 'tts'] as const).map((kind) => {
    const entries = discovered.filter((e) => e.kind === kind);
    return {
      kind,
      hasResident: (name) => registry.audio.get(name)?.kind === kind,
      listModels: () =>
        entries.map((entry) => ({
          id: entry.name,
          object: 'model' as const,
          created,
          owned_by: 'mlx-node',
          modality: 'audio' as const,
          task: entry.kind,
          model_family: entry.modelType,
          capabilities: entry.capabilities,
        })),
      disposeAll: () => registry.audio.disposeAll(kind),
      async resolveModel(name) {
        const entry = entries.find((e) => e.name === name);
        if (!entry) throw new AudioError(404, 'model_not_found', `Model ${name} not found`, 'model');
        if (registry.audio.get(name)) return;
        await registry.audio.disposeAll(kind);
        const loaded = await loader(entry);
        try {
          if (loaded.kind !== entry.kind) throw new Error('Audio loader returned the wrong task');
          if (loaded.kind === 'tts') {
            const actual = ttsCapabilities(loaded.model.capabilities);
            for (const key of Object.keys(actual))
              if (JSON.stringify(actual[key]) !== JSON.stringify(entry.capabilities[key]))
                throw new Error(`TTS catalogue does not match loaded checkpoint: ${key}`);
          }
          registry.audio.register(name, loaded);
        } catch (error) {
          await loaded.model.dispose();
          throw error;
        }
      },
    };
  });
}
/** Compatibility entry point for callers composing only chat and audio. */
export function withAudioModels(
  chat: SwapController,
  discovered: DiscoveredAudioModel[],
  registry: ModelRegistry,
  loader: (entry: DiscoveredAudioModel) => Promise<AudioModel>,
  residency: 'capability' | 'single',
  hasChat: boolean,
): SwapController {
  return withCapabilityModels(
    [...(hasChat ? [chatCapability(chat, registry)] : []), ...audioCapabilities(discovered, registry, loader)],
    residency,
  );
}
