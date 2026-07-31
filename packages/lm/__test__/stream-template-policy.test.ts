import type { ChatMessage } from '@mlx-node/core';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const tokenizerMocks = vi.hoisted(() => ({
  applyChatTemplate: vi.fn(),
  fromPretrained: vi.fn(),
}));

vi.mock('@mlx-node/core', () => {
  class UnusedNativeModel {}
  return {
    Gemma4Model: UnusedNativeModel,
    Lfm2Model: UnusedNativeModel,
    Qwen3Model: UnusedNativeModel,
    Qwen35Model: UnusedNativeModel,
    Qwen35MoeModel: UnusedNativeModel,
    Qwen3Tokenizer: {
      fromPretrained: tokenizerMocks.fromPretrained,
    },
  };
});

import { makeStreamingModel } from '../src/stream.js';

class NativeStreamingStub {
  static async load(_modelPath: string): Promise<NativeStreamingStub> {
    return new NativeStreamingStub();
  }

  chatStreamSessionStart(..._args: never[]): never {
    throw new Error('not used');
  }

  chatStreamSessionContinue(..._args: never[]): never {
    throw new Error('not used');
  }

  chatStreamSessionContinueTool(..._args: never[]): never {
    throw new Error('not used');
  }
}

const messages: ChatMessage[] = [
  {
    role: 'user',
    content: 'Read <image>',
    images: [new Uint8Array([1, 2, 3])],
  },
];

describe('makeStreamingModel template content policy', () => {
  beforeEach(() => {
    tokenizerMocks.applyChatTemplate.mockReset();
    tokenizerMocks.applyChatTemplate.mockResolvedValue(new Uint32Array([1, 2]));
    tokenizerMocks.fromPretrained.mockReset();
    tokenizerMocks.fromPretrained.mockResolvedValue({
      applyChatTemplate: tokenizerMocks.applyChatTemplate,
    });
  });

  it('forwards Qianfan image order and manual-placeholder suppression', async () => {
    const QianfanLikeModel = makeStreamingModel(NativeStreamingStub, {
      recordModelPath: true,
      templateContentPolicy: {
        order: 'imagesThenText',
        existingImagePlaceholder: '<image>',
      },
    });
    const model = await QianfanLikeModel.load('/models/qianfan');

    await model.applyChatTemplate(messages, true, null, false);

    expect(tokenizerMocks.fromPretrained).toHaveBeenCalledWith('/models/qianfan/tokenizer.json');
    expect(tokenizerMocks.applyChatTemplate).toHaveBeenCalledWith(
      messages,
      true,
      null,
      false,
      'imagesThenText',
      '<image>',
    );
  });

  it('keeps the existing four-argument tokenizer call for default families', async () => {
    const DefaultModel = makeStreamingModel(NativeStreamingStub, {
      recordModelPath: true,
    });
    const model = await DefaultModel.load('/models/default');

    await model.applyChatTemplate(messages, true, null, false);

    expect(tokenizerMocks.applyChatTemplate).toHaveBeenCalledWith(messages, true, null, false);
  });

  it('exposes raw assistant replay only for checkpoint templates that require it', async () => {
    const DefaultModel = makeStreamingModel(NativeStreamingStub, {
      recordModelPath: true,
    });
    const RawReplayModel = makeStreamingModel(NativeStreamingStub, {
      recordModelPath: true,
      replayAssistantRawText: true,
    });

    const defaultModel = await DefaultModel.load('/models/default');
    const rawReplayModel = await RawReplayModel.load('/models/raw-replay');

    expect(defaultModel.replaysAssistantRawText?.()).toBe(false);
    expect(rawReplayModel.replaysAssistantRawText?.()).toBe(true);
  });
});
