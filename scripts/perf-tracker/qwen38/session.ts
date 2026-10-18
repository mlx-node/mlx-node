/// <reference types="node" />

import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { createInterface } from 'node:readline';

import { writeJson, type FixtureCase, type MlxSpeculationRequest } from '../common.js';

export interface StreamEvent {
  done: boolean;
  promptTokens?: number;
  cachedTokens?: number;
  [field: string]: unknown;
}

export interface SessionModel {
  resetCaches(): Promise<void>;
  chatStreamSessionStart(messages: unknown[], config: Record<string, unknown>): AsyncGenerator<StreamEvent>;
}

export interface Tokenizer {
  applyChatTemplate(
    messages: unknown[],
    addGenerationPrompt: boolean,
    tools: null,
    enableThinking: boolean,
  ): Promise<Uint32Array>;
}

// Native sessions render with preserve_thinking=true, the standalone tokenizer with
// false; a private copy of the loaded template renders what the session renders.
export async function sessionTokenizer(
  assets: string,
  out: string,
  load: (path: string) => Promise<Tokenizer>,
): Promise<Tokenizer> {
  const tokenizerDir = join(out, 'session-tokenizer');
  await mkdir(tokenizerDir, { recursive: true });
  await copyFile(join(assets, 'tokenizer.json'), join(tokenizerDir, 'tokenizer.json'));
  const config = JSON.parse(await readFile(join(assets, 'tokenizer_config.json'), 'utf8')) as {
    chat_template?: unknown;
  };
  const template =
    typeof config.chat_template === 'string'
      ? config.chat_template
      : await readFile(join(assets, 'chat_template.jinja'), 'utf8');
  config.chat_template = `{% set preserve_thinking = true %}${template}`;
  await writeFile(join(tokenizerDir, 'tokenizer_config.json'), JSON.stringify(config));
  return load(join(tokenizerDir, 'tokenizer.json'));
}

// One stdin line per request until {"action":"exit"}; one stdout result line each.
export async function serveSamples(
  model: SessionModel,
  tokenizer: Tokenizer,
  cases: readonly FixtureCase[],
  out: string,
  memoryStats: () => unknown,
  speculation: MlxSpeculationRequest,
): Promise<void> {
  for await (const line of createInterface({ input: process.stdin })) {
    const request = JSON.parse(line) as { action?: string; name: string; run: number | string; maxNewTokens: number };
    if (request.action === 'exit') break;
    const item = cases.find((candidate) => candidate.name === request.name);
    if (!item) throw new Error(`Unknown case ${request.name}`);
    await model.resetCaches();
    const tokens = Array.from(await tokenizer.applyChatTemplate(item.messages, true, null, true));
    if (JSON.stringify(tokens) !== JSON.stringify(item.tokenIds)) throw new Error('Prompt token mismatch');
    const started = performance.now();
    let result: StreamEvent | undefined;
    let firstEventMs: number | undefined;
    for await (const event of model.chatStreamSessionStart(item.messages, {
      maxNewTokens: request.maxNewTokens,
      temperature: 0,
      topK: 0,
      topP: 1,
      minP: 0,
      repetitionPenalty: 1,
      presencePenalty: 0,
      frequencyPenalty: 0,
      reasoningEffort: 'high',
      includeReasoning: true,
      ...speculation,
      reuseCache: true,
      reportPerformance: true,
      cacheSalt: `benchmark-${process.pid}-${request.run}-${request.name}`,
    })) {
      firstEventMs ??= performance.now() - started;
      if (event.done) result = event;
    }
    if (!result) throw new Error('No terminal result');
    const report: Record<string, unknown> = {
      event: 'result',
      runtime: 'mlx-node',
      name: request.name,
      run: request.run,
      pid: process.pid,
      wallMs: performance.now() - started,
      firstEventMs,
      ...result,
      memory: memoryStats(),
    };
    await writeJson(join(out, 'raw', `mlx-${request.run}-${request.name}.json`), report);
    if (result.promptTokens !== item.tokenIds.length || result.cachedTokens !== 0) {
      throw new Error(
        `Prompt/cache accounting mismatch: expected ${item.tokenIds.length}/0; received ${result.promptTokens}/${result.cachedTokens}`,
      );
    }
    const { text: _text, rawText: _rawText, thinking: _thinking, publicRawText: _publicRawText, ...summary } = report;
    console.log(JSON.stringify(summary));
  }
}
