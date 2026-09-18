#!/usr/bin/env oxnode

/// <reference types="node" />

// K2-Horizon coding-agent workload benchmark (mlx-node native path).
// Replays a real Devin session shape: large system prompt + 8 turns of
// user/tool-result-sized content, multi-turn with delta prefill.
// Normal user path: ChatSession.send (no flags, no path tricks).
// Run: K2_MXFP8=... K2_WORKLOAD=... oxnode scripts/k2-bench-session-node.ts

import { createHash } from 'node:crypto';
import { closeSync, openSync, readFileSync, writeSync } from 'node:fs';
import { join } from 'node:path';

import { Qwen3Tokenizer, getMemorySnapshot, type ChatMessage, type ChatResult } from '@mlx-node/core';
import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const MODEL = process.env.K2_MXFP8;
const WORKLOAD = process.env.K2_WORKLOAD;
const MAX_TOKENS = Number(process.env.K2_MAX_TOKENS ?? '200');
const REPETITIONS = Number(process.env.K2_REPETITIONS ?? '1');
if (!MODEL || !WORKLOAD) throw new Error('Set K2_MXFP8 and K2_WORKLOAD to the model and frozen workload paths');
if (!Number.isInteger(MAX_TOKENS) || MAX_TOKENS < 1 || MAX_TOKENS > 200) {
  throw new Error('K2_MAX_TOKENS must be 1..200 so neither runtime reaches the low-effort reasoning budget');
}
if (!Number.isInteger(REPETITIONS) || REPETITIONS < 1) throw new Error('K2_REPETITIONS must be a positive integer');
const workloadBytes = readFileSync(WORKLOAD);
const workload = JSON.parse(workloadBytes.toString('utf8')) as { system: string; turns: string[] };
const referenceLines = process.env.K2_REFERENCE_INPUTS
  ? readFileSync(process.env.K2_REFERENCE_INPUTS, 'utf8').trim().split('\n')
  : [];
const referenceTurns = referenceLines
  .slice(1)
  .map(
    (line) =>
      JSON.parse(line) as {
        repetition: number;
        turn: number;
        promptTokenIds: number[];
        result: ChatResult;
      },
  )
  .filter((turn) => turn.repetition === 0);
const hash = (data: string | Buffer) => createHash('sha256').update(data).digest('hex');
const capture = process.env.K2_CAPTURE ? openSync(process.env.K2_CAPTURE, 'wx', 0o600) : undefined;
const record = (value: unknown) => {
  if (capture !== undefined) writeSync(capture, JSON.stringify(value) + '\n');
};

try {
  const metadata = {
    kind: 'metadata',
    runtime: 'mlx-node',
    pid: process.pid,
    model: MODEL,
    workloadSha256: hash(workloadBytes),
    configSha256: hash(readFileSync(join(MODEL, 'config.json'))),
    tokenizerSha256: hash(readFileSync(join(MODEL, 'tokenizer.json'))),
    maxTokens: MAX_TOKENS,
    repetitions: REPETITIONS,
    effort: 'low',
    temperature: 0,
    readyToTrace: process.env.K2_WAIT_FOR_TRACE === '1',
  };
  if (referenceLines.length > 0) {
    const expected = JSON.parse(referenceLines[0]!) as typeof metadata;
    for (const key of [
      'workloadSha256',
      'configSha256',
      'tokenizerSha256',
      'maxTokens',
      'effort',
      'temperature',
    ] as const) {
      if (metadata[key] !== expected[key]) throw new Error(`Reference capture differs in ${key}`);
    }
    if (referenceTurns.length !== workload.turns.length) throw new Error('Reference capture is incomplete');
  }
  console.log(JSON.stringify(metadata));
  record(metadata);
  if (metadata.readyToTrace) {
    await new Promise<void>((resolve) => {
      const keepAlive = setInterval(() => {}, 1000);
      process.once('SIGUSR2', () => {
        clearInterval(keepAlive);
        resolve();
      });
    });
  }

  const model = (await loadModel(MODEL)) as unknown as SessionCapableModel;
  const tokenizer = await Qwen3Tokenizer.fromPretrained(join(MODEL, 'tokenizer.json'));
  console.log(JSON.stringify({ loaded: MODEL, paged: model.hasBlockPagedCache?.() }));
  const frozenPromptIds: number[][] = [];
  for (let repetition = 0; repetition < REPETITIONS; repetition++) {
    if (repetition > 0) await model.resetCaches();
    const session = new ChatSession(model, { system: workload.system });
    const history: ChatMessage[] = [{ role: 'system', content: workload.system }];
    for (const [turnIndex, turn] of workload.turns.entries()) {
      const messages: ChatMessage[] = [...history, { role: 'user', content: turn }];
      const promptTokens = await tokenizer.applyChatTemplate(
        messages,
        true,
        undefined,
        true,
        undefined,
        undefined,
        'low',
      );
      const rendered = await tokenizer.decode(promptTokens, false);
      const promptTokenIds = Array.from(promptTokens);
      const promptBytes = Buffer.alloc(promptTokenIds.length * 4);
      promptTokenIds.forEach((token, index) => promptBytes.writeUInt32LE(token, index * 4));
      const promptTokenSha256 = hash(promptBytes);
      const expected = referenceTurns.find((entry) => entry.turn === turnIndex);
      if (
        referenceLines.length > 0 &&
        (!expected || JSON.stringify(promptTokenIds) !== JSON.stringify(expected.promptTokenIds))
      ) {
        throw new Error(`Prompt IDs differ from the frozen baseline at turn ${turnIndex}`);
      }
      if (repetition === 0) frozenPromptIds.push(promptTokenIds);
      else if (JSON.stringify(promptTokenIds) !== JSON.stringify(frozenPromptIds[turnIndex])) {
        throw new Error(`Session history changed prompt IDs in repetition ${repetition}, turn ${turnIndex}`);
      }
      const t0 = performance.now();
      const result = await session.send(turn, {
        config: {
          maxNewTokens: MAX_TOKENS,
          temperature: 0,
          reasoningEffort: 'low',
          reportPerformance: true,
        },
      });
      const wallMs = performance.now() - t0;
      if (expected) {
        for (const field of [
          'rawText',
          'text',
          'thinking',
          'thinkingEnabled',
          'numTokens',
          'reasoningTokens',
          'finishReason',
          'promptTokens',
          'cachedTokens',
        ] as const) {
          if (result[field] !== expected.result[field])
            throw new Error(`Output ${field} differs from the frozen baseline at turn ${turnIndex}`);
        }
      }
      if (result.toolCalls.some((call) => call.status === 'ok')) {
        throw new Error(
          'This text-session workload emitted a tool call; freeze a tool-aware replay before comparing it',
        );
      }
      record({
        kind: 'turn',
        repetition,
        turn: turnIndex,
        messages,
        rendered,
        promptTokenIds,
        promptTokenSha256,
        result,
        wallMs,
        memory: getMemorySnapshot(),
      });
      history.push(messages[messages.length - 1]!, {
        role: 'assistant',
        content: result.text,
        reasoningContent: result.thinking ?? '',
        thinkingEnabled: result.thinkingEnabled,
      });
      console.log(
        JSON.stringify({
          turn: turnIndex,
          repetition,
          phase: repetition === 0 ? 'first-pass' : 'steady-state',
          promptChars: Array.from(turn).length,
          renderedPromptTokens: promptTokenIds.length,
          cachedTokens: result.cachedTokens,
          numTokens: result.numTokens,
          finishReason: result.finishReason,
          ttftMs: result.performance?.ttftMs ?? null,
          prefillTps: result.performance?.prefillTokensPerSecond ?? null,
          decodeTps: result.performance?.decodeTokensPerSecond ?? null,
          wallMs: Math.round(wallMs),
          text: result.text.slice(0, 50),
        }),
      );
    }
  }
} finally {
  if (capture !== undefined) closeSync(capture);
}
