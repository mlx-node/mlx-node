#!/usr/bin/env oxnode
/// <reference types="node" />

import { copyFile, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

import type { ChatMessage, ToolDefinition } from '../../../packages/core/index.cjs';
import { ROOT, assertLoadedAddon, sha256 } from '../common.js';
import { MODEL_DIR, readFixture } from './suite.js';

type Core = typeof import('../../../packages/core/index.cjs');

// mlx-node renders the prompt from the model folder while llama.cpp receives the
// pinned IDs, so the folder's tokenizer and template must still produce them.
// The session renders with preserve_thinking = true (SESSION_PRESERVE_THINKING
// in engine/session.rs); applyChatTemplate has no such option, so the flag is
// set inside a temporary copy of the template. enableThinking = true is what
// reasoningEffort 'high' resolves to.
const { manifest, cases } = await readFixture();
const temporary = await mkdtemp(join(tmpdir(), 'perf-tracker-gemma4-'));
try {
  const config = JSON.parse(await readFile(join(MODEL_DIR, 'tokenizer_config.json'), 'utf8')) as {
    chat_template?: unknown;
  };
  const template =
    typeof config.chat_template === 'string'
      ? config.chat_template
      : await readFile(join(MODEL_DIR, 'chat_template.jinja'), 'utf8');
  config.chat_template = `{% set preserve_thinking = true %}${template}`;
  await copyFile(join(MODEL_DIR, 'tokenizer.json'), join(temporary, 'tokenizer.json'));
  await writeFile(join(temporary, 'tokenizer_config.json'), JSON.stringify(config));
  const { Qwen3Tokenizer } = (await import(pathToFileURL(join(ROOT, 'packages/core/index.cjs')).href)) as Core;
  assertLoadedAddon();
  const tokenizer = await Qwen3Tokenizer.fromPretrained(join(temporary, 'tokenizer.json'));
  for (const [index, item] of cases.entries()) {
    const ids = await tokenizer.applyChatTemplate(
      item.messages as ChatMessage[],
      true,
      item.tools as ToolDefinition[] | null,
      true,
    );
    if (
      sha256(JSON.stringify(Array.from(ids))) !== manifest.cases[index].tokenIdsSha256 ||
      (await tokenizer.decode(ids, false)) !== item.rendered
    ) {
      throw new Error(`${item.name}: the model folder's tokenizer/template no longer renders the pinned token IDs`);
    }
  }
} finally {
  await rm(temporary, { recursive: true, force: true });
}
console.log(
  JSON.stringify({
    preserveThinking: true,
    enableThinking: true,
    cases: manifest.cases.map(({ name, tokenIdsSha256 }) => ({ name, tokenIdsSha256 })),
  }),
);
