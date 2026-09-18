#!/usr/bin/env oxnode

import { mkdtemp, readdir, readFile, stat, symlink, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const SRC = process.env.K2_SRC ?? '/Users/brooklyn/workspace/github/mlx-node/.cache/models/k2-horizon-7b-fp8';
const PROMPT = 'What is 17 * 23? Give the final number.';
const EXPECT = '391';

async function flatClone(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'k2-flat-'));
  const cfg = JSON.parse(await readFile(join(SRC, 'config.json'), 'utf-8'));
  cfg.use_block_paged_cache = false;
  await writeFile(join(dir, 'config.json'), JSON.stringify(cfg));
  for (const name of await readdir(SRC)) {
    if (name === 'config.json') continue;
    if ((await stat(join(SRC, name))).isFile()) {
      await symlink(join(SRC, name), join(dir, name));
    }
  }
  return dir;
}

async function runOnce(path: string, label: string, maxTok = 256) {
  const t0 = performance.now();
  const model = await loadModel(path);
  const loadMs = Math.round(performance.now() - t0);
  const native = model as unknown as { hasBlockPagedCache?: () => boolean };
  console.log(JSON.stringify({ label, loadMs, paged: native.hasBlockPagedCache?.() ?? 'n/a' }));
  const session = new ChatSession(model as unknown as SessionCapableModel);
  const t1 = performance.now();
  const result = await session.send(PROMPT, {
    config: { maxNewTokens: maxTok, temperature: 0, reasoningEffort: 'low', reportPerformance: true },
  });
  return {
    label,
    loadMs,
    genMs: Math.round(performance.now() - t1),
    finishReason: result.finishReason,
    numTokens: result.numTokens,
    reasoningTokens: result.reasoningTokens,
    has391: (result.text + (result.rawText ?? '')).includes(EXPECT),
    text: result.text.slice(0, 400),
    thinking: (result.thinking ?? '').slice(0, 200),
    perf: result.performance,
    session,
  };
}

const mode = process.argv[2] ?? 'all';

if (mode === 'paged' || mode === 'all') {
  const r = await runOnce(SRC, 'paged');
  console.log(JSON.stringify(r, null, 1));
}

if (mode === 'flat' || mode === 'all') {
  const flatDir = await flatClone();
  try {
    const r = await runOnce(flatDir, 'flat');
    console.log(JSON.stringify(r, null, 1));
  } finally {
    await rm(flatDir, { recursive: true, force: true });
  }
}

if (mode === 'concurrent' || mode === 'all') {
  // Two sessions on one loaded model — paged scheduler overlap.
  const model = await loadModel(SRC);
  const s1 = new ChatSession(model as unknown as SessionCapableModel);
  const s2 = new ChatSession(model as unknown as SessionCapableModel);
  const t0 = performance.now();
  const [r1, r2] = await Promise.all([
    s1.send('What is 11 * 13? Final number only.', {
      config: { maxNewTokens: 128, temperature: 0, reasoningEffort: 'low' },
    }),
    s2.send('What is 7 * 19? Final number only.', {
      config: { maxNewTokens: 128, temperature: 0, reasoningEffort: 'low' },
    }),
  ]);
  console.log(
    JSON.stringify({
      label: 'concurrent',
      ms: Math.round(performance.now() - t0),
      r1: { text: r1.text.slice(0, 200), ok: (r1.text + (r1.rawText ?? '')).includes('143') },
      r2: { text: r2.text.slice(0, 200), ok: (r2.text + (r2.rawText ?? '')).includes('133') },
    }),
  );
}
