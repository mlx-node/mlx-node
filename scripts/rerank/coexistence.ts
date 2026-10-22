/** Compare chat alone, chat with an idle reranker, and simultaneous inference. */
import assert from 'node:assert/strict';
import { mkdtemp, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { getMemorySnapshot, resetPeakMemory } from '@mlx-node/core';

import { createInferenceHost } from '../../packages/server/src/host/index.js';
const [chatPath, rerankPath, output] = process.argv.slice(2);
if (!output) throw new Error('Usage: oxnode scripts/rerank/coexistence.ts CHAT RERANK REPORT');
const root = await mkdtemp(join(tmpdir(), 'mlx-rerank-coexist-'));
await symlink(resolve(chatPath), join(root, 'chat'));
await symlink(resolve(rerankPath), join(root, 'rerank'));
const memory = () =>
  Object.fromEntries(Object.entries(getMemorySnapshot()).map(([k, v]) => [k, typeof v === 'bigint' ? Number(v) : v]));
const report: Record<string, unknown> = { recordedAt: new Date().toISOString(), chatPath, rerankPath };
const host = await createInferenceHost({
  modelsDir: root,
  model: 'chat',
  port: 0,
  disableStore: true,
  sweepOrphanTempRoots: false,
  logDir: join(root, 'logs'),
});
async function post(path: string, body: unknown) {
  const start = performance.now();
  const r = await fetch(host.url + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const value = await r.json();
  assert.equal(r.status, 200, JSON.stringify(value));
  return { latencyMs: performance.now() - start, value };
}
const chat = () =>
  post('/v1/messages', {
    model: 'chat',
    messages: [{ role: 'user', content: 'Write one sentence explaining unified memory. /no_think' }],
    temperature: 0,
    max_tokens: 8,
  });
const rank = () =>
  post('/v1/rerank', {
    model: 'rerank',
    query: 'What is unified memory?',
    documents: Array.from(
      { length: 50 },
      (_, i) => `Passage ${i}. ` + 'Apple Silicon uses unified memory for local machine learning inference. '.repeat(4),
    ),
  });
try {
  report.coldChat = await chat();
  console.log('cold chat completed');
  const alone = [];
  for (let i = 0; i < 3; i++) alone.push(await chat());
  report.chatAlone = alone;
  console.log('chat control completed');
  report.aloneMemory = memory();
  const start = performance.now();
  await host.loadModel('rerank');
  report.rerankerLoadMs = performance.now() - start;
  assert.ok(host.server.registry.getAny('chat'));
  assert.ok(host.server.registry.rerank.get('rerank'));
  const resident = [];
  for (let i = 0; i < 3; i++) resident.push(await chat());
  report.chatWithIdleReranker = resident;
  console.log('idle co-residency completed');
  report.residentMemory = memory();
  report.rerankAlone = await rank();
  const simultaneous = [];
  resetPeakMemory();
  for (let i = 0; i < 3; i++) {
    const [c, r] = await Promise.all([chat(), rank()]);
    simultaneous.push({ chat: c, rerank: r });
    console.log('concurrent round', i + 1);
  }
  report.simultaneous = simultaneous;
  report.concurrentMemory = memory();
  const expected = JSON.stringify(alone[0].value.content);
  for (const r of [...alone, ...resident, ...simultaneous.map((r) => r.chat)])
    assert.equal(JSON.stringify(r.value.content), expected, 'chat output changed under co-residency');
} finally {
  await host.close();
  await rm(root, { recursive: true, force: true });
}
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ output, done: true }));
