/**
 * Replay a recorded mlx agent context through its production provider.
 * Historical tools are data only: this script never executes them.
 *
 * MLX_COLD_CACHE_DIR=/an/isolated/cache oxnode scripts/benchmark-qwen35-agent-session.ts \
 *   MODEL SESSION.jsonl OUTPUT.json capture|restore [ASSISTANT_ENTRY_ID]
 *
 * Defaults to the last successful assistant entry's input context. Pi resolves
 * the parent chain and compaction. The original system prompt is not recorded;
 * reconstruct it with current Pi tools/prompt and retain a context hash.
 * Capture measures full prefill followed by three genuine resident continuations.
 * Restore runs in a NEW process and requires an installed recurrent sidecar.
 * MLX_AGENT_BENCH_WARM_TURNS overrides the continuation count (0..12).
 * With MLX_AGENT_BENCH_GATE_DIR, a controller creates turn-N.go to release each
 * turn, waits for turn-N.done, and creates finish.go after the final turn.
 */
import { createHash } from 'node:crypto';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { basename, resolve } from 'node:path';
import { setTimeout } from 'node:timers/promises';

import type { Api, AssistantMessage, Context, Model, ThinkingLevel } from '@earendil-works/pi-ai';
import {
  buildSessionContext,
  convertToLlm,
  createCodingTools,
  parseSessionEntries,
} from '@earendil-works/pi-coding-agent';
import { coldCacheDrain, coldCacheStats, coldSidecarStats, getMemorySnapshot, resetPeakMemory } from '@mlx-node/core';
import { detectModelType, PagedConfigOverrideManager } from '@mlx-node/lm';

import { MlxModelHost } from '../packages/agent/src/provider/model-host.js';
import { makeMlxStreamSimple, type TurnRecorder } from '../packages/agent/src/provider/stream-adapter.js';

const [modelArg, sessionArg, outputArg, mode = 'capture', entryId] = process.argv.slice(2);
if (!modelArg || !sessionArg || !outputArg || !['capture', 'restore'].includes(mode) || !process.env.MLX_COLD_CACHE_DIR)
  throw new Error('Expected MODEL SESSION.jsonl OUTPUT.json capture|restore [ENTRY_ID] and MLX_COLD_CACHE_DIR');
if (process.platform === 'darwin' && process.arch !== 'arm64') throw new Error('Native ARM64 Node is required');
process.env.MLX_PAGED_PREFILL_CHUNK_SIZE ??= '2048';
const warmTurns = Number(process.env.MLX_AGENT_BENCH_WARM_TURNS ?? '3');
if (!Number.isInteger(warmTurns) || warmTurns < 0 || warmTurns > 12)
  throw new Error('MLX_AGENT_BENCH_WARM_TURNS must be an integer from 0 to 12');
// Optional controller handshake for alternating resident processes. The done
// marker is published only after the native producer and SSD writer are idle.
const gateDir = process.env.MLX_AGENT_BENCH_GATE_DIR;
if (gateDir) await mkdir(gateDir, { recursive: true });
async function waitForGate(name: string) {
  if (!gateDir) return;
  const deadline = performance.now() + 600_000;
  while (true) {
    try {
      await stat(resolve(gateDir, `${name}.go`));
      return;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    }
    if (performance.now() > deadline) throw new Error(`Controller timed out before ${name}`);
    await setTimeout(100);
  }
}
const sha256 = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const addonPath = (process.report.getReport() as { sharedObjects: string[] }).sharedObjects.find((path) =>
  /mlx-core.*\.node$/.test(path),
);
if (!addonPath) throw new Error('Cannot identify the loaded native addon');
const modelPath = resolve(modelArg);
const sessionText = await readFile(resolve(sessionArg), 'utf8');
const allEntries = parseSessionEntries(sessionText);
const header = allEntries.find((e) => e.type === 'session');
if (!header) throw new Error('No session header');
const entries = allEntries.filter((e) => e.type !== 'session');
const selected = [...entries]
  .reverse()
  .find(
    (e) =>
      e.type === 'message' &&
      e.message.role === 'assistant' &&
      (entryId ? e.id === entryId : e.message.stopReason === 'stop'),
  );
if (!selected || selected.type !== 'message' || selected.message.role !== 'assistant')
  throw new Error('No matching assistant entry');
const session = buildSessionContext(entries, selected.parentId);
const reasoning = session.thinkingLevel === 'off' ? undefined : session.thinkingLevel;
if (reasoning !== undefined && !['minimal', 'low', 'medium', 'high', 'xhigh'].includes(reasoning))
  throw new Error(`Unknown recorded thinking level: ${reasoning}`);
const codingTools = createCodingTools(header.cwd);
// Pi does not export this pure prompt builder from its public entrypoint.
const { buildSystemPrompt } = await import(
  new URL('./core/system-prompt.js', import.meta.resolve('@earendil-works/pi-coding-agent')).href
);
const initial: Context = {
  systemPrompt: buildSystemPrompt({
    cwd: header.cwd,
    selectedTools: codingTools.map((tool) => tool.name),
    toolSnippets: Object.fromEntries(codingTools.map((tool) => [tool.name, tool.description.split('\n')[0]])),
  }),
  tools: codingTools.map(({ name, description, parameters }) => ({ name, description, parameters })),
  messages: convertToLlm(session.messages),
};
const discovered = { name: basename(modelPath), path: modelPath, modelType: await detectModelType(modelPath) };
if (discovered.modelType !== 'qwen3_5') throw new Error(`Expected Qwen3.5 dense, got ${discovered.modelType}`);
const overlays = new PagedConfigOverrideManager();
const host = new MlxModelHost([discovered], {
  resolveModelPathFn: (model, policy) => overlays.resolve(model.path, model.modelType, policy?.persistPagedCache),
  requirePagedCache: true,
  persistPagedCache: true,
});
const model: Model<Api> = {
  id: discovered.name,
  name: discovered.name,
  api: 'mlx',
  provider: 'mlx',
  baseUrl: 'mlx://local',
  reasoning: true,
  input: ['text'],
  contextWindow: 262144,
  maxTokens: 256,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};
let telemetry: Parameters<TurnRecorder>[0] | undefined;
const stream = makeMlxStreamSimple(
  host,
  undefined,
  () => 'recorded-session-replay',
  (record) => {
    telemetry = record;
  },
);
const metadata = {
  startedAt: new Date().toISOString(),
  addonPath,
  addonSha256: sha256(await readFile(addonPath)),
  harnessSha256: sha256(await readFile(new URL(import.meta.url))),
  modelPath,
  checkpointConfigSha256: sha256(await readFile(resolve(modelPath, 'config.json'))),
  sessionPath: resolve(sessionArg),
  sessionSha256: sha256(sessionText),
  entryId: selected.id,
  historicalUsage: selected.message.usage,
  historicalModel: session.model,
  contextSha256: sha256(JSON.stringify(initial)),
  messages: initial.messages.length,
  reasoning: session.thinkingLevel,
  systemPromptReconstructed: true,
  runtime: { node: process.version, arch: process.arch, pid: process.pid },
  mode,
  env: Object.fromEntries(Object.entries(process.env).filter(([key]) => key.startsWith('MLX_'))),
};
const rows: unknown[] = [];
const drains: boolean[] = [];
const delta = (before: object, after: object) =>
  Object.fromEntries(
    Object.entries(after)
      .filter(
        ([key, value]) => typeof value === 'number' && typeof (before as Record<string, unknown>)[key] === 'number',
      )
      .map(([key, value]) => [key, (value as number) - (before as Record<string, number>)[key]!]),
  );
const save = () =>
  writeFile(
    resolve(outputArg),
    JSON.stringify({ ...metadata, rows, drains, cold: coldCacheStats(), sidecar: coldSidecarStats() }, null, 2),
  );
async function turn(name: string, context: Context): Promise<AssistantMessage> {
  telemetry = undefined;
  resetPeakMemory();
  const before = { cold: coldCacheStats(), sidecar: coldSidecarStats() };
  const start = performance.now();
  let message: AssistantMessage | undefined;
  let firstEventMs: number | undefined;
  for await (const event of stream(model, context, {
    temperature: 0,
    maxTokens: 256,
    reasoning: reasoning as ThinkingLevel | undefined,
    sessionId: 'recorded-session-replay',
  })) {
    if (event.type === 'text_delta' || event.type === 'thinking_delta' || event.type === 'toolcall_delta')
      firstEventMs ??= performance.now() - start;
    if (event.type === 'done') message = event.message;
    if (event.type === 'error') message = event.error;
  }
  const wallMs = performance.now() - start;
  // Pi's final event can precede completion of the detached native producer.
  // Queue a no-op on the same host before signaling another GPU process.
  if (host.residentId === model.id) await host.runWithResident(model.id, async () => {});
  const recorded = telemetry as Parameters<TurnRecorder>[0] | undefined;
  const row = {
    name,
    wallMs,
    firstEventMs,
    telemetry: recorded,
    message,
    coldDelta: delta(before.cold, coldCacheStats()),
    sidecarDelta: delta(before.sidecar, coldSidecarStats()),
    memory: getMemorySnapshot(),
  };
  rows.push(row);
  await save();
  process.stderr.write(JSON.stringify({ ...row, message: undefined, memory: undefined }) + '\n');
  if (!message || ['error', 'aborted'].includes(message.stopReason) || !recorded)
    throw new Error(`Failed turn ${name}`);
  if (mode === 'restore' && rows.length === 1 && !(row.sidecarDelta.installed > 0))
    throw new Error('SSD restart did not install a recurrent sidecar');
  return message;
}
let finalDrain = false;
try {
  const messages = [...initial.messages];
  for (let i = 0; i <= warmTurns; i++) {
    await waitForGate(`turn-${i}`);
    const message = await turn(
      i === 0 ? (mode === 'restore' ? 'ssd-restore' : 'full-prefill') : `warm-continuation-${i}`,
      { ...initial, messages },
    );
    messages.push(message);
    // Finish any generated calls without executing them. That exercises the
    // tool-result continuation path while keeping historical actions read-only.
    for (const call of message.content)
      if (call.type === 'toolCall')
        messages.push({
          role: 'toolResult',
          toolCallId: call.id,
          toolName: call.name,
          content: [
            {
              type: 'text',
              text: 'This is an offline replay. The recorded tool results above are available; no new command was executed.',
            },
          ],
          isError: true,
          timestamp: 0,
        });
    messages.push({
      role: 'user',
      content: [
        'From the recorded evidence, explain the most important correctness risk and a regression test for it. Do not execute more tools.',
        'Explain the likely performance impact of the changes discussed above, and which measurements would verify it. Use the recorded results only.',
        'Finish with the concrete remaining validation steps and explain what each one proves. Do not execute more tools.',
      ][i % 3]!,
      timestamp: 0,
    });
    const drained = coldCacheDrain(5000);
    drains.push(drained);
    if (!drained) throw new Error('SSD writer did not drain between turns');
    await save();
    if (gateDir) await writeFile(resolve(gateDir, `turn-${i}.done`), 'idle\n');
  }
  await waitForGate('finish');
} finally {
  finalDrain = coldCacheDrain(5000);
  drains.push(finalDrain);
  await save();
  await overlays.cleanup();
}
if (!finalDrain) throw new Error('SSD writer did not drain before exit');
