import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { LoadableModel } from '@mlx-node/lm';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { sidecarHostOptions } from '../../packages/desktop/src/inference/sidecar.js';
import { createHandler, type HandlerOptions, type PublicModelEntry } from '../../packages/server/src/handler.js';
import { createInferenceHost, type InferenceHost } from '../../packages/server/src/host/index.js';
import { reasoningEffortsFor } from '../../packages/server/src/host/thinking-levels.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';

type Json = Record<string, unknown>;

const row = (id: string, extra: Partial<PublicModelEntry>): PublicModelEntry => ({
  id,
  object: 'model',
  created: 1,
  owned_by: 'mlx-node',
  ...extra,
});

const ROWS: PublicModelEntry[] = [
  row('q35', { api_backend: 'messages', model_family: 'qwen3_5', context_window: 262_144 }),
  row('hy4', { api_backend: 'messages', model_family: 'hy_v4_flash' }),
  row('g4', { api_backend: 'messages', model_family: 'gemma4' }),
  row('asr', { modality: 'audio', task: 'asr', model_family: 'qwen3_asr' }),
  row('tts', { modality: 'audio', task: 'tts', model_family: 'qwen3_tts' }),
  row('rerank', { modality: 'text', task: 'rerank', model_family: 'jina_reranker' }),
  row('plain', {}),
];

/** The agent's effort values (`ReasoningEffort::from_str`). */
const AGENT_EFFORTS = new Set(['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max']);

/**
 * The agent's `/v1/models` reader for `reasoningEfforts` (`remote/client.rs`, `parse_reasoning_effort_options`): a
 * bare value string, or a table with a `value` and optional `id`, `label`, `description` strings and a `default` bool.
 * An entry it cannot read is skipped, so a skipped entry here is a level the agent would never offer.
 */
function agentReads(entries: unknown): Array<{ id: string; value: string; label: string; default: boolean }> {
  if (!Array.isArray(entries)) return [];
  const out: Array<{ id: string; value: string; label: string; default: boolean }> = [];
  for (const entry of entries) {
    if (typeof entry === 'string') {
      if (AGENT_EFFORTS.has(entry)) out.push({ id: entry, value: entry, label: entry, default: false });
      continue;
    }
    if (entry === null || typeof entry !== 'object') continue;
    const { value, id, label, description, default: isDefault } = entry as Json;
    if (typeof value !== 'string' || !AGENT_EFFORTS.has(value)) continue;
    if (id !== undefined && typeof id !== 'string') continue;
    if (label !== undefined && typeof label !== 'string') continue;
    if (description !== undefined && typeof description !== 'string') continue;
    if (isDefault !== undefined && typeof isDefault !== 'boolean') continue;
    out.push({ id: id ?? value, value, label: label ?? value, default: isDefault === true });
  }
  return out;
}

const servers: Server[] = [];
const hosts: InferenceHost[] = [];
const dirs: string[] = [];
afterEach(async () => {
  for (const server of servers.splice(0)) await new Promise<void>((done) => server.close(() => done()));
  for (const host of hosts.splice(0)) await host.close({ timeoutMs: 250 }).catch(() => undefined);
  for (const dir of dirs.splice(0)) await rm(dir, { recursive: true, force: true });
});

async function listOf(options: HandlerOptions, listModels?: () => PublicModelEntry[]): Promise<string> {
  const registry = new ModelRegistry();
  registry.register('loaded', {} as never);
  const handler = createHandler(registry, { ...(listModels ? { listModels } : {}), ...options });
  const server = createServer((req, res) => void handler(req, res));
  servers.push(server);
  await new Promise<void>((done) => server.listen(0, '127.0.0.1', () => done()));
  const res = await fetch(`http://127.0.0.1:${(server.address() as AddressInfo).port}/v1/models`);
  expect(res.status).toBe(200);
  return res.text();
}

const dataOf = (body: string): Json[] => (JSON.parse(body) as { data: Json[] }).data;

describe('GET /v1/models: thinking levels', () => {
  it('with levels, chat rows carry reasoningEfforts in the shape the agent reads; audio, rerank and familyless rows do not', async () => {
    const data = dataOf(await listOf({ thinkingLevels: true }, () => ROWS));
    const byId = new Map(data.map((entry) => [entry.id, entry]));
    expect(byId.get('q35')!.reasoningEfforts).toEqual(reasoningEffortsFor('qwen3_5'));
    expect(agentReads(byId.get('q35')!.reasoningEfforts)).toEqual([
      { id: 'none', value: 'none', label: 'Off', default: false },
      { id: 'low', value: 'low', label: 'Low', default: false },
      { id: 'medium', value: 'medium', label: 'Medium', default: false },
      { id: 'high', value: 'high', label: 'High', default: true },
    ]);
    expect(agentReads(byId.get('hy4')!.reasoningEfforts).map((option) => option.value)).toEqual([
      'none',
      'low',
      'high',
      'max',
    ]);
    for (const id of ['g4', 'asr', 'tts', 'rerank', 'plain'])
      expect('reasoningEfforts' in byId.get(id)!, id).toBe(false);
    for (const entry of data) {
      const { reasoningEfforts, ...rest } = entry;
      expect(rest, String(entry.id)).toEqual(ROWS.find((r) => r.id === entry.id));
      if (reasoningEfforts !== undefined) {
        expect(agentReads(reasoningEfforts)).toHaveLength((reasoningEfforts as unknown[]).length);
      }
    }
  });

  it('without levels the body is byte-equal to the rows as listed', async () => {
    const expected = JSON.stringify({
      object: 'list',
      data: ROWS,
      models: ROWS.map((entry) => ({ name: entry.id, description: 'Local MLX model', release_date: '' })),
    });
    expect(await listOf({}, () => ROWS)).toBe(expected);
    expect(await listOf({ thinkingTokenBudget: 12_000, thinkingCycleGuard: true }, () => ROWS)).toBe(expected);
  });

  it('a registry list names no family, so it is the same with levels on', async () => {
    expect(await listOf({ thinkingLevels: true })).toBe(await listOf({}));
  });
});

describe('GET /v1/models: the host option', () => {
  async function modelsDir(types: Record<string, string>): Promise<string> {
    const root = await mkdtemp(join(tmpdir(), 'mlx-models-list-'));
    dirs.push(root);
    for (const [name, modelType] of Object.entries(types)) {
      await mkdir(join(root, name), { recursive: true });
      await writeFile(join(root, name, 'config.json'), JSON.stringify({ model_type: modelType }));
      await writeFile(join(root, name, 'model.safetensors'), 'not-real-weights');
    }
    return root;
  }

  async function hostRows(thinkingLevels: boolean | undefined): Promise<Json[]> {
    const host = await createInferenceHost({
      port: 0,
      modelsDir: await modelsDir({ alpha: 'qwen3_5', beta: 'gemma4' }),
      model: 'alpha',
      disableStore: true,
      sweepOrphanTempRoots: false,
      setResidencyKeepAlive: () => undefined,
      loadModel: async () => ({}) as unknown as LoadableModel,
      estimateContextWindow: async () => null,
      ...(thinkingLevels !== undefined ? { thinkingLevels } : {}),
    });
    hosts.push(host);
    const res = await fetch(`${host.url}/v1/models`);
    return dataOf(await res.text());
  }

  it('the desktop engine lists the levels of its chat models; a host without the option lists none', async () => {
    expect(sidecarHostOptions({}).thinkingLevels).toBe(true);
    const on = await hostRows(sidecarHostOptions({}).thinkingLevels);
    expect(on.find((entry) => entry.id === 'alpha')!.reasoningEfforts).toEqual(reasoningEffortsFor('qwen3_5'));
    expect('reasoningEfforts' in on.find((entry) => entry.id === 'beta')!).toBe(false);
    for (const entry of await hostRows(undefined)) expect('reasoningEfforts' in entry, String(entry.id)).toBe(false);
  });

  // Agent 0.3.5 sends `extra_body.reasoning_effort` only to a row that has both; any other endpoint gets 0.3.4's bytes.
  it("every host row says owned_by mlx-node, with or without the option: the agent's test for extra_body", async () => {
    for (const thinkingLevels of [true, undefined]) {
      const rows = await hostRows(thinkingLevels);
      expect(rows.map((entry) => String(entry.id)).sort((a, b) => a.localeCompare(b))).toEqual(['alpha', 'beta']);
      for (const entry of rows) expect(entry.owned_by, `${String(entry.id)} ${thinkingLevels}`).toBe('mlx-node');
    }
  });
});
