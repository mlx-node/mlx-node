/**
 * The PDF tool in Work on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no model. The host
 * writes the saved workflow and its reader agent at the agent start; a 12-page PDF is staged and placed as the
 * service does on Send. The model starts `read-pdf` by name from the note (Ask: one card naming the PDF; Auto and
 * Bypass: no card), and the host starts it with the agent's own `/workflow` command (no model call, no permission
 * request). Readers have `read_file` only, open their 4 page files as images, and the report names the pages read.
 * Skipped without the variable. `MLX_AGENT_E2E_RECORD` (a folder) keeps each case's raw wire and engine requests.
 */

import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { spawnAgent, type SpawnAgent } from '../src/chat/acp/process.js';
import { readPdfCommand } from '../src/chat/agent/pdf-tool.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import type { AttachmentInput } from '../src/chat/attachment-rule.js';
import { PdfStore, readPdfMeta } from '../src/chat/pdf-store.js';
import type { PermissionInteraction, PermissionMode } from '../src/chat/types.js';
import { fakeHost, noProviders, until, type FakeHost, type Json } from './helpers/chat-fake.js';
import {
  hasToolResult,
  lastMessage,
  messageText,
  startScriptedEngine,
  type ModelReply,
  type ModelRequest,
  type ScriptedEngine,
} from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const RECORD = process.env.MLX_AGENT_E2E_RECORD;
const MODEL = 'qwen3.5-4b';
const NAME = 'report.pdf';
const PAGES = 12;
const QUESTION = 'PDF-PROBE: what are the findings of this report?';
const SHORT_LINE = 'I am reading the PDF now.';
const ANSWER = 'The report has one finding (p. 1).';
const COMBINED = 'COMBINED: the findings are on page 1 (p. 1).';
const AGENT_WAIT_MS = 60_000;

/** A 32x32 grey baseline JPEG whose comment segment holds `label`; the agent drops images under 512 pixels. */
function jpeg(label: string): Buffer {
  const segment = (marker: number, body: Buffer): Buffer => {
    const length = Buffer.alloc(2);
    length.writeUInt16BE(body.length + 2);
    return Buffer.concat([Buffer.from([0xff, marker]), length, body]);
  };
  const huffman = (tableClass: number): Buffer =>
    Buffer.from([tableClass << 4, 1, ...Array.from({ length: 15 }, () => 0), 0]);
  // Sixteen 8x8 blocks, each a 1-bit "no DC change" and a 1-bit end of block.
  const scan = Buffer.alloc(4);
  return Buffer.concat([
    Buffer.from([0xff, 0xd8]),
    segment(0xfe, Buffer.from(label, 'latin1')),
    segment(0xdb, Buffer.concat([Buffer.from([0]), Buffer.alloc(64, 1)])),
    segment(0xc0, Buffer.from([8, 0, 32, 0, 32, 1, 1, 0x11, 0])),
    segment(0xc4, huffman(0)),
    segment(0xc4, huffman(1)),
    segment(0xda, Buffer.from([1, 1, 0x00, 0, 63, 0])),
    scan,
    Buffer.from([0xff, 0xd9]),
  ]);
}

/** The label of a page image the engine got: the text of its comment segment. */
function labelOf(data: string): string | null {
  const bytes = Buffer.from(data, 'base64');
  if (bytes[0] !== 0xff || bytes[1] !== 0xd8 || bytes[2] !== 0xff || bytes[3] !== 0xfe) return null;
  return bytes.subarray(6, 4 + bytes.readUInt16BE(4)).toString('latin1');
}

const pageLabel = (page: number): string => `page-${String(page).padStart(3, '0')}`;

/** The labels of every image the request carries, in order, wherever the agent put it. */
function imageLabels(q: ModelRequest): string[] {
  const out: string[] = [];
  const walk = (value: unknown): void => {
    if (Array.isArray(value)) {
      for (const item of value) walk(item);
      return;
    }
    if (typeof value !== 'object' || value === null) return;
    const block = value as Json;
    const source = block.source as Json | undefined;
    if (block.type === 'image' && typeof source?.data === 'string') {
      out.push(labelOf(source.data) ?? '?');
      return;
    }
    for (const inner of Object.values(block)) walk(inner);
  };
  walk(q.messages);
  return out;
}

interface Wire {
  out: Json[];
}

function teeSpawn(wire: Wire): SpawnAgent {
  return (command) => {
    const child = spawnAgent(command);
    let partial = '';
    child.stdout?.on('data', (chunk: Buffer) => {
      partial += chunk.toString();
      let nl = partial.indexOf('\n');
      while (nl !== -1) {
        const line = partial.slice(0, nl);
        partial = partial.slice(nl + 1);
        if (line.trim() !== '') {
          try {
            wire.out.push(JSON.parse(line) as Json);
          } catch {
            wire.out.push({ unparsed: line });
          }
        }
        nl = partial.indexOf('\n');
      }
    });
    return child;
  };
}

const isTitle = (q: ModelRequest): boolean => q.tools.length === 1 && q.tools[0] === 'session_title';
const isParent = (q: ModelRequest): boolean => q.tools.includes('workflow');
const isTextAgent = (q: ModelRequest): boolean => messageText(q.messages[0]).includes('Call no tool.');
const isReader = (q: ModelRequest): boolean => !isTitle(q) && !isParent(q) && !isTextAgent(q);

function engineReply(launchArgs: () => Json): (q: ModelRequest) => ModelReply {
  return (q) => {
    if (isTitle(q)) {
      return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'PDF probe' } }] };
    }
    const last = lastMessage(q);
    if (isParent(q)) {
      if (hasToolResult(last)) return { blocks: [{ type: 'text', text: SHORT_LINE }] };
      if (messageText(last).includes(QUESTION)) {
        const input = { source: { type: 'name', name: 'read-pdf' }, args: launchArgs() };
        return { blocks: [{ type: 'tool_use', name: 'workflow', input }] };
      }
      return { blocks: [{ type: 'text', text: ANSWER }] };
    }
    if (isTextAgent(q)) return { blocks: [{ type: 'text', text: COMBINED }] };
    if (!hasToolResult(last)) {
      const files = messageText(q.messages[0])
        .split('\n')
        .filter((line) => line.endsWith('.jpg'));
      return {
        blocks: files.map((file) => ({ type: 'tool_use', name: 'read_file', input: { target_file: file } })),
      };
    }
    const notes = imageLabels(q).map(
      (label) => `Page ${Number(label.slice(5))}: ${label} seen (p. ${Number(label.slice(5))}).`,
    );
    return { blocks: [{ type: 'text', text: notes.join('\n') }] };
  };
}

interface Rig {
  fake: FakeHost;
  engine: ScriptedEngine;
  wire: Wire;
  mode: { current: PermissionMode };
  allowed: Json[];
  outer: string;
}

async function rig(mode: PermissionMode, launchArgs: () => Json): Promise<Rig> {
  const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-pdf-')));
  const agentTmp = join(outer, 'tmp');
  mkdirSync(agentTmp);
  const wire: Wire = { out: [] };
  const state = { current: mode };
  const allowed: Json[] = [];
  let engine: ScriptedEngine | null = null;
  const paths: { workPdf: string | null } = { workPdf: null };
  const fake = await fakeHost('work-workflow-run', {
    mode: 'work',
    providers: noProviders(),
    launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
    startEngine: async () => {
      engine = await startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], engineReply(launchArgs));
      return engine;
    },
    env: { TMPDIR: `${agentTmp}/` },
    spawn: teeSpawn(wire),
    permission: { modeOf: () => state.current, watched: () => true },
    onModeAllowed: async (_id, mark) => {
      allowed.push(mark as unknown as Json);
      return true;
    },
    pdfFacts: async (_sessionId, pdf) => {
      const meta = paths.workPdf === null ? null : await readPdfMeta(paths.workPdf, pdf);
      return meta === null
        ? null
        : { id: pdf, session: meta.session, name: meta.name, pages: meta.pages, images: true, tool: true };
    },
  });
  paths.workPdf = fake.paths.workPdf;
  return { fake, engine: engine!, wire, mode: state, allowed, outer };
}

/** Stages the 12 pages as the page sends them and places them for `session` as Send does: the attachment. */
async function placedPdf(fake: FakeHost, session: string): Promise<{ id: string; attachment: AttachmentInput }> {
  const store = new PdfStore(fake.paths.pdfStage);
  let stage: string | null = null;
  for (let page = 1; page <= PAGES; page += 1) {
    stage = await store.stagePage(stage, page, PAGES, jpeg(pageLabel(page)));
    expect(stage).not.toBeNull();
  }
  const id = await store.place(stage!, PAGES, fake.paths.workPdf, session, NAME);
  expect(id).not.toBeNull();
  return { id: id!, attachment: { kind: 'pdf', name: NAME, stage: stage!, pages: PAGES } };
}

function xaiUpdates(wire: Wire, kind: string): Json[] {
  return wire.out
    .filter((m) => m.method === '_x.ai/session_notification')
    .map((m) => (m.params as Json).update as Json)
    .filter((u) => u.sessionUpdate === kind);
}

function permissionRequests(wire: Wire): Json[] {
  return wire.out.filter((m) => m.method === 'session/request_permission');
}

function entriesOf(fake: FakeHost, id: string): TranscriptEntry[] {
  const reducer = new TranscriptReducer();
  for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
  return reducer.snapshot().entries;
}

/** Waits for the run's end and the wake turn's answer; the run's last update. */
async function runEnd(r: Rig, id: string): Promise<Json> {
  const ended = (): Json | undefined =>
    xaiUpdates(r.wire, 'workflow_updated').find((u) => u.status !== 'active' && u.status !== 'pending');
  await until(() => ended() !== undefined, AGENT_WAIT_MS);
  await until(() => entriesOf(r.fake, id).some((e) => e.kind === 'assistant' && e.text === ANSWER), AGENT_WAIT_MS);
  return xaiUpdates(r.wire, 'workflow_updated').at(-1)!;
}

/** The pages each reader got as images, by the order the readers ran; its reads end in any order. */
function readerPages(engine: ScriptedEngine): string[][] {
  return engine.modelRequests
    .filter((q) => isReader(q) && hasToolResult(lastMessage(q)))
    .map((q) => imageLabels(q).toSorted());
}

function report(fake: FakeHost, id: string): string {
  const run = entriesOf(fake, id).find((e) => e.kind === 'workflow');
  return run?.kind === 'workflow' ? (run.report ?? '') : '';
}

async function finish(name: string, r: Rig, extra: Json = {}): Promise<void> {
  if (RECORD !== undefined) {
    mkdirSync(RECORD, { recursive: true });
    const record = {
      model: r.engine.modelRequests.map((q) => ({
        index: q.index,
        tools: q.tools,
        first: messageText(q.messages[0]).slice(0, 2_000),
        shape: JSON.stringify(q.messages).slice(0, 6_000),
        images: imageLabels(q),
        at: q.at,
      })),
      wire: r.wire.out,
      interactions: r.fake.events.interactions,
      allowed: r.allowed,
      ...extra,
    };
    writeFileSync(join(RECORD, `${name}.json`), `${JSON.stringify(record, null, 2)}\n`);
  }
  await r.fake.close();
  rmSync(r.outer, { recursive: true, force: true });
}

async function session(fake: FakeHost): Promise<string> {
  return (await fake.host.createSession({ modelId: MODEL, cwd: fake.project })).sessionId;
}

describe.skipIf(BIN === undefined)('The PDF tool in Work on the real agent', () => {
  // Mutation: the reader file without its tool limits (search_tool and use_tool come back) → red.
  it('Ask: the model starts read-pdf from the note, one card names the PDF, readers read pages 1-8 and 12', async () => {
    const held: { id: string } = { id: '' };
    const r = await rig('ask', () => ({ pdf: held.id, question: QUESTION, pages: '1-8, 12' }));
    try {
      const { fake, engine, wire } = r;
      const id = await session(fake);
      const placed = await placedPdf(fake, id);
      held.id = placed.id;
      const turn = fake.host.startTurn(id, QUESTION, { modelId: MODEL, cwd: fake.project }, [placed.attachment]);
      await until(() => fake.events.interactions.length > 0, AGENT_WAIT_MS);
      const card = fake.events.interactions[0]!.interaction as PermissionInteraction;
      expect([card.toolName, card.pdfRead]).toEqual(['workflow', { name: NAME, pages: 9, readers: 3 }]);
      const parent = engine.modelRequests.find(isParent)!;
      expect(messageText(lastMessage(parent))).toContain(`id: ${placed.id}`);
      expect(parent.system).toContain('Call the workflow tool with name read-pdf');
      fake.host.answer(id, card.id, 'allow-once');
      expect((await turn.done).outcome).toBe('completed');
      const end = await runEnd(r, id);
      expect(end.status).toBe('complete');
      expect(fake.events.interactions).toHaveLength(1);
      expect(permissionRequests(wire)).toHaveLength(1);
      for (const q of engine.modelRequests.filter((x) => !isTitle(x) && !isParent(x))) {
        expect(q.tools).toEqual(['read_file']);
      }
      expect(readerPages(engine)).toEqual([[1, 2, 3, 4].map(pageLabel), [5, 6, 7, 8].map(pageLabel), [pageLabel(12)]]);
      expect(report(fake, id).startsWith(`PDF read: 9 of 9 pages.\n\n${COMBINED}`)).toBe(true);
    } finally {
      await finish('launch-ask', r);
    }
  }, 180_000);

  it("Auto: the launch on this conversation's pages runs with no card, and no reader read asks", async () => {
    const held: { id: string } = { id: '' };
    const r = await rig('auto', () => ({ pdf: held.id, question: QUESTION }));
    try {
      const { fake, engine, wire } = r;
      const id = await session(fake);
      const placed = await placedPdf(fake, id);
      held.id = placed.id;
      const turn = fake.host.startTurn(id, QUESTION, { modelId: MODEL, cwd: fake.project }, [placed.attachment]);
      expect((await turn.done).outcome).toBe('completed');
      const end = await runEnd(r, id);
      expect(end.status).toBe('complete');
      expect(fake.events.interactions).toEqual([]);
      expect(permissionRequests(wire)).toHaveLength(1);
      expect(r.allowed.map((m) => (m.card as Json | undefined)?.pdfRead)).toEqual([
        { name: NAME, pages: PAGES, readers: 3 },
      ]);
      expect(readerPages(engine)).toEqual([
        [1, 2, 3, 4].map(pageLabel),
        [5, 6, 7, 8].map(pageLabel),
        [9, 10, 11, 12].map(pageLabel),
      ]);
      expect(report(fake, id).startsWith(`PDF read: ${PAGES} of ${PAGES} pages.`)).toBe(true);
    } finally {
      await finish('launch-auto', r);
    }
  }, 180_000);

  it("Bypass: the model's workflow call runs with no card", async () => {
    const held: { id: string } = { id: '' };
    const r = await rig('bypass', () => ({ pdf: held.id, question: QUESTION }));
    try {
      const { fake, engine, wire } = r;
      const id = await session(fake);
      const placed = await placedPdf(fake, id);
      held.id = placed.id;
      const turn = fake.host.startTurn(id, QUESTION, { modelId: MODEL, cwd: fake.project }, [placed.attachment]);
      expect((await turn.done).outcome).toBe('completed');
      const end = await runEnd(r, id);
      expect(end.status).toBe('complete');
      expect(fake.events.interactions).toEqual([]);
      expect(permissionRequests(wire)).toHaveLength(1);
      expect(readerPages(engine)).toHaveLength(3);
      expect(report(fake, id).startsWith(`PDF read: ${PAGES} of ${PAGES} pages.`)).toBe(true);
    } finally {
      await finish('launch-bypass', r);
    }
  }, 180_000);

  it('the host start: /workflow read-pdf runs with no model call before it and no permission request', async () => {
    const r = await rig('ask', () => ({}));
    try {
      const { fake, engine, wire } = r;
      const id = await session(fake);
      const placed = await placedPdf(fake, id);
      const turn = fake.host.startTurn(id, readPdfCommand(placed.id, QUESTION, NAME), {
        modelId: MODEL,
        cwd: fake.project,
      });
      expect((await turn.done).outcome).toBe('completed');
      const end = await runEnd(r, id);
      expect([end.status, end.objective]).toEqual(['complete', `Read ${NAME}`]);
      expect(fake.events.interactions).toEqual([]);
      expect(permissionRequests(wire)).toEqual([]);
      const parents = engine.modelRequests.filter(isParent);
      const firstReader = engine.modelRequests.find(isReader)!;
      expect(parents.every((q) => q.index > firstReader.index)).toBe(true);
      expect(readerPages(engine)).toHaveLength(3);
      expect(report(fake, id).startsWith(`PDF read: ${PAGES} of ${PAGES} pages.`)).toBe(true);
    } finally {
      await finish('host-start', r);
    }
  }, 180_000);
});
