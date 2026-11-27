/**
 * The agent host against the scripted fake agent: lifecycle, turns and their outcomes, both web call shapes, the
 * refusals that end a turn, crash and reopen, idle stop, stale restart, and a Work session.
 */

import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { spawnAgent } from '../src/chat/acp/process.js';
import { AgentHostError, launchFromProbe } from '../src/chat/agent/agent-host.js';
import { LAUNCHD_PATH } from '../src/chat/agent/login-path.js';
import { CHAT_CONFIG_TOML, CHAT_REQUIREMENTS_TOML, promptParams } from '../src/chat/agent/profiles.js';
import { pdfNote, type AttachmentInput } from '../src/chat/attachment-rule.js';
import { REDACTED } from '../src/chat/redact.js';
import { CONTINUE_PROMPT } from '../src/chat/types.js';
import {
  endpointModel,
  fakeHost,
  INFERENCE_TOKEN,
  noProviders,
  until,
  type FakeHost,
  type Json,
} from './helpers/chat-fake.js';
import { gate } from './helpers/fake-engine.js';
import { runView } from './helpers/run-views.js';

const MODEL = 'qwen3.5-4b';
/** The agent's own lines for a Stop of one workflow agent (mlx-agent v0.3.6, recorded). */
const STOP_RECORD = JSON.parse(
  readFileSync(fileURLToPath(new URL('./fixtures/workflow-agent-stop.json', import.meta.url)), 'utf8'),
) as { lines: Array<{ params?: { update?: Json } }> };
const FILE: AttachmentInput = { kind: 'file', name: 'notes.md', text: '# notes\n' };
const PICTURE: AttachmentInput = { kind: 'image', mediaType: 'image/png', data: 'iVBORw0KGgoAAAANSUhEUg==' };
const PDF: AttachmentInput = { kind: 'pdf', name: 'paper.pdf', images: ['/9j/4AAQ', '/9j/4AAR'] };

async function withFake(
  name: string,
  run: (fake: FakeHost) => Promise<void>,
  options: Parameters<typeof fakeHost>[1] = {},
) {
  const fake = await fakeHost(name, options);
  try {
    await run(fake);
  } finally {
    await fake.close();
  }
}

/** Creates a Chat session and runs one turn to its end. */
async function turn(fake: FakeHost, text: string, sessionId?: string) {
  const id = sessionId ?? (await fake.host.createSession({ modelId: MODEL })).sessionId;
  const started = fake.host.startTurn(id, text, { modelId: MODEL });
  const result = await started.done;
  return { id, turnId: started.turnId, result };
}

function permissionAnswers(fake: FakeHost): unknown[] {
  return fake
    .sent()
    .filter(
      (m) =>
        m.method === undefined &&
        m.result !== undefined &&
        typeof m.result === 'object' &&
        'outcome' in (m.result as Json),
    )
    .map((m) => (m.result as Json).outcome);
}

/** The `MLX_WEB_LEDGER` path of every `session/new` and `session/load` the fake agent received, in order. */
function ledgerPaths(fake: FakeHost): string[] {
  return fake
    .sent()
    .filter((m) => m.method === 'session/new' || m.method === 'session/load')
    .map((m) => {
      const servers = (m.params as Json).mcpServers as Array<{ env: Array<{ name: string; value: string }> }>;
      return servers[0]!.env.find((e) => e.name === 'MLX_WEB_LEDGER')!.value;
    });
}

function ledgerFile(path: string): unknown {
  return JSON.parse(readFileSync(path, 'utf8'));
}

function texts(fake: FakeHost): string {
  return fake.events.updates
    .map((u) => {
      const content = u.line.update.content as Json | undefined;
      return typeof content?.text === 'string' ? content.text : '';
    })
    .join('');
}

describe('promptParams', () => {
  const note = (name: string, pages: number, from: number) => ({
    type: 'resource',
    resource: {
      uri: `file:///attached/${name}`,
      mimeType: 'text/plain',
      text: pdfNote({ name, pages, way: 'message', from }),
    },
  });
  const page = (data: string) => ({ type: 'image', data, mimeType: 'image/jpeg' });

  // Mutation: a note's images range that leaves out the pictures before the pages → red.
  it('sends the text, the files, a note per PDF, the pictures, then each PDF page in order', () => {
    const second: AttachmentInput = { kind: 'pdf', name: 'b.pdf', images: ['/9j/4AAS'] };
    expect(promptParams('s1', 'read these', [PDF, PICTURE, second, FILE])).toEqual({
      sessionId: 's1',
      prompt: [
        { type: 'text', text: 'read these' },
        {
          type: 'resource',
          resource: { uri: 'file:///attached/notes.md', mimeType: 'text/plain', text: '# notes\n' },
        },
        note('paper.pdf', 2, 2),
        note('b.pdf', 1, 4),
        { type: 'image', data: 'iVBORw0KGgoAAAANSUhEUg==', mimeType: 'image/png' },
        page('/9j/4AAQ'),
        page('/9j/4AAR'),
        page('/9j/4AAS'),
      ],
    });
    const prompt = promptParams('s1', 'read', [PICTURE, PDF]).prompt as Array<{ resource?: { text: string } }>;
    expect(prompt[1]!.resource!.text.split('\n').slice(4)).toEqual([
      'images: 2-3',
      'Images 2-3 of this message are pages 1-2 of this PDF, in order.',
    ]);
  });
});

describe('launchFromProbe', () => {
  it('maps the probe to a launch or a typed failure', () => {
    expect(launchFromProbe({ source: 'none' })).toEqual({ failure: 'agent-missing' });
    expect(launchFromProbe({ source: 'none', managedTag: 'mlx-agent-v0.2.0' })).toEqual({ failure: 'agent-too-old' });
    expect(
      launchFromProbe({
        source: 'bundled',
        path: '/b',
        bundledTag: 'mlx-agent-v0.2.0',
        effectiveTag: 'mlx-agent-v0.2.0',
      }),
    ).toEqual({
      failure: 'agent-too-old',
    });
    // Mutation: the minimum left at v0.3.5 → Work starts an agent that has no `GROK_CLAUDE_PLUGINS_ENABLED`.
    expect(launchFromProbe({ source: 'managed', path: '/m', effectiveTag: 'mlx-agent-v0.3.5' })).toEqual({
      failure: 'agent-too-old',
    });
    // Mutation: the minimum left at v0.3.6 → Work starts an agent that reads `.grok` project folders and sends no
    // `tool_failure_count`.
    expect(launchFromProbe({ source: 'managed', path: '/m', effectiveTag: 'mlx-agent-v0.3.6' })).toEqual({
      failure: 'agent-too-old',
    });
    expect(launchFromProbe({ source: 'managed', path: '/m', effectiveTag: 'mlx-agent-v0.3.7' })).toEqual({
      command: '/m',
      args: ['--disable-web-search', 'agent', '--no-leader', 'stdio'],
      source: 'managed',
      tag: 'mlx-agent-v0.3.7',
      versionVerified: true,
    });
    expect(launchFromProbe({ source: 'env', path: '/e' })).toMatchObject({
      command: '/e',
      versionVerified: false,
      tag: null,
    });
  });
});

describe('Chat host', () => {
  it('starts lazily with the Chat env and initialize params, and streams a turn', async () => {
    await withFake('chat-basic', async (fake) => {
      expect(fake.records()).toEqual([]);
      const { id, result } = await turn(fake, 'hello there');
      expect(result.outcome).toBe('completed');
      const start = fake.records().find((r) => r.kind === 'start')!;
      // macOS adds __CF_USER_TEXT_ENCODING to every process it starts.
      expect((start.envNames as string[]).filter((n) => n !== '__CF_USER_TEXT_ENCODING')).toEqual(
        [
          'DISABLE_TELEMETRY',
          'GROK_CLAUDE_HOOKS_ENABLED',
          'GROK_CLAUDE_MCPS_ENABLED',
          'GROK_CLAUDE_PLUGINS_ENABLED',
          'GROK_CLAUDE_SKILLS_ENABLED',
          'GROK_CLI_CHAT_PROXY_BASE_URL',
          'GROK_CODEX_HOOKS_ENABLED',
          'GROK_CODEX_MCPS_ENABLED',
          'GROK_COMPACTION_WALL_CLOCK_SECS',
          'GROK_CONFIG',
          'GROK_CURSOR_HOOKS_ENABLED',
          'GROK_CURSOR_MCPS_ENABLED',
          'GROK_CURSOR_SKILLS_ENABLED',
          'GROK_HOME',
          'GROK_IMAGE_EDIT',
          'GROK_IMAGE_GEN',
          'GROK_LOOPBACK_CONNECT_MAX_ATTEMPTS',
          'GROK_MODELS_BASE_URL',
          'GROK_SESSION_SEARCH',
          'GROK_TELEMETRY_ENABLED',
          'GROK_TELEMETRY_TRACE_UPLOAD',
          'GROK_TURN_SUMMARY',
          'GROK_TWO_PASS_COMPACTION',
          'GROK_VIDEO_GEN',
          'GROK_WORKFLOWS',
          'GROK_XAI_API_BASE_URL',
          'HOME',
          'LANG',
          'LOGNAME',
          'PATH',
          'SHELL',
          'TERM',
          'TMPDIR',
          'USER',
          'XAI_API_KEY',
        ].sort(),
      );
      expect(realpathSync(start.cwd as string)).toBe(realpathSync(fake.paths.chatWorkspace));
      expect(JSON.stringify(start.argv)).not.toContain(INFERENCE_TOKEN);
      const sent = fake.sent();
      expect(sent[0]).toMatchObject({ method: 'initialize', params: { _meta: { clientType: 'generic' } } });
      expect(sent.some((m) => m.method === 'authenticate')).toBe(false);
      const prompt = sent.find((m) => m.method === 'session/prompt')!;
      expect(prompt.params).toEqual({ sessionId: id, prompt: [{ type: 'text', text: 'hello there' }] });
      expect(texts(fake)).toContain('Hello! How can I help?');
      expect(fake.events.titles).toEqual([{ sessionId: id, title: 'hello there' }]);
      expect(fake.events.updates.some((u) => u.line.update.sessionUpdate === 'model_changed')).toBe(false);
      expect(fake.events.turns.map((t) => t.turn?.phase ?? null)).toEqual(['starting', 'running', null]);
    });
  });

  // Mutation: the login PATH awaited or applied in both modes → the Chat start hangs on the pending one, or its sealed
  // PATH changes.
  it('never waits for the login PATH and keeps its sealed PATH', async () => {
    const envs: Array<Record<string, string>> = [];
    await withFake(
      'chat-basic',
      async (fake) => {
        expect((await turn(fake, 'hello there')).result.outcome).toBe('completed');
        expect(envs.map((env) => env.PATH)).toEqual([LAUNCHD_PATH]);
      },
      {
        loginPath: new Promise<string | null>(() => {}),
        spawn: (command) => {
          envs.push(command.env);
          return spawnAgent(command);
        },
      },
    );
  });

  it('writes the deny-rule files and removes the search index and prompt history before the spawn', async () => {
    await withFake('chat-basic', async (fake) => {
      const sessions = fake.paths.chatSessions;
      mkdirSync(join(sessions, 'some-cwd'), { recursive: true });
      for (const f of ['session_search.sqlite', 'session_search.sqlite-wal', 'session_search.sqlite-shm']) {
        writeFileSync(join(sessions, f), 'x');
      }
      writeFileSync(join(sessions, 'some-cwd', 'prompt_history.jsonl'), 'typed prompt\n');
      await fake.host.start();
      expect(readFileSync(fake.paths.chatConfig, 'utf8')).toBe(CHAT_CONFIG_TOML);
      expect(readFileSync(fake.paths.chatRequirements, 'utf8')).toBe(CHAT_REQUIREMENTS_TOML);
      for (const f of ['session_search.sqlite', 'session_search.sqlite-wal', 'session_search.sqlite-shm']) {
        expect(existsSync(join(sessions, f))).toBe(false);
      }
      expect(existsSync(join(sessions, 'some-cwd', 'prompt_history.jsonl'))).toBe(false);
      expect(existsSync(join(sessions, 'some-cwd'))).toBe(true);
    });
  });

  it('call shape A (use_tool) is approved and its result arrives as web data', async () => {
    await withFake('chat-web-search', async (fake) => {
      const { result } = await turn(fake, 'newest rust?');
      expect(result.outcome).toBe('completed');
      expect(permissionAnswers(fake)).toEqual([{ outcome: 'selected', optionId: 'allow-once' }]);
      expect(fake.events.web.map((w) => w.result.kind)).toEqual(['search']);
      expect(texts(fake)).toContain('Rust 1.99 is the newest stable release');
    });
  });

  it('call shape B (direct) is approved, and a fetch of a search result address goes on', async () => {
    await withFake('chat-web-search-direct', async (fake) => {
      const { id, result } = await turn(fake, 'newest rust?');
      expect(result.outcome).toBe('completed');
      expect(permissionAnswers(fake)).toEqual([
        { outcome: 'selected', optionId: 'allow-once' },
        { outcome: 'selected', optionId: 'allow-once' },
      ]);
      expect(fake.events.web.map((w) => w.result.kind)).toEqual(['search', 'page']);
      expect(fake.host.ledger(id)?.has('https://releases.rs')).toBe(true);
    });
  });

  it('chat-wrong-tool-name: the wrong guess is approved and the turn goes on to an answer', async () => {
    await withFake('chat-wrong-tool-name', async (fake) => {
      const { result } = await turn(fake, 'search please');
      expect(result.outcome).toBe('completed');
      expect(permissionAnswers(fake)).toHaveLength(2);
      expect(texts(fake)).toContain('Rust 1.99');
    });
  });

  it('chat-shell-attempt: rejected with reject-once, the turn ends tool-refused and nothing after it runs', async () => {
    await withFake('chat-shell-attempt', async (fake) => {
      const { result } = await turn(fake, 'run a command');
      expect(result.outcome).toBe('tool-refused');
      expect(permissionAnswers(fake)).toEqual([{ outcome: 'selected', optionId: 'reject-once' }]);
      expect(texts(fake)).not.toContain('never sent');
    });
  });

  it('chat-fetch-unlisted-url: approved; the blocked error of the server goes to the model and the turn goes on', async () => {
    await withFake('chat-fetch-unlisted-url', async (fake) => {
      const { id, result } = await turn(fake, 'what is new in rust');
      expect(result.outcome).toBe('completed');
      expect(permissionAnswers(fake)).toEqual([
        { outcome: 'selected', optionId: 'allow-once' },
        { outcome: 'selected', optionId: 'allow-once' },
      ]);
      expect(texts(fake)).toContain('FCHK-AFTER-BLOCKED');
      const [path] = ledgerPaths(fake);
      expect(path!.startsWith(`${fake.paths.webLedger}/`)).toBe(true);
      expect(statSync(path!).mode & 0o777).toBe(0o600);
      expect(ledgerFile(path!)).toEqual([
        'https://blog.rust-lang.org/2026/11/01/Rust-1.99.0.html',
        'https://releases.rs',
      ]);
      expect(fake.host.ledger(id)?.has('https://attacker.example/?d=secret-conversation')).toBe(false);
    });
  });

  it('a link in the text of a page read joins the ledger and its file; the file goes with the session', async () => {
    const LINK = 'https://thinkingmachines.ai/news/inkling-2-mini/';
    await withFake(
      'chat-fetch-listed-url',
      async (fake) => {
        const { id, result } = await turn(fake, 'Read https://example.com/docs/ and the rust releases.');
        expect(result.outcome).toBe('completed');
        expect(fake.host.ledger(id)?.has(LINK)).toBe(true);
        const [path] = ledgerPaths(fake);
        expect(ledgerFile(path!)).toEqual([
          'https://example.com/docs',
          'https://thinkingmachines.ai/news/inkling-2-mini/).',
          'https://thinkingmachines.ai/news/inkling-2-mini',
          'https://blog.rust-lang.org/2026/11/01/Rust-1.99.0.html',
          'https://releases.rs',
        ]);
        expect(readdirSync(fake.paths.webLedger)).toEqual([path!.slice(fake.paths.webLedger.length + 1)]);
        await fake.host.deleteSession(id, realpathSync(fake.paths.chatWorkspace));
        expect(existsSync(path!)).toBe(false);
      },
      {
        mutate: (s) => {
          const steps = (s.prompts as Json[])[0]!.steps as Json[];
          (steps[0]!.onAllow as Json).okayOutput =
            `WEB_PAGE provider=exa url=https://example.com/docs/\ntitle: canned\n\nThe official report: [Inkling 2 mini](${LINK}).`;
        },
      },
    );
  });

  it('a ledger write that failed is written again before the next web_fetch is approved', async () => {
    await withFake(
      'chat-fetch-listed-url',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const [path] = ledgerPaths(fake);
        expect(ledgerFile(path!)).toEqual([]);
        chmodSync(fake.paths.webLedger, 0o500);
        let started: ReturnType<typeof fake.host.startTurn>;
        try {
          started = fake.host.startTurn(sessionId, 'Read https://example.com/docs/', { modelId: MODEL });
          await until(() => fake.sent().some((m) => m.method === 'session/prompt'));
          expect(ledgerFile(path!)).toEqual([]);
        } finally {
          chmodSync(fake.paths.webLedger, 0o700);
        }
        expect((await started.done).outcome).toBe('completed');
        expect(permissionAnswers(fake)).toEqual([{ outcome: 'selected', optionId: 'allow-once' }]);
        expect(ledgerFile(path!)).toEqual(['https://example.com/docs']);
      },
      {
        mutate: (s) => {
          const prompt = (s.prompts as Json[])[0]!;
          const steps = prompt.steps as Json[];
          (steps[0]!.onAllow as Json).okayOutput = 'not a web result';
          prompt.steps = [{ sleepMs: 200 }, steps[0]!, ...steps.slice(3)];
        },
      },
    );
  });

  it('chat-web-bad-input: an input text no reader can make an object of is approved; the tool error goes to the model', async () => {
    await withFake('chat-web-bad-input', async (fake) => {
      const { id, result } = await turn(fake, 'cut search');
      expect(result.outcome).toBe('completed');
      expect((await turn(fake, 'direct cut', id)).result.outcome).toBe('completed');
      expect((await turn(fake, 'cut read', id)).result.outcome).toBe('completed');
      expect((await turn(fake, 'python read', id)).result.outcome).toBe('completed');
      expect((await turn(fake, 'quoted search', id)).result.outcome).toBe('completed');
      expect(permissionAnswers(fake)).toEqual(Array(5).fill({ outcome: 'selected', optionId: 'allow-once' }));
      for (const after of [
        'FCHK-AFTER-CUT-SEARCH',
        'FCHK-AFTER-DIRECT-CUT',
        'FCHK-AFTER-CUT-READ',
        'FCHK-AFTER-PYTHON-READ',
        'FCHK-AFTER-QUOTED-SEARCH',
      ])
        expect(texts(fake)).toContain(after);
    });
  });

  it('chat-web-bad-input: an input text that could still become an object the app cannot read ends the turn arguments-unreadable', async () => {
    await withFake('chat-web-bad-input', async (fake) => {
      const { id, result } = await turn(fake, 'duplicate url');
      expect(result.outcome).toBe('arguments-unreadable');
      expect((await turn(fake, 'smuggled tool', id)).result.outcome).toBe('arguments-unreadable');
      expect(permissionAnswers(fake)).toEqual(Array(2).fill({ outcome: 'selected', optionId: 'reject-once' }));
      expect(texts(fake)).not.toContain('never sent');
    });
  });

  it('chat-web-bad-input: a clean input text is approved whatever its address; the server checks it', async () => {
    await withFake('chat-web-bad-input', async (fake) => {
      const { id, result } = await turn(fake, 'clean read stranger');
      expect(result.outcome).toBe('completed');
      const listed = await turn(fake, 'clean read listed https://example.com/docs/', id);
      expect(listed.result.outcome).toBe('completed');
      expect(permissionAnswers(fake)).toEqual(Array(2).fill({ outcome: 'selected', optionId: 'allow-once' }));
      expect(texts(fake)).toContain('FCHK-AFTER-STRANGER');
      expect(texts(fake)).toContain('FCHK-AFTER-LISTED');
      expect(ledgerFile(ledgerPaths(fake)[0]!)).toEqual(['https://example.com/docs']);
    });
  });

  it('chat-fetch-listed-url: an address from the user message and one from a search result are fetched', async () => {
    await withFake('chat-fetch-listed-url', async (fake) => {
      const { result } = await turn(fake, 'Read https://example.com/docs/ and the rust releases.');
      expect(result.outcome).toBe('completed');
      expect(permissionAnswers(fake)).toEqual(Array(3).fill({ outcome: 'selected', optionId: 'allow-once' }));
    });
  });

  it('chat-use-tool-file-input: the tool_input_file and file forms and an extra key end the turn tool-refused', async () => {
    await withFake('chat-use-tool-file-input', async (fake) => {
      const { id, result } = await turn(fake, 'file form');
      expect(result.outcome).toBe('tool-refused');
      expect((await turn(fake, 'second form', id)).result.outcome).toBe('tool-refused');
      expect((await turn(fake, 'extra key', id)).result.outcome).toBe('tool-refused');
      expect(permissionAnswers(fake)).toEqual(Array(3).fill({ outcome: 'selected', optionId: 'reject-once' }));
    });
  });

  it('chat-enable-always-approve: the offered option is never chosen; the turn ends tool-refused', async () => {
    await withFake('chat-enable-always-approve', async (fake) => {
      const { result } = await turn(fake, 'search');
      expect(result.outcome).toBe('tool-refused');
      expect(permissionAnswers(fake)).toEqual([{ outcome: 'selected', optionId: 'reject-once' }]);
    });
  });

  it('chat-tool-loop: the ninth request is cancelled and the turn ends tool-loop', async () => {
    await withFake('chat-tool-loop', async (fake) => {
      const { id, result } = await turn(fake, 'loop');
      expect(result.outcome).toBe('tool-loop');
      const answers = permissionAnswers(fake);
      expect(answers).toHaveLength(9);
      expect(answers[8]).toEqual({ outcome: 'cancelled' });
      await until(() => fake.sent().some((m) => m.method === 'session/cancel' && (m.params as Json).sessionId === id));
    });
  });

  it('cancel: the user stop ends the turn as cancelled', async () => {
    await withFake('cancel', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const started = fake.host.startTurn(sessionId, 'think long', { modelId: MODEL });
      await until(() => texts(fake).includes('Thinking'));
      expect(fake.host.cancel(sessionId)).toBe(true);
      expect((await started.done).outcome).toBe('cancelled');
      expect(fake.host.cancel(sessionId)).toBe(false);
    });
  });

  it('a second prompt while a turn runs is refused as conversation-busy', async () => {
    await withFake('cancel', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const started = fake.host.startTurn(sessionId, 'one', { modelId: MODEL });
      expect(() => fake.host.startTurn(sessionId, 'two', { modelId: MODEL })).toThrow(AgentHostError);
      fake.host.cancel(sessionId);
      await started.done;
    });
  });

  it('the attachments go out in the prompt, after its text block', async () => {
    await withFake('chat-basic', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const started = fake.host.startTurn(sessionId, 'look at these', { modelId: MODEL }, [FILE, PICTURE]);
      expect((await started.done).outcome).toBe('completed');
      const prompts = fake.sent().filter((m) => m.method === 'session/prompt');
      expect(prompts.map((m) => m.params)).toEqual([promptParams(sessionId, 'look at these', [FILE, PICTURE])]);
    });
  });

  it('an agent that does not take embedded resources: a file or PDF turn ends agent-incompatible unsent, a text turn runs', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const result = await fake.host.startTurn(sessionId, 'read this', { modelId: MODEL }, [FILE]).done;
        expect(result).toEqual({
          turnId: expect.any(String),
          failure: { code: 'agent-incompatible', message: 'This agent build does not match this app.' },
          unsent: true,
        });
        expect(fake.sent().filter((m) => m.method === 'session/prompt')).toEqual([]);
        const pdf = await fake.host.startTurn(sessionId, 'read this', { modelId: MODEL }, [PDF]).done;
        expect(pdf).toMatchObject({ failure: { code: 'agent-incompatible' }, unsent: true });
        expect(fake.sent().filter((m) => m.method === 'session/prompt')).toEqual([]);
        expect((await fake.host.startTurn(sessionId, 'hello', { modelId: MODEL }).done).outcome).toBe('completed');
        expect(fake.sent().filter((m) => m.method === 'session/prompt')).toHaveLength(1);
      },
      { mutate: (s) => (s.initialize = { ...(s.initialize as Json | undefined), embeddedContext: false }) },
    );
  });

  it('max-tokens ends as output-limit; an error text never reaches the result', async () => {
    await withFake('max-tokens', async (fake) => {
      expect((await turn(fake, 'long')).result.outcome).toBe('output-limit');
    });
    await withFake('error-with-marker', async (fake) => {
      const { result } = await turn(fake, 'boom');
      expect(result.outcome).toBe('model-error');
      expect(JSON.stringify(result)).not.toContain('marker-KEY');
      expect(JSON.stringify(fake.events)).not.toContain('marker-KEY');
    });
  });

  it('survives a crash: the turn ends agent-crashed, the session goes cold and reopens with a load', async () => {
    await withFake('crash-mid-turn', async (fake) => {
      const { id, result } = await turn(fake, 'crash now');
      expect(result).toMatchObject({ outcome: 'agent-crashed', failure: { code: 'agent-crashed' } });
      expect(fake.host.modeState()).toMatchObject({ state: 'crashed', failure: { code: 'agent-crashed' } });
      expect(fake.events.cold).toEqual([[id]]);
      expect(fake.host.isHot(id)).toBe(false);
    });
  });

  it('reopens a cold conversation: session/load resends everything with noReplay, the guard runs, no model entry', async () => {
    await withFake(
      'new-and-load-model-changed',
      async (fake) => {
        const { id } = await turn(fake, 'first');
        await until(() => fake.events.cold.length > 0);
        expect(fake.host.modeState().state).toBe('stopped');
        const second = fake.host.startTurn(id, 'second https://example.org/x', { modelId: MODEL });
        expect((await second.done).outcome).toBe('completed');
        const sent = fake.sent();
        const load = sent.find((m) => m.method === 'session/load')!;
        expect(load.params).toMatchObject({
          sessionId: id,
          mcpServers: [{ name: 'mlx_web' }],
          _meta: { noReplay: true, agentProfile: { name: 'mlx-chat' }, yoloMode: false },
        });
        expect(sent.filter((m) => m.method === '_x.ai/session/info')).toHaveLength(2);
        expect(fake.events.updates.some((u) => u.line.update.sessionUpdate === 'model_changed')).toBe(false);
        expect(fake.host.ledger(id)?.has('https://example.org/x')).toBe(true);
      },
      { timings: { idleStopMs: 50 } },
    );
  });

  it('seeds the ledger of a cold conversation from its disk entries, page links too, and sweeps stale files', async () => {
    await withFake('chat-fetch-listed-url', async (fake) => {
      mkdirSync(fake.paths.webLedger, { recursive: true });
      writeFileSync(join(fake.paths.webLedger, 'stale.json'), '["https://old.example"]');
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      expect(existsSync(join(fake.paths.webLedger, 'stale.json'))).toBe(false);
      await fake.host.restart();
      const page = 'WEB_PAGE provider=exa url=https://example.com/docs\ntitle: t\n\nsee https://linked.example/report.';
      const search = 'WEB_SEARCH provider=exa results=1\n[1] T https://title.example\nhttps://found.example/a\nsnippet';
      const started = fake.host.startTurn(sessionId, 'again', {
        modelId: MODEL,
        entries: [
          { kind: 'user', ts: 1, text: 'see https://example.com/docs', promptIndex: 0 },
          { kind: 'tool', ts: 2, id: 't1', name: 'mlx_web__web_fetch', title: 'f', status: 'completed', output: page },
          { kind: 'tool', ts: 3, id: 't2', name: 'use_tool', title: 's', status: 'completed', output: search },
          {
            kind: 'tool',
            ts: 4,
            id: 't3',
            name: 'other',
            title: 'o',
            status: 'completed',
            output: page.replace('linked', 'other'),
          },
        ],
      });
      expect((await started.done).outcome).toBe('completed');
      const paths = ledgerPaths(fake);
      expect(paths).toHaveLength(2);
      expect(paths[1]).toBe(paths[0]);
      expect(ledgerFile(paths[0]!)).toEqual([
        'https://example.com/docs',
        'https://linked.example/report.',
        'https://linked.example/report',
        'https://found.example/a',
        'https://blog.rust-lang.org/2026/11/01/Rust-1.99.0.html',
        'https://releases.rs',
      ]);
      await fake.host.close();
      expect(existsSync(paths[0]!)).toBe(false);
    });
  });

  it('renames and deletes a hot conversation through the agent; the session folder goes away', async () => {
    await withFake('chat-basic', async (fake) => {
      const { id } = await turn(fake, 'to be renamed');
      await fake.host.rename(id, 'New title');
      const cwd = realpathSync(fake.paths.chatWorkspace);
      const dir = join(fake.paths.chatSessions, encodeURIComponent(cwd), id);
      expect(JSON.parse(readFileSync(join(dir, 'summary.json'), 'utf8')).generated_title).toBe('New title');
      await fake.host.deleteSession(id, cwd);
      expect(existsSync(dir)).toBe(false);
      expect(fake.sent().find((m) => m.method === '_x.ai/session/delete')?.params).toEqual({ sessionId: id, cwd });
    });
  });

  it('switches the model between turns and confirms it with session info', async () => {
    await withFake('chat-basic', async (fake) => {
      const { id } = await turn(fake, 'hi');
      await fake.host.setModel(id, 'qwen3.5-9b');
      expect(fake.host.sessionModel(id)).toBe('qwen3.5-9b');
      await expect(fake.host.setModel(id, 'nope')).rejects.toMatchObject({ code: 'model-unavailable' });
    });
  });

  it('passes a real model switch during a turn and drops one to the current model', async () => {
    await withFake('model-switch-in-turn', async (fake) => {
      const { id } = await turn(fake, 'switch');
      const switches = fake.events.updates.filter((u) => u.line.update.sessionUpdate === 'model_changed');
      expect(switches.map((u) => u.line.update.model_id)).toEqual(['qwen3.5-9b']);
      expect(fake.events.models).toEqual([{ sessionId: id, modelId: 'qwen3.5-9b' }]);
    });
  });

  it('stops after the idle time and marks its sessions cold', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { id } = await turn(fake, 'hi');
        await until(() => fake.host.modeState().state === 'stopped');
        expect(fake.events.cold).toEqual([[id]]);
      },
      { timings: { idleStopMs: 50 } },
    );
  });

  it('a stale host finishes the running turn on the old process, then restarts', async () => {
    await withFake('cancel', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const started = fake.host.startTurn(sessionId, 'long', { modelId: MODEL });
      await until(() => texts(fake).includes('Thinking'));
      fake.host.markStale();
      expect(fake.host.modeState()).toMatchObject({ stale: true, state: 'ready' });
      fake.host.cancel(sessionId);
      await started.done;
      await until(() => fake.host.modeState().state === 'stopped');
      expect(fake.host.modeState().stale).toBe(false);
    });
  });

  it('a missing endpoint file during a turn cancels it as inference-lost', async () => {
    await withFake('cancel', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const started = fake.host.startTurn(sessionId, 'long', { modelId: MODEL });
      await until(() => texts(fake).includes('Thinking'));
      fake.endpoint.current = null;
      await fake.host.checkEndpoint();
      expect((await started.done).outcome).toBe('inference-lost');
    });
  });

  it('no endpoint file: the start fails as inference-stopped and nothing is spawned', async () => {
    await withFake('chat-basic', async (fake) => {
      fake.endpoint.current = null;
      await expect(fake.host.createSession({ modelId: MODEL })).rejects.toMatchObject({ code: 'inference-stopped' });
      expect(fake.records()).toEqual([]);
    });
  });

  it('a prompt reply that does not follow a cancel inside the grace time is a protocol error', async () => {
    await withFake(
      'hang-on-cancel',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const started = fake.host.startTurn(sessionId, 'x', { modelId: MODEL });
        await until(() => texts(fake).includes('x'));
        fake.host.cancel(sessionId);
        expect((await started.done).outcome).toBe('agent-crashed');
        expect(fake.host.modeState()).toMatchObject({ state: 'crashed', failure: { code: 'agent-protocol-error' } });
      },
      { timings: { cancelReplyMs: 100 } },
    );
  });

  it('a few garbage lines are dropped; a flood of them is a protocol error', async () => {
    await withFake('garbage-stdout', async (fake) => {
      expect((await turn(fake, 'hi')).result.outcome).toBe('completed');
      expect(JSON.stringify(fake.events)).not.toContain('marker-SECRET');
    });
    await withFake('garbage-flood', async (fake) => {
      await expect(fake.host.createSession({ modelId: MODEL })).rejects.toMatchObject({ code: 'agent-protocol-error' });
    });
  });

  it('a flood stays a protocol error when the agent answers before its late exit', async () => {
    await withFake(
      'garbage-flood',
      async (fake) => {
        await expect(fake.host.createSession({ modelId: MODEL })).rejects.toMatchObject({
          code: 'agent-protocol-error',
        });
        expect(fake.sent().map((m) => m.method)).not.toContain('session/new');
      },
      {
        mutate: (scenario) => {
          (scenario.faults as Json).exitAfterTermMs = 150;
        },
      },
    );
  });

  it('three crashes inside the window fail the mode until restart', async () => {
    await withFake('crash-mid-turn', async (fake) => {
      for (let i = 0; i < 3; i++) await turn(fake, `crash ${i}`);
      expect(fake.host.modeState()).toMatchObject({ state: 'failed', failure: { code: 'agent-failed' } });
      await expect(fake.host.createSession({ modelId: MODEL })).rejects.toMatchObject({ code: 'agent-failed' });
      await fake.host.restart();
      expect(fake.host.modeState()).toMatchObject({ state: 'stopped', failure: null });
      await fake.host.createSession({ modelId: MODEL });
    });
  });

  it('close cancels the running turn as interrupted and stops the process', async () => {
    const fake = await fakeHost('cancel');
    const { sessionId } = await fake.host.createSession({ modelId: MODEL });
    const started = fake.host.startTurn(sessionId, 'long', { modelId: MODEL });
    await until(() => texts(fake).includes('Thinking'));
    await fake.close();
    expect((await started.done).outcome).toBe('interrupted');
  });
});

describe('Chat host: the model select call before the prompt', () => {
  const prompts = (fake: FakeHost): Json[] => fake.sent().filter((m) => m.method === 'session/prompt');
  const starts = (fake: FakeHost): number => fake.records().filter((r) => r.kind === 'start').length;
  const waits = (fake: FakeHost): Array<string | null> =>
    fake.events.turns.flatMap((t) => (t.turn === null ? [] : [`${t.turn.phase}:${t.turn.wait ?? '-'}`]));

  it('the select call ends before session/prompt is written; a resident model shows no wait', async () => {
    await withFake('chat-basic', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      expect(fake.engine.selects()).toHaveLength(0);
      const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
      expect(result).toEqual({ turnId: expect.any(String), outcome: 'completed' });
      expect(fake.engine.selects()).toHaveLength(1);
      expect(JSON.parse(fake.engine.selects()[0]!.body)).toMatchObject({ model: MODEL });
      expect(waits(fake)).toEqual(['starting:-', 'running:-']);
    });
  });

  it('beside a held answer on the same model the select returns, the prompt goes out, and the turn runs with no output', async () => {
    const held = gate();
    await withFake(
      'chat-basic',
      async (fake) => {
        const answer = fetch(`${fake.engine.url}/v1/messages`, { method: 'POST', body: '{}' });
        await until(() => fake.engine.requests.some((r) => r.path === '/v1/messages'));
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const started = fake.host.startTurn(sessionId, 'hi', { modelId: MODEL });
        await until(() => prompts(fake).length === 1);
        expect(fake.engine.selects()).toHaveLength(1);
        expect(fake.engine.selects()[0]!.answered).toBe(true);
        expect(fake.host.turnState(sessionId)).toMatchObject({ phase: 'running', wait: null });
        expect(waits(fake)).toEqual(['starting:-', 'running:-']);
        expect(texts(fake)).toBe('');
        expect(fake.host.cancel(sessionId)).toBe(true);
        expect((await started.done).outcome).toBe('cancelled');
        held.open();
        expect((await answer).status).toBe(200);
      },
      {
        // The held answer stays held until the end, so a select that waited for it would pass this mark.
        timings: { modelWaitShowMs: 2_000 },
        mutate: (scenario) => {
          (scenario.prompts as Json[])[0]!.steps = [{ sleepMs: 60_000 }, { end: { stopReason: 'end_turn' } }];
        },
        engine: (r) => (r.path === '/v1/messages' ? { until: held.promise, json: {} } : { json: { input_tokens: 1 } }),
      },
    );
  });

  it('a held load shows wait model, keeps the prompt back, and clears the wait before running', async () => {
    const hold = gate();
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const started = fake.host.startTurn(sessionId, 'hi', { modelId: MODEL });
        await until(() => fake.host.turnState(sessionId)?.wait === 'model');
        expect(fake.host.turnState(sessionId)?.phase).toBe('starting');
        expect(prompts(fake)).toHaveLength(0);
        hold.open();
        expect((await started.done).outcome).toBe('completed');
        expect(prompts(fake)).toHaveLength(1);
        expect(waits(fake)).toEqual(['starting:-', 'starting:model', 'starting:-', 'running:-']);
      },
      {
        timings: { modelWaitShowMs: 20 },
        engine: (r) => (r.path.endsWith('count_tokens') ? { until: hold.promise, json: {} } : { json: {} }),
      },
    );
  });

  it('Stop during the load ends the turn cancelled and unsent: no session/cancel, no kill, the socket closes', async () => {
    const hold = gate();
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const started = fake.host.startTurn(sessionId, 'hi', { modelId: MODEL });
        await until(() => fake.engine.selects().length === 1);
        expect(fake.host.cancel(sessionId)).toBe(true);
        expect(await started.done).toEqual({ turnId: started.turnId, outcome: 'cancelled', unsent: true });
        await until(() => fake.engine.selects()[0]!.closedEarly);
        hold.open();
        fake.engine.script(() => ({ json: {} }));
        expect((await fake.host.startTurn(sessionId, 'again', { modelId: MODEL }).done).outcome).toBe('completed');
        const methods = fake.sent().map((m) => m.method);
        expect(methods).not.toContain('session/cancel');
        expect(prompts(fake)).toHaveLength(1);
        expect(starts(fake)).toBe(1);
      },
      {
        timings: { cancelReplyMs: 1 },
        engine: (r) => (r.path.endsWith('count_tokens') ? { until: hold.promise, json: {} } : { json: {} }),
      },
    );
  });

  it('a failed load ends the turn model-load-failed with no prompt', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
        expect(result).toEqual({
          turnId: expect.any(String),
          failure: { code: 'model-load-failed', message: 'The model did not load.' },
          unsent: true,
        });
        expect(prompts(fake)).toHaveLength(0);
      },
      { engine: () => ({ status: 500, json: {} }) },
    );
  });

  it('two conversations on two models: a model another select swapped out after its load still gets the prompt', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const one = await fake.host.createSession({ modelId: 'qwen3.5-4b' });
        const two = await fake.host.createSession({ modelId: 'qwen3.5-9b' });
        const results = await Promise.all([
          fake.host.startTurn(one.sessionId, 'one', { modelId: 'qwen3.5-4b' }).done,
          fake.host.startTurn(two.sessionId, 'two', { modelId: 'qwen3.5-9b' }).done,
        ]);
        expect(results.map((r) => r.outcome)).toEqual(['completed', 'completed']);
        expect(fake.engine.selects()).toHaveLength(2);
        expect(prompts(fake)).toHaveLength(2);
      },
      {
        endpointModels: [
          endpointModel('qwen3.5-4b', '/models/qwen3.5-4b'),
          endpointModel('qwen3.5-9b', '/models/qwen3.5-9b'),
        ],
        engine: () => ({ status: 400, json: {} }),
      },
    );
  });

  it('the agent exiting during the load ends the turn at once, agent-crashed and unsent', async () => {
    const hold = gate();
    await withFake(
      'chat-basic',
      async (fake) => {
        try {
          const { sessionId } = await fake.host.createSession({ modelId: MODEL });
          const started = fake.host.startTurn(sessionId, 'hi', { modelId: MODEL });
          await until(() => fake.engine.selects().length === 1);
          const pid = fake.records().find((r) => r.kind === 'start')!.pid as number;
          let ended: unknown = null;
          void started.done.then((result) => (ended = result));
          process.kill(pid, 'SIGKILL');
          await until(() => ended !== null, 5_000);
          expect(ended).toEqual({
            turnId: started.turnId,
            failure: { code: 'agent-crashed', message: expect.any(String) },
            unsent: true,
          });
          await until(() => fake.engine.selects()[0]!.closedEarly);
          expect(prompts(fake)).toHaveLength(0);
        } finally {
          hold.open();
        }
      },
      { engine: (r) => (r.path.endsWith('count_tokens') ? { until: hold.promise, json: {} } : { json: {} }) },
    );
  });

  it('Stop while the turn reads the endpoint sends no select call', async () => {
    await withFake('chat-basic', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const endpoint = fake.endpoint.current;
      let stopOnRead = true;
      Object.defineProperty(fake.endpoint, 'current', {
        configurable: true,
        get: () => {
          if (stopOnRead && fake.host.turnState(sessionId) !== null) {
            stopOnRead = false;
            fake.host.cancel(sessionId);
          }
          return endpoint;
        },
      });
      const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
      expect(result).toMatchObject({ outcome: 'cancelled', unsent: true });
      expect(stopOnRead).toBe(false);
      expect((await fake.host.startTurn(sessionId, 'again', { modelId: MODEL }).done).outcome).toBe('completed');
      expect(fake.engine.selects()).toHaveLength(1);
    });
  });

  it('a name the endpoint file does not list is model-unavailable with no engine request', async () => {
    await withFake('chat-basic', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      fake.endpoint.current = { ...fake.endpoint.current!, models: [endpointModel('other', '/models/other')] };
      const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
      expect(result.failure?.code).toBe('model-unavailable');
      expect(fake.engine.requests).toHaveLength(0);
      expect(prompts(fake)).toHaveLength(0);
    });
  });

  it('a name the engine does not serve is model-unavailable', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
        expect(result.failure?.code).toBe('model-unavailable');
      },
      { engine: () => ({ status: 404, json: {} }) },
    );
  });

  it('a refused token restarts the agent once, then fails inference-failed', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
        expect(result.failure?.code).toBe('inference-failed');
        expect(fake.engine.selects()).toHaveLength(2);
        expect(starts(fake)).toBe(2);
        expect(prompts(fake)).toHaveLength(0);
      },
      { engine: () => ({ status: 401, json: {} }) },
    );
  });

  it('an engine that is gone is inference-failed, or inference-stopped when its file went away', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const first = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
        expect(first.failure?.code).toBe('inference-failed');
        const endpoint = fake.endpoint.current;
        Object.defineProperty(fake.endpoint, 'current', {
          configurable: true,
          get: () => (fake.engine.selects().length > 1 ? null : endpoint),
        });
        const second = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
        expect(second.failure?.code).toBe('inference-stopped');
      },
      { engine: () => ({ drop: true }) },
    );
  });
});

describe('Work host', () => {
  const work = { mode: 'work' as const, providers: noProviders() };

  it('runs a Work session in the chosen folder with the parent env and a temp agent home', async () => {
    await withFake(
      'work-permission',
      async (fake) => {
        const created = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        expect(created).toMatchObject({ agentName: 'grok-build-plan', toolCount: 20, cwd: realpathSync(fake.project) });
        const newCall = fake.sent().find((m) => m.method === 'session/new')!;
        expect(newCall.params).toMatchObject({
          cwd: realpathSync(fake.project),
          mcpServers: [],
          _meta: { yoloMode: false, autoMode: false },
        });
        const start = fake.records().find((r) => r.kind === 'start')!;
        expect(start.envNames).toEqual(
          expect.arrayContaining(['GROK_HOME', 'HOME', 'PATH', 'XAI_API_KEY', 'GROK_RESUME_COMMAND']),
        );
        expect(start.envNames).not.toContain('ELECTRON_RUN_AS_NODE');
        // Mutation: the process started in the user's home again (its `.grok` is grok-build's) → red.
        expect(realpathSync(start.cwd as string)).toBe(realpathSync(fake.paths.workProcess));
        expect(readdirSync(fake.paths.workProcess)).toEqual([]);
        const started = fake.host.startTurn(created.sessionId, 'list files', { modelId: MODEL, cwd: fake.project });
        await until(() => fake.events.interactions.length === 1);
        const { interaction } = fake.events.interactions[0]!;
        expect(interaction).toMatchObject({ kind: 'permission', access: 'bash', toolName: 'run_terminal_command' });
        expect(interaction.kind === 'permission' && interaction.options.map((o) => o.optionId)).toEqual([
          'always-allow',
          'allow-once',
          'reject-once',
        ]);
        expect(() => fake.host.answer(created.sessionId, interaction.id, 'reject-always')).toThrow();
        fake.host.answer(created.sessionId, interaction.id, 'allow-once');
        expect((await started.done).outcome).toBe('completed');
        expect(fake.events.closed).toEqual([{ sessionId: created.sessionId, id: interaction.id, reason: 'answered' }]);
      },
      work,
    );
  });

  // Mutations: the login PATH not awaited (the agent starts with launchd's PATH, so a hook's `node` is not found), or
  // put after the inherited entries.
  it('waits for the pending login PATH and starts the agent with its entries first', async () => {
    let answer: (path: string | null) => void = () => {};
    const loginPath = new Promise<string | null>((resolve) => {
      answer = resolve;
    });
    const envs: Array<Record<string, string>> = [];
    await withFake(
      'work-plan',
      async (fake) => {
        const created = fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        await new Promise((resolve) => setTimeout(resolve, 100));
        expect(envs).toEqual([]);
        answer('/Users/dev/.vite-plus/bin:/usr/bin:/opt/homebrew/bin');
        await created;
        expect(envs.map((env) => env.PATH)).toEqual([
          `/Users/dev/.vite-plus/bin:/usr/bin:/opt/homebrew/bin:/bin:/usr/sbin:/sbin`,
        ]);
      },
      {
        ...work,
        env: { PATH: LAUNCHD_PATH },
        loginPath,
        spawn: (command) => {
          envs.push(command.env);
          return spawnAgent(command);
        },
      },
    );
  });

  it('a login PATH that failed leaves the inherited PATH as it is', async () => {
    const envs: Array<Record<string, string>> = [];
    await withFake(
      'work-plan',
      async (fake) => {
        await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        expect(envs.map((env) => env.PATH)).toEqual([LAUNCHD_PATH]);
      },
      {
        ...work,
        env: { PATH: LAUNCHD_PATH },
        loginPath: Promise.resolve(null),
        spawn: (command) => {
          envs.push(command.env);
          return spawnAgent(command);
        },
      },
    );
  });

  it('Deny ends the turn as denied, never cancelled; enable-always-approve is never offered to the page', async () => {
    await withFake(
      'work-deny',
      async (fake) => {
        const created = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        const started = fake.host.startTurn(created.sessionId, 'clean', { modelId: MODEL, cwd: fake.project });
        await until(() => fake.events.interactions.length === 1);
        const { interaction } = fake.events.interactions[0]!;
        expect(JSON.stringify(interaction)).not.toContain('enable-always-approve');
        fake.host.answer(created.sessionId, interaction.id, 'reject-once');
        expect((await started.done).outcome).toBe('denied');
      },
      work,
    );
  });

  it('a Work delete starts a stopped host and goes through the agent', async () => {
    await withFake(
      'work-plan',
      async (fake) => {
        await fake.host.deleteSession('01a1cb42-0000-7000-8000-000000000001', fake.project);
        const del = fake.sent().find((m) => m.method === '_x.ai/session/delete');
        expect(del?.params).toEqual({ sessionId: '01a1cb42-0000-7000-8000-000000000001', cwd: fake.project });
      },
      work,
    );
  });

  it('a provider key in a Work permission request is redacted before it reaches the page', async () => {
    const KEY = 'exa-KEY-feedfacecafebeef';
    const oneKey = noProviders({ workKeys: () => ({ EXA_API_KEY: KEY }), secretNeedles: () => [KEY] });
    await withFake(
      'work-permission',
      async (fake) => {
        const created = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        expect(fake.records().find((r) => r.kind === 'start')?.envNames).toContain('EXA_API_KEY');
        expect(JSON.stringify(fake.records().find((r) => r.kind === 'start'))).not.toContain(KEY);
        const started = fake.host.startTurn(created.sessionId, 'curl', { modelId: MODEL, cwd: fake.project });
        await until(() => fake.events.interactions.length === 1);
        const { interaction } = fake.events.interactions[0]!;
        expect(JSON.stringify(interaction)).not.toContain(KEY);
        expect(JSON.stringify(interaction)).toContain(REDACTED);
        fake.host.answer(created.sessionId, interaction.id, 'reject-once');
        expect((await started.done).outcome).toBe('denied');
      },
      {
        mode: 'work',
        providers: oneKey,
        mutate: (s) => {
          const step = ((s.prompts as Json[])[0]!.steps as Json[])[0]!.permission as Json;
          step.title = `Execute \`curl -H x-api-key:${KEY}\``;
          step.rawInput = {
            variant: 'Bash',
            command: `curl -H x-api-key:${KEY} https://api.exa.ai`,
            description: 'call',
            is_background: false,
          };
        },
      },
    );
  });

  it('a Work folder that does not exist is cwd-invalid', async () => {
    await withFake(
      'work-plan',
      async (fake) => {
        await expect(
          fake.host.createSession({ modelId: MODEL, cwd: join(fake.root, 'missing') }),
        ).rejects.toMatchObject({
          code: 'cwd-invalid',
        });
      },
      work,
    );
  });
});

describe('Work host: the length stop', () => {
  const work = { mode: 'work' as const, providers: noProviders() };
  const prompts = (fake: FakeHost): Json[] => fake.sent().filter((m) => m.method === 'session/prompt');
  const steps = (s: Json): Json[] => ((s.prompts as Json[])[1]!.steps as Json[])!;
  const setLength = (s: Json, length: Json): void => {
    steps(s)[1] = { lengthError: length };
  };
  const thought = (fake: FakeHost): boolean =>
    fake.events.updates.some((u) => u.line.update.sessionUpdate === 'agent_thought_chunk');
  const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

  async function start(fake: FakeHost) {
    const created = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
    const started = fake.host.startTurn(created.sessionId, 'fix it', { modelId: MODEL, cwd: fake.project });
    return { id: created.sessionId, done: started.done };
  }

  // Mutation: send the continue without waiting for the agent's end line → red (the order).
  it('continues once after the end line, with the button message, inside one running turn', async () => {
    await withFake(
      'work-length',
      async (fake) => {
        const { id, done } = await start(fake);
        expect(await done).toEqual({
          turnId: expect.any(String),
          outcome: 'completed',
          continues: 1,
        });
        const sent = prompts(fake);
        expect(sent).toHaveLength(2);
        expect(sent[1]!.params).toEqual(promptParams(id, CONTINUE_PROMPT));
        expect(CONTINUE_PROMPT).toBe('continue');
        expect(fake.events.continues).toEqual([{ sessionId: id, n: 1 }]);
        const order = fake.events.timeline.filter((k) => k === 'turn_completed' || k.startsWith('continue'));
        expect(order).toEqual(['turn_completed', 'continue 1', 'turn_completed']);
        const states = fake.events.turns.filter((t) => t.sessionId === id).map((t) => t.turn);
        const from = states.findIndex((t) => t?.phase === 'running');
        expect(states.slice(from, -1).every((t) => t?.phase === 'running')).toBe(true);
        expect(states.at(-1)).toBeNull();
        expect(states.at(-2)).toMatchObject({ phase: 'running', continues: 1 });
      },
      { ...work, mutate: (s) => setLength(s, { endAfterMs: 200 }) },
    );
  });

  // Mutation: send the attachments again on the continue prompt → red.
  it('the attachments go out with the first prompt only; the continue is one text block', async () => {
    await withFake(
      'work-length',
      async (fake) => {
        const created = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        const id = created.sessionId;
        const started = fake.host.startTurn(id, 'fix it', { modelId: MODEL, cwd: fake.project }, [FILE]);
        expect(await started.done).toMatchObject({ outcome: 'completed', continues: 1 });
        expect(prompts(fake).map((m) => m.params)).toEqual([
          promptParams(id, 'fix it', [FILE], { mode: 'agent' }),
          { sessionId: id, prompt: [{ type: 'text', text: CONTINUE_PROMPT }] },
        ]);
      },
      { ...work, mutate: (s) => setLength(s, { endAfterMs: 200 }) },
    );
  });

  // Mutation: count tool calls in `trackCall` → red (the read call of an ungranted session is missed).
  it('a second length stop with no tool call ends the turn; a read call between gives one more continue', async () => {
    await withFake(
      'work-length',
      async (fake) => {
        const { done } = await start(fake);
        expect(await done).toEqual({
          turnId: expect.any(String),
          outcome: 'output-limit',
          lengthStops: 2,
          continues: 1,
          noContinue: 'second-length',
        });
        expect(prompts(fake)).toHaveLength(2);
      },
      {
        ...work,
        mutate: (s) => {
          (s.prompts as Json[])[0] = { match: 'continue', steps: [{ thought: 'Again.' }, { lengthError: {} }] };
        },
      },
    );
    await withFake(
      'work-length',
      async (fake) => {
        const { done } = await start(fake);
        expect(await done).toEqual({ turnId: expect.any(String), outcome: 'completed', continues: 2 });
        expect(prompts(fake)).toHaveLength(3);
      },
      {
        ...work,
        mutate: (s) => {
          const after = (s.prompts as Json[])[0]!;
          (s.prompts as Json[]).unshift({
            match: 'continue',
            once: true,
            steps: [
              { tool: { toolName: 'read_file', kind: 'read', title: 'Read b.txt', output: 'y' } },
              { lengthError: {} },
            ],
          });
          expect(after.match).toBe('continue');
        },
      },
    );
  });

  // Mutation: remove the `between` branch of `cancelTurn` → red (a `session/cancel` goes out).
  it('Stop between the two prompts ends output-limit, sends nothing and arms no kill timer', async () => {
    await withFake(
      'work-length',
      async (fake) => {
        const { id, done } = await start(fake);
        await until(() => thought(fake));
        await sleep(150);
        expect(fake.host.cancel(id)).toBe(true);
        expect(await done).toEqual({
          turnId: expect.any(String),
          outcome: 'output-limit',
          lengthStops: 1,
          noContinue: 'stop',
          late: 'cancelled',
        });
        await sleep(150);
        expect(fake.sent().some((m) => m.method === 'session/cancel')).toBe(false);
        expect(prompts(fake)).toHaveLength(1);
        expect(fake.host.modeState().state).toBe('ready');
        expect(fake.records().filter((r) => r.kind === 'start')).toHaveLength(1);
      },
      { ...work, timings: { cancelReplyMs: 50 }, mutate: (s) => setLength(s, { noEnd: true }) },
    );
    let blocked: (() => void) | null = null;
    let block = false;
    await withFake(
      'work-length',
      async (fake) => {
        const { id, done } = await start(fake);
        await until(() => thought(fake));
        block = true;
        await until(() => blocked !== null);
        fake.host.cancel(id);
        blocked!();
        expect(await done).toMatchObject({ outcome: 'output-limit', noContinue: 'stop', late: 'cancelled' });
        expect(fake.sent().some((m) => m.method === 'session/cancel')).toBe(false);
        expect(prompts(fake)).toHaveLength(1);
      },
      {
        ...work,
        mutate: (s) => setLength(s, { endAfterMs: 200 }),
        readEndpoint: async (endpoint) => {
          if (block) await new Promise<void>((resolve) => (blocked = resolve));
          return endpoint.current;
        },
      },
    );
  });

  // Mutation: skip `checkEndpoint` before the continue → red (the continue goes to the old address).
  it('no continue when the agent exits or the endpoint changed in the gap', async () => {
    await withFake(
      'work-length',
      async (fake) => {
        const { done } = await start(fake);
        expect(await done).toMatchObject({ outcome: 'output-limit', noContinue: 'agent-exit' });
        expect(prompts(fake)).toHaveLength(1);
      },
      { ...work, mutate: (s) => setLength(s, { noEnd: true, exitAfterMs: 100 }) },
    );
    await withFake(
      'work-length',
      async (fake) => {
        const { done } = await start(fake);
        await until(() => thought(fake));
        fake.endpoint.current = { ...fake.endpoint.current!, token: 'tok-OTHER-0123456789abcdef' };
        expect(await done).toMatchObject({ outcome: 'output-limit', noContinue: 'host-stale' });
        expect(prompts(fake)).toHaveLength(1);
      },
      { ...work, mutate: (s) => setLength(s, { endAfterMs: 200 }) },
    );
  });

  // Mutation: the gate's wait also ends on a timer → red (the continue goes out after the clock moves).
  it('no timer ends the wait for the end line', async () => {
    const tick = (): Promise<void> => new Promise((resolve) => setImmediate(resolve));
    const spin = async (ms: number, check: () => boolean = () => false): Promise<void> => {
      const end = Date.now() + ms;
      while (!check() && Date.now() < end) await tick();
    };
    await withFake(
      'work-length',
      async (fake) => {
        const { id, done } = await start(fake);
        vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
        try {
          await spin(3_000, () => thought(fake));
          expect(thought(fake)).toBe(true);
          await spin(150);
          for (let hour = 0; hour < 3; hour += 1) {
            vi.advanceTimersByTime(3_600_000);
            await spin(100);
          }
          expect(fake.host.isBusy(id)).toBe(true);
          expect(prompts(fake)).toHaveLength(1);
        } finally {
          vi.useRealTimers();
        }
        fake.host.cancel(id);
        expect(await done).toMatchObject({ outcome: 'output-limit', noContinue: 'stop' });
      },
      { ...work, mutate: (s) => setLength(s, { noEnd: true }) },
    );
  });

  // Mutation: `markStale` does not wake the gate → red (the turn waits for Stop).
  it('a config change while the end line never comes ends the turn host-stale', async () => {
    await withFake(
      'work-length',
      async (fake) => {
        const { done } = await start(fake);
        let result: Json | null = null;
        void done.then((r) => (result = r as unknown as Json));
        await until(() => thought(fake));
        await sleep(150);
        fake.host.markStale();
        await until(() => result !== null, 2_000);
        expect(result).toMatchObject({ outcome: 'output-limit', noContinue: 'host-stale' });
        expect(prompts(fake)).toHaveLength(1);
      },
      { ...work, mutate: (s) => setLength(s, { noEnd: true }) },
    );
  });

  // Mutation: keep `toolSeen` across the continue → red (a third prompt goes out).
  it('a tool call before the first length stop does not count for the continue that calls none', async () => {
    await withFake(
      'work-length',
      async (fake) => {
        const { done } = await start(fake);
        expect(await done).toEqual({
          turnId: expect.any(String),
          outcome: 'output-limit',
          lengthStops: 2,
          continues: 1,
          noContinue: 'second-length',
        });
        expect(prompts(fake)).toHaveLength(2);
      },
      {
        ...work,
        mutate: (s) => {
          const list = s.prompts as Json[];
          const first = list[1]!;
          first.steps = [
            { tool: { toolName: 'read_file', kind: 'read', title: 'Read a.txt', output: 'x' } },
            { lengthError: {} },
          ];
          list.unshift({ match: 'continue', once: true, steps: [{ thought: 'Again.' }, { lengthError: {} }] });
        },
      },
    );
  });

  // Mutation: drop the mode check of the loop → red (Chat sends a second prompt).
  it('Chat never continues by itself', async () => {
    await withFake(
      'max-tokens',
      async (fake) => {
        const { result } = await turn(fake, 'long');
        expect(result).toEqual({ turnId: expect.any(String), outcome: 'output-limit' });
        await sleep(100);
        expect(prompts(fake)).toHaveLength(1);
        expect(fake.events.continues).toEqual([]);
      },
      {
        mutate: (s) => {
          const list = ((s.prompts as Json[])[0]!.steps as Json[])!;
          list[1] = { lengthError: {} };
        },
      },
    );
  });
});

describe('compaction state', () => {
  /** One prompt: a text, a compaction (`compact` step of the fake agent), then `after`. */
  const compacting =
    (compact: Json, after: Json[] = [{ text: ' done.', chunk: 6 }, { end: { stopReason: 'end_turn' } }]) =>
    (scenario: Json): void => {
      scenario.prompts = [{ match: '', steps: [{ text: 'Working', chunk: 7 }, { compact }, ...after] }];
    };
  const kinds = (fake: FakeHost, id: string) =>
    fake.events.compactions.filter((c) => c.sessionId === id).map((c) => (c.compaction === null ? null : 'running'));

  it('runs from the started update to the completed one, with the context the agent counted', async () => {
    await withFake(
      'cancel',
      async (fake) => {
        const { id, result } = await turn(fake, 'go');
        expect(result.outcome).toBe('completed');
        const states = fake.events.compactions.filter((c) => c.sessionId === id).map((c) => c.compaction);
        expect(states).toEqual([{ phase: 'running', startedAt: expect.any(Number), tokensBefore: 222_822 }, null]);
        const order = fake.events.timeline.filter((k) => k.includes('compact') || k === 'turn end');
        expect(order).toEqual([
          'auto_compact_started',
          'compaction start',
          'auto_compact_completed',
          'compaction end',
          'turn end',
        ]);
        expect(fake.host.compaction(id)).toBeNull();
      },
      { mutate: compacting({ ms: 200 }) },
    );
  }, 20_000);

  it('a failed compaction ends the state; the turn goes on', async () => {
    await withFake(
      'cancel',
      async (fake) => {
        const { id, result } = await turn(fake, 'go');
        expect(result.outcome).toBe('completed');
        expect(kinds(fake, id)).toEqual(['running', null]);
        expect(fake.events.timeline.indexOf('compaction end')).toBe(
          fake.events.timeline.indexOf('auto_compact_failed') + 1,
        );
      },
      { mutate: compacting({ ms: 100, outcome: 'failed' }) },
    );
  }, 20_000);

  it('Stop during a compaction: the agent cancels it, the state ends, the turn ends cancelled', async () => {
    await withFake(
      'cancel',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const started = fake.host.startTurn(sessionId, 'go', { modelId: MODEL });
        await until(() => fake.host.compaction(sessionId) !== null, 5_000);
        expect(fake.host.cancel(sessionId)).toBe(true);
        expect((await started.done).outcome).toBe('cancelled');
        expect(kinds(fake, sessionId)).toEqual(['running', null]);
        expect(fake.events.timeline).toContain('auto_compact_cancelled');
      },
      { mutate: compacting({ ms: 10_000 }) },
    );
  }, 20_000);

  it('the end of the turn ends a compaction the agent never closed', async () => {
    await withFake(
      'cancel',
      async (fake) => {
        const { id } = await turn(fake, 'go');
        const order = fake.events.timeline.filter((k) => k.startsWith('compaction') || k === 'turn end');
        expect(order).toEqual(['compaction start', 'compaction end', 'turn end']);
        expect(fake.host.compaction(id)).toBeNull();
      },
      { mutate: compacting({ outcome: 'none' }) },
    );
  }, 20_000);

  it("the agent's turn_completed line ends it before the prompt's reply", async () => {
    await withFake(
      'cancel',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const started = fake.host.startTurn(sessionId, 'go', { modelId: MODEL });
        await until(() => fake.events.timeline.includes('compaction end'), 5_000);
        expect(fake.host.modeState().turnsInFlight).toBe(1);
        expect(fake.host.compaction(sessionId)).toBeNull();
        fake.host.cancel(sessionId);
        await started.done;
      },
      { mutate: compacting({ outcome: 'none' }, [{ turnCompleted: {} }, { sleepMs: 10_000 }]) },
    );
  }, 20_000);

  it('the exit of the agent ends it at once, before the turn ends', async () => {
    await withFake(
      'cancel',
      async (fake) => {
        const { result } = await turn(fake, 'go');
        expect(result.outcome).toBe('agent-crashed');
        const order = fake.events.timeline.filter(
          (k) => k.startsWith('compaction') || k === 'cold' || k === 'turn end',
        );
        expect(order).toEqual(['compaction start', 'compaction end', 'cold', 'turn end']);
      },
      { mutate: compacting({ outcome: 'none' }, [{ exit: 9 }]) },
    );
  }, 20_000);
});

describe('model_auto_switched (#16): taken by the host, carried on the turn end', () => {
  const BLOCKED = 'Model "qwen3.5-4b" is not on the allowed list of this agent.';
  const switches = (fake: FakeHost) =>
    fake.events.updates.filter((u) => u.line.update.sessionUpdate === 'model_auto_switched');
  const prompts = (fake: FakeHost) => fake.sent().filter((m) => m.method === 'session/prompt');

  // Mutation: drop `turn.recorded = 'no-model'` → the blocked prompt ends `completed` with no line.
  it('a blocked prompt (both ids empty) ends no-model; the reason rides the end with null ids; no update', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { id, turnId, result } = await turn(fake, 'nomodel here');
        expect(result).toEqual({
          turnId,
          outcome: 'no-model',
          model: { previous: null, next: null, reason: BLOCKED },
        });
        expect(fake.events.ends).toEqual([{ sessionId: id, result }]);
        expect(switches(fake)).toEqual([]);
        // The next prompt runs as before.
        expect((await turn(fake, 'hi', id)).result).toEqual({ turnId: expect.any(String), outcome: 'completed' });
      },
      {
        mutate: (scenario) => {
          scenario.prompts = [{ match: 'nomodel', noModel: BLOCKED, steps: [] }, ...(scenario.prompts as Json[])];
        },
      },
    );
  });

  it('the reason is cut at 300 characters', async () => {
    const long = `${'r'.repeat(320)}`;
    await withFake(
      'chat-basic',
      async (fake) => {
        const { result } = await turn(fake, 'nomodel here');
        expect(result.model?.reason).toBe(`${'r'.repeat(300)}…`);
      },
      {
        mutate: (scenario) => {
          scenario.prompts = [{ match: 'nomodel', noModel: long, steps: [] }, ...(scenario.prompts as Json[])];
        },
      },
    );
  });

  it('one for a session the host does not know is dropped; model_changed still passes', async () => {
    await withFake(
      'model-switch-in-turn',
      async (fake) => {
        const { result } = await turn(fake, 'switch');
        expect(result).toEqual({ turnId: expect.any(String), outcome: 'completed' });
        expect(switches(fake)).toEqual([]);
        const changed = fake.events.updates.filter((u) => u.line.update.sessionUpdate === 'model_changed');
        expect(changed.map((u) => u.line.update.model_id)).toEqual(['qwen3.5-9b']);
      },
      {
        mutate: (scenario) => {
          const script = (scenario.prompts as Json[])[0]!;
          script.steps = [
            {
              notify: {
                method: '_x.ai/session_notification',
                params: {
                  sessionId: '01a20000-0000-7000-8000-00000000dead',
                  update: {
                    sessionUpdate: 'model_auto_switched',
                    previous_model_id: '',
                    new_model_id: '',
                    reason: 'x',
                  },
                },
              },
            },
            ...(script.steps as Json[]),
          ];
        },
      },
    );
  });

  it('a load whose stored model is gone with no fallback ends no-model unsent; later prompts stay blocked', async () => {
    const gone = 'Model "qwen3.5-4b" is no longer available. Please start a new session.';
    await withFake(
      'new-and-load-model-changed',
      async (fake) => {
        const { id } = await turn(fake, 'first');
        await until(() => fake.events.cold.length > 0);
        const second = fake.host.startTurn(id, 'second', { modelId: MODEL });
        expect(await second.done).toEqual({
          turnId: second.turnId,
          outcome: 'no-model',
          unsent: true,
          model: { previous: MODEL, next: null, reason: gone },
        });
        expect(prompts(fake)).toHaveLength(1);
        const third = await turn(fake, 'third', id);
        expect(third.result).toMatchObject({ outcome: 'no-model', model: { previous: null, next: null } });
        expect(switches(fake)).toEqual([]);
      },
      {
        timings: { idleStopMs: 50 },
        mutate: (scenario) => {
          (scenario.session as Json).modelGoneOnLoad = { reason: gone };
        },
      },
    );
  });

  it('a load the agent moved to another model is refused as before, with what it wanted on the end', async () => {
    const swapped = 'Model "qwen3.5-4b" is no longer available for your account.';
    await withFake(
      'new-and-load-model-changed',
      async (fake) => {
        const { id } = await turn(fake, 'first');
        await until(() => fake.events.cold.length > 0);
        const second = fake.host.startTurn(id, 'second', { modelId: MODEL });
        expect(await second.done).toMatchObject({
          outcome: 'refused',
          failure: { code: 'model-unavailable' },
          unsent: true,
          model: { previous: MODEL, next: 'qwen3.5-9b', reason: swapped },
        });
        expect(prompts(fake)).toHaveLength(1);
      },
      {
        timings: { idleStopMs: 50 },
        mutate: (scenario) => {
          (scenario.session as Json).modelGoneOnLoad = { next: 'qwen3.5-9b', reason: swapped };
        },
      },
    );
  });
});

describe('hook_run_started (#11): taken by the host, ended by its batch, the next words, the turn end or the exit', () => {
  const STOP = { event_name: 'stop', prompt_id: true, count: 2 };
  const LINT_FAILED = {
    event_name: 'stop',
    prompt_id: true,
    runs: [
      { name: 'global/lint:stop[0].hooks[0]', status: { status: 'failed', error: 'exit code 1', elapsed_ms: 900 } },
    ],
  };
  const hooking =
    (steps: Json[]) =>
    (scenario: Json): void => {
      scenario.prompts = [{ match: '', steps }];
    };
  const hookOrder = (fake: FakeHost) =>
    fake.events.timeline.filter(
      (k) => k === 'hook start' || k === 'hook end' || k === 'turn end' || k === 'cold' || k.startsWith('compaction'),
    );
  const started = (fake: FakeHost) =>
    fake.events.updates.filter((u) => u.line.update.sessionUpdate === 'hook_run_started');

  // Mutation: pass `hook_run_started` on (no `return`) → the update list holds it. Ignore the prompt id in
  // `hookRunEnds` → the other turn's line ends it.
  it("sets the state and never reaches the reducer; another turn's batch keeps it; its own hook_execution ends it", async () => {
    const gateFile = join(realpathSync(process.env.TMPDIR ?? '/tmp'), `th5-hook-${process.pid}-${Date.now()}`);
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const run = fake.host.startTurn(sessionId, 'go', { modelId: MODEL });
        await until(() => fake.host.hookRun(sessionId) !== null, 5_000);
        expect(fake.host.hookRun(sessionId)).toEqual({ event: 'stop', count: 2, startedAt: expect.any(Number) });
        expect(fake.events.hookRuns).toEqual([
          { sessionId, hookRun: { event: 'stop', count: 2, startedAt: expect.any(Number) } },
        ]);
        writeFileSync(gateFile, '');
        expect((await run.done).outcome).toBe('completed');
        expect(fake.host.hookRun(sessionId)).toBeNull();
        expect(started(fake)).toEqual([]);
        expect(hookOrder(fake)).toEqual(['hook start', 'hook end', 'turn end']);
        const executions = fake.events.timeline.flatMap((k, i) => (k === 'hook_execution' ? [i] : []));
        expect(executions).toHaveLength(2);
        expect(fake.events.timeline.indexOf('hook end')).toBe(executions[1]! - 1);
      },
      {
        mutate: hooking([
          { text: 'Done.', chunk: 5 },
          { hookStarted: STOP },
          {
            hook: {
              event_name: 'stop',
              prompt_id: 'an-earlier-turn',
              runs: [{ name: 'global/lint:stop[0].hooks[0]', status: { status: 'success', elapsed_ms: 30 } }],
            },
          },
          { holdOn: gateFile },
          { hook: LINT_FAILED },
          { end: { stopReason: 'end_turn' } },
        ]),
      },
    ).finally(() => rmSync(gateFile, { force: true }));
  }, 20_000);

  it('the tool hook carries its tool; another batch, a tool line and a chunk stamped before it keep it; a later chunk ends it', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { id, result } = await turn(fake, 'go');
        expect(result.outcome).toBe('completed');
        expect(fake.events.hookRuns.map((h) => h.hookRun)).toEqual([
          { event: 'pre_tool_use', tool: 'run_terminal_command', count: 1, startedAt: expect.any(Number) },
          null,
        ]);
        const kinds = fake.events.timeline.filter((k) => k !== 'session_info_update');
        const start = kinds.indexOf('hook start');
        expect(kinds.slice(start, kinds.indexOf('hook end') + 2)).toEqual([
          'hook start',
          'hook_execution',
          'tool_call',
          'tool_call_update',
          'agent_message_chunk',
          'hook end',
          'agent_message_chunk',
        ]);
        expect(fake.host.hookRun(id)).toBeNull();
      },
      {
        mutate: hooking([
          { hookStarted: { event_name: 'pre_tool_use', tool_name: 'run_terminal_command' } },
          {
            hook: {
              event_name: 'pre_tool_use',
              tool_name: 'read_file',
              runs: [{ name: 'global/audit:pre_tool_use[0].hooks[0]', status: { status: 'success', elapsed_ms: 12 } }],
            },
          },
          { tool: { toolName: 'read_file', rawInput: { path: 'a.txt' }, output: 'a' } },
          {
            notify: {
              method: 'session/update',
              params: {
                update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'made before. ' } },
                _meta: { agentTimestampMs: 1 },
              },
            },
          },
          { sleepMs: 20 },
          { text: 'Now.', chunk: 4 },
          { end: { stopReason: 'end_turn' } },
        ]),
      },
    );
  }, 20_000);

  it("the turn's end line ends a batch with no outcome; so does a compaction start", async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        await turn(fake, 'go');
        expect(hookOrder(fake)).toEqual(['hook start', 'hook end', 'turn end']);
        const end = fake.events.timeline.indexOf('hook end');
        expect(fake.events.timeline[end + 1]).toBe('turn_completed');
      },
      { mutate: hooking([{ hookStarted: STOP }, { turnCompleted: {} }, { end: { stopReason: 'end_turn' } }]) },
    );
    await withFake(
      'chat-basic',
      async (fake) => {
        await turn(fake, 'go');
        expect(hookOrder(fake)).toEqual(['hook start', 'hook end', 'compaction start', 'compaction end', 'turn end']);
      },
      {
        mutate: hooking([
          { hookStarted: { event_name: 'pre_compact', count: 1 } },
          { compact: { ms: 50 } },
          { end: { stopReason: 'end_turn' } },
        ]),
      },
    );
  }, 20_000);

  it("the agent's exit ends it before the turn ends; the reply ends one the agent never closed", async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { result } = await turn(fake, 'go');
        expect(result.outcome).toBe('agent-crashed');
        expect(hookOrder(fake)).toEqual(['hook start', 'hook end', 'cold', 'turn end']);
      },
      { mutate: hooking([{ hookStarted: STOP }, { exit: 9 }]) },
    );
    await withFake(
      'chat-basic',
      async (fake) => {
        const { id } = await turn(fake, 'go');
        expect(hookOrder(fake)).toEqual(['hook start', 'hook end', 'turn end']);
        expect(fake.host.hookRun(id)).toBeNull();
      },
      { mutate: hooking([{ hookStarted: STOP }, { end: { stopReason: 'end_turn' } }]) },
    );
  }, 20_000);

  // Mutation: drop the `HookDenied` branch of `turnOutcome` → the turn ends `interrupted`.
  it('a prompt a hook blocked ends hook-stopped; its hook lines and the stored end go to the reducer', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { id, result } = await turn(fake, 'deploy to prod');
        expect(result).toEqual({ turnId: expect.any(String), outcome: 'hook-stopped' });
        const lines = fake.events.updates.filter((u) => u.sessionId === id).map((u) => u.line);
        expect(lines.map((l) => l.update.sessionUpdate)).toEqual([
          'hook_execution',
          'hook_annotation',
          'turn_completed',
        ]);
        expect(lines[2]!.meta).toMatchObject({ cancellationCategory: 'HookDenied' });
        expect(hookOrder(fake)).toEqual(['hook start', 'hook end', 'turn end']);
        // A plain agent cancel with no category stays interrupted.
        expect((await turn(fake, 'agent-stop', id)).result.outcome).not.toBe('hook-stopped');
      },
      {
        mutate: (scenario) => {
          scenario.prompts = [
            {
              match: 'deploy',
              promptBlocked: { hook: 'global/policy:user_prompt_submit[0].hooks[0]', reason: 'no deploys on Friday' },
              steps: [],
            },
            ...(scenario.prompts as Json[]),
          ];
        },
      },
    );
  }, 20_000);
});

describe('Work host: workflow runs that left active', () => {
  const dirs: string[] = [];
  afterEach(() => {
    for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  });
  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  /** Paths in a fresh folder; the fake agent waits for each to exist. */
  function gateFiles() {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-wf-ended-'));
    dirs.push(dir);
    const at = (name: string) => join(dir, name);
    return { child: at('child'), complete: at('complete'), drain: at('drain'), wakeOn: at('wake'), wakeEnd: at('end') };
  }

  /** A Work host on `work-workflow-run` with the run's options merged in; `more` adds prompts after the first. */
  function work(
    run: Json,
    more: Json[] = [],
    extra: Parameters<typeof fakeHost>[1] = {},
  ): Parameters<typeof fakeHost>[1] {
    return {
      mode: 'work',
      providers: noProviders(),
      ...extra,
      mutate: (scenario) => {
        const prompts = scenario.prompts as Json[];
        const steps = prompts[0]!.steps as Json[];
        steps[1] = { workflow: { ...(steps[1]!.workflow as Json), ...run } };
        scenario.prompts = [{ ...prompts[0], match: 'p1' }, ...more];
      },
    };
  }

  /** The main agent's turn `p2`: its `workflow` call pauses or stops the run, then the turn ends. */
  function control(call: Json, end: Json = {}): Json {
    return { match: 'p2', steps: [{ runControl: call }, { end: { stopReason: 'end_turn', ...end } }] };
  }

  async function session(fake: FakeHost): Promise<string> {
    return (await fake.host.createSession({ modelId: MODEL, cwd: fake.project })).sessionId;
  }

  function start(fake: FakeHost, id: string, text: string) {
    return fake.host.startTurn(id, text, { modelId: MODEL, cwd: fake.project });
  }

  /** Turn p1 to its end with the `workflow` start allowed; the run is live after it. */
  async function startRun(fake: FakeHost, id: string): Promise<void> {
    const turn = start(fake, id, 'p1');
    await until(() => fake.events.interactions.length === 1);
    fake.host.answer(id, fake.events.interactions[0]!.interaction.id, 'allow-once');
    expect((await turn.done).outcome).toBe('completed');
    expect(fake.host.runs(id)).toEqual([runView('wf_1')]);
  }

  const first = (fake: FakeHost, id: string) => fake.host.runs(id)[0];

  // Mutation: keep a run that left `active` in `session.runs` → its late ask opens a card, and it still reads live.
  it('a run that ends moves to the ended list; a late ask of it opens no card', async () => {
    const g = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        writeFileSync(g.complete, '');
        await until(() => fake.records().some((r) => r.kind === 'child-answer'));
        // No wake turn comes in this scenario: the run's wake stays due.
        expect(fake.host.runs(id)).toEqual([runView('wf_1', { live: false, status: 'complete', wakeDue: true })]);
        expect(fake.records().find((r) => r.kind === 'child-answer')?.answer).toBeNull();
        expect(fake.events.interactions).toHaveLength(1);
        expect(fake.host.modeState().background).toBe(0);
      },
      work({ completeOn: g.complete, askAfterEnd: true, wake: false }),
    );
  }, 20_000);

  // Mutation: count only live runs in the idle hold → the idle stop ends the process under a paused run, which no
  // later process can resume.
  it("a run its script paused holds the idle stop until the main agent's stop of it; a failed run holds nothing", async () => {
    const g = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        writeFileSync(g.complete, '');
        await until(() => first(fake, id)?.live === false);
        expect(fake.host.runs(id)).toEqual([runView('wf_1', { live: false, status: 'user_paused', waiting: true })]);
        await sleep(800);
        expect(fake.host.modeState().state).toBe('ready');
        expect((await start(fake, id, 'p2').done).outcome).toBe('completed');
        expect(fake.host.runs(id)).toEqual([runView('wf_1', { live: false, status: 'cancelled', by: 'main-agent' })]);
        await until(() => fake.host.modeState().state === 'stopped', 5_000);
        expect(fake.events.turns.some((t) => t.turn?.origin === 'wake')).toBe(false);
      },
      work({ ask: false, completeOn: g.complete, status: 'user_paused' }, [control({ verb: 'stop' })], {
        timings: { idleStopMs: 300 },
      }),
    );
    const failed = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        writeFileSync(failed.complete, '');
        await until(() => fake.events.ends.length === 2, 5_000);
        expect(fake.host.runs(id)).toEqual([runView('wf_1', { live: false, status: 'failed' })]);
        await until(() => fake.host.modeState().state === 'stopped', 5_000);
      },
      work({ ask: false, completeOn: failed.complete, status: 'failed', wakeEndAfterMs: 0 }, [], {
        timings: { idleStopMs: 300 },
      }),
    );
  }, 30_000);

  // Mutation: drop `emitRuns` when the wake starts → the page's run still says its wake is due, so the block says
  // "Writing up the result" after the wake row came.
  it("a run's end makes its wake due on its view; the wake's start clears it and tells the page", async () => {
    const g = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        writeFileSync(g.complete, '');
        await until(() => fake.events.turns.some((t) => t.sessionId === id && t.turn?.origin === 'wake'));
        const sent = fake.events.runs.filter((e) => e.sessionId === id).map((e) => e.runs[0]);
        expect(sent.some((run) => run?.status === 'complete' && run.wakeDue)).toBe(true);
        expect(sent.at(-1)).toEqual(runView('wf_1', { live: false, status: 'complete' }));
        expect(fake.host.runs(id)).toEqual([runView('wf_1', { live: false, status: 'complete' })]);
        writeFileSync(g.wakeEnd, '');
        await until(() => fake.events.ends.length === 2, 5_000);
      },
      work({ completeOn: g.complete, ask: false, wakeEndOn: g.wakeEnd }),
    );
  }, 20_000);

  // Mutation: make a wake due for the main agent's own stop (`wakeAfter('cancelled', false)`) → the wake stays due,
  // and the next wake, which names no run, is titled as this run.
  it("the main agent's own stop of a run makes no wake due; the next wake is not titled as the run", async () => {
    const g = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        expect((await start(fake, id, 'p2').done).outcome).toBe('completed');
        expect(fake.host.wakeDue(id)).toBe(false);
        await until(() => first(fake, id)?.live === false);
        expect(first(fake, id)).toMatchObject({ live: false, status: 'cancelled', by: 'main-agent' });
        const hidden = () =>
          fake.events.updates.filter((u) => u.sessionId === id && u.line.update.sessionUpdate === 'user_message_chunk');
        writeFileSync(g.wakeOn, '');
        await until(() => hidden().length === 1);
        expect(hidden()[0]!.line.update._meta).toEqual({ hideFromScrollback: true });
        writeFileSync(g.wakeEnd, '');
        await until(() => fake.events.ends.length === 3, 5_000);
      },
      work({ childOn: g.child }, [control({ verb: 'stop' }, { wake: { startOn: g.wakeOn, endOn: g.wakeEnd } })]),
    );
  }, 20_000);

  // Mutation: drop the clear at the call's end → the wake the run's update made due stays, though none comes.
  it('a stop whose input comes only with its end clears the wake its update made due, and names the main agent', async () => {
    const g = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        expect((await start(fake, id, 'p2').done).outcome).toBe('completed');
        expect(fake.host.wakeDue(id)).toBe(false);
        expect(first(fake, id)).toMatchObject({ live: false, status: 'cancelled', by: 'main-agent' });
      },
      work({ childOn: g.child }, [control({ verb: 'stop', key: 'wf_1', inputLate: true })]),
    );
  }, 20_000);

  // Mutation: drop the typed `/workflow` rule → the person's own stop reads as no one's.
  it('a typed /workflow stop names you, and its wake stays due: the agent wakes for it', async () => {
    const g = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        expect((await start(fake, id, '/workflow stop deep-research').done).outcome).toBe('completed');
        expect(first(fake, id)).toMatchObject({ live: false, status: 'cancelled', by: 'you', stopping: false });
        expect(fake.host.wakeDue(id)).toBe(true);
      },
      work({ childOn: g.child, wake: false }),
    );
  }, 20_000);

  // Mutation: drop the rule for a stop of a run that waits → its wake reads due for good: the run's watcher ended with
  // its helpers, and the agent's stop from there sends only the update.
  it('a typed /workflow stop of a paused run whose helpers ended makes no wake due', async () => {
    const g = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        expect((await start(fake, id, '/workflow pause deep-research').done).outcome).toBe('completed');
        await until(() =>
          fake.events.updates.some((u) => u.sessionId === id && u.line.update.sessionUpdate === 'subagent_finished'),
        );
        expect(first(fake, id)).toMatchObject({ live: false, status: 'user_paused', by: 'you' });
        expect((await start(fake, id, '/workflow stop deep-research').done).outcome).toBe('completed');
        expect(first(fake, id)).toMatchObject({ live: false, status: 'cancelled', by: 'you', wakeDue: false });
        expect(fake.host.wakeDue(id)).toBe(false);
      },
      work({ childOn: g.child }),
    );
  }, 20_000);

  // Mutation: make no wake due for every stop of a run that waits → a stop during a pause's drain reads no wake, though
  // the run's watcher reports it once the helpers end.
  it("a typed /workflow stop during a pause's drain keeps its wake; the agent sends it once the helpers end", async () => {
    const g = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        expect((await start(fake, id, '/workflow pause deep-research').done).outcome).toBe('completed');
        expect(first(fake, id)).toMatchObject({ live: false, status: 'user_paused', by: 'you' });
        expect((await start(fake, id, '/workflow stop deep-research').done).outcome).toBe('completed');
        expect(first(fake, id)).toMatchObject({ live: false, status: 'cancelled', by: 'you', wakeDue: true });
        writeFileSync(g.drain, '');
        await until(() => fake.events.turns.some((t) => t.sessionId === id && t.turn?.origin === 'wake'));
        expect(first(fake, id)?.wakeDue).toBe(false);
        const ended = fake.events.ends.length;
        writeFileSync(g.wakeEnd, '');
        await until(() => fake.events.ends.length === ended + 1, 5_000);
        expect(fake.host.wakeDue(id)).toBe(false);
      },
      work({ childOn: g.child, pauseDrainOn: g.drain, wakeEndOn: g.wakeEnd }),
    );
  }, 20_000);

  // Mutation: send the cancel with the id the press named (the first attempt's) → the agent answers already_finished:
  // the Stop reads not-live and the retry runs on.
  it('Stop on a retried agent reaches its newest attempt: it ends stopped by you, and the run goes on (T41)', async () => {
    const g = gateFiles();
    const FIRST = 'run-agent-1';
    const RETRY = `${FIRST}-retry`;
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        expect(fake.host.liveActivity(id).agents).toEqual([
          { childId: RETRY, run: 'wf_1', attempts: [FIRST, RETRY], stopping: false, messages: 'no-address' },
        ]);
        expect(await fake.host.stopSubagent(id, FIRST)).toBe('stopped');
        const cancels = fake.sent().filter((m) => m.method === '_x.ai/subagent/cancel');
        expect(cancels.map((m) => m.params)).toEqual([{ subagentId: RETRY }]);
        await until(() => fake.host.liveActivity(id).agents.length === 0, 5_000);
        // The fake's lines are the agent's (recorded): the end line's fields, and the roster says failed.
        const [, , realEnd, realRoster] = STOP_RECORD.lines.map((l) => l.params?.update as Json);
        const lines = () => fake.events.updates.filter((u) => u.sessionId === id).map((u) => u.line.update as Json);
        const rowOf = (u: Json) => (u.agents as Json[] | undefined)?.find((a) => a.agent_id === RETRY);
        const isEnd = (u: Json) => u.sessionUpdate === 'subagent_finished' && u.subagent_id === RETRY;
        // As recorded: the roster update right after the end line.
        const rosterAfter = () => {
          const all = lines();
          const at = all.findIndex(isEnd);
          return at < 0 ? undefined : all.slice(at + 1).find((u) => u.sessionUpdate === 'workflow_updated');
        };
        await until(() => rosterAfter() !== undefined);
        const end = lines().find(isEnd)!;
        expect(Object.keys(end).sort()).toEqual(Object.keys(realEnd!).sort());
        expect([end.status, end.error, end.will_wake]).toEqual([realEnd!.status, realEnd!.error, realEnd!.will_wake]);
        const row = rowOf(rosterAfter()!)!;
        const realRow = (realRoster!.agents as Json[]).find((a) => a.agent_id === realEnd!.subagent_id)!;
        expect(Object.keys(row)).toEqual(expect.arrayContaining(Object.keys(realRow)));
        expect(row.state).toBe(realRow.state);
        expect(fake.host.liveActivity(id).stoppedByYou).toEqual([RETRY]);
        expect(first(fake, id)).toMatchObject({ live: true, status: 'active', stopping: false });
        writeFileSync(g.complete, '');
        await until(() => first(fake, id)?.status === 'complete', 5_000);
        expect(fake.host.liveActivity(id).stoppedByYou).toEqual([RETRY]);
      },
      work({ ask: false, retry: true, childSessionId: FIRST, completeOn: g.complete, wake: false }),
    );
  }, 20_000);

  // Mutation: drop by the pressed id alone → the agent's already_finished for the first attempt ends the retry's row
  // while the retry still runs.
  it('a Stop that reaches the first attempt after its end leaves the retry that took its row running', async () => {
    const g = gateFiles();
    const FIRST = 'run-agent-1';
    const RETRY = `${FIRST}-retry`;
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        expect(fake.host.liveActivity(id).agents.map((a) => a.childId)).toEqual([FIRST]);
        expect(await fake.host.stopSubagent(id, FIRST)).toBe('not-live');
        const cancels = fake.sent().filter((m) => m.method === '_x.ai/subagent/cancel');
        expect(cancels.map((m) => m.params)).toEqual([{ subagentId: FIRST }]);
        expect(fake.host.liveActivity(id).agents).toEqual([
          { childId: RETRY, run: 'wf_1', attempts: [FIRST, RETRY], stopping: false, messages: 'no-address' },
        ]);
        expect(fake.host.liveActivity(id).stoppedByYou).toEqual([]);
        expect(await fake.host.stopSubagent(id, FIRST)).toBe('stopped');
        expect(
          fake
            .sent()
            .filter((m) => m.method === '_x.ai/subagent/cancel')
            .map((m) => m.params),
        ).toEqual([{ subagentId: FIRST }, { subagentId: RETRY }]);
      },
      work({ ask: false, retry: 'on-cancel', childSessionId: FIRST, childOn: g.child, wake: false }),
    );
  }, 20_000);

  // Mutation: refuse a run that left active in `stopRunAs` → the Stop of a paused run reads not-live, nothing is sent.
  // Mutation: end `stopping` only at a drain update → the stopped paused run reads stopping until the cap (no drain
  // update follows: its script no longer runs).
  it('Stop run of a paused run: the stop is sent and the run reads stopped by you; a budget-limited run is not stopped', async () => {
    const g = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        writeFileSync(g.complete, '');
        await until(() => first(fake, id)?.status === 'user_paused');
        expect(await fake.host.stopRun(id, 'wf_1')).toBe('stopped');
        await until(() => first(fake, id)?.status === 'cancelled');
        await until(() => first(fake, id)?.stopping === false, 3_000);
        expect(fake.host.runs(id)).toEqual([runView('wf_1', { live: false, status: 'cancelled', by: 'you' })]);
        expect(fake.host.wakeDue(id)).toBe(false);
      },
      work({ ask: false, completeOn: g.complete, status: 'user_paused' }, [], { timings: { stopDrainMs: 60_000 } }),
    );
    const budget = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        writeFileSync(budget.complete, '');
        await until(() => first(fake, id)?.status === 'budget_limited');
        expect(await fake.host.stopRun(id, 'wf_1')).toBe('not-live');
        expect(fake.sent().filter((m) => m.method === '_x.ai/workflows/stop')).toEqual([]);
      },
      work({ ask: false, completeOn: budget.complete, status: 'budget_limited', wake: false }),
    );
  }, 30_000);

  // Mutation: end `stopping` at the `cancelled` status → the run reads stopped while its agents still drain.
  it('Stop run: the run reads stopping until its drain update, then stopped by you, with no wake due', async () => {
    const g = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        expect(await fake.host.stopRun(id, 'wf_1')).toBe('stopped');
        await until(() => first(fake, id)?.live === false);
        const stopped = { live: false, status: 'cancelled', by: 'you' } as const;
        expect(fake.host.runs(id)).toEqual([runView('wf_1', { ...stopped, stopping: true })]);
        await sleep(200);
        expect(first(fake, id)?.stopping).toBe(true);
        writeFileSync(g.drain, '');
        await until(() => first(fake, id)?.stopping === false);
        expect(fake.host.runs(id)).toEqual([runView('wf_1', stopped)]);
        expect(fake.host.wakeDue(id)).toBe(false);
      },
      work({ childOn: g.child, drainOn: g.drain }),
    );
  }, 20_000);

  // Mutation: end `stopping` only at the drain update → a run that cannot resume reads stopping until the cap.
  it('a drain too slow turns the stopped run interrupted: not stopping, not waiting, nothing due', async () => {
    const g = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        expect(await fake.host.stopRun(id, 'wf_1')).toBe('stopped');
        await until(() => first(fake, id)?.status === 'interrupted');
        expect(fake.host.runs(id)).toEqual([runView('wf_1', { live: false, status: 'interrupted', by: 'you' })]);
        expect(fake.host.wakeDue(id)).toBe(false);
      },
      work({ childOn: g.child, drainStatus: 'interrupted' }),
    );
  }, 20_000);

  // Mutation: no cap → a run whose drain update never comes reads stopping for good.
  it('a drain update that never comes ends stopping at the cap', async () => {
    const g = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        expect(await fake.host.stopRun(id, 'wf_1')).toBe('stopped');
        await until(() => first(fake, id)?.live === false);
        expect(first(fake, id)?.stopping).toBe(true);
        await until(() => first(fake, id)?.stopping === false, 3_000);
        expect(fake.host.runs(id)).toEqual([runView('wf_1', { live: false, status: 'cancelled', by: 'you' })]);
      },
      work({ childOn: g.child, drainOn: g.drain }, [], { timings: { stopDrainMs: 400 } }),
    );
  }, 20_000);

  // Mutation: make a run's wake due for every reportable status → a run that ends while the goal loop runs reads
  // "writing up" for good: the agent starts no wake then (`run_loop.rs`, `goal_loop_active`).
  it('a run that ends while the goal loop runs makes no wake due', async () => {
    const g = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        expect((await start(fake, id, 'p2').done).outcome).toBe('completed');
        writeFileSync(g.complete, '');
        await until(() => first(fake, id)?.live === false);
        expect(fake.host.runs(id)).toEqual([runView('wf_1', { live: false, status: 'complete' })]);
        expect(fake.host.wakeDue(id)).toBe(false);
      },
      work({ completeOn: g.complete, ask: false, wake: false }, [
        { match: 'p2', goal: { objective: 'Keep the tree green.' }, steps: [{ end: { stopReason: 'end_turn' } }] },
      ]),
    );
  }, 20_000);

  // Mutation: make a run's wake due for every reportable status → the second run takes the due from the first, whose
  // wake the agent did queue (it queues one run wake at a time: `run_loop.rs`, `workflow_wake_queued`).
  it("a run that ends while another run's wake is queued gets none; the first keeps its own", async () => {
    const g = gateFiles();
    const second = `${g.complete}-2`;
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id, 'p1');
        await until(() => fake.events.interactions.length === 1);
        fake.host.answer(id, fake.events.interactions[0]!.interaction.id, 'allow-once');
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.host.runs(id).map((run) => run.runId)).toEqual(['wf_1', 'wf_2']);
        const view = (runId: string) => fake.host.runs(id).find((run) => run.runId === runId);
        writeFileSync(g.complete, '');
        await until(() => view('wf_1')?.live === false);
        expect(view('wf_1')?.wakeDue).toBe(true);
        writeFileSync(second, '');
        await until(() => view('wf_2')?.live === false);
        expect(fake.host.runs(id)).toEqual([
          runView('wf_1', { live: false, status: 'complete', wakeDue: true }),
          runView('wf_2', { name: 'second-look', live: false, status: 'complete' }),
        ]);
      },
      {
        mode: 'work',
        providers: noProviders(),
        mutate: (scenario) => {
          const prompts = scenario.prompts as Json[];
          const steps = prompts[0]!.steps as Json[];
          const run = { ...(steps[1]!.workflow as Json), ask: false, wake: false };
          steps[1] = { workflow: { ...run, completeOn: g.complete } };
          steps.splice(2, 0, {
            workflow: { ...run, runId: 'wf_2', name: 'second-look', label: 'researcher-1', completeOn: second },
          });
          scenario.prompts = [{ ...prompts[0], match: 'p1' }];
        },
      },
    );
  }, 20_000);

  // Mutation: keep a queued run wake through every later turn → a run that a later turn reported (its start drains the
  // run's end: `drain_between_turn_completions`) reads "writing up" for good.
  // Mutation: drop it at any turn's end → the turn that ran when the run ended takes the wake away.
  it("the first turn that starts after a run's end and is not its wake drops the wake; the turn running then does not", async () => {
    const g = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        const running = start(fake, id, 'p2');
        await until(() => fake.sent().some((m) => m.method === 'session/prompt' && JSON.stringify(m).includes('"p2"')));
        await sleep(50);
        writeFileSync(g.complete, '');
        await until(() => first(fake, id)?.live === false);
        expect(fake.host.wakeDue(id)).toBe(true);
        writeFileSync(g.wakeEnd, '');
        expect((await running.done).outcome).toBe('completed');
        expect(first(fake, id)?.wakeDue).toBe(true);
        await sleep(50);
        expect((await start(fake, id, 'p3').done).outcome).toBe('completed');
        expect(fake.host.runs(id)).toEqual([runView('wf_1', { live: false, status: 'complete' })]);
        expect(fake.host.wakeDue(id)).toBe(false);
        expect(fake.events.runs.at(-1)?.runs).toEqual([runView('wf_1', { live: false, status: 'complete' })]);
      },
      work({ completeOn: g.complete, ask: false, wake: false }, [
        {
          match: 'p2',
          steps: [{ holdOn: g.wakeEnd }, { turnCompleted: { elapsed: 'turn' } }, { end: { stopReason: 'end_turn' } }],
        },
        { match: 'p3', steps: [{ turnCompleted: { elapsed: 'turn' } }, { end: { stopReason: 'end_turn' } }] },
      ]),
    );
  }, 20_000);

  // Mutation: send the runs event before the wake row (as `startWake` did) → for one flush the page holds the run's
  // report with no wake due and no row, and draws it in the block before it moves to the end row.
  it("the runs event that ends a run's due wake comes after the wake row; the run's end comes before its due", async () => {
    const g = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        writeFileSync(g.complete, '');
        await until(() => fake.events.turns.some((t) => t.sessionId === id && t.turn?.origin === 'wake'));
        await until(() => fake.host.runs(id)[0]?.wakeDue === false);
        const order = fake.events.order.filter((e) => e.sessionId === id);
        const ended = order.findIndex(
          (e) =>
            e.kind === 'update' &&
            e.line.update.sessionUpdate === 'workflow_updated' &&
            e.line.update.status === 'complete',
        );
        const end = order[ended];
        const revision = end?.kind === 'update' ? end.line.update.revision : null;
        const due = order.findIndex((e) => e.kind === 'runs' && e.runs[0]?.wakeDue === true);
        const row = order.findIndex(
          (e) =>
            e.kind === 'update' &&
            e.line.update.sessionUpdate === 'user_message_chunk' &&
            (e.line.update._meta as Json | undefined)?.mlxWakePromptId ===
              `workflow-completed-wf_1-${String(revision)}`,
        );
        const cleared = order.findIndex((e, n) => n > due && e.kind === 'runs' && e.runs[0]?.wakeDue === false);
        expect(ended).toBeGreaterThan(-1);
        expect(due).toBe(ended + 1);
        expect(row).toBeGreaterThan(due);
        expect(cleared).toBeGreaterThan(row);
        writeFileSync(g.wakeEnd, '');
        await until(() => fake.events.ends.length === 2, 5_000);
      },
      work({ completeOn: g.complete, ask: false, wakeEndOn: g.wakeEnd }),
    );
  }, 20_000);

  // Mutation: read on each new revision with no wait → the launch's two updates give two reads.
  // Mutation: no read → the thread never gets the run's history or its steps' details.
  it("a run's new revisions read its state.json once after the wait: a burst gives one read, with history and details", async () => {
    const g = gateFiles();
    const dir = mkdtempSync(join(tmpdir(), 'mlx-wf-state-'));
    dirs.push(dir);
    mkdirSync(join(dir, 'workflows', 'wf_1'), { recursive: true });
    const at = new Date().toISOString();
    writeFileSync(
      join(dir, 'workflows', 'wf_1', 'state.json'),
      JSON.stringify({
        version: 4,
        state: {
          run_id: 'wf_1',
          revision: 1,
          name: 'deep-research',
          status: 'active',
          phases: [
            { title: 'Plan', detail: 'Read the question.' },
            { title: 'Research', detail: 'Find the files.' },
          ],
          current_phase: 'Research',
          agents: [],
          history: [
            { event: 'phase_entered', detail: 'Research', at },
            { event: 'log', detail: 'Searching the tree.', at },
          ],
        },
      }),
    );
    const reads: string[] = [];
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        const launched = Date.now();
        await until(() => fake.events.runStates.length > 0, 3_000);
        expect(Date.now() - launched).toBeGreaterThanOrEqual(150);
        await sleep(400);
        expect(reads).toEqual([id]);
        const read = fake.events.runStates[0]!;
        expect(read.sessionId).toBe(id);
        expect(read.state.history.map((line) => [line.event, line.detail])).toEqual([
          ['phase_entered', 'Research'],
          ['log', 'Searching the tree.'],
        ]);
        expect([...read.state.details]).toEqual([
          ['Plan', 'Read the question.'],
          ['Research', 'Find the files.'],
        ]);
        writeFileSync(g.complete, '');
        await until(() => fake.events.runStates.length > 1, 3_000);
        await sleep(400);
        expect(reads).toEqual([id, id]);
      },
      work({ completeOn: g.complete, ask: false, wake: false }, [], {
        timings: { runStateMs: 300 },
        sessionDir: (sessionId) => {
          reads.push(sessionId);
          return dir;
        },
      }),
    );
  }, 20_000);

  // Mutation: the endpoint change's stop as the user's → a run the person never stopped reads stopped by you.
  it("an endpoint change stops the run as the engine's", async () => {
    const g = gateFiles();
    await withFake(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        fake.endpoint.current = { ...fake.endpoint.current!, url: 'http://127.0.0.1:9' };
        await fake.host.checkEndpoint();
        await until(() => fake.host.modeState().state === 'stopped', 5_000);
        const ended = fake.events.runs.flatMap((e) => e.runs).filter((r) => !r.live);
        expect(ended.map((r) => [r.status, r.by])).toContainEqual(['cancelled', 'engine']);
        expect(ended.some((r) => r.by === 'you')).toBe(false);
        expect(fake.host.runs(id)).toEqual([]);
      },
      work({ childOn: g.child }),
    );
  }, 20_000);
});
