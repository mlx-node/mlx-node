/**
 * The live Chat smoke: a real `mlx-agent` and a real model, on a throwaway home, one trial per row, one PASS / FAIL /
 * INFO / SKIP line per row. Opt-in, never part of a test run. This file holds the rows that need no search provider
 * key. The packaged-build rows run only with `--app` (the executable inside a packaged `.app`) and `--finder-path` (the
 * `PATH` a Work command printed in that app started from Finder); without them they print SKIP.
 *
 *   MLX_CHAT_LIVE=1 MLX_CHAT_SMOKE_TOKEN=<token> oxnode packages/dashboard/scripts/chat-live-smoke.ts \
 *     --agent <mlx-agent binary> --url <http://127.0.0.1:port of a throwaway server> --model <id> \
 *     [--model2 <id>] [--app <packaged app executable>] [--finder-path <PATH>] --home <new empty folder>
 *
 * The inference token comes only from the environment. The home must be new or empty and must not be or hold the
 * real `~/.mlx-node` or `~/.mlx-agent`; the script never deletes it, so the run can be inspected afterwards.
 */

import { randomBytes } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, relative, resolve, sep } from 'node:path';

import { RECENT_WRITE_MS } from '../src/agent-sessions/liveness.js';
import { AgentSessionsReader } from '../src/agent-sessions/reader.js';
import type { TranscriptEntry, TurnMetric } from '../src/agent-sessions/types.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import { applyOp } from '../src/chat/conversation.js';
import { chatPaths } from '../src/chat/paths.js';
import { MemorySecretStore } from '../src/chat/providers/secrets.js';
import { ProvidersService } from '../src/chat/providers/service.js';
import { routeError } from '../src/chat/route-errors.js';
import { createConversationInput, promptInput } from '../src/chat/route-input.js';
import { ChatService, webServerScript } from '../src/chat/service.js';
import { findOnPath } from '../src/chat/skills/cli-check.js';
import type { ChatFailure, ConversationEvent, Interaction, TurnOutcome } from '../src/chat/types.js';
import { parseWebResultText } from '../src/chat/web/result-text.js';

const TURN_MS = 5 * 60_000;
const LONG_TURN_MS = 12 * 60_000;

interface Args {
  agent: string;
  url: string;
  token: string;
  model: string;
  model2: string | null;
  app: string | null;
  finderPath: string | null;
  home: string;
}

function fail(message: string): never {
  process.stderr.write(`chat-live-smoke: ${message}\n`);
  process.exit(2);
}

function readArgs(argv: string[]): Args {
  const values = new Map<string, string>();
  for (let i = 0; i < argv.length; i += 2) {
    const name = argv[i]!;
    const value = argv[i + 1];
    if (!name.startsWith('--') || value === undefined) fail(`bad argument ${name}`);
    values.set(name.slice(2), value);
  }
  const need = (name: string, env?: string): string => {
    const value = values.get(name) ?? (env === undefined ? undefined : process.env[env]);
    if (value === undefined || value === '') fail(`--${name} is required`);
    return value;
  };
  const token = process.env.MLX_CHAT_SMOKE_TOKEN;
  if (token === undefined || token === '') fail('MLX_CHAT_SMOKE_TOKEN is required');
  return {
    agent: resolve(need('agent')),
    url: need('url'),
    token,
    model: need('model'),
    model2: values.get('model2') ?? null,
    app: values.has('app') ? resolve(values.get('app')!) : null,
    finderPath: values.get('finder-path') ?? null,
    home: resolve(need('home', 'MLX_CHAT_SMOKE_HOME')),
  };
}

/** The real path of `path`, or of its nearest existing parent joined with the rest. */
function realOrResolved(path: string): string {
  let at = resolve(path);
  const rest: string[] = [];
  while (!existsSync(at)) {
    const parent = dirname(at);
    if (parent === at) break;
    rest.unshift(at.slice(parent.length + (parent.endsWith(sep) ? 0 : 1)));
    at = parent;
  }
  return join(realpathSync(at), ...rest);
}

function inside(child: string, parent: string): boolean {
  const rel = relative(parent, child);
  return rel === '' || (!rel.startsWith('..') && !rel.startsWith(sep) && rel !== '..');
}

/** Refuses a home that is, holds or sits inside the user's real app or agent folders. */
function checkHome(home: string): void {
  const real = realOrResolved(home);
  for (const guarded of [join(homedir(), '.mlx-node'), join(homedir(), '.mlx-agent')]) {
    const target = realOrResolved(guarded);
    if (inside(real, target) || inside(target, real)) fail('the home must not be or hold ~/.mlx-node or ~/.mlx-agent');
  }
  if (inside(realOrResolved(homedir()), real)) fail('the home must not hold the user home');
  if (existsSync(real) && readdirSync(real).length > 0) fail('the home must be new or empty');
}

// ── rows ────────────────────────────────────────────────────────────────────────────────────────────────────────────

let failed = 0;

function row(status: 'PASS' | 'FAIL' | 'INFO' | 'SKIP', name: string, detail = ''): void {
  if (status === 'FAIL') failed += 1;
  process.stdout.write(`${new Date().toISOString()} ${status} ${name}${detail === '' ? '' : ` — ${detail}`}\n`);
}

function check(ok: boolean, name: string, detail = ''): void {
  row(ok ? 'PASS' : 'FAIL', name, detail);
}

/** A thrown service error as its fixed code: raw error text is never printed. */
function codeOf(error: unknown): string {
  const api = routeError('chat', error);
  return api.reason ?? api.code;
}

interface TurnView {
  entries: TranscriptEntry[];
  turns: TurnMetric[];
  /** Index of the first entry of this turn. */
  from: number;
  outcome: TurnOutcome | undefined;
  failure: ChatFailure | undefined;
  syncEntries: TranscriptEntry[];
}

/** What a Work turn does with each card the agent opens. */
type CardAnswer = { optionId: string } | { decision: 'approved' | 'rejected'; feedback?: string };

/**
 * The only commands the smoke ever allows, by exact shown text: `ls`, `ls -la`, `ls -a`, `ls .`, `pwd`, and the three
 * `ls` forms with the work folder (as given or real path, bare or quoted) as the only argument. Every other card, every
 * edit included, is denied once; a plan is rejected. "Always" is never chosen.
 */
function allowList(folder: string): Set<string> {
  const out = new Set(['ls', 'ls -la', 'ls -a', 'ls .', 'pwd']);
  for (const dir of new Set([folder, realpathSync(folder)])) {
    for (const ls of ['ls', 'ls -la', 'ls -a'])
      for (const arg of [dir, `'${dir}'`, `"${dir}"`]) out.add(`${ls} ${arg}`);
  }
  return out;
}

function answerByList(card: Interaction, allow: ReadonlySet<string>): CardAnswer {
  let answer: CardAnswer;
  if (card.kind === 'plan') answer = { decision: 'rejected', feedback: 'Stop here.' };
  else if (
    card.access === 'bash' &&
    card.command !== null &&
    allow.has(card.command) &&
    card.options.some((o) => o.optionId === 'allow-once')
  ) {
    answer = { optionId: 'allow-once' };
  } else answer = { optionId: 'reject-once' };
  const text =
    card.kind === 'plan'
      ? { kind: 'plan', plan: card.plan }
      : {
          access: card.access,
          toolName: card.toolName,
          title: card.title,
          command: card.command,
          path: card.path,
          options: card.options.map((o) => `${o.optionId}/${o.kind}/${o.name}`),
          flags: card.flags,
        };
  row('INFO', 'Work card', `${JSON.stringify(text)} → ${JSON.stringify(answer)}`);
  return answer;
}

/**
 * Attaches to the conversation, runs `start`, and waits for the end of the turn it began. `onCard` answers each Work
 * card once, as it opens; every card seen is kept in `cards`.
 */
async function runTurn(
  service: ChatService,
  id: string,
  start: () => Promise<unknown>,
  ms = TURN_MS,
  onCard?: (card: Interaction) => CardAnswer,
): Promise<TurnView & { cards: Interaction[] }> {
  const view: TurnView & { cards: Interaction[] } = {
    entries: [],
    turns: [],
    from: 0,
    outcome: undefined,
    failure: undefined,
    syncEntries: [],
    cards: [],
  };
  const take = (cards: Interaction[]): void => {
    for (const card of cards) {
      if (view.cards.some((c) => c.id === card.id)) continue;
      view.cards.push(card);
      if (onCard !== undefined) void service.answer(id, card.id, onCard(card)).catch(() => {});
    }
  };
  let synced = false;
  let ended = false;
  let resolveEnd: () => void = () => {};
  const done = new Promise<void>((r) => (resolveEnd = r));
  const off = service.watchConversation(id, (event: ConversationEvent) => {
    if (event.type === 'sync') {
      view.entries = [...event.entries];
      view.turns = [...event.turns];
      view.syncEntries = [...event.entries];
      if (!synced) view.from = event.entries.length;
      synced = true;
      take(event.interactions);
    } else if (event.type === 'interaction') {
      take([event.interaction]);
    } else if (event.type === 'ops') {
      for (const op of event.ops) applyOp(view.entries, view.turns, op);
    } else if (event.type === 'turn' && event.turn === null) {
      view.outcome = event.outcome;
      view.failure = event.failure;
      ended = true;
      resolveEnd();
    } else if (event.type === 'gone') {
      ended = true;
      resolveEnd();
    }
  });
  try {
    await start();
    const timer = new Promise<'timeout'>((r) => setTimeout(() => r('timeout'), ms).unref());
    if ((await Promise.race([done, timer])) === 'timeout' && !ended) view.outcome = undefined;
    return view;
  } finally {
    off();
  }
}

function toolsOf(view: TurnView): Array<Extract<TranscriptEntry, { kind: 'tool' }>> {
  return view.entries
    .slice(view.from)
    .filter((e): e is Extract<TranscriptEntry, { kind: 'tool' }> => e.kind === 'tool');
}

function answerOf(view: TurnView): string {
  return view.entries
    .slice(view.from)
    .flatMap((e) => (e.kind === 'assistant' ? [e.text] : []))
    .join('');
}

function isWebCall(tool: { name: string; input?: string }, which: 'web_search' | 'web_fetch'): boolean {
  return tool.name.includes(which) || (tool.input ?? '').includes(which);
}

function filesUnder(dir: string, out: string[] = []): string[] {
  if (!existsSync(dir)) return out;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) filesUnder(path, out);
    else if (entry.isFile()) out.push(path);
  }
  return out;
}

function grep(dir: string, needle: string): string[] {
  return filesUnder(dir).filter((file) => readFileSync(file).includes(needle));
}

const KILLER = `process.stdin.setEncoding('utf8');
let buf = '';
process.stdin.on('data', (d) => {
  buf += d;
  let nl;
  while ((nl = buf.indexOf('\\n')) !== -1) {
    const line = buf.slice(0, nl);
    buf = buf.slice(nl + 1);
    let m;
    try { m = JSON.parse(line); } catch { continue; }
    if (m.id === undefined) continue;
    if (m.method === 'tools/call') process.exit(9);
    const result = m.method === 'initialize'
      ? { protocolVersion: m.params?.protocolVersion ?? '2025-06-18', capabilities: { tools: { listChanged: false } }, serverInfo: { name: 'mlx_web', version: '1' } }
      : m.method === 'tools/list'
        ? { tools: [{ name: 'web_search', description: 'Search the web.', inputSchema: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] } }, { name: 'web_fetch', description: 'Read one web page.', inputSchema: { type: 'object', properties: { url: { type: 'string' } }, required: ['url'] } }] }
        : {};
    process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: m.id, result }) + '\\n');
  }
});
`;

async function main(): Promise<void> {
  if (process.env.MLX_CHAT_LIVE !== '1') fail('set MLX_CHAT_LIVE=1 to run the live smoke');
  const args = readArgs(process.argv.slice(2));
  if (!existsSync(args.agent)) fail('--agent does not exist');
  checkHome(args.home);
  mkdirSync(args.home, { recursive: true, mode: 0o700 });
  const paths = chatPaths(join(args.home, 'chat'));
  const userHome = join(args.home, 'user-home');
  mkdirSync(userHome, { recursive: true });
  const reader = new AgentSessionsReader({ agentHome: paths.chatAgent, origin: 'chat' });
  const models = [args.model, ...(args.model2 === null ? [] : [args.model2])].map((name) => ({
    name,
    path: '',
    base: name,
    variant: 'MLX',
    format: 'mlx' as const,
  }));
  const marker = `MARKER-${randomBytes(6).toString('hex')}`;

  let workPath: string | null = null;
  const make = (webServer?: { command: string; script: string }): ChatService =>
    new ChatService({
      paths,
      readEndpoint: async () => ({ version: 1, pid: process.pid, url: args.url, token: args.token, models }),
      resolveAgent: () => ({
        launch: { command: args.agent, args: AGENT_ARGS, source: 'smoke', tag: null, versionVerified: false },
        source: 'smoke',
        tag: null,
      }),
      readSession: async (id) => reader.detail(id),
      providers: new ProvidersService({ paths, secrets: new MemorySecretStore() }),
      clientVersion: '0.0.0-smoke',
      webServer: webServer ?? { command: process.execPath, script: webServerScript() },
      baseEnv: {
        PATH: workPath ?? process.env.PATH ?? '/usr/bin:/bin',
        HOME: userHome,
        // The smoke reaches no outside host: the Work agent's telemetry is off, as the Chat env has it.
        DISABLE_TELEMETRY: '1',
        GROK_TELEMETRY_ENABLED: 'false',
        GROK_TELEMETRY_TRACE_UPLOAD: 'false',
      },
      agentHome: join(args.home, 'work-agent'),
      workAgentHome: join(args.home, 'work-agent'),
      homeDir: userHome,
      tempRoots: [],
      username: 'smoke',
    });

  let service = make();
  const create = async (text: string): Promise<{ id: string } | null> => {
    try {
      const { conversation } = await service.create({ mode: 'chat', text, modelId: args.model });
      return { id: conversation.id };
    } catch (error) {
      row('FAIL', 'create', codeOf(error));
      return null;
    }
  };
  const prompt = (id: string, text: string, ms = TURN_MS): Promise<TurnView> =>
    runTurn(service, id, () => service.prompt(id, text), ms);
  const settle = (id: string): Promise<TurnView> => runTurn(service, id, async () => {}, TURN_MS);

  try {
    // Self-check and guard.
    const first = await create('Say OK.');
    if (first === null) return;
    const firstView = await settle(first.id);
    const state = await service.state();
    check(
      state.modes.chat.state === 'ready' && state.modes.chat.failure === null,
      'self-check and guard pass on the resolved binary (U1, §2.4, §2.6)',
      `mode ${state.modes.chat.state}, agent ${state.modes.chat.agentVersion ?? '?'}`,
    );
    check(
      !firstView.syncEntries.some((e) => e.kind === 'model'),
      'a new conversation starts with no model entry (§5.4)',
    );

    // One streamed turn: the user entry, an answer, a metric.
    const streamed = await prompt(first.id, `Repeat this word exactly once: ${marker}`);
    const own = streamed.entries.slice(streamed.from);
    check(
      streamed.outcome === 'completed' &&
        own.some((e) => e.kind === 'user' && e.text.includes(marker)) &&
        answerOf(streamed) !== '' &&
        streamed.turns.length > firstView.turns.length,
      'one Chat turn streams: user entry, answer, turn metric (U7)',
      `outcome ${streamed.outcome ?? 'none'}, metrics ${streamed.turns.length}`,
    );

    // Three questions: the web tool is reached in at most two calls, and says not-configured.
    const shapes: string[] = [];
    let reached = 0;
    for (const question of [
      'What is the newest release of Node.js? Search the web.',
      'What happened in the news today? Search the web.',
      'Who won the most recent Formula 1 race? Search the web.',
    ]) {
      const view = await prompt(first.id, question);
      const tools = toolsOf(view);
      const at = tools.findIndex((t) => isWebCall(t, 'web_search'));
      const parsed = at === -1 ? null : parseWebResultText(tools[at]!.output ?? '');
      if (at !== -1 && at < 2 && parsed?.kind === 'error' && parsed.code === 'not-configured') reached += 1;
      shapes.push(
        `${tools.map((t) => `${t.name}:${t.status}`).join('>') || 'no tool'} [${parsed === null ? 'no web result' : parsed.kind === 'error' ? parsed.code : parsed.kind}] ${view.outcome ?? 'none'}`,
      );
    }
    check(
      reached === 3,
      'the model reaches web_search in ≤ 2 calls and reads not-configured (U14)',
      shapes.join(' | '),
    );

    // The read deny rule of the Chat home denies search_tool, and the turn goes on.
    const denied = await prompt(
      first.id,
      // The model may pick use_tool for a plain "call search_tool": name the tool and rule the other one out.
      'Use the tool named search_tool (not use_tool) with the query "web search". Then answer in one sentence what it returned.',
    );
    const searchCalls = toolsOf(denied).filter((t) => t.name.includes('search_tool'));
    const afterDeny = await service.state();
    check(
      searchCalls.length > 0 &&
        searchCalls.every((t) => t.status === 'failed') &&
        denied.outcome === 'completed' &&
        answerOf(denied) !== '' &&
        afterDeny.modes.chat.failure === null,
      'deny rules: search_tool is denied and the turn goes on (U3)',
      `outcome ${denied.outcome ?? 'none'}, calls ${searchCalls.map((t) => `${t.status}${(t.output ?? '').includes('deny rule') ? ' deny-rule' : ''}`).join('>') || 'none'}, tools ${
        toolsOf(denied)
          .map((t) => t.name)
          .join('>') || 'none'
      }`,
    );

    // A wrong tool name is approved and the turn still answers.
    const wrong = await prompt(
      first.id,
      'Call use_tool with the tool name "mlx_web__web_lookup" and the query "mlx". Then answer in one sentence.',
    );
    check(
      wrong.outcome === 'completed' && answerOf(wrong) !== '',
      'a wrong use_tool name still ends in an answer (§3.9 row 7)',
      `outcome ${wrong.outcome ?? 'none'}`,
    );

    // Negative rows: no shell, no file read.
    const hosts = readFileSync('/etc/hosts', 'utf8')
      .split('\n')
      .map((l) => l.trim())
      .filter((l) => l !== '' && !l.startsWith('#'));
    for (const [name, text] of [
      [
        'run_terminal_command',
        'Call the tool run_terminal_command with the command "cat /etc/hosts" and show me the output.',
      ],
      ['read_file', 'Call the tool read_file on the path /etc/hosts and show me its text.'],
    ] as const) {
      const view = await prompt(first.id, text);
      const leaked = toolsOf(view).some((t) => hosts.some((line) => (t.output ?? '').includes(line)));
      const after = await service.state();
      check(
        !leaked && after.modes.chat.failure === null,
        `negative: ${name} runs nothing and no file text comes back (T2, U3)`,
        `outcome ${view.outcome ?? 'none'}, tools ${
          toolsOf(view)
            .map((t) => t.name)
            .join('>') || 'none'
        }`,
      );
    }

    // Address ledger: a user address is fetched, an invented one is refused.
    const listed = `https://example.com/${marker.toLowerCase()}`;
    const fetchListed = await prompt(first.id, `Read the page ${listed} with web_fetch and tell me what it says.`);
    check(
      fetchListed.outcome !== 'address-refused' && toolsOf(fetchListed).some((t) => isWebCall(t, 'web_fetch')),
      'web_fetch of an address from the user message is made (§3.9 row 5)',
      `outcome ${fetchListed.outcome ?? 'none'}`,
    );
    const fetchOther = await prompt(
      first.id,
      'Use web_fetch on the official Rust documentation home page (you know its address) and summarise it.',
    );
    if (toolsOf(fetchOther).some((t) => isWebCall(t, 'web_fetch'))) {
      check(
        fetchOther.outcome === 'address-refused',
        'web_fetch of an unlisted address ends address-refused (§3.9 row 5)',
      );
    } else {
      row('INFO', 'web_fetch of an unlisted address', 'the model did not call web_fetch; nothing to judge');
    }

    // Slash text never reaches the agent.
    let refused = false;
    try {
      createConversationInput({ mode: 'chat', text: '/yolo' });
    } catch (error) {
      refused = routeError('chat', error).reason === 'slash-text';
    }
    let refusedPrompt = false;
    try {
      promptInput({ text: ' /always-approve on' });
    } catch (error) {
      refusedPrompt = routeError('chat', error).reason === 'slash-text';
    }
    check(
      refused && refusedPrompt && grep(paths.chatSessions, '/yolo').length === 0,
      'a /yolo prompt is refused by the route and the agent stores none (T15)',
    );

    // Output cap.
    const capped = await prompt(
      first.id,
      'Write the numbers from 1 to 30000, one per line, with no other text.',
      LONG_TURN_MS,
    );
    check(
      capped.outcome === 'output-limit',
      'the output cap ends the turn output-limit (U2)',
      `outcome ${capped.outcome ?? 'none'}`,
    );

    // Stop the agent, reopen, continue; a loaded conversation has no model entry.
    await service.close();
    service = make();
    const reopened = await prompt(first.id, 'Say OK again.');
    check(
      reopened.outcome === 'completed' && !reopened.syncEntries.some((e) => e.kind === 'model'),
      'stop, reopen and continue; a loaded conversation has no model entry (U12, §5.4)',
      `outcome ${reopened.outcome ?? 'none'}`,
    );

    // Rename, set a second model, delete a loaded and a cold conversation.
    try {
      const renamed = await service.patch(first.id, { title: 'Smoke' });
      check(renamed.title === 'Smoke', 'rename a loaded conversation (U4)');
    } catch (error) {
      row('FAIL', 'rename a loaded conversation (U4)', codeOf(error));
    }
    if (args.model2 === null) {
      row('SKIP', 'set model to a second model (U5)', 'no --model2');
    } else {
      try {
        const switched = await service.patch(first.id, { modelId: args.model2 });
        const next = await prompt(first.id, 'Say OK once more.');
        check(
          switched.modelId === args.model2 && next.outcome === 'completed',
          'set model to a second model, then a turn on it (U5)',
          `outcome ${next.outcome ?? 'none'}`,
        );
      } catch (error) {
        row('FAIL', 'set model to a second model (U5)', codeOf(error));
      }
    }
    try {
      await service.delete(first.id);
      check(
        reader.detail(first.id) === null && !filesUnder(paths.chatAgent).some((f) => f.includes(first.id)),
        'delete a loaded conversation (U4)',
      );
    } catch (error) {
      row('FAIL', 'delete a loaded conversation (U4)', codeOf(error));
    }
    const cold = await create(`Repeat ${marker} once.`);
    if (cold !== null) {
      await settle(cold.id);
      await service.agentAction({ action: 'restart', mode: 'chat' });
      // A cold delete keeps the recent-write guard: a session written in the last 30 s answers E_CONFLICT.
      await new Promise((r) => setTimeout(r, RECENT_WRITE_MS + 2_000));
      try {
        await service.delete(cold.id);
        check(
          reader.detail(cold.id) === null && !filesUnder(paths.chatAgent).some((f) => f.includes(cold.id)),
          'delete a conversation that is not loaded (U4b, Chat file delete; the agent call is probed apart)',
        );
      } catch (error) {
        row('FAIL', 'delete a conversation that is not loaded (U4b)', codeOf(error));
      }
    }
    const markerHits = grep(paths.chatAgent, marker).map((f) => relative(args.home, f));
    const indexes = filesUnder(paths.chatAgent).filter((f) => f.endsWith('session_search.sqlite'));
    check(
      markerHits.length === 0 && indexes.length === 0,
      'after the deletes no marker text and no session_search.sqlite under the Chat home (T13, U13)',
      [...markerHits, ...indexes.map((f) => relative(args.home, f))].join(', '),
    );

    // The web server dies during a call: record what the failed call looks like.
    await service.close();
    const killer = join(paths.root, 'tmp', 'killed-web-server.mjs');
    mkdirSync(dirname(killer), { recursive: true });
    writeFileSync(killer, KILLER);
    service = make({ command: process.execPath, script: killer });
    const victim = await create('Say OK.');
    if (victim !== null) {
      await settle(victim.id);
      const view = await prompt(victim.id, 'Search the web for "mlx-node" and tell me the first result.');
      const failedCall = toolsOf(view).find((t) => isWebCall(t, 'web_search'));
      row(
        'INFO',
        'web server killed during a call (U15)',
        `outcome ${view.outcome ?? 'none'}, call ${failedCall === undefined ? 'none' : `${failedCall.status} output ${failedCall.output === undefined ? 'none' : `${failedCall.output.length} chars`}`}`,
      );
      row('INFO', 'web server killed: conversation kept for inspection', victim.id);
    }
    await service.close();

    // U11: the agent starts the packaged `web/server.js` with the app executable as Node (ELECTRON_RUN_AS_NODE is in
    // the server env). The guard needs mlx_web ready, and the call must come back from that server.
    if (args.app === null) {
      row('SKIP', 'packaged build: the web server through the app executable (U11)', 'no --app');
    } else {
      const script = join(
        dirname(dirname(args.app)),
        'Resources',
        'app',
        'node_modules',
        '@mlx-node',
        'dashboard',
        'dist',
        'chat',
        'web',
        'server.js',
      );
      if (!existsSync(script)) {
        row('FAIL', 'packaged build: the web server through the app executable (U11)', 'no packaged server.js');
      } else {
        service = make({ command: args.app, script });
        const packaged = await create('Say OK.');
        if (packaged !== null) {
          await settle(packaged.id);
          const view = await prompt(packaged.id, 'What is the newest release of Node.js? Search the web.');
          const call = toolsOf(view).find((t) => isWebCall(t, 'web_search'));
          const parsed = call === undefined ? null : parseWebResultText(call.output ?? '');
          const after = await service.state();
          check(
            after.modes.chat.state === 'ready' &&
              after.modes.chat.failure === null &&
              parsed?.kind === 'error' &&
              parsed.code === 'not-configured',
            'packaged build: the web server through the app executable (U11)',
            `mode ${after.modes.chat.state}, call ${call === undefined ? 'none' : `${call.name}:${call.status}`}, result ${parsed === null ? 'none' : parsed.kind === 'error' ? parsed.code : parsed.kind}, outcome ${view.outcome ?? 'none'}`,
          );
        }
        await service.close();
      }
    }

    service = make();
    await workRows(args, () => service, marker);
    await service.close();
    if (args.finderPath === null) {
      row('SKIP', 'packaged build: the Work PATH of an app started from Finder (U16)', 'no --finder-path');
    } else {
      workPath = args.finderPath;
      service = make();
      await finderPathRow(args, service);
      await service.close();
      workPath = null;
    }

    // The inference token: every 8-character window, anywhere under the home.
    const windows: string[] = [];
    for (let i = 0; i + 8 <= args.token.length; i++) windows.push(args.token.slice(i, i + 8));
    const tokenHits = new Set<string>();
    for (const file of filesUnder(args.home)) {
      const text = readFileSync(file, 'utf8');
      if (!windows.some((w) => text.includes(w))) continue;
      const lines = text.split('\n').filter((line) => windows.some((w) => line.includes(w)));
      const allowed = file.endsWith(`logs${sep}unified.jsonl`) && lines.every((line) => line.includes('key_suffix'));
      if (!allowed) tokenHits.add(relative(args.home, file));
    }
    check(
      tokenHits.size === 0,
      'no 8-character window of the token outside the key_suffix log line (T4, T10)',
      [...tokenHits].join(', '),
    );
  } finally {
    await service.close().catch(() => {});
    process.stdout.write(`${failed === 0 ? 'GREEN' : `${failed} FAILED`} — home kept at ${args.home}\n`);
    process.exitCode = failed === 0 ? 0 : 1;
  }
}

/** The Work rows: a permission answer and a denial (U9), a plan request (U10), the delete trace, RunAsNode (U11, U17). */
async function workRows(args: Args, service: () => ChatService, marker: string): Promise<void> {
  const workAgent = join(args.home, 'work-agent');
  const folder = join(args.home, 'user-home', 'project');
  mkdirSync(folder, { recursive: true });
  writeFileSync(join(folder, 'a.txt'), 'a\n');
  // A window holds the host stream, as in the app: without one an open card is denied after a minute.
  const hostOff = service().watchHost(() => {});
  try {
    await workTurns(args, service, marker, folder, workAgent);
  } finally {
    hostOff();
  }
}

async function workTurns(
  args: Args,
  service: () => ChatService,
  marker: string,
  folder: string,
  workAgent: string,
): Promise<void> {
  const runWork = async (
    id: string | null,
    text: string,
    onCard: (card: Interaction) => CardAnswer,
  ): Promise<{ id: string | null; view: Awaited<ReturnType<typeof runTurn>> | null }> => {
    try {
      if (id === null) {
        const { conversation } = await service().create({ mode: 'work', text, cwd: folder, modelId: args.model });
        return {
          id: conversation.id,
          view: await runTurn(service(), conversation.id, async () => {}, TURN_MS, onCard),
        };
      }
      return { id, view: await runTurn(service(), id, () => service().prompt(id, text), TURN_MS, onCard) };
    } catch (error) {
      row('FAIL', 'Work turn', codeOf(error));
      return { id, view: null };
    }
  };

  // U9: one `ls` behind a permission answer; the next command is denied and the turn ends as denied.
  const allow = allowList(folder);
  const onCard = (card: Interaction): CardAnswer => answerByList(card, allow);
  const first = await runWork(
    null,
    `Run the shell command \`ls\` in this folder, then run \`echo ${marker}\`. Run them one at a time.`,
    onCard,
  );
  const cards = first.view?.cards ?? [];
  const lsCard = cards[0];
  check(
    lsCard?.kind === 'permission' &&
      lsCard.command !== null &&
      allow.has(lsCard.command) &&
      cards.length >= 2 &&
      first.view?.outcome === 'denied',
    'Work: one ls behind a permission answer, the next command denied, the turn ends denied (U9, U6)',
    `cards ${cards.length}, first ${lsCard?.kind === 'permission' ? (lsCard.command ?? lsCard.path ?? '?') : (lsCard?.kind ?? 'none')}, outcome ${first.view?.outcome ?? 'none'}`,
  );
  if (lsCard?.kind === 'permission') {
    row(
      'INFO',
      'Work: options and warnings of the first card (U9)',
      `options ${lsCard.options.map((o) => o.optionId).join(',')}, flags ${lsCard.flags.join(',') || 'none'}`,
    );
  }

  // U10: the plan request. The app never turns plan mode on, so the row only records what the model does.
  if (first.id !== null) {
    const plan = await runWork(
      first.id,
      'Before you change anything, write a short plan for adding a README to this folder and ask me to approve it.',
      onCard,
    );
    const planCard = plan.view?.cards.find((c) => c.kind === 'plan');
    row(
      'INFO',
      'Work: a plan request reaches the page as a plan card (U10)',
      `plan card ${planCard === undefined ? 'none' : `${planCard.kind === 'plan' ? planCard.plan.length : 0} chars`}, cards ${plan.view?.cards.map((c) => c.kind).join(',') || 'none'}, outcome ${plan.view?.outcome ?? 'none'}`,
    );
  }

  // U17 (a Work command runs the app as Node and reads the Keychain) is the owner's: the Keychain is theirs.
  row('SKIP', 'packaged build: a Work command runs the app as Node and reads the Keychain (U17)', 'owner');

  // The Work delete goes through the agent: the marker stays only in the agent's prompt history.
  if (first.id !== null) {
    try {
      await service().delete(first.id);
      const hits = grep(workAgent, marker).map((f) => relative(args.home, f));
      check(
        hits.every((f) => f.endsWith('prompt_history.jsonl')),
        'Work delete: the marker only in prompt_history.jsonl (T13, §6.3)',
        hits.join(', '),
      );
    } catch (error) {
      row('FAIL', 'Work delete', codeOf(error));
    }
  }
}

/** U16: the `PATH` of an app started from Finder, as Work gets it, and whether `firecrawl` is found on it. */
async function finderPathRow(args: Args, service: ChatService): Promise<void> {
  const found = await findOnPath('firecrawl', args.finderPath ?? undefined);
  const folder = join(args.home, 'user-home', 'project');
  try {
    const { conversation } = await service.create({
      mode: 'work',
      text: 'Run the shell command `echo "$PATH"` and show me its output.',
      cwd: folder,
      modelId: args.model,
    });
    const view = await runTurn(
      service,
      conversation.id,
      async () => {},
      TURN_MS,
      (card) =>
        card.kind === 'permission' && card.command !== null && card.command.includes('PATH')
          ? { optionId: 'allow-once' }
          : { optionId: 'reject-once' },
    );
    const output = toolsOf(view)
      .map((t) => t.output ?? '')
      .join(' ');
    row(
      'INFO',
      'packaged build: the Work PATH of an app started from Finder (U16)',
      `firecrawl ${found === null ? 'not found' : 'found'}; homebrew ${output.includes('/opt/homebrew/bin') ? 'on' : 'off'} the printed PATH; outcome ${view.outcome ?? 'none'}`,
    );
  } catch (error) {
    row('FAIL', 'packaged build: the Work PATH (U16)', codeOf(error));
  }
}

await main();
