// A scripted stand-in for `mlx-agent … stdio`: speaks ACP (newline-delimited JSON-RPC) on stdio from a scenario file.
// Plain Node, no imports from the repo. Every rule below is copied from recorded real traffic (spike runs f-chat,
// f-u12, f-work): web calls raise a permission request in one of two shapes; any rejection ends the whole turn;
// `session/new` and `session/load` send a live `model_changed` before the reply and `server_status ready` after it.
//
// Usage: node fake-agent.mjs <scenario.json>
// The scenario's `recordTo` file receives one JSON line per received message, plus the argv and the env NAMES, each
// request the agent sends (`ask`), and each permission-gated tool it runs (`exec`) with the request that allowed it.

import { randomUUID } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { stripVTControlCharacters } from 'node:util';
import { crc32, deflateSync } from 'node:zlib';

const scenario = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const init = scenario.initialize ?? {};
const sess = scenario.session ?? {};
const faults = scenario.faults ?? {};
const prompts = scenario.prompts ?? [];
const models = init.models ?? [{ modelId: 'qwen3.5-4b', totalContextTokens: 32768 }];
const grokHome = process.env.GROK_HOME ?? '';
// Work sessions (no `agentProfile` in `_meta`) take the scenario's `work` section when it has one.
const workSection = scenario.work ?? null;
function profile(session) {
  return session?.work && workSection?.session ? { ...sess, ...workSection.session } : sess;
}
function scriptsOf(session) {
  return session?.work && workSection?.prompts ? workSection.prompts : prompts;
}
// Saved "always" grants, as the agent keeps them in its home: the exact command, per folder, and the request that gave it.
const grants = new Map();

function record(entry) {
  if (scenario.recordTo) appendFileSync(scenario.recordTo, `${JSON.stringify(entry)}\n`);
}

record({
  kind: 'start',
  pid: process.pid,
  argv: process.argv.slice(2),
  envNames: Object.keys(process.env).sort(),
  // Not a secret: the address the firecrawl CLI would use.
  firecrawlApiUrl: process.env.FIRECRAWL_API_URL ?? null,
  // Not secrets: the Claude Code switches, by name.
  claudeSwitches: Object.fromEntries(Object.entries(process.env).filter(([name]) => name.startsWith('GROK_CLAUDE_'))),
  cwd: process.cwd(),
});

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
// Lines held for one stdout write while non-null.
let batch = null;
function send(message) {
  const line = `${JSON.stringify({ jsonrpc: '2.0', ...message })}\n`;
  if (batch === null) process.stdout.write(line);
  else batch.push(line);
}
function reply(id, result) {
  send({ id, result });
}
function replyError(id, code, message, data) {
  send({ id, error: data === undefined ? { code, message } : { code, message, data } });
}

// As the agent reads `clientCapabilities._meta`; the scenario's `initialize.incrementalBashOutput` overrides it.
let incrementalBashOutput = false;

let nextAgentRequestId = 0;
const pendingAgentRequests = new Map();
function agentRequest(method, params) {
  const id = nextAgentRequestId++;
  record({
    kind: 'ask',
    id,
    method,
    toolCallId: params?.toolCall?.toolCallId ?? params?.toolCallId ?? null,
    hookAsk: params?._meta?.hookAsk?.hookName ?? null,
  });
  send({ id, method, params });
  return new Promise((resolve) => pendingAgentRequests.set(id, resolve));
}

const sessions = new Map();
let seq = 0;

function persistDir(session) {
  return join(grokHome, 'sessions', encodeURIComponent(session.cwd), session.id);
}

function persistLine(session, method, params) {
  if (!scenario.persist || grokHome === '') return;
  const dir = persistDir(session);
  mkdirSync(dir, { recursive: true });
  appendFileSync(
    join(dir, 'updates.jsonl'),
    `${JSON.stringify({ timestamp: Math.floor(Date.now() / 1000), method, params })}\n`,
  );
}

/** A fork's provenance in its stored summary: the agent never restamps a persisted kind, so later writes keep it. */
function storedFork(dir) {
  let stored;
  try {
    stored = JSON.parse(readFileSync(join(dir, 'summary.json'), 'utf8'));
  } catch {
    return {};
  }
  if (stored?.session_kind !== 'fork') return {};
  return { parent_session_id: stored.parent_session_id, forked_at: stored.forked_at, session_kind: 'fork' };
}

function persistSummary(session) {
  if (!scenario.persist || grokHome === '') return;
  const dir = persistDir(session);
  mkdirSync(dir, { recursive: true });
  const fork = session.parentId === undefined ? storedFork(dir) : {};
  const now = new Date().toISOString();
  writeFileSync(
    join(dir, 'summary.json'),
    JSON.stringify({
      info: { id: session.id, cwd: session.cwd },
      session_summary: session.title,
      created_at: session.createdAt,
      updated_at: now,
      last_active_at: now,
      num_messages: session.messages,
      num_chat_messages: session.messages,
      current_model_id: session.model,
      generated_title: session.title,
      title_is_manual: session.titleIsManual === true,
      ...fork,
      ...(session.parentId !== undefined ? { parent_session_id: session.parentId, session_kind: 'subagent' } : {}),
    }),
  );
}

// A `subagent_spawned` line's `agentAddress` is live only: the agent stores the line without it
// (`extensions/notification.rs` `without_live_agent_address`).
function update(session, sessionUpdate, fields, xai = false, extraMeta = {}) {
  if (typeof extraMeta.totalTokens === 'number') session.used = extraMeta.totalTokens;
  const params = {
    sessionId: session.id,
    update: { sessionUpdate, ...fields },
    _meta: { eventId: `${session.id}-${++seq}`, agentTimestampMs: Date.now(), ...extraMeta },
  };
  send({ method: xai ? '_x.ai/session_notification' : 'session/update', params });
  const stored = { ...params.update };
  delete stored.agentAddress;
  persistLine(session, xai ? '_x.ai/session/update' : 'session/update', { ...params, update: stored });
}

// As the agent's `send_model_auto_switched` (`agent_ops.rs`): gateway only, no `_meta`, never stored. `''` is no model.
function modelAutoSwitched(session, previous, next, reason) {
  send({
    method: '_x.ai/session_notification',
    params: {
      sessionId: session.id,
      update: { sessionUpdate: 'model_auto_switched', previous_model_id: previous, new_model_id: next, reason },
    },
  });
}

// As the agent's `announce_hook_run` (`hook_dispatch.rs`): the start of an awaited hook batch, sent at once with only an
// `agentTimestampMs` stamp (`send_xai_notification_transient`), never stored.
function hookRunStarted(session, fields) {
  send({
    method: '_x.ai/session_notification',
    params: {
      sessionId: session.id,
      update: { sessionUpdate: 'hook_run_started', count: 1, ...fields },
      _meta: { agentTimestampMs: Date.now() },
    },
  });
}

// A hook's name as the agent's user-facing copy shows it: without its `:{event}[i].hooks[j]` tail (`config.rs`).
function hookShown(name) {
  const colon = name.lastIndexOf(':');
  return colon > 0 && name.slice(colon + 1).includes('].hooks[') ? name.slice(0, colon) : name;
}

// A plain RGB PNG of the given sides, tinted by `tint`, as base64.
function pngBase64(width, height, tint) {
  const chunk = (type, data) => {
    const length = Buffer.alloc(4);
    length.writeUInt32BE(data.length);
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(body));
    return Buffer.concat([length, body, crc]);
  };
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 2;
  const rows = [];
  for (let y = 0; y < height; y += 1) {
    const row = Buffer.alloc(1 + width * 3);
    for (let x = 0; x < width; x += 1) {
      const line = y > 8 && y % 6 < 2 && x > 6 && x < width - 6;
      row[1 + x * 3] = line ? 140 : (240 + tint * 9) & 255;
      row[2 + x * 3] = line ? 140 : 244;
      row[3 + x * 3] = line ? 150 : (250 - tint * 23) & 255;
    }
    rows.push(row);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(Buffer.concat(rows))),
    chunk('IEND', Buffer.alloc(0)),
  ]).toString('base64');
}

// A `read_file` of a picture or a PDF (`acp_conversion.rs` `ReadFile`): the call, its title update, then its result with
// one image block per picture and the same bytes again in `rawOutput` (`ImageContent`, or `PdfPageImages` with page
// numbers, the file's page count and size). A PDF page is `image/jpeg` as the agent renders it; the fake's bytes are a
// PNG. `r.pages`: the model's page range, which the agent requires for a PDF of more than 10 pages (`pdf.rs`).
function pictureReadStep(session, r, toolCallId) {
  const tool = { version: 1, name: 'read_file', kind: 'read', namespace: 'grok_build', label: 'Read', read_only: true };
  const asked = r.pages === undefined ? {} : { pages: r.pages };
  update(session, 'tool_call', {
    toolCallId,
    title: 'read_file',
    rawInput: { target_file: r.path, ...asked },
    _meta: { 'x.ai/tool': tool },
  });
  update(session, 'tool_call_update', {
    toolCallId,
    kind: 'read',
    title: `Read \`${r.path}\``,
    locations: [{ path: r.path }],
    rawInput: { variant: 'ReadFile', target_file: r.path, ...asked },
    _meta: { 'x.ai/tool': { ...tool, input: { path: r.path } } },
  });
  const pages = r.pdf?.pages ?? null;
  const images =
    pages === null
      ? [{ data: pngBase64(r.width ?? 64, r.height ?? 48, 0), mimeType: 'image/png' }]
      : pages.map((page) => ({ data: pngBase64(60, 80, page), mimeType: 'image/jpeg' }));
  const rawOutput =
    pages === null
      ? { type: 'ReadFile', ImageContent: { data: images[0].data, mime_type: images[0].mimeType } }
      : {
          type: 'ReadFile',
          PdfPageImages: {
            pages: images.map((image, i) => ({ data: image.data, mime_type: image.mimeType, page_number: pages[i] })),
            total_pages: r.pdf.totalPages,
            file_size: r.pdf.fileSize,
          },
        };
  update(session, 'tool_call_update', {
    toolCallId,
    status: 'completed',
    content: images.map((image) => ({
      type: 'content',
      content: { type: 'image', data: image.data, mimeType: image.mimeType },
    })),
    rawOutput,
  });
}

// A call a `pre_tool_use` hook denies (`tool_calls.rs` gate, `hooks.rs` `block_tool_call`): the call, the batch's
// start, its `hook_execution` with the run `blocked`, the call failed "Hook denied: <reason>", then the `tool_outcome`
// note. `verdictFirst`: the note comes before the failed update, as when the agent's buffer holds the update back.
async function hookDeniedStep(session, d, toolCallId) {
  const meta = { 'x.ai/tool': xaiTool(d.toolName) };
  update(session, 'tool_call', { toolCallId, title: d.title ?? d.toolName, rawInput: d.rawInput ?? {}, _meta: meta });
  hookRunStarted(session, { event_name: 'pre_tool_use', tool_name: d.toolName });
  if (d.waitMs !== undefined) await sleep(d.waitMs);
  update(
    session,
    'hook_execution',
    {
      event_name: 'pre_tool_use',
      tool_name: d.toolName,
      runs: [
        {
          name: d.hook,
          status: { status: 'failed', error: `denied: ${d.reason}`, elapsed_ms: d.waitMs ?? 40, blocked: true },
        },
      ],
    },
    true,
  );
  const failed = () =>
    update(session, 'tool_call_update', {
      toolCallId,
      status: 'failed',
      content: [{ type: 'content', content: { type: 'text', text: `Hook denied: ${d.reason}` } }],
    });
  const verdict = () =>
    update(
      session,
      'hook_annotation',
      { message: `\`${d.toolName}\` blocked by ${hookShown(d.hook)}: ${d.reason}`, kind: 'tool_outcome' },
      true,
    );
  if (d.verdictFirst === true) {
    verdict();
    failed();
  } else {
    failed();
    verdict();
  }
}

/** A wake turn's prompt: the real agent stores it in updates.jsonl and never sends it live; `live` sends it too. */
function wakePrompt(session, text, live = false) {
  if (live) {
    const meta = { hideFromScrollback: true, promptIndex: session.messages };
    update(session, 'user_message_chunk', { content: { type: 'text', text }, _meta: meta });
    return;
  }
  persistLine(session, 'session/update', {
    sessionId: session.id,
    update: {
      sessionUpdate: 'user_message_chunk',
      content: { type: 'text', text },
      _meta: { hideFromScrollback: true },
    },
    _meta: { eventId: `${session.id}-${++seq}`, agentTimestampMs: Date.now() },
  });
}

// The goal loop (`/goal`, `goal.rs`, `goal_orchestrator.rs`). `goal_updated` is sent live on
// `_x.ai/session_notification` and stored at each state change (`emit_goal_updated`); a tick (`stored: false`, the
// agent's `emit_goal_updated_ephemeral`) is live only. A cleared goal is a line with an empty id.
const GOAL_PAUSED = ['user_paused', 'back_off_paused', 'no_progress_paused', 'infra_paused', 'blocked'];
function goalLine(session, stored = true, extra = {}) {
  const g = session.goal;
  const fields =
    g === null
      ? { goal_id: '', objective: '', status: 'cleared', phase: 'idle', tokens_used: 0, elapsed_ms: 0 }
      : {
          goal_id: g.id,
          objective: g.objective,
          status: g.status,
          phase: 'executing',
          ...(g.budget === undefined ? {} : { token_budget: g.budget }),
          tokens_used: g.tokens,
          elapsed_ms: g.elapsed,
        };
  const line = {
    ...fields,
    total_deliverables: 0,
    completed_deliverables: 0,
    total_worker_rounds: g?.rounds ?? 0,
    total_verify_rounds: 0,
    token_baseline: 0,
    finished_subagent_tokens: 0,
    ...(g?.pauseMessage === undefined ? {} : { pause_message: g.pauseMessage }),
    ...extra,
  };
  if (stored) return update(session, 'goal_updated', line, true);
  send({
    method: '_x.ai/session_notification',
    params: {
      sessionId: session.id,
      update: { sessionUpdate: 'goal_updated', ...line },
      _meta: { eventId: `${session.id}-${++seq}`, agentTimestampMs: Date.now() },
    },
  });
}
// The prompt the agent runs and stores for `/goal <objective>` and for a resume that runs (`goal_rules.md`).
function goalRules(objective, resume) {
  const body = `A goal has been set: ${objective}\n\nYou are working directly on this goal across multiple turns.`;
  return resume
    ? `<system-reminder>\n${body}\nContinue working now.\n</system-reminder>`
    : `<system-reminder>\n${body}\nStart now.\n</system-reminder>\n\n`;
}
// `/goal pause|clear|status|resume` as the built-in runs them (`slash_exec.rs`): the words it answers, after the state
// line it stores. A resume that runs is no host turn ({@link goalResume}).
function goalCommand(session, word) {
  const g = session.goal;
  if (word === 'pause') {
    if (g === null) return 'No goal is currently set.';
    if (g.status === 'active') {
      g.status = 'user_paused';
      goalLine(session, true, { last_event: 'goal_paused' });
      return 'Goal paused. Use /goal resume to continue.';
    }
    if (GOAL_PAUSED.includes(g.status)) return 'Goal is already paused.';
    return g.status === 'complete' ? 'Goal is already complete.' : 'Goal is budget-limited.';
  }
  if (word === 'clear') {
    session.goal = null;
    goalLine(session);
    return 'Goal cleared.';
  }
  if (word === 'resume') {
    if (g === null) return 'No goal set. Use /goal <objective> to start one.';
    return g.status === 'complete'
      ? 'Goal is already complete. Use /goal <objective> to start a new one.'
      : 'Goal is budget-limited. Use /goal clear, then /goal <objective>.';
  }
  return g === null
    ? 'No goal is currently set. Use /goal <objective> to start one.'
    : `Goal: ${g.objective}\nStatus: ${g.status}`;
}
// A resume of a paused goal (`resume_goal`): the active line, "Goal resumed." (live and stored, no `hostTurn`), then
// the goal rules as the stored prompt, and the turn runs on (here: it ends; `wakeRounds` left run after it).
function goalResume(id, session) {
  session.goal.status = 'active';
  delete session.goal.pauseMessage;
  goalLine(session, true, { last_event: 'goal_resumed' });
  update(session, 'agent_message_chunk', { content: { type: 'text', text: 'Goal resumed.' } });
  session.messages += 1;
  persistLine(session, 'session/update', {
    sessionId: session.id,
    update: {
      sessionUpdate: 'user_message_chunk',
      content: { type: 'text', text: goalRules(session.goal.objective, true) },
      _meta: { modelId: session.model, promptIndex: session.messages - 1 },
    },
    _meta: { eventId: `${session.id}-${++seq}`, agentTimestampMs: Date.now() },
  });
  persistSummary(session);
  reply(id, { stopReason: 'end_turn', _meta: { sessionId: session.id } });
  if (session.goal.wakeRounds > 0 || session.goal.after !== undefined) void goalWakes(session);
}
// One round of the goal inside the turn: `ticks` live-only lines while it works (`ms`), then its stored end.
async function goalRound(session, r) {
  const g = session.goal;
  const ms = r.ms ?? 20;
  const ticks = r.ticks ?? 0;
  const tokens = r.tokens ?? 1_200;
  const start = g.tokens;
  const since = g.elapsed;
  for (let i = 1; i <= ticks; i += 1) {
    await sleep(ms / (ticks + 1));
    g.tokens = start + Math.floor((tokens * i) / (ticks + 1));
    g.elapsed = since + Math.floor((ms * i) / (ticks + 1));
    goalLine(session, false, { last_event: 'worker_started' });
  }
  await sleep(ms / (ticks + 1));
  g.tokens = start + tokens;
  g.elapsed = since + ms;
  g.rounds += 1;
  goalLine(session, true, { last_event: 'worker_completed' });
}
// Rounds the agent starts by itself after a turn's end (`maybe_queue_goal_continuation`): each its own turn
// `goal-summary-<uuid>`, `gapMs` after the turn before; Stop of one pauses the goal. The last one ends the goal as
// `end` says (default: it stays active); `after` changes its state that long after the last round, with no turn.
async function goalWakes(session) {
  const g = session.goal;
  while (session.goal === g && g.status === 'active' && g.wakeRounds > 0) {
    if (g.gapMs !== undefined) await sleep(g.gapMs);
    if (session.goal !== g || g.status !== 'active' || session.turn) break;
    g.wakeRounds -= 1;
    g.round += 1;
    const cancelled = await wakeTurn(session, `goal-summary-${randomUUID()}`, {
      wakeText: 'Goal NOT complete — continue working.',
      wakeAnswer: `Round ${g.round}.`,
      ...(g.wakeEndOn === undefined ? { wakeEndAfterMs: g.wakeRoundMs ?? 40 } : { wakeEndOn: g.wakeEndOn }),
      beforeEnd: () => {
        if (session.goal !== g || g.status !== 'active') return;
        g.tokens += 1_200;
        g.elapsed += g.wakeRoundMs ?? 40;
        g.rounds += 1;
        if (g.wakeRounds === 0 && g.end !== undefined) {
          g.status = g.end;
          if (g.endMessage !== undefined) g.pauseMessage = g.endMessage;
        }
        goalLine(session, true, { last_event: 'worker_completed' });
      },
    });
    if (cancelled && session.goal === g && g.status === 'active') {
      g.status = 'user_paused';
      goalLine(session, true, { last_event: 'goal_paused' });
    }
  }
  if (g.after === undefined || session.goal !== g || g.status !== 'active') return;
  const after = g.after;
  delete g.after;
  await sleep(after.ms);
  if (session.goal !== g || g.status !== 'active') return;
  g.status = after.status;
  if (after.pauseMessage !== undefined) g.pauseMessage = after.pauseMessage;
  goalLine(session, true, { last_event: 'goal_paused' });
}

function xaiTool(name) {
  if (name === 'use_tool')
    return {
      version: 1,
      name: 'use_tool',
      kind: 'use_tool',
      namespace: 'grok_build',
      label: 'Use Tool',
      read_only: false,
    };
  if (name.includes('__'))
    return { version: 1, name, kind: 'other', namespace: 'mcp', label: 'Tool', read_only: false };
  return { version: 1, name, kind: name, namespace: 'grok_build', label: name, read_only: false };
}

/** Live-only kinds the real agent sends around model output and permission requests; the product ignores them. */
const liveExtras = sess.liveExtras !== false;
function xaiOnly(session, sessionUpdate, fields) {
  if (!liveExtras) return;
  send({
    method: '_x.ai/session_notification',
    params: { sessionId: session.id, update: { sessionUpdate, ...fields } },
  });
}
// The agent's built-ins with the gate each needs, from its source (`__test__/fixtures/agent-builtins.json`).
const BUILTINS = JSON.parse(readFileSync(new URL('../fixtures/agent-builtins.json', import.meta.url), 'utf8')).commands;
// The gates a Work session of the app opens (workflows, the goal tool, the scheduler, hooks, plugins, feedback), as
// agent 0.3.4 lists them for the app's Work environment; Chat: none.
const WORK_GATES = [
  'AlwaysOn',
  'Goal',
  'Scheduler',
  'WorkflowLaunches',
  'WorkflowManagement',
  'Hooks',
  'Plugins',
  'Feedback',
];
// The session's command list as `available_commands` builds it: built-ins whose gate is on, then the scenario's
// `commands.skills` (`_meta` scope, path, bareName, qualifiedName) and `commands.workflows` (`_meta.workflowSource`;
// `inHome`: its `workflowPath` is the agent home's `workflows/<name>.rhai`).
function commandRows(session) {
  const c = session.commands ?? profile(session).commands ?? {};
  const gates = c.gates ?? (session.work ? WORK_GATES : ['AlwaysOn']);
  // `/deep-research` is also the agent's built-in workflow of that name: its row carries the workflow's `_meta`.
  const rows = BUILTINS.filter((b) => gates.includes(b.gate)).map((b) => ({
    name: b.name,
    description: b.description,
    input: b.hint === null ? null : { hint: b.hint },
    ...(b.name === 'deep-research' ? { _meta: { workflowSource: 'builtin' } } : {}),
  }));
  for (const skill of c.skills ?? []) {
    rows.push({
      name: skill.name,
      description: skill.description ?? '',
      input: skill.hint === undefined ? null : { hint: skill.hint },
      _meta: {
        scope: 'user',
        path: `/skills/${skill.name}/SKILL.md`,
        bareName: skill.name,
        qualifiedName: skill.qualifiedName ?? `user:${skill.name}`,
      },
    });
  }
  for (const flow of c.workflows ?? []) {
    rows.push({
      name: flow.name,
      description: `Workflow: ${flow.description ?? ''}`,
      input: { hint: '[--agent-budget N] [--effort LEVEL] [args]' },
      _meta: {
        workflowSource: 'user',
        workflowPath: flow.inHome === true ? `${grokHome}/workflows/${flow.name}.rhai` : `/workflows/${flow.name}.md`,
      },
    });
  }
  return rows;
}
// As `send_available_commands_update`: live only, never stored; after a load, around tool calls and at a turn's end.
// After an answer the agent sends it to carry the answer's token count (`totalTokens`, `UsageMeta`).
function commandsUpdate(session, totalTokens) {
  if (!liveExtras) return;
  if (typeof totalTokens === 'number') session.used = totalTokens;
  send({
    method: 'session/update',
    params: {
      sessionId: session.id,
      update: {
        sessionUpdate: 'available_commands_update',
        availableCommands: commandRows(session),
        _meta: { tools: [] },
      },
      _meta: {
        eventId: `${session.id}-${++seq}`,
        agentTimestampMs: Date.now(),
        ...(typeof totalTokens === 'number' ? { totalTokens } : {}),
      },
    },
  });
}
// The built-in a typed prompt runs, as `resolve_human_intent` reads it (`slash_authority.rs` `parse_slash_prefix`):
// listed for the session, by name or alias, lower-cased. `/loop` and `/goal <objective>` (`GoalSet`, `turn.rs`) are
// model turns, so they are not one.
const GOAL_WORDS = ['', 'status', 'pause', 'resume', 'clear'];
function hostTurnCommand(session, text) {
  const trimmed = text.trim();
  if (!trimmed.startsWith('/')) return null;
  const rest = trimmed.slice(1);
  let end = 0;
  while (end < rest.length && rest[end].trim() !== '') end += 1;
  const name = rest.slice(0, end).toLowerCase();
  if (name === '' || name.includes('/') || name.includes('\\') || name === 'loop') return null;
  if (name === 'goal' && !GOAL_WORDS.includes(rest.slice(end).trim().toLowerCase())) return null;
  const builtin = BUILTINS.find((b) => b.name === name || b.aliases.includes(name));
  if (builtin === undefined || !commandRows(session).some((r) => r.name === builtin.name)) return null;
  return builtin.name;
}
function callDeltas(session, toolCallId, name, input) {
  xaiOnly(session, 'tool_call_delta_chunk', { tool_call_id: toolCallId, tool_index: 0, name });
  xaiOnly(session, 'tool_call_delta_chunk', { tool_index: 0, arguments_delta: JSON.stringify(input) });
  commandsUpdate(session);
}

const PERMISSION_OPTIONS = [
  { optionId: 'always-allow', name: 'always allow', kind: 'allow_always' },
  { optionId: 'allow-once', name: 'allow once', kind: 'allow_once' },
  { optionId: 'reject-once', name: 'reject once', kind: 'reject_once' },
];

function infoResult(session) {
  const sess = profile(session);
  const categories = [];
  if ((sess.skills ?? 1) > 0) categories.push({ label: 'Skills', tokens: 94, detail: `${sess.skills ?? 1} skill` });
  categories.push({ label: 'Workflows', tokens: 119, detail: '1 workflow' });
  if (session.mcpReady && (sess.mcpServers ?? 1) > 0) {
    categories.push({ label: 'MCP servers', tokens: 61, detail: `${sess.mcpServers ?? 1} server` });
  }
  return {
    result: {
      sessionId: session.id,
      cwd: session.cwd,
      agentName: sess.agentName ?? 'mlx-chat',
      model: session.model,
      apiBackend: sess.apiBackend ?? 'Messages',
      turns: session.messages,
      turnIndex: Math.max(0, session.messages - 1),
      context: contextOf(session, categories),
    },
  };
}

// The numbers of `ContextInfo` as the agent fills them (`build_session_info`): `used` is the last `totalTokens` it sent,
// `total` the model's window, `messageTokens` an estimate apart from both. The scenario's `contextInfo` replaces any.
function contextOf(session, categories) {
  const total = models.find((m) => m.modelId === session.model)?.totalContextTokens ?? 32768;
  const used = session.used ?? 432;
  return {
    used,
    total,
    systemPromptTokens: 432,
    toolDefinitionsCount: profile(session).toolDefinitionsCount ?? 2,
    toolDefinitionsTokens: 261,
    compactionCount: session.compactions ?? 0,
    turnCount: session.messages,
    toolCallCount: 0,
    messageCount: session.messages * 2 + 1,
    messageTokens: Math.max(0, used - 693),
    freeTokens: Math.max(0, total - used),
    usagePct: Math.min(100, Math.round((used / total) * 100)),
    autoCompactThresholdPercent: 85,
    usageCategories: categories,
    ...scenario.contextInfo,
  };
}

// A model's reasoning efforts as the agent reads them from the engine's `/v1/models` row (`reasoningEfforts`, each
// `{value, label?, default?}`) and lists them again: its serialized `ReasoningEffortOption` (`id` = `value` when the row
// names none).
function effortOptions(model) {
  return (model?.reasoningEfforts ?? []).map((e) => ({
    id: e.id ?? e.value,
    value: e.value,
    label: e.label ?? e.value,
    description: null,
    default: e.default === true,
  }));
}

function modelsBlock(current) {
  return {
    currentModelId: current,
    availableModels: models.map((m) => {
      const efforts = effortOptions(m);
      return {
        modelId: m.modelId,
        name: m.modelId,
        _meta: {
          totalContextTokens: m.totalContextTokens ?? 32768,
          ...(efforts.length === 0 ? {} : { supportsReasoningEffort: true, reasoningEfforts: efforts }),
        },
      };
    }),
  };
}

// The level the process last took (`set_current_reasoning_effort`): a session the process loads or makes starts at it
// when its model lists it, else at the model's default (`session_config.rs` `build_acp_config_options`).
let processEffort = null;

function effortStart(session) {
  const options = effortOptions(models.find((m) => m.modelId === session.model));
  if (options.length === 0) return null;
  if (processEffort !== null && options.some((o) => o.id === processEffort)) return processEffort;
  return (options.find((o) => o.default) ?? options[0]).id;
}

// `configOptions` of a session (`build_acp_config_options`): the model select, then the effort select only when the
// session's model lists efforts.
function configOptions(session) {
  const out = [
    {
      id: 'model',
      name: 'Model',
      category: 'model',
      type: 'select',
      currentValue: session.model,
      options: models.map((m) => ({ value: m.modelId, name: m.modelId })),
    },
  ];
  const options = effortOptions(models.find((m) => m.modelId === session.model));
  if (options.length > 0) {
    out.push({
      id: 'reasoning_effort',
      name: 'Reasoning Effort',
      category: 'thought_level',
      type: 'select',
      currentValue: session.effort ?? effortStart(session),
      options: options.map((o) => ({ value: o.id, name: o.label })),
    });
  }
  return out;
}

// After a set or a model switch (`notify_model_changed`, `notify_config_options`): a live `model_changed` and a live
// `config_option_update`, neither stored.
function configChanged(session) {
  send({
    method: '_x.ai/session_notification',
    params: {
      sessionId: session.id,
      update: {
        sessionUpdate: 'model_changed',
        model_id: session.model,
        ...(session.effort == null ? {} : { reasoning_effort: session.effort }),
      },
    },
  });
  send({
    method: 'session/update',
    params: {
      sessionId: session.id,
      update: { sessionUpdate: 'config_option_update', configOptions: configOptions(session) },
    },
  });
}

// `session/set_config_option` (`handlers/config_option.rs`): `reasoning_effort` by option id, checked against the
// session's model; the reply is the full `configOptions`. `session.refuseEffort` refuses every set as an old build
// with no such option would (a number: only that many first sets).
let effortRefusals = 0;
function setConfigOption(id, params) {
  record({ kind: 'set-config', params });
  const session = sessions.get(params?.sessionId);
  if (session === undefined) return replyError(id, -32602, 'Invalid params', 'unknown session id');
  const refuse =
    sess.refuseEffort === true || (typeof sess.refuseEffort === 'number' && effortRefusals++ < sess.refuseEffort);
  if (params.configId !== 'reasoning_effort' || refuse) {
    return replyError(id, -32602, 'Invalid params', `unknown config option: ${params.configId}`);
  }
  const options = effortOptions(models.find((m) => m.modelId === session.model));
  if (!options.some((o) => o.id === params.value)) {
    return replyError(id, -32602, 'Invalid params', 'unknown reasoning_effort value');
  }
  session.effort = params.value;
  processEffort = params.value;
  configChanged(session);
  return reply(id, { configOptions: configOptions(session) });
}

function scheduleReady(session, mcpServers) {
  const hasWeb = Array.isArray(mcpServers) && mcpServers.some((s) => s && s.name === 'mlx_web');
  const after = sess.mcpReadyAfterMs === undefined ? 70 : sess.mcpReadyAfterMs;
  if (!hasWeb || after === null) return;
  setTimeout(() => {
    session.mcpReady = true;
    send({
      method: '_x.ai/mcp/server_status',
      params: {
        sessionId: session.id,
        name: 'mlx_web',
        source: 'local',
        status: 'ready',
        reason: 'initialized',
        tools: null,
      },
    });
  }, after);
}

function requestedModel(params) {
  const id = params?._meta?.modelId;
  return models.some((m) => m.modelId === id) ? id : models[0].modelId;
}

// As the real agent with the app's guard plugin among `pluginDirs`: its `PreToolUse` hook answers "ask", so a shell
// request carries `_meta.hookAsk` and the hook's line at the end of its title.
function guardHook(dirs) {
  if (!Array.isArray(dirs)) return null;
  for (const dir of dirs) {
    let manifest;
    try {
      manifest = JSON.parse(readFileSync(join(dir, 'plugin.json'), 'utf8'));
    } catch {
      continue;
    }
    const group = manifest?.hooks?.hooks?.PreToolUse?.[0];
    const command = group?.hooks?.[0]?.command;
    if (
      typeof group?.matcher === 'string' &&
      group.matcher.split('|').includes('Bash') &&
      typeof command === 'string' &&
      command.includes('"permissionDecision":"ask"')
    )
      return `plugin/${manifest.name}/plugin:pre_tool_use[0].hooks[0]`;
  }
  return null;
}

/** A loaded session's prompt index, as the agent restores it: one past the last stored prompt, rewinds applied. */
function storedPromptIndex(id, cwd) {
  if (!scenario.persist || grokHome === '') return 0;
  let text;
  try {
    text = readFileSync(join(persistDir({ id, cwd }), 'updates.jsonl'), 'utf8');
  } catch {
    return 0;
  }
  let index = 0;
  for (const line of text.split('\n')) {
    if (line === '') continue;
    let update;
    try {
      update = JSON.parse(line).params?.update;
    } catch {
      continue;
    }
    if (update?.sessionUpdate === 'rewind_marker') index = update.target_prompt_index;
    else if (update?.sessionUpdate === 'user_message_chunk' && typeof update._meta?.promptIndex === 'number')
      index = update._meta.promptIndex + 1;
  }
  return index;
}

/**
 * The schedules a stored session holds, as the agent restores them at load (`scheduler/actor.rs`
 * `announce_existing_tasks`): each `scheduled_task_created` not followed by a `scheduled_task_deleted` of another reason
 * than `shutdown` (the process stopping), with the words of its last created line.
 */
function storedSchedules(id, cwd) {
  const held = new Map();
  if (!scenario.persist || grokHome === '') return held;
  let text;
  try {
    text = readFileSync(join(persistDir({ id, cwd }), 'updates.jsonl'), 'utf8');
  } catch {
    return held;
  }
  for (const line of text.split('\n')) {
    if (line === '') continue;
    let update;
    try {
      update = JSON.parse(line).params?.update;
    } catch {
      continue;
    }
    if (typeof update?.task_id !== 'string') continue;
    if (update.sessionUpdate === 'scheduled_task_created') held.set(update.task_id, update);
    else if (update.sessionUpdate === 'scheduled_task_deleted' && update.reason !== 'shutdown')
      held.delete(update.task_id);
  }
  return held;
}

// The `slowLoad` step's marker: this load waits its time, once.
async function slowLoad(id, cwd) {
  if (!scenario.persist || grokHome === '' || typeof id !== 'string' || typeof cwd !== 'string') return;
  let real = cwd;
  try {
    real = realpathSync(cwd);
  } catch {}
  const file = join(persistDir({ id, cwd: real }), 'slow_load.json');
  let ms = 0;
  try {
    ms = JSON.parse(readFileSync(file, 'utf8')).ms;
    rmSync(file, { force: true });
  } catch {
    return;
  }
  if (typeof ms === 'number' && ms > 0) await sleep(ms);
}

// After a `session/load` of a session this process did not hold, the agent's scheduler restores the stored schedules and
// announces each again with `scheduled_task_created`, stored and live.
function announceSchedules(session) {
  if (session.schedules !== undefined) return;
  session.schedules = new Set();
  for (const [taskId, created] of storedSchedules(session.id, session.cwd)) {
    session.schedules.add(taskId);
    xaiUpdate(session, '_x.ai/scheduled_task_created', 'scheduled_task_created', {
      task_id: taskId,
      prompt: created.prompt ?? '',
      human_schedule: created.human_schedule ?? '',
      next_fire_at: created.next_fire_at ?? null,
    });
  }
}

function newSession(id, params, loaded) {
  let cwd = params.cwd;
  try {
    cwd = realpathSync(params.cwd);
  } catch {}
  const existing = sessions.get(id);
  const session = existing ?? {
    id,
    cwd,
    model: requestedModel(params),
    title: null,
    titled: loaded,
    createdAt: new Date().toISOString(),
    messages: loaded ? storedPromptIndex(id, cwd) : 0,
    filePoints: loaded ? storedPoints(id, cwd) : new Map(),
    mcpReady: false,
    turn: null,
  };
  session.work = !(params?._meta && typeof params._meta.agentProfile === 'object');
  if (profile(session).currentModelId) session.model = profile(session).currentModelId;
  session.mcpReady = false;
  session.noReplay = params?._meta?.noReplay === true;
  session.guard = guardHook(params?._meta?.pluginDirs);
  sessions.set(id, session);
  return session;
}

async function handle(message) {
  const { id, method, params } = message;
  if (method === undefined) {
    const resolve = pendingAgentRequests.get(id);
    pendingAgentRequests.delete(id);
    resolve?.(message.result ?? { outcome: { outcome: 'cancelled' } });
    return;
  }
  if (faults.hangOn === method) return;
  const delay = faults.delayMs?.[method];
  if (delay !== undefined) await sleep(delay);
  switch (method) {
    case 'initialize':
      incrementalBashOutput =
        init.incrementalBashOutput ?? params?.clientCapabilities?._meta?.['x.ai/incrementalBashOutput'] === true;
      reply(id, {
        protocolVersion: init.protocolVersion ?? 1,
        agentCapabilities: {
          loadSession: init.loadSession ?? true,
          promptCapabilities: { image: false, audio: false, embeddedContext: init.embeddedContext ?? true },
          mcpCapabilities: { http: true, sse: true },
          sessionCapabilities: { list: {}, resume: {}, close: {} },
        },
        authMethods: [{ id: 'xai.api_key', name: 'xai.api_key' }],
        _meta: {
          'x.ai/pluginDirs': init.pluginDirs ?? true,
          ...(init.workflowControl === false ? {} : { 'x.ai/workflowControl': true }),
          ...(init.agentVersion === null ? {} : { agentVersion: init.agentVersion ?? '1.0.45' }),
          modelState: { currentModelId: models[0]?.modelId ?? 'mlx-local', ...modelsBlock(models[0]?.modelId) },
        },
      });
      return;
    case 'session/new': {
      const session = newSession(randomUUID(), params, false);
      // Before the reply, as the agent's session start: the client does not know the session yet.
      commandsUpdate(session);
      // `sessionStart`: the agent's `session_start` hook batch (`DispatchSessionStartHook`, run off the command loop),
      // stored: one `hook_execution` with `runs`, then a `hook_annotation` for each of `notes` (a hook's own
      // `systemMessage`, `send_hook_execution`), before the reply, or just after it with `afterReply`.
      const start = profile(session).sessionStart;
      const startHook = () => {
        update(session, 'hook_execution', { event_name: 'session_start', runs: start.runs ?? [] }, true);
        for (const message of start.notes ?? []) update(session, 'hook_annotation', { message }, true);
      };
      if (start !== undefined && start.afterReply !== true) startHook();
      // Live only: the real agent never stores this one in updates.jsonl.
      if (sess.modelChangedOnNewAndLoad !== false) {
        send({
          method: '_x.ai/session_notification',
          params: {
            sessionId: session.id,
            update: { sessionUpdate: 'model_changed', model_id: session.model },
            _meta: { eventId: `${session.id}-${++seq}`, agentTimestampMs: Date.now() },
          },
        });
      }
      persistSummary(session);
      session.effort = effortStart(session);
      reply(id, {
        sessionId: session.id,
        models: modelsBlock(session.model),
        configOptions: configOptions(session),
        _meta: {
          currentWorkingDirectory: profile(session).cwdOverride ?? session.cwd,
          'x.ai/sessionDetail': {
            sessionId: session.id,
            kind: 'build',
            cwd: session.cwd,
            currentModelId: session.model,
          },
        },
      });
      if (start !== undefined && start.afterReply === true) startHook();
      scheduleReady(session, params.mcpServers);
      return;
    }
    case 'session/load': {
      await slowLoad(params.sessionId, params.cwd);
      const session = newSession(params.sessionId, params, true);
      commandsUpdate(session);
      // `modelGoneOnLoad`: the stored model is gone (`session_setup.rs`). With a `next` the session takes it; with
      // none every later prompt is blocked with the agent's words.
      const gone = sess.modelGoneOnLoad;
      if (gone !== undefined) {
        modelAutoSwitched(session, session.model, gone.next ?? '', gone.reason);
        if (gone.next) session.model = gone.next;
        else
          session.noModel =
            'Your previous model is no longer available and could not be switched to a compatible model. Please start a new session.';
      }
      if (sess.replayOnLoad && !session.noReplay) {
        update(session, 'user_message_chunk', { content: { type: 'text', text: 'replayed prompt' } }, false, {
          isReplay: true,
        });
      }
      if (sess.modelChangedOnNewAndLoad !== false) {
        send({
          method: '_x.ai/session_notification',
          params: { sessionId: session.id, update: { sessionUpdate: 'model_changed', model_id: session.model } },
        });
      }
      if (session.effort === undefined) session.effort = effortStart(session);
      reply(id, {
        models: modelsBlock(session.model),
        configOptions: configOptions(session),
        _meta: {
          sessionId: session.id,
          'x.ai/sessionDetail': {
            sessionId: session.id,
            kind: 'build',
            cwd: profile(session).cwdOverride ?? session.cwd,
            currentModelId: session.model,
          },
        },
      });
      announceSchedules(session);
      scheduleReady(session, params.mcpServers);
      return;
    }
    case '_x.ai/session/info': {
      const session = sessions.get(params?.sessionId);
      if (sess.infoMissing || session === undefined) return replyError(id, -32601, 'Method not found');
      // `infoLateMs`: each read after the guard's first one answers this late.
      if (typeof sess.infoLateMs === 'number' && session.infoRead === true) await sleep(sess.infoLateMs);
      session.infoRead = true;
      return reply(id, infoResult(session));
    }
    case 'session/set_model': {
      const session = sessions.get(params?.sessionId);
      if (session === undefined || !models.some((m) => m.modelId === params.modelId)) {
        return replyError(id, -32602, 'Invalid params', { message: 'unknown model id' });
      }
      session.model = params.modelId;
      // `SwitchEffort::Preserve`: the level stays when the new model lists it, else the model's default.
      const kept = effortOptions(models.find((m) => m.modelId === session.model));
      if (!kept.some((o) => o.id === session.effort))
        session.effort = (kept.find((o) => o.default) ?? kept[0])?.id ?? null;
      persistSummary(session);
      configChanged(session);
      return reply(id, { _meta: { model: { Ok: params.modelId } } });
    }
    case '_x.ai/session/rename': {
      const session = sessions.get(params?.sessionId);
      if (session !== undefined) {
        session.title = params.title;
        session.titleIsManual = true;
        persistSummary(session);
      }
      return reply(id, { success: true });
    }
    case '_x.ai/session/delete': {
      const session = sessions.get(params?.sessionId) ?? { id: params?.sessionId, cwd: params?.cwd };
      if (scenario.persist && grokHome !== '' && typeof session.id === 'string')
        rmSync(persistDir(session), { recursive: true, force: true });
      sessions.delete(params?.sessionId);
      return reply(id, { success: true });
    }
    case 'session/prompt':
      return runPrompt(id, params);
    case 'session/set_config_option':
      return setConfigOption(id, params);
    case 'session/cancel': {
      if (faults.ignoreCancel) return;
      const session = sessions.get(params?.sessionId);
      // As the agent: the turn task is dropped at once, and a later answer to its open request goes nowhere.
      if (faults.abortOnCancel && session?.turn?.asking)
        return endCancelled(session.turn.rpcId, session, 'MidTurnAbort');
      if (session?.turn) session.turn.cancelled = true;
      else if (session?.compacting) session.compacting.cancel();
      else session?.wakeTurn?.cancel();
      session?.turn?.wake?.();
      return;
    }
    case '_x.ai/workflows/stop':
      return stopRun(id, params);
    case '_x.ai/compact_conversation':
      return compactNow(id, params);
    case '_x.ai/subagent/list_running':
      return listRunning(id, params);
    case '_x.ai/task/kill':
      return killTask(id, params);
    case '_x.ai/subagent/cancel':
      return cancelHelper(id, params);
    case '_x.ai/subagent/message':
      return messageHelper(id, params);
    case '_x.ai/scheduler/delete':
      return deleteSchedule(id, params);
    case '_x.ai/rewind/execute':
      return rewind(id, params);
    case '_x.ai/session/fork':
      return fork(id, params);
    case '_x.ai/recap':
      return recap(id, params);
    case '_x.ai/interject':
      return interject(id, params);
    case '_x.ai/commands/list': {
      // `handle_commands_list`: the session's rows (`{commands, tools}`, no envelope); `kind: "chat"` is the agent's
      // own chat product, which the app never asks for.
      record({ kind: 'commands-list', params });
      const session = sessions.get(params?.sessionId);
      if (session === undefined)
        return replyError(id, -32600, 'Invalid request', `unknown session id: ${params?.sessionId}`);
      return reply(id, { commands: commandRows(session), tools: [] });
    }
    default:
      if (id !== undefined) replyError(id, -32601, 'Method not found');
  }
}

// `_x.ai/compact_conversation` (Compact now), as the agent runs it with no turn: no `auto_compact_started`; on success
// one stored `auto_compact_completed` (no elapsed time, no preview), then `{}`; a failure or a `session/cancel` is a
// JSON-RPC error whose `data.kind` names it. The scenario's `compactNow` holds it until `holdOn` exists or `ms`
// passed, `fail` makes it fail, and `cancelled` ends it cancelled with no `session/cancel` from the app (another cancel
// of the session). A session that is not loaded never answers the session task. `compactionCount` of
// `_x.ai/session/info` grows as the run starts (`compaction.rs` `run_compact_inner` → `record_compaction`), so a stopped
// or failed run counts too.
async function compactNow(id, params) {
  const session = sessions.get(params?.sessionId);
  if (session === undefined) return replyError(id, -32603, 'Internal error', 'session failed to respond');
  const c = scenario.compactNow ?? {};
  let cancelled = c.cancelled === true;
  session.compactions = (session.compactions ?? 0) + 1;
  session.compacting = {
    cancel: () => {
      cancelled = true;
    },
  };
  if (typeof c.holdOn === 'string') while (!existsSync(c.holdOn) && !cancelled) await sleep(10);
  else if (typeof c.ms === 'number') {
    const until = Date.now() + c.ms;
    while (Date.now() < until && !cancelled) await sleep(10);
  }
  session.compacting = null;
  if (cancelled) {
    return replyError(id, -32603, 'Internal error', { kind: 'compact_cancelled', message: 'compact cancelled' });
  }
  if (typeof c.fail === 'string') {
    return replyError(id, -32603, 'Internal error', { kind: 'compact_failed', message: c.fail });
  }
  update(
    session,
    'auto_compact_completed',
    {
      tokens_before: c.tokensBefore ?? 41_000,
      tokens_after: c.tokensAfter ?? 9_000,
      elapsed_ms: null,
      summary_preview: null,
    },
    true,
  );
  session.used = c.tokensAfter ?? 9_000;
  reply(id, {});
}

// `_x.ai/recap` (`extensions/recap.rs`, `acp_session_impl/recap.rs`): the ack comes at once, `{result: {ok: true}}`, or
// `{result: {ok: true, disabled: true}}` with recaps off (then nothing more). Then one stored xAI line:
// `session_recap {summary, auto}`, or (a manual one only) `session_recap_unavailable` when the session has no finished
// prompt, another recap runs, or a typed prompt came while it was written. The scenario's `recap` turns it off
// (`disabled`), sets the words (`summary`), writes none (`unavailable`), holds the line until `holdOn` exists or for
// `ms`, sends no line at all (`silent`), never acks (`noAck`), or answers a JSON-RPC error (`rpcError`).
async function recap(id, params) {
  record({ kind: 'recap', params });
  const r = scenario.recap ?? {};
  const session = sessions.get(params?.sessionId);
  if (session === undefined) return replyError(id, -32602, 'Invalid params', `session not found: ${params?.sessionId}`);
  if (r.rpcError === true) return replyError(id, -32603, 'Internal error', 'recap failed');
  if (r.disabled === true) return reply(id, { result: { ok: true, disabled: true } });
  if (r.noAck !== true) reply(id, { result: { ok: true } });
  const auto = params?.auto === true;
  const none = () => {
    if (!auto) update(session, 'session_recap_unavailable', {}, true);
  };
  if (session.messages === 0 || session.recapping === true) return none();
  const epoch = session.recapEpoch ?? 0;
  session.recapping = true;
  if (typeof r.holdOn === 'string') while (!existsSync(r.holdOn)) await sleep(10);
  if (typeof r.ms === 'number') await sleep(r.ms);
  session.recapping = false;
  if (r.silent === true) return;
  if (r.unavailable === true || (session.recapEpoch ?? 0) !== epoch) return none();
  update(session, 'session_recap', { summary: r.summary ?? 'The fix is in and its test passes.', auto }, true);
}

// The agent's file rewind points (`xai-grok-workspace` `session/file_state.rs`) per session: for each prompt index, the
// files its own edit tool wrote, in the order first written. A command's or a helper's change is never in them, and a
// fork's copy starts with none. Kept beside `updates.jsonl` when the scenario persists, so a load after an agent
// restart has them again, as the agent's `rewind_points.jsonl` (`session/storage/jsonl/mod.rs`).
function fileWritten(session, path) {
  const index = Math.max(0, session.messages - 1);
  session.filePoints ??= new Map();
  const point = session.filePoints.get(index) ?? [];
  if (!point.includes(path)) point.push(path);
  session.filePoints.set(index, point);
  savePoints(session);
}

function savePoints(session) {
  if (!scenario.persist || grokHome === '') return;
  const dir = persistDir(session);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'rewind_points.json'), JSON.stringify([...(session.filePoints ?? new Map())]));
}

/** The file points a stored session holds, as the agent reads them back at its first rewind after a load. */
function storedPoints(id, cwd) {
  if (!scenario.persist || grokHome === '') return new Map();
  try {
    return new Map(JSON.parse(readFileSync(join(persistDir({ id, cwd }), 'rewind_points.json'), 'utf8')));
  } catch {
    return new Map();
  }
}

/** The files a restore to `target` puts back: each one written in a prompt at or past it, once. */
function filesFrom(session, target) {
  const out = [];
  const points = [...(session.filePoints ?? new Map())].sort((a, b) => a[0] - b[0]);
  for (const [index, paths] of points)
    if (index >= target) for (const path of paths) if (!out.includes(path)) out.push(path);
  return out;
}

/** A restore drops the points at or past `target`; a conversation-only rewind folds them into the one before it. */
function dropPoints(session, target, fold) {
  const points = session.filePoints ?? new Map();
  const folded = fold && target > 0 ? filesFrom(session, target) : [];
  for (const index of points.keys()) if (index >= target) points.delete(index);
  if (folded.length > 0) {
    const before = points.get(target - 1) ?? [];
    points.set(target - 1, [...before, ...folded.filter((path) => !before.includes(path))]);
  }
  savePoints(session);
}

// `_x.ai/rewind/execute` (`extensions/rewind.rs`, `acp_session_impl/rewind.rs`): a bare `RewindResponse`. `force`
// absent or false is a dry run (`success: false`, nothing changes): for `all` and `files_only` it names the files of the
// rewind points at or past the target, `clean_files` and `conflicts` (the scenario's `rewind.changed`, path → conflict
// type, says which changed outside the agent; `changedFile`, a JSON file of that map read at each dry run, takes its
// place while it exists; the error is set when one did). A commit with `all` or `files_only` answers those files as
// `reverted_files` (but the scenario's `unwritable` paths: the agent skips a file it cannot write and still answers
// success) and drops the points; one that keeps the conversation's prompts `0..target-1`
// stores a `rewind_marker` (disk only, never sent live) and answers `success: true` with the target's prompt text.
// Outside `files_only`, a target at or past the current prompt index answers `success: false` with an error. The
// scenario's `rewind` holds a commit until `holdOn` exists or for `ms`, refuses it (`refuse`), or answers a JSON-RPC
// error (`rpcError`); `previewMs` holds a dry run. A session that is not loaded is `resource_not_found`.
async function rewind(id, params) {
  const session = sessions.get(params?.sessionId);
  const target = params?.targetPromptIndex;
  const force = params?.force === true;
  const mode = params?.mode ?? 'all';
  record({
    kind: 'rewind',
    target,
    force,
    mode,
    hasMode: params?.mode !== undefined,
    hasForce: params?.force !== undefined,
  });
  if (session === undefined) return replyError(id, -32002, 'Resource not found', { message: 'session not found' });
  const r = scenario.rewind ?? {};
  const answer = (success, fields = {}) =>
    reply(id, {
      success,
      target_prompt_index: target,
      mode,
      reverted_files: [],
      clean_files: [],
      conflicts: [],
      prompt_text: null,
      error: null,
      ...fields,
    });
  const outOfRange = mode !== 'files_only' && target >= session.messages;
  const files = mode === 'conversation_only' ? [] : filesFrom(session, target);
  if (!force) {
    if (typeof r.previewMs === 'number') await sleep(r.previewMs);
    if (outOfRange) return answer(false, { error: `Cannot rewind to prompt #${target}` });
    const changed =
      typeof r.changedFile === 'string' && existsSync(r.changedFile)
        ? JSON.parse(readFileSync(r.changedFile, 'utf8'))
        : (r.changed ?? {});
    const conflicts = files
      .filter((path) => typeof changed[path] === 'string')
      .map((path) => ({ path, conflict_type: changed[path] }));
    return answer(false, {
      clean_files: files.filter((path) => typeof changed[path] !== 'string'),
      conflicts,
      error: conflicts.length > 0 ? 'External modifications detected. Confirm to revert anyway.' : null,
    });
  }
  if (typeof r.holdOn === 'string') while (!existsSync(r.holdOn)) await sleep(10);
  if (typeof r.ms === 'number') await sleep(r.ms);
  if (r.rpcError === true) return replyError(id, -32603, 'Internal error', 'Rewind failed');
  if (r.refuse === true || outOfRange) {
    return answer(false, { error: `Cannot rewind to prompt #${target}` });
  }
  dropPoints(session, target, mode === 'conversation_only');
  const reverted = files.filter((path) => !(r.unwritable ?? []).includes(path));
  if (mode === 'files_only') return answer(true, { reverted_files: reverted });
  const text = session.promptTexts?.[target] ?? null;
  session.messages = target;
  session.promptTexts = (session.promptTexts ?? []).slice(0, target);
  persistLine(session, '_x.ai/session/update', {
    sessionId: session.id,
    update: { sessionUpdate: 'rewind_marker', target_prompt_index: target, created_at: new Date().toISOString() },
  });
  answer(true, { prompt_text: text, reverted_files: reverted });
}

// `_x.ai/session/fork` (`session/fork.rs`, `extensions/session_admin.rs`): the session copied on disk under a new id and
// not loaded; a bare camelCase `ForkSessionResponse`. Needs no loaded session. `targetPromptIndex` keeps prompts
// `0..target` and their answers (inclusive; rewinds applied first); without it every line is copied. The copy's summary
// keeps the source's title and adds `session_kind: "fork"`, `parent_session_id` and `forked_at`. The scenario's `fork`
// holds it until `holdOn` exists or for `ms`, or answers a JSON-RPC error (`rpcError`).
async function fork(id, params) {
  record({ kind: 'fork', params });
  const f = scenario.fork ?? {};
  if (typeof f.holdOn === 'string') while (!existsSync(f.holdOn)) await sleep(10);
  if (typeof f.ms === 'number') await sleep(f.ms);
  if (f.rpcError === true) return replyError(id, -32603, 'Internal error', 'fork failed');
  const newId = randomUUID();
  let copied = 0;
  if (scenario.persist && grokHome !== '') {
    const source = persistDir({ id: params.sourceSessionId, cwd: params.sourceCwd });
    let summary;
    try {
      summary = JSON.parse(readFileSync(join(source, 'summary.json'), 'utf8'));
    } catch {
      return replyError(id, -32603, 'Internal error', 'No such file or directory (os error 2)');
    }
    let lines = [];
    try {
      lines = readFileSync(join(source, 'updates.jsonl'), 'utf8')
        .split('\n')
        .filter((l) => l !== '');
    } catch {}
    if (typeof params.targetPromptIndex === 'number') lines = forkCut(lines, params.targetPromptIndex);
    const target = persistDir({ id: newId, cwd: params.newCwd });
    mkdirSync(target, { recursive: true });
    const moved = lines.map((line) => {
      const parsed = JSON.parse(line);
      if (parsed.params?.sessionId !== undefined) parsed.params.sessionId = newId;
      return JSON.stringify(parsed);
    });
    writeFileSync(join(target, 'updates.jsonl'), moved.length === 0 ? '' : `${moved.join('\n')}\n`);
    copied = moved.length;
    const now = new Date().toISOString();
    writeFileSync(
      join(target, 'summary.json'),
      JSON.stringify({
        ...summary,
        info: { id: newId, cwd: params.newCwd },
        created_at: now,
        updated_at: now,
        parent_session_id: params.sourceSessionId,
        forked_at: now,
        session_kind: 'fork',
      }),
    );
  }
  reply(id, {
    newSessionId: newId,
    chatMessagesCopied: 0,
    updatesCopied: copied,
    planStateCopied: false,
    newCwd: params.newCwd,
    parentSessionId: params.sourceSessionId,
  });
}

/** The stored lines a fork to `target` keeps: rewinds applied, then cut at the first prompt past `target`. */
function forkCut(lines, target) {
  const promptOf = (line) => {
    try {
      const update = JSON.parse(line).params?.update;
      if (update?.sessionUpdate === 'rewind_marker') return { rewind: update.target_prompt_index };
      if (update?.sessionUpdate === 'user_message_chunk' && typeof update._meta?.promptIndex === 'number')
        return { prompt: update._meta.promptIndex };
    } catch {}
    return {};
  };
  const kept = [];
  for (const line of lines) {
    const step = promptOf(line);
    if (step.rewind === undefined) {
      kept.push(line);
      continue;
    }
    const at = kept.findIndex((l) => (promptOf(l).prompt ?? -1) >= step.rewind);
    if (at !== -1) kept.length = at;
  }
  const cut = kept.findIndex((l) => (promptOf(l).prompt ?? -1) > target);
  return cut === -1 ? kept : kept.slice(0, cut);
}

// `_x.ai/scheduler/delete` (`extensions/task.rs`): `{result: {taskId, deleted}}`, then `scheduled_task_deleted` with
// reason `deleted` for a schedule it held; the scenario's `faults.scheduleDeleteReply: 'error'` answers the error
// envelope, `'gone'` answers `deleted: false` as if the schedule had ended with no line the client read.
function deleteSchedule(id, params) {
  const session = sessions.get(params?.sessionId);
  const taskId = params?.taskId;
  record({ kind: 'schedule-delete', taskId });
  if (session === undefined || faults.scheduleDeleteReply === 'error')
    return reply(id, { result: null, error: 'session not found' });
  if (faults.scheduleDeleteReply === 'gone') {
    session.schedules?.delete(taskId);
    return reply(id, { result: { taskId, deleted: false } });
  }
  const held = session.schedules?.has(taskId) === true;
  reply(id, { result: { taskId, deleted: held } });
  if (!held) return;
  session.schedules.delete(taskId);
  xaiUpdate(session, '_x.ai/scheduled_task_deleted', 'scheduled_task_deleted', { task_id: taskId, reason: 'deleted' });
}

// `_x.ai/interject` (`extensions/interject.rs`, `run_loop.rs`): `{result: {status: "queued"}}` at once (the scenario's
// `faults.interjectError` answers `session not found`), then the live `_x.ai/session/interjection` broadcast
// (`interjection.rs` `broadcast_interjection`, never stored). Into a running turn the text is buffered and stored only
// when a `readSteers` step drains it, as the agent stores it (`_meta.interjection`, the wrapped text of
// `xai-interjection-core` `format_interjection`, the typed text in `content._meta.displayText`; no live copy); one left
// at the turn's end, or sent with no turn, runs as its own turn (`interject-fallback-<uuid>`, its prompt stored only).
// `session/cancel` drops what was not read.
function interject(id, params) {
  const session = sessions.get(params?.sessionId);
  const text = typeof params?.text === 'string' ? params.text : '';
  const interjectionId = typeof params?.interjectionId === 'string' ? params.interjectionId : null;
  record({ kind: 'interject', text, interjectionId });
  if (session === undefined || faults.interjectError === true) {
    return replyError(id, -32602, 'Invalid params', `session not found: ${params?.sessionId}`);
  }
  reply(id, { result: { status: 'queued' } });
  send({
    method: '_x.ai/session/interjection',
    params: { sessionId: session.id, text, ...(interjectionId === null ? {} : { interjectionId }) },
  });
  if (session.turn !== null && session.turn !== undefined) {
    session.turn.steers ??= [];
    session.turn.steers.push(text);
    return;
  }
  // Into the agent's own running turn: the model reads it before that turn ends.
  if (session.wakeTurn !== null && session.wakeTurn !== undefined) {
    session.wakeTurn.steers.push(text);
    return;
  }
  void steerTurns(session, [text]);
}

/** A `readSteers` step: the buffered steers reach the model and are stored. */
function readSteers(session) {
  const steers = session.turn.steers ?? [];
  session.turn.steers = [];
  for (const text of steers) {
    persistLine(session, 'session/update', {
      sessionId: session.id,
      update: {
        sessionUpdate: 'user_message_chunk',
        content: {
          type: 'text',
          text: `The user sent a message while you were working:\n<user_query>\n${text}\n</user_query>\nIf the user is asking for a response, address the user first. After replying, complete any unfinished tasks from previous turns.`,
          _meta: { displayText: text },
        },
        _meta: { modelId: session.model, interjection: true },
      },
      _meta: { eventId: `${session.id}-${++seq}`, agentTimestampMs: Date.now() },
    });
    record({ kind: 'steer-read', text });
  }
}

/** Steers the turn never read: each runs as its own turn after it, as the agent's stranded-interjection flush does. */
async function steerTurns(session, texts) {
  for (const text of texts) {
    while (session.turn !== null && session.turn !== undefined) await sleep(10);
    const promptId = `interject-fallback-${randomUUID()}`;
    session.messages += 1;
    persistLine(session, 'session/update', {
      sessionId: session.id,
      update: {
        sessionUpdate: 'user_message_chunk',
        content: { type: 'text', text },
        _meta: { modelId: session.model, promptIndex: session.messages - 1 },
      },
      _meta: { eventId: `${session.id}-${++seq}`, agentTimestampMs: Date.now() },
    });
    update(session, 'agent_message_chunk', { content: { type: 'text', text: 'Read your message.' } });
    // Recorded before the end line: a test that saw the turn end reads the record file next.
    record({ kind: 'steer-turn', text, promptId });
    update(
      session,
      'turn_completed',
      {
        prompt_id: promptId,
        stop_reason: 'end_turn',
        usage: { inputTokens: 1, outputTokens: 1, modelCalls: 1 },
        elapsed_ms: 10,
      },
      true,
    );
  }
}

function endCancelled(id, session, category) {
  update(session, 'turn_completed', { prompt_id: session.turn.promptId, stop_reason: 'cancelled' }, true, {
    cancellationCategory: category,
  });
  session.turn = null;
  reply(id, { stopReason: 'cancelled', _meta: { sessionId: session.id, cancellationCategory: category } });
  // Stop of a goal's turn pauses the goal, after the turn's end (`run_loop.rs`).
  if (session.goal?.status === 'active') {
    session.goal.status = 'user_paused';
    goalLine(session, true, { last_event: 'goal_paused' });
  }
}

// A turn the agent starts by itself after a background task: once `startOn` exists it sends a hidden prompt and an
// answer; once `endOn` exists, its end line. With `together`, the prompt and answer go at once, in the same stdout
// write as the typed turn's end line and reply.
async function wakeLater(session, wake) {
  const fileAppears = async (path) => {
    while (!existsSync(path)) await sleep(10);
  };
  if (wake.together !== true) await fileAppears(wake.startOn);
  const promptId = `task-${randomUUID()}`;
  wakePrompt(session, 'A background task finished.');
  update(session, 'agent_message_chunk', { content: { type: 'text', text: 'Looked at it.' } });
  if (batch !== null) {
    process.stdout.write(batch.join(''));
    batch = null;
  }
  await fileAppears(wake.endOn);
  update(
    session,
    'turn_completed',
    {
      prompt_id: promptId,
      stop_reason: 'end_turn',
      usage: { inputTokens: 1, outputTokens: 1, modelCalls: 1 },
      elapsed_ms: 10,
    },
    true,
  );
}

// Workflow runs by id, as the agent's tracker keeps them.
const runs = new Map();

// As the agent's `build_workflow_updated` (notify.rs): the phases before the current one are done, the current one is
// done only when the run is `complete` (else active, also once it failed or was cancelled), the rest pending; the
// current phase stays named after the run ends. Fields the agent has no value for are left out.
function workflowUpdate(run) {
  run.revision += 1;
  const complete = run.status === 'complete';
  const running = run.agents.find((a) => a.state === 'running');
  update(
    run.session,
    'workflow_updated',
    {
      run_id: run.id,
      revision: run.revision,
      name: run.name,
      objective: run.objective,
      status: run.status,
      foreground: false,
      phases: run.phases.map((title, k) => ({
        title,
        state: k < run.phase || (k === run.phase && complete) ? 'done' : k === run.phase ? 'active' : 'pending',
      })),
      current_phase: run.phases[run.phase],
      agent_budget: run.budget,
      agents_used: run.agents.length,
      agents_reserved: 0,
      agents_remaining: run.budget - run.agents.length,
      agent_usage_incomplete: false,
      elapsed_ms: Date.now() - run.startedAt,
      active_agents: run.agents.filter((a) => a.state === 'running').length,
      ...(running === undefined ? {} : { current_agent_label: running.label }),
      agents: run.agents.map((a) => ({
        agent_id: a.id,
        label: a.label,
        phase: a.phase,
        model: run.session.model,
        state: a.state,
        tokens_used: a.tokens,
        duration_ms: a.durationMs,
      })),
      last_event: run.lastEvent,
      last_event_timestamp: new Date().toISOString(),
      ...(run.pauseMessage === null ? {} : { pause_message: run.pauseMessage }),
      ...(complete ? { result_summary: run.report } : {}),
    },
    true,
  );
}

/** Waits for the gate file, or `ms`; false when the run was stopped meanwhile. */
async function runGate(run, path, ms) {
  if (typeof path === 'string') while (!existsSync(path) && !run.stopped) await sleep(10);
  else if (typeof ms === 'number') await sleep(ms);
  return !run.stopped;
}

// A `workflow` step: the run starts inside the typed turn (`workflow_updated` active, `subagent_spawned`, a second
// `workflow_updated` with the helper row); the rest runs on its own after the step, each part behind its gate (a file
// path, else a delay): the helper's shell call and its permission request with `x.ai/requester` (`childOn`/`childAfterMs`),
// the helper's end and the terminal `workflow_updated` with `result_summary` (`completeOn`/`completeAfterMs`), then the
// wake turn (`wakeEndOn`/`wakeEndAfterMs` holds its end line). `_x.ai/workflows/stop` ends the run at any gate
// (`stopOn` holds the stop until its file exists; `stopRejects` answers every stop as rejected, `stopFails` with a
// JSON-RPC error); once its helpers
// ended the run's drain update follows (`drainOn` holds it; `drainStatus: 'interrupted'` turns the run interrupted, as
// the agent's watcher does after a slow drain); `pauseDrainOn` holds the helper ends of a typed `/workflow pause`
// (`pauseDrain`). A `status` other than `complete` ends the run so (a paused one sends
// `pause_message` and no wake, as the agent sends none for a pause). `retry: true`: the helper's first attempt ends
// at once and the run retries it as its output contract does (a new child, `resumed_from` the first, the same label;
// the roster names the newest attempt). `retry: 'on-cancel'`: the first attempt runs until a cancel of it comes; then
// it ends `completed`, its retry spawns, and the cancel is answered `already_finished`, as when the cancel reaches the
// agent in the 47 ms between the first attempt's end and its retry's spawn. `_x.ai/subagent/cancel` of the helper
// ends it `cancelled` (the roster says `failed`) and the run goes on to its end, as the agent's script gets a failed
// call. `askAfterEnd`: the
// helper asks only after the terminal update, as a late ask of a run that has ended. `elicit` (an `elicit` step's
// fields): before its permission request the helper's MCP server asks for input ({@link runElicit}). A `complete` run
// reaches its last phase first; `failed` keeps its phase and sends `pause_message`. `more`: agents spawned beside the
// first (`label`, `phase`; `command` opens a tool call of theirs after `toolAfterMs`), each cancelled alone or with
// the run, and ended `completed` at the run's end. `tickEveryMs`: every running agent's `subagent_progress` ticks.
function startWorkflow(session, w, n) {
  const run = {
    id: w.runId ?? `wf_${n}`,
    session,
    name: w.name ?? 'deep-research',
    objective: w.objective ?? 'Answer the question.',
    status: 'active',
    revision: 0,
    phases: w.phases ?? ['Plan', 'Research', 'Report'],
    phase: 0,
    budget: w.budget ?? 8,
    agents: [],
    lastEvent: 'run_started',
    report: w.report ?? '# Findings\n\nThe probe found three files.',
    startedAt: Date.now(),
    stopped: false,
    stopOn: w.stopOn ?? null,
    stopRejects: w.stopRejects === true,
    stopFails: w.stopFails === true,
    drainOn: w.drainOn ?? null,
    drainStatus: w.drainStatus ?? null,
    pauseDrainOn: w.pauseDrainOn ?? null,
    flow: w,
    pauseMessage: null,
  };
  runs.set(run.id, run);
  workflowUpdate(run);
  // The real agent names a helper once: its subagent id, its child session id and the run row's agent id are equal.
  const childId = w.childSessionId ?? `${session.id}-child-${n}`;
  const agent = {
    id: childId,
    child: childId,
    label: w.label ?? 'researcher-0',
    phase: run.phases[1] ?? run.phases[0],
    state: 'running',
    tokens: 0,
    durationMs: 0,
  };
  run.agents.push(agent);
  const spawnOne = (a, resumedFrom) => {
    update(
      session,
      'subagent_spawned',
      {
        subagent_id: a.id,
        child_session_id: a.child,
        parent_session_id: session.id,
        subagent_type: 'general-purpose',
        description: a.label,
        workflow_run_id: run.id,
        ...(resumedFrom === null ? {} : { resumed_from: resumedFrom, effective_context_source: 'resumed' }),
      },
      true,
    );
    helpers.set(a.child, {
      session,
      description: a.label,
      startedAt: Date.now(),
      address: null,
      run: run.id,
      cancel: () => cancelRunAgent(run, a),
    });
    session.lastHelper = a.child;
  };
  const spawn = (resumedFrom) => spawnOne(agent, resumedFrom);
  spawn(null);
  const more = (w.more ?? []).map((m, k) => {
    const id = m.childSessionId ?? `${childId}-${k + 2}`;
    const a = {
      id,
      child: id,
      label: m.label,
      phase: m.phase ?? agent.phase,
      state: 'running',
      tokens: 0,
      durationMs: 0,
    };
    run.agents.push(a);
    spawnOne(a, null);
    return { a, m };
  });
  run.phase = Math.min(1, run.phases.length - 1);
  run.lastEvent = 'agent_started';
  workflowUpdate(run);
  for (const { a, m } of more) void runMoreAgent(run, a, m, w.childCwd ?? session.cwd);
  if (typeof w.tickEveryMs === 'number') void tickRunAgents(run, w.tickEveryMs);
  const retry = () => {
    const first = agent.child;
    endRunAgent(run, agent, 'completed', { output: 'I looked at the files; the answer is in prose.' });
    agent.id = agent.child = `${first}-retry`;
    agent.state = 'running';
    spawn(first);
    workflowUpdate(run);
  };
  if (w.retry === true) retry();
  else if (w.retry === 'on-cancel') helpers.get(agent.child).endsFirst = retry;
  void runWorkflow(run, agent, w);
}

/** The PDF id a prompt's PDF tool note names (a `resource` block, `id: xxxxxxxx-N` after its marker), or null. */
function promptPdf(blocks) {
  for (const block of Array.isArray(blocks) ? blocks : []) {
    const text = block?.type === 'resource' ? block.resource?.text : undefined;
    if (typeof text !== 'string' || !text.startsWith('Portato PDF note 1\n')) continue;
    for (const line of text.split('\n')) if (line.startsWith('id: ')) return line.slice('id: '.length);
  }
  return null;
}

/** `step` with each string `$PDF` in it set to the prompt's PDF id, and each `$RUN` to the turn's run id. */
function withPdf(step, pdf, run) {
  if (pdf === null) return step;
  if (step === '$PDF') return pdf;
  if (step === '$RUN') return run;
  if (Array.isArray(step)) return step.map((item) => withPdf(item, pdf, run));
  if (step === null || typeof step !== 'object') return step;
  return Object.fromEntries(Object.entries(step).map(([key, value]) => [key, withPdf(value, pdf, run)]));
}

/** Pages as the script's `range_text` writes them: runs `a-b`, joined by `, `. */
function pageRange(pages) {
  const parts = [];
  for (let k = 0; k < pages.length;) {
    let end = k;
    while (end + 1 < pages.length && pages[end + 1] === pages[end] + 1) end += 1;
    parts.push(end === k ? `${pages[k]}` : `${pages[k]}-${pages[end]}`);
    k = end + 1;
  }
  return parts.join(', ');
}

// A run of the PDF tool (`read-pdf`), as the saved workflow plays it: in "Read pages" one reader after another, 4
// pages each ("pages 1-4"), each opening its pages with `read_file`, `pageMs` apart; group `fails` (from 1) returns no
// notes and is read once more at the end ("pages 9-12 again"), which also fails with `failsTwice`; then "combine" in
// "Combine notes", the report ("PDF read: R of T pages." and the pages not read) and the wake. `pdf`: the PDF id;
// `pages`: the pages asked for (all when unset). Stop ends it at any time.
function readPdfRun(session, w, n, pdf, pages) {
  const total = Number(pdf.slice(pdf.lastIndexOf('-') + 1));
  const chosen = pages ?? Array.from({ length: total }, (_, k) => k + 1);
  const groups = [];
  for (let k = 0; k < chosen.length; k += 4) groups.push(chosen.slice(k, k + 4));
  const run = {
    id: w.runId ?? `wf_pdf_${n}`,
    session,
    name: 'read-pdf',
    objective: w.objective ?? 'Read the PDF.',
    status: 'active',
    revision: 0,
    phases: ['Read pages', 'Combine notes'],
    phase: 0,
    budget: 2 * groups.length + 2,
    agents: [],
    lastEvent: 'run_started',
    report: '',
    startedAt: Date.now(),
    stopped: false,
    stopOn: null,
    stopRejects: false,
    stopFails: false,
    drainOn: null,
    drainStatus: null,
    pauseDrainOn: null,
    flow: w,
    pauseMessage: null,
  };
  runs.set(run.id, run);
  workflowUpdate(run);
  void playReadPdf(run, w, n, pdf, groups);
  return run;
}

/** One agent of a `read-pdf` run: spawned, its pages opened one by one, ended; false when the run was stopped. */
async function readPdfAgent(run, n, label, pages, pdf, ms, fails) {
  const session = run.session;
  const id = `${session.id}-pdf-${n}-${run.agents.length + 1}`;
  const a = { id, child: id, label, phase: run.phases[run.phase], state: 'running', tokens: 0, durationMs: 0 };
  const started = Date.now();
  run.agents.push(a);
  update(
    session,
    'subagent_spawned',
    {
      subagent_id: id,
      child_session_id: id,
      parent_session_id: session.id,
      subagent_type: 'pdf-reader',
      description: label,
      workflow_run_id: run.id,
    },
    true,
  );
  helpers.set(id, {
    session,
    description: label,
    startedAt: started,
    address: null,
    run: run.id,
    cancel: () => cancelRunAgent(run, a),
  });
  run.lastEvent = 'agent_started';
  workflowUpdate(run);
  const child = { id, cwd: session.cwd };
  const meta = { 'x.ai/tool': xaiTool('read_file') };
  const step = pages.length === 0 ? ms : ms / pages.length;
  // The agent's live tick of the reader (never stored): what it does now shows only once one came.
  const tick = (calls) =>
    send({
      method: '_x.ai/session_notification',
      params: {
        sessionId: session.id,
        update: {
          sessionUpdate: 'subagent_progress',
          subagent_id: id,
          parent_session_id: session.id,
          child_session_id: id,
          duration_ms: Date.now() - started,
          context_window_tokens: 32_768,
          tools_used: ['read_file'],
          turn_count: 1,
          tool_call_count: calls,
          tokens_used: 2_400 * calls,
          context_usage_pct: Math.min(99, 7 * calls),
          error_count: 0,
        },
      },
    });
  tick(0);
  for (const [k, page] of pages.entries()) {
    const toolCallId = `${id}-page-${page}`;
    const path = `/pdf/${pdf}/page-${String(page).padStart(3, '0')}.jpg`;
    update(child, 'tool_call', { toolCallId, title: `Read \`${path}\``, rawInput: { path }, _meta: meta });
    tick(k + 1);
    if (!(await runGate(run, undefined, step)) || a.state !== 'running') return false;
    update(child, 'tool_call_update', { toolCallId, status: 'completed' });
  }
  if (pages.length === 0 && !(await runGate(run, undefined, ms))) return false;
  if (a.state !== 'running') return !run.stopped;
  a.state = fails ? 'failed' : 'done';
  a.tokens = 2_400 * Math.max(1, pages.length);
  a.durationMs = Date.now() - started;
  endRunAgent(
    run,
    a,
    fails ? 'failed' : 'completed',
    fails ? { error: 'The agent returned no output.' } : { output: `Notes of pages ${pageRange(pages)}.` },
  );
  workflowUpdate(run);
  return true;
}

async function playReadPdf(run, w, n, pdf, groups) {
  const ms = w.pageMs ?? 900;
  const read = groups.map(() => false);
  for (const [k, pages] of groups.entries()) {
    const fails = w.fails === k + 1;
    if (!(await readPdfAgent(run, n, `pages ${pageRange(pages)}`, pages, pdf, ms * pages.length, fails))) return;
    read[k] = !fails && run.agents.at(-1).state === 'done';
  }
  for (const [k, pages] of groups.entries()) {
    if (read[k]) continue;
    const fails = w.failsTwice === true;
    if (!(await readPdfAgent(run, n, `pages ${pageRange(pages)} again`, pages, pdf, ms * pages.length, fails))) return;
    read[k] = !fails && run.agents.at(-1).state === 'done';
  }
  run.phase = 1;
  run.lastEvent = 'phase_started';
  workflowUpdate(run);
  if (!(await readPdfAgent(run, n, 'combine', [], pdf, w.combineMs ?? 3_000, false))) return;
  const unread = groups.filter((_, k) => !read[k]).flat();
  const done = groups.filter((_, k) => read[k]).flat().length;
  const head = `PDF read: ${done} of ${groups.flat().length} pages.${unread.length === 0 ? '' : ` Not read: ${pageRange(unread)}.`}`;
  run.report = `${head}\n\n${w.answer ?? 'The findings are on pages 3 and 7 (p. 3, p. 7).'}\n\nNotes by page\n…`;
  run.status = 'complete';
  run.lastEvent = 'run_finished';
  workflowUpdate(run);
  await wakeTurn(run.session, `workflow-completed-${run.id}-${run.revision}`, w);
}

/** An agent beside a run's first: one tool call of its own after its gate, left open until it ends. */
async function runMoreAgent(run, a, m, cwd) {
  if (typeof m.command !== 'string') return;
  if (!(await runGate(run, undefined, m.toolAfterMs ?? 0)) || a.state !== 'running') return;
  const toolCallId = `${a.child}-call-1`;
  const rawInput = { variant: 'Bash', command: m.command, description: 'run', is_background: false };
  update({ id: a.child, cwd }, 'tool_call', {
    toolCallId,
    title: m.command,
    status: 'in_progress',
    rawInput,
    _meta: { 'x.ai/tool': xaiTool('run_terminal_command') },
  });
}

/** Each running agent's `subagent_progress` tick, `everyMs` apart, while the run goes on (live only, never stored). */
async function tickRunAgents(run, everyMs) {
  const startedAt = Date.now();
  let turn = 0;
  while (!run.stopped && run.status === 'active') {
    turn += 1;
    for (const [k, a] of run.agents.entries()) {
      if (a.state !== 'running') continue;
      send({
        method: '_x.ai/session_notification',
        params: {
          sessionId: run.session.id,
          update: {
            sessionUpdate: 'subagent_progress',
            subagent_id: a.id,
            parent_session_id: run.session.id,
            child_session_id: a.child,
            duration_ms: Date.now() - startedAt,
            context_window_tokens: 131_072,
            tools_used: ['run_terminal_command'],
            turn_count: 1,
            tool_call_count: turn + k,
            tokens_used: 4_000 * turn + 1_000 * k,
            context_usage_pct: Math.min(99, 3 * turn + k),
            error_count: 0,
          },
        },
      });
    }
    await sleep(everyMs);
  }
}

/** A run helper's end line; it is no longer running. */
function endRunAgent(run, agent, status, fields = {}) {
  helpers.delete(agent.child);
  helpersEnded.set(agent.child, status);
  update(
    run.session,
    'subagent_finished',
    {
      subagent_id: agent.id,
      attempt_id: `at1.${agent.child}`,
      child_session_id: agent.child,
      status,
      tool_calls: 1,
      turns: 1,
      duration_ms: Date.now() - run.startedAt,
      tokens_used: agent.tokens,
      will_wake: false,
      ...fields,
    },
    true,
  );
}

/**
 * `_x.ai/subagent/cancel` of a run's helper: it ends `cancelled`, the roster says `failed`, the run goes on with its
 * last event unchanged (`fixtures/workflow-agent-stop.json`, recorded from the agent).
 */
function cancelRunAgent(run, agent) {
  agent.state = 'failed';
  endRunAgent(run, agent, 'cancelled', { error: 'Subagent was cancelled', tool_calls: 0, tokens_used: 0 });
  workflowUpdate(run);
}

async function runWorkflow(run, agent, w) {
  const session = run.session;
  const child = { id: agent.child, cwd: w.childCwd ?? session.cwd };
  if (!(await runGate(run, w.childOn, w.childAfterMs))) return;
  if (agent.state !== 'running') return endWorkflow(run, w);
  const toolCallId = `${run.id}-call-1`;
  const meta = { 'x.ai/tool': xaiTool('run_terminal_command') };
  const rawInput = { variant: 'Bash', command: w.command ?? 'ls', description: 'run', is_background: false };
  update(child, 'tool_call', { toolCallId, title: 'run_terminal_command', rawInput, _meta: meta });
  const ask = async () => {
    const answer = await agentRequest('session/request_permission', {
      sessionId: session.id,
      toolCall: { toolCallId, kind: 'execute', title: 'run', rawInput, _meta: meta },
      options: PERMISSION_OPTIONS,
      _meta: {
        ...(session.guard === null ? {} : { hookAsk: { hookName: session.guard } }),
        'x.ai/requester': { sessionId: child.id, label: agent.label, runId: run.id, cwd: child.cwd },
      },
    });
    const outcome = answer?.outcome;
    const optionId = outcome?.outcome === 'selected' ? outcome.optionId : null;
    // A note lands in the helper's own session, as the agent's gateway routes it.
    const note = typeof answer?._meta?.followup_message === 'string' ? answer._meta.followup_message : null;
    record({ kind: 'child-answer', runId: run.id, toolCallId, answer: optionId, ...(note === null ? {} : { note }) });
    return optionId === 'allow-once' || optionId === 'always-allow';
  };
  if (w.elicit !== undefined) {
    await runElicit(session, w.elicit);
    if (run.stopped) return;
  }
  let allowed = true;
  if (w.ask !== false && w.askAfterEnd !== true) {
    allowed = await ask();
    if (run.stopped) return;
  }
  if (w.askAfterEnd !== true)
    update(child, 'tool_call_update', { toolCallId, status: allowed ? 'completed' : 'failed' });
  await endWorkflow(run, w, async () => {
    const late = await ask();
    update(child, 'tool_call_update', { toolCallId, status: late ? 'completed' : 'failed' });
  });
}

/** The run's end behind its gate: its agents' ends (unless they ended already), the terminal update, the wake. */
async function endWorkflow(run, w, lateAsk = async () => {}) {
  const session = run.session;
  if (!(await runGate(run, w.completeOn, w.completeAfterMs))) return;
  for (const a of run.agents) {
    if (a.state !== 'running') continue;
    a.state = 'done';
    a.tokens = 1200;
    a.durationMs = Date.now() - run.startedAt;
    endRunAgent(run, a, 'completed');
  }
  run.status = w.status ?? 'complete';
  run.lastEvent = 'run_finished';
  if (run.status === 'complete') run.phase = run.phases.length - 1;
  if (run.status === 'failed') run.pauseMessage = w.error ?? 'The script failed.';
  if (RUN_WAITING.has(run.status)) run.pauseMessage = w.error ?? 'Check the repro log before resuming.';
  workflowUpdate(run);
  if (w.askAfterEnd === true) await lateAsk();
  if (w.wake === false || (RUN_WAITING.has(run.status) && run.status !== 'budget_limited')) return;
  await wakeTurn(session, `workflow-completed-${run.id}-${run.revision}`, w);
}

/**
 * The agent's own turn after a run, a command or a helper ended: a hidden prompt (`w.wakeText`), an answer, then its
 * end line (`session/cancel` ends it).
 */
async function wakeTurn(session, promptId, w) {
  let cancel = () => {};
  let wasCancelled = false;
  const cancelled = new Promise((resolve) => {
    cancel = () => {
      wasCancelled = true;
      resolve(true);
    };
  });
  session.wakeTurn = { cancel, steers: [] };
  const steers = session.wakeTurn.steers;
  wakePrompt(session, w.wakeText ?? 'A workflow finished. Read its report.', w.wakeEcho === true);
  update(session, 'agent_message_chunk', { content: { type: 'text', text: w.wakeAnswer ?? 'The report is ready.' } });
  if (w.wakeAsk === true) {
    const toolCallId = `${promptId}-ask`;
    const meta = { 'x.ai/tool': xaiTool('run_terminal_command') };
    const rawInput = { variant: 'Bash', command: 'pwd', description: 'run', is_background: false };
    update(session, 'tool_call', { toolCallId, title: 'run_terminal_command', rawInput, _meta: meta });
    const answer = await Promise.race([
      agentRequest('session/request_permission', {
        sessionId: session.id,
        toolCall: { toolCallId, kind: 'execute', title: 'run', rawInput, _meta: meta },
        options: PERMISSION_OPTIONS,
        ...(session.guard === null ? {} : { _meta: { hookAsk: { hookName: session.guard } } }),
      }),
      cancelled,
    ]);
    record({ kind: 'wake-answer', answer: answer === true ? 'cancelled-turn' : (answer?.outcome ?? null) });
  }
  if (!wasCancelled) {
    await Promise.race([
      (async () => {
        if (typeof w.wakeEndOn === 'string') while (!existsSync(w.wakeEndOn) && !wasCancelled) await sleep(10);
        else if (typeof w.wakeEndAfterMs === 'number') await sleep(w.wakeEndAfterMs);
      })(),
      cancelled,
    ]);
  }
  session.wakeTurn = null;
  if (!wasCancelled) {
    if (steers.length > 0) {
      session.turn = { steers: [...steers] };
      readSteers(session);
      session.turn = null;
    }
    w.beforeEnd?.();
  }
  update(
    session,
    'turn_completed',
    {
      prompt_id: promptId,
      stop_reason: wasCancelled ? 'cancelled' : 'end_turn',
      usage: { inputTokens: 1, outputTokens: 1, modelCalls: 1 },
      elapsed_ms: 10,
    },
    true,
  );
  return wasCancelled;
}

/**
 * `x.ai/workflows/stop`: a run that is not terminal is cancelled with no wake (an active one drains its helpers, one
 * that waits sends only its update); a `budget_limited` run or a terminal one is a rejection.
 */
async function stopRun(id, params) {
  const sessionId = params?.sessionId;
  const runId = params?.runId;
  if (typeof sessionId !== 'string' || sessionId.trim() === '' || typeof runId !== 'string' || runId.trim() === '')
    return replyError(id, -32602, 'Invalid params');
  if (!sessions.has(sessionId)) return replyError(id, -32602, 'Invalid params', `unknown session id: ${sessionId}`);
  record({ kind: 'stop', runId });
  const run = runs.get(runId);
  if (typeof run?.stopOn === 'string') while (!existsSync(run.stopOn)) await sleep(10);
  if (run === undefined || run.session.id !== sessionId)
    return reply(id, { outcome: 'rejected', code: 'workflow_control_unknown_run', detail: `no run ${runId}` });
  if (run.stopFails) return replyError(id, -32603, 'Internal error');
  if (RUN_DONE.includes(run.status) || run.stopRejects)
    return reply(id, { outcome: 'rejected', code: 'workflow_control_not_applicable', detail: run.status });
  const active = run.status === 'active';
  const cancelled = haltRun(run, 'cancelled');
  reply(id, { outcome: 'stopped', runId: run.id, name: run.name });
  workflowUpdate(run);
  if (active) void drainRun(run, cancelled);
}

// The statuses of a run that waits on the person.
const RUN_WAITING = new Set([
  'user_paused',
  'back_off_paused',
  'no_progress_paused',
  'infra_paused',
  'blocked',
  'budget_limited',
]);

/** The agent's cancel of a run (a stop, or the cancel-based pause): its running helpers turn cancelled. */
function haltRun(run, status) {
  run.stopped = true;
  run.status = status;
  run.lastEvent = status === 'cancelled' ? 'workflow_cancelled' : 'workflow_paused';
  const cancelled = run.agents.filter((a) => a.state === 'running');
  for (const a of cancelled) a.state = 'cancelled';
  return cancelled;
}

/**
 * After a stop the agent waits for the run's helpers (`cancel_children_for_run`), each one's end its own line; then
 * the run's own outcome arrives and is ignored, a stored update (`workflow_outcome_ignored`).
 */
async function drainRun(run, cancelled) {
  endHelpers(run, cancelled);
  if (run.status !== 'cancelled') return;
  if (typeof run.drainOn === 'string') while (!existsSync(run.drainOn)) await sleep(10);
  if (run.drainStatus === 'interrupted') {
    run.status = 'interrupted';
    run.lastEvent = 'workflow_interrupted';
    run.pauseMessage = 'workflow cleanup timed out; the run cannot resume';
  } else {
    run.lastEvent = 'workflow_outcome_ignored';
  }
  workflowUpdate(run);
}

/**
 * A pause's drain: the agent's watcher waits for the run's helpers (`pauseDrainOn` holds their ends), then ends. A stop
 * that came meanwhile is the status it reports, with a wake (grok `manager.rs`); a stop after it sends none.
 */
async function pauseDrain(run, cancelled) {
  if (typeof run.pauseDrainOn === 'string') while (!existsSync(run.pauseDrainOn)) await sleep(10);
  endHelpers(run, cancelled);
  if (run.status !== 'cancelled' || run.flow.wake === false) return;
  await wakeTurn(run.session, `workflow-completed-${run.id}-${run.revision}`, run.flow);
}

/** The end line of each helper a stop or a pause cancelled. */
function endHelpers(run, cancelled) {
  for (const a of cancelled) {
    helpers.delete(a.child);
    helpersEnded.set(a.child, 'cancelled');
    update(
      run.session,
      'subagent_finished',
      {
        subagent_id: a.id,
        child_session_id: a.child,
        status: 'cancelled',
        tool_calls: 0,
        turns: 0,
        duration_ms: Date.now() - run.startedAt,
        tokens_used: 0,
        will_wake: false,
      },
      true,
    );
  }
}

/**
 * `/workflow pause|stop <name>` (`manage_workflow_run`): the run of that id or name, if the op applies, is paused or
 * stopped as the tool does, but not marked reported, so a stop still wakes the agent. A stop of a run that is not
 * active sends only its update (grok `manager.rs` `cancel`). The agent's words; null for any other `/workflow`.
 */
function manageRun(session, text) {
  const words = text
    .trim()
    .split(' ')
    .filter((w) => w !== '');
  const op = words[1]?.toLowerCase();
  if ((op !== 'pause' && op !== 'stop') || words.length !== 3) return null;
  const run = [...runs.values()].find((r) => r.session === session && (r.id === words[2] || r.name === words[2]));
  if (run === undefined || (op === 'pause' ? run.status !== 'active' : RUN_DONE.includes(run.status))) {
    return `No workflow run matches '${words[2]}'.`;
  }
  const active = run.status === 'active';
  const cancelled = haltRun(run, op === 'stop' ? 'cancelled' : 'user_paused');
  workflowUpdate(run);
  if (op === 'pause') void pauseDrain(run, cancelled);
  else if (active) void drainRun(run, cancelled);
  return op === 'stop' ? `Stopped ${run.name}.` : `Paused ${run.name}.`;
}

/**
 * `/workflow read-pdf {"pdf": …}` (the app's Read it now): the agent starts the PDF tool's run itself, played as
 * {@link readPdfRun} with the Work scenario's `readPdf`. Null for any other text, and with no `readPdf` in the scenario.
 */
function readPdfStart(session, text) {
  const head = '/workflow read-pdf ';
  if (workSection?.readPdf === undefined || !text.trim().startsWith(head)) return null;
  let args = null;
  try {
    args = JSON.parse(text.trim().slice(head.length));
  } catch {
    return 'The args are not JSON.';
  }
  if (typeof args?.pdf !== 'string') return 'read-pdf needs a pdf.';
  session.steps = (session.steps ?? 0) + 1;
  const run = readPdfRun(session, { ...workSection.readPdf, objective: args.objective }, session.steps, args.pdf, null);
  return `Started workflow read-pdf (${run.id}).`;
}

// The statuses a stop does not apply to.
const RUN_DONE = ['complete', 'failed', 'cancelled', 'interrupted', 'budget_limited'];

/**
 * The main agent's own `workflow` call that pauses or stops a run (`source.type`, `run_id` an id or a name): a pause
 * applies to an active run, a stop to one that is not terminal; any other is a failed call. `inputLate`: the call's
 * `rawInput` comes only with its end, after the run's update.
 */
function runControlStep(session, c, toolCallId) {
  const meta = { 'x.ai/tool': xaiTool('workflow') };
  const key = c.key ?? 'deep-research';
  const rawInput = { variant: 'Workflow', source: { type: c.verb, run_id: key }, validate_only: false };
  update(session, 'tool_call', {
    toolCallId,
    title: 'workflow',
    kind: 'other',
    ...(c.inputLate === true ? {} : { rawInput }),
    _meta: meta,
  });
  const run = [...runs.values()].find((r) => r.session === session && (r.id === key || r.name === key));
  const applies = run !== undefined && (c.verb === 'pause' ? run.status === 'active' : !RUN_DONE.includes(run.status));
  let cancelled = [];
  if (applies) {
    cancelled = haltRun(run, c.verb === 'stop' ? 'cancelled' : 'user_paused');
    if (c.verb === 'pause') run.pauseMessage = 'Paused by the model.';
    workflowUpdate(run);
  }
  update(session, 'tool_call_update', {
    toolCallId,
    status: applies ? 'completed' : 'failed',
    ...(c.inputLate === true ? { rawInput } : {}),
    ...(applies ? { rawOutput: { type: 'Workflow', run_id: run.id, name: run.name } } : {}),
    content: [{ type: 'content', content: { type: 'text', text: applies ? `${c.verb} ${run.name}` : 'no such run' } }],
  });
  if (applies) void drainRun(run, cancelled);
}

async function streamText(session, kind, text, chunk, delayMs, totalTokens) {
  const size = chunk ?? text.length;
  const meta = totalTokens === undefined ? {} : { totalTokens };
  for (let i = 0; i < text.length; i += size) {
    if (session.turn.cancelled) return;
    update(session, kind, { content: { type: 'text', text: text.slice(i, i + size) } }, false, meta);
    if (delayMs) await sleep(delayMs);
  }
}

const QUESTION_CANCEL_TEXT =
  'User declined to answer the questions. Continue with the task using your best judgment, or ask different questions.';

/** The tool result of a question call for the client's reply, worded as `ask_user_question/format.rs` words it. */
function questionResult(questions, answer) {
  const invalid = (why) => ({ status: 'failed', text: `Client returned an invalid response to user question: ${why}` });
  const lines = (partial) =>
    questions
      .map((q) =>
        typeof partial?.[q.question] === 'string'
          ? `- "${q.question}"\n  Answer: ${partial[q.question]}`
          : `- "${q.question}"\n  (No answer provided)`,
      )
      .join('\n');
  if (answer === null || typeof answer !== 'object') return invalid('invalid type');
  // A plan reply's `partial_answers` is a map of one label string per question, or left out (`types.rs`).
  const partial = answer.partial_answers;
  const partialBad =
    partial !== undefined &&
    (partial === null ||
      typeof partial !== 'object' ||
      Array.isArray(partial) ||
      Object.values(partial).some((label) => typeof label !== 'string'));
  if ((answer.outcome === 'chat_about_this' || answer.outcome === 'skip_interview') && partialBad) {
    return invalid('invalid type');
  }
  switch (answer.outcome) {
    case 'cancelled':
      return { status: 'completed', text: QUESTION_CANCEL_TEXT };
    case 'accepted': {
      const answers = answer.answers;
      if (answers === null || typeof answers !== 'object' || Array.isArray(answers)) return invalid('missing field');
      const entries = [];
      for (const [text, value] of Object.entries(answers)) {
        const labels = typeof value === 'string' ? [value] : value;
        if (!Array.isArray(labels) || labels.some((label) => typeof label !== 'string')) return invalid('invalid type');
        const note = answer.annotations?.[text];
        const parts = [`"${text}"="${labels.join(', ')}"`];
        if (typeof note?.preview === 'string') parts.push(`selected preview:\n${note.preview}`);
        if (typeof note?.notes === 'string') parts.push(`user notes: ${note.notes}`);
        entries.push(parts.join(' '));
      }
      return {
        status: 'completed',
        text: `User has answered your questions: ${entries.join(', ')}. You can now continue with the user's answers in mind.`,
      };
    }
    case 'chat_about_this':
      return {
        status: 'completed',
        text:
          'The user wants to clarify these questions.\n    This means they may have additional information, context or questions for you.\n' +
          '    Take their response into account and then reformulate the questions if appropriate.\n' +
          `    Start by asking them what they would like to clarify.\n\n    Questions asked:\n${lines(answer.partial_answers)}`,
      };
    case 'skip_interview':
      return {
        status: 'completed',
        text:
          'The user has indicated they have provided enough answers for the plan interview.\n' +
          'Stop asking clarifying questions and proceed to finish the plan with the information you have.\n\n' +
          `Questions asked and answers provided:\n${lines(answer.partial_answers)}`,
      };
    default:
      return invalid('unknown variant');
  }
}

// `ask_user_question`, as the agent runs it: the model's call (its own input says `multi_select`), the agent's copy
// of the input (`variant`, `multiSelect`, title `Ask: <question>` or `Ask N questions`), live `pending_interaction`,
// the request `_x.ai/ask_user_question` (`mode: "plan"` in plan mode), `interaction_resolved` once its wait ends, then
// the stored result sentence of `format.rs`; a reply that is none of the four shapes fails the call. `giveUpOn`: once
// that file exists the agent stops waiting (its time limit) and the call ends "declined"; the host's late answer is
// recorded and goes nowhere. Stop ends the turn with no result for the call. Returns 'dropped' when the turn was.
async function questionStep(session, q, toolCallId) {
  const meta = {
    'x.ai/tool': {
      version: 1,
      name: 'ask_user_question',
      kind: 'ask_user',
      namespace: 'grok_build',
      label: 'Ask User',
      read_only: true,
    },
  };
  const questions = q.questions.map((x) => ({
    question: x.question,
    options: (x.options ?? []).map((o) => ({
      label: o.label,
      description: o.description ?? '',
      ...(o.preview === undefined ? {} : { preview: o.preview }),
    })),
    multiSelect: x.multi === true ? true : null,
  }));
  const modelInput = {
    questions: questions.map(({ question, options }, i) => ({
      question,
      options,
      ...(q.questions[i].multi === true ? { multi_select: true } : {}),
    })),
  };
  callDeltas(session, toolCallId, 'ask_user_question', modelInput);
  update(session, 'tool_call', { toolCallId, title: 'ask_user_question', rawInput: modelInput, _meta: meta });
  update(session, 'tool_call_update', {
    toolCallId,
    kind: 'other',
    title: questions.length === 1 ? `Ask: ${questions[0].question}` : `Ask ${questions.length} questions`,
    locations: [],
    rawInput: { variant: 'AskUserQuestion', questions },
    _meta: meta,
  });
  xaiOnly(session, 'pending_interaction', { tool_call_id: toolCallId, kind: 'question' });
  const turn = session.turn;
  turn.asking = true;
  let answered = false;
  let late = false;
  const asked = agentRequest('_x.ai/ask_user_question', {
    sessionId: session.id,
    toolCallId,
    questions,
    mode: session.planMode === 'plan' ? 'plan' : 'default',
  });
  void asked.then((answer) => {
    answered = true;
    record({ kind: 'question-answer', toolCallId, answer: answer ?? null, late });
  });
  if (typeof q.giveUpOn === 'string') while (!answered && !existsSync(q.giveUpOn)) await sleep(10);
  const answer = answered || typeof q.giveUpOn !== 'string' ? await asked : undefined;
  late = !answered;
  turn.asking = false;
  if (session.turn !== turn) return 'dropped';
  xaiOnly(session, 'interaction_resolved', { tool_call_id: toolCallId });
  if (turn.cancelled) return undefined;
  const result = late ? { status: 'completed', text: QUESTION_CANCEL_TEXT } : questionResult(questions, answer);
  update(session, 'tool_call_update', {
    toolCallId,
    status: result.status,
    content: [{ type: 'content', content: { type: 'text', text: result.text } }],
    ...(result.status === 'completed'
      ? { rawOutput: { type: 'AskUserQuestion', UserAnswered: { message: result.text } } }
      : {}),
  });
  return undefined;
}

// An MCP server asks for input while its tool runs, as the agent passes it on (`session/mcp_elicitation.rs`, checked on
// v0.3.4): live `pending_interaction` (kind `mcp_elicitation`) under a made-up `mcp-elicit-<uuid>` id no tool row has,
// the request `_x.ai/mcp/elicit` from the main session (no requester: a helper's ask looks the same), then
// `interaction_resolved` once the wait ends. Nothing of it is stored. `giveUpOn`: once that file exists the server
// stops waiting; the agent sends `interaction_resolved` at once and the host's late answer goes nowhere. The answer is
// recorded as `elicit-answer`. Returns 'dropped' when the turn was.
// A link request (`mode: 'url'`) with `complete` set: once it is answered (any answer), the server says the browser
// part is done, as the agent passes on its `notifications/elicitation/complete` (`_x.ai/mcp/elicit_complete`,
// `{sessionId, elicitationId, serverName}`). `complete` (one, or a list sent in order) may name another
// `elicitationId`, `sessionId` or `server` (null: left out, as an older agent does), and wait `afterMs` first.
async function elicitStep(session, e) {
  const toolCallId = e.id ?? `mcp-elicit-${randomUUID()}`;
  xaiOnly(session, 'pending_interaction', { tool_call_id: toolCallId, kind: 'mcp_elicitation' });
  const turn = session.turn;
  turn.asking = true;
  let answered = false;
  let late = false;
  const mode = e.mode ?? 'form';
  const server = e.server ?? 'github';
  const elicitationId = e.elicitationId ?? 'e-1';
  const asked = agentRequest('_x.ai/mcp/elicit', elicitParams(session, e, toolCallId));
  void asked.then(async (answer) => {
    answered = true;
    record({ kind: 'elicit-answer', toolCallId, answer: answer ?? null, late });
    if (mode !== 'url' || e.complete === undefined) return;
    for (const item of Array.isArray(e.complete) ? e.complete : [e.complete]) {
      const done = typeof item === 'object' && item !== null ? item : {};
      if (typeof done.afterMs === 'number') await sleep(done.afterMs);
      const name = done.server === undefined ? server : done.server;
      send({
        method: '_x.ai/mcp/elicit_complete',
        params: {
          sessionId: done.sessionId ?? session.id,
          elicitationId: done.elicitationId ?? elicitationId,
          ...(name === null ? {} : { serverName: name }),
        },
      });
    }
  });
  if (typeof e.giveUpOn === 'string') while (!answered && !existsSync(e.giveUpOn)) await sleep(10);
  if (answered || typeof e.giveUpOn !== 'string') await asked;
  late = !answered;
  turn.asking = false;
  if (session.turn !== turn) return 'dropped';
  xaiOnly(session, 'interaction_resolved', { tool_call_id: toolCallId });
  return undefined;
}

/** The `_x.ai/mcp/elicit` params of an `elicit` step, from the main session, as the agent sends every one. */
function elicitParams(session, e, toolCallId) {
  const mode = e.mode ?? 'form';
  return {
    sessionId: session.id,
    toolCallId,
    serverName: e.server ?? 'github',
    message: e.message ?? 'The server needs input.',
    mode,
    ...(mode === 'url'
      ? { url: e.url ?? 'https://example.com/auth', elicitationId: e.elicitationId ?? 'e-1' }
      : e.schema === undefined
        ? {}
        : { requestedSchema: e.schema }),
  };
}

// A workflow helper's MCP server asks for input (`workflow` option `elicit`, after the helper's gate): the request
// comes from the main session with no requester, as every MCP request does, so while no typed turn or wake runs only
// the run waits on it. The answer is recorded as `elicit-answer`.
async function runElicit(session, e) {
  const toolCallId = e.id ?? `mcp-elicit-${randomUUID()}`;
  xaiOnly(session, 'pending_interaction', { tool_call_id: toolCallId, kind: 'mcp_elicitation' });
  const answer = await agentRequest('_x.ai/mcp/elicit', elicitParams(session, e, toolCallId));
  record({ kind: 'elicit-answer', toolCallId, answer: answer ?? null, late: false });
  xaiOnly(session, 'interaction_resolved', { tool_call_id: toolCallId });
}

/** One permission round trip: the selected option id, null when refused, undefined when the turn was dropped. */
async function permissionStep(session, toolCallId, toolCall, options, requestMeta) {
  session.lastAsk = nextAgentRequestId;
  const turn = session.turn;
  turn.asking = true;
  const answer = await agentRequest('session/request_permission', {
    sessionId: session.id,
    toolCall: { toolCallId, ...toolCall },
    options,
    ...(requestMeta === undefined ? {} : { _meta: requestMeta }),
  });
  turn.asking = false;
  if (session.turn !== turn) return undefined;
  xaiOnly(session, 'interaction_resolved', { tool_call_id: toolCallId });
  const outcome = answer?.outcome;
  if (session.turn) session.turn.permissionCancelled = outcome?.outcome !== 'selected';
  const optionId = outcome?.outcome === 'selected' ? outcome.optionId : null;
  // As the agent: a deny once with a note that is not blank is a follow-up message, not a rejection.
  const followup = answer?._meta?.followup_message;
  if (session.turn) {
    session.turn.note =
      optionId === 'reject-once' && typeof followup === 'string' && followup.trim() !== '' ? followup : null;
  }
  return ['allow-once', 'always-allow', 'allow-edits-session'].includes(optionId) ? optionId : null;
}

const EDIT_TOOLS = new Set(['write', 'search_replace']);
function grantKey(session, command) {
  return `${session.cwd}\0${command}`;
}

function rejected(id, session, toolCallId, name) {
  update(session, 'tool_call_update', {
    toolCallId,
    status: 'failed',
    content: [{ type: 'content', content: { type: 'text', text: `User rejected the execution for tool \`${name}\`` } }],
  });
  // As the agent: a cancelled answer ends the turn PermissionCancelled, a rejected one PermissionRejected.
  endCancelled(id, session, session.turn.permissionCancelled ? 'PermissionCancelled' : 'PermissionRejected');
}

// An edit as mlx-agent v0.3.4 plays it (`tool_calls.rs` send_tool_call_start, `acp_conversion.rs`, the search_replace
// and write tools): a `tool_call`, a start update with the asked-for `diff`, the request (unless `ask: false`), then the
// result with the made `diff` and its `_meta.details`, or a failed text. `before`: the file before the call (null: none).
const EDIT_OPTIONS = [
  { optionId: 'allow-edits-session', name: 'Yes, allow all edits this session', kind: 'allow_always' },
  { optionId: 'allow-once', name: 'Yes', kind: 'allow_once' },
  { optionId: 'reject-once', name: 'No', kind: 'reject_once' },
];

/** Rust `str::split_inclusive('\n')`. */
function inclusiveLines(text) {
  const lines = [];
  let from = 0;
  while (from < text.length) {
    const end = text.indexOf('\n', from);
    const stop = end === -1 ? text.length : end + 1;
    lines.push(text.slice(from, stop));
    from = stop;
  }
  return lines;
}

/** `build_edit_details` of the search_replace tool: each replaced match's line in the new file and 3 lines around it. */
function editDetails(before, oldString, newString, replaceAll) {
  let text = '';
  let last = 0;
  const starts = [];
  for (let at = before.indexOf(oldString); at !== -1; at = before.indexOf(oldString, last)) {
    text += before.slice(last, at);
    starts.push(text.length);
    text += newString;
    last = at + oldString.length;
    if (!replaceAll) break;
  }
  text += before.slice(last);
  const lines = inclusiveLines(text);
  return starts.map((start) => {
    const startLine = inclusiveLines(text.slice(0, start)).filter((line) => line.endsWith('\n')).length;
    const endLine = startLine + Math.max(1, inclusiveLines(newString).length) - 1;
    const from = Math.max(0, startLine - 3);
    const to = Math.min(endLine + 3, Math.max(0, lines.length - 1));
    return {
      old_string: oldString,
      old_line: startLine + 1,
      new_string: newString,
      new_line: startLine + 1,
      context_before: lines.slice(from, startLine).join(''),
      context_after: endLine < to ? lines.slice(endLine + 1, to + 1).join('') : '',
      line_prefix: text.slice(text.lastIndexOf('\n', start - 1) + 1, start),
    };
  });
}

async function editStep(id, session, e, toolCallId) {
  const write = e.toolName === 'write';
  const path = join(session.cwd, e.file_path);
  const before = e.before ?? null;
  const rawInput = write
    ? { variant: 'Write', file_path: e.file_path, content: e.content }
    : {
        variant: 'SearchReplace',
        file_path: e.file_path,
        old_string: e.old_string,
        new_string: e.new_string,
        ...(e.replace_all === true ? { replace_all: true } : {}),
      };
  const meta = { 'x.ai/tool': xaiTool(e.toolName) };
  const title = `${write ? 'Write' : 'Edit'} \`${e.file_path}\``;
  update(session, 'tool_call', {
    toolCallId,
    title: e.toolName,
    kind: 'other',
    status: 'pending',
    rawInput,
    _meta: meta,
  });
  const at = write || e.old_string === '' || before === null ? -1 : before.indexOf(e.old_string);
  const line = at === -1 ? null : before.slice(0, at).split('\n').length;
  const asked = write
    ? { type: 'diff', path, oldText: '', newText: e.content }
    : {
        type: 'diff',
        path,
        oldText: e.old_string,
        newText: e.new_string,
        ...(line === null ? {} : { _meta: { old_line: line, new_line: line } }),
      };
  update(session, 'tool_call_update', {
    toolCallId,
    title,
    kind: 'edit',
    locations: [{ path: e.file_path }],
    content: [asked],
    rawInput,
    _meta: meta,
  });
  let via = null;
  if (e.ask !== false && session.editsAllowed !== true) {
    const allowed = await permissionStep(
      session,
      toolCallId,
      { kind: 'edit', title, rawInput, _meta: meta },
      e.options ?? EDIT_OPTIONS,
    );
    if (allowed === undefined) return 'dropped';
    if (allowed === null) {
      rejected(id, session, toolCallId, e.toolName);
      return 'rejected';
    }
    via = session.lastAsk;
    if (allowed === 'allow-edits-session') {
      session.editsAllowed = true;
      session.editsGrant = via;
    }
  }
  record({ kind: 'exec', toolCallId, via, grantedBy: via === null ? (session.editsGrant ?? null) : null });
  const failure =
    e.fail ??
    (!write && (before === null || !before.includes(e.old_string)) ? 'No matches found for old_string.' : null);
  if (failure !== null) {
    update(session, 'tool_call_update', {
      toolCallId,
      status: 'failed',
      content: [{ type: 'content', content: { type: 'text', text: failure } }],
      rawOutput: { type: 'SearchReplace', NoMatchesFound: { message: failure, file_path: path } },
    });
    return 'failed';
  }
  const oldString = write ? (before ?? '') : e.old_string;
  const newString = write ? e.content : e.new_string;
  const details = write
    ? [
        {
          old_string: oldString,
          old_line: 1,
          new_string: newString,
          new_line: 1,
          context_before: '',
          context_after: '',
          line_prefix: '',
        },
      ]
    : editDetails(before, oldString, newString, e.replace_all === true);
  const said = !write
    ? 'The file was edited.'
    : before !== null
      ? `Wrote file successfully to ${path}.`
      : `The file ${path} has been created.`;
  fileWritten(session, e.file_path);
  update(session, 'tool_call_update', {
    toolCallId,
    status: 'completed',
    content: [{ type: 'diff', path, oldText: oldString, newText: newString, _meta: { details } }],
    rawOutput: {
      type: 'SearchReplace',
      EditsApplied: {
        old_string: oldString,
        new_string: newString,
        tool_output_for_prompt: said,
        absolute_path: path,
        edits: { details },
      },
    },
  });
  return 'completed';
}

/**
 * As the agent for a deny with a note: the call ends "not executed", the note becomes a user message stored twice
 * (one direct write, one through the event pipeline) with no `promptIndex` and never sent live, and the turn goes on.
 */
function noteDenied(session, toolCallId, name, note) {
  update(session, 'tool_call_update', {
    toolCallId,
    status: 'failed',
    content: [
      {
        type: 'content',
        content: {
          type: 'text',
          text: `The user elected to avoid running the ${name} tool. The tool was not executed. Please refer to the user's message for next steps.`,
        },
      },
    ],
  });
  for (let copy = 0; copy < 2; copy += 1) {
    persistLine(session, 'session/update', {
      sessionId: session.id,
      update: {
        sessionUpdate: 'user_message_chunk',
        content: { type: 'text', text: note },
        _meta: { modelId: session.model },
      },
      _meta: { eventId: `${session.id}-${++seq}`, agentTimestampMs: Date.now() },
    });
  }
  session.turn.note = null;
  record({ kind: 'note', toolCallId, note });
}

// The session mode as the agent keeps it (`default`, `plan`, `ask`): each move is one `current_mode_update`, sent and
// stored; a mode it already has sends none.
function setMode(session, mode) {
  if ((session.planMode ?? 'default') === mode) return;
  session.planMode = mode;
  update(session, 'current_mode_update', { currentModeId: mode });
}

// As the agent's `reconcile_plan_mode_with_prompt`: `_meta.mode` "plan" enters plan mode; any other value leaves it
// (to "ask" for "ask", else "default") and changes nothing outside plan mode. A prompt with no `_meta.mode` keeps the
// mode. The scenario's `noPlanMode` plays an older agent that reads no `_meta.mode`.
function promptMode(session, asked) {
  if (asked === 'plan') setMode(session, 'plan');
  else if (session.planMode === 'plan') setMode(session, asked === 'ask' ? 'ask' : 'default');
}

async function runPrompt(id, params) {
  const session = sessions.get(params?.sessionId);
  if (session === undefined) return replyError(id, -32602, 'Invalid params', { message: 'unknown session' });
  const block = Array.isArray(params.prompt) ? params.prompt[0] : null;
  const text = typeof block?.text === 'string' ? block.text : '';
  if (faults.exitOnPrompt !== undefined && faults.exitOnPrompt !== null) process.exit(faults.exitOnPrompt);
  // A script with `once` answers one prompt of the session; the next match takes the prompt after it.
  session.usedOnce ??= new Set();
  const scripts = scriptsOf(session);
  const index = scripts.findIndex((p, i) => text.includes(p.match ?? '') && !(p.once && session.usedOnce.has(i)));
  if (index !== -1 && scripts[index].once) session.usedOnce.add(index);
  // A script's `followUp` answers the session's next prompt that holds its `match`.
  const pending = session.followUp;
  session.followUp = undefined;
  const script = (pending !== undefined && text.includes(pending.match ?? '') ? pending : scripts[index]) ?? {
    steps: [{ text: 'ok' }, { end: { stopReason: 'end_turn' } }],
  };
  if (script.followUp !== undefined) session.followUp = script.followUp;
  // A blocked prompt (`acp_agent.rs`): one live line, `end_turn` at once, nothing stored. `noModel` is the agent's words.
  const blocked = script.noModel ?? session.noModel;
  if (blocked !== undefined) {
    record({ kind: 'no-model', text });
    modelAutoSwitched(session, '', '', blocked);
    return reply(id, { stopReason: 'end_turn' });
  }
  // A prompt a `user_prompt_submit` hook blocks (`turn.rs`): the batch's start, its `hook_execution` with the run
  // `blocked`, the agent's note, then a cancelled end with the category `HookDenied`. The prompt's chunk is never stored.
  if (script.promptBlocked !== undefined) {
    const b = script.promptBlocked;
    const promptId = randomUUID();
    record({ kind: 'prompt-blocked', text });
    hookRunStarted(session, { event_name: 'user_prompt_submit', prompt_id: promptId });
    if (b.waitMs !== undefined) await sleep(b.waitMs);
    update(
      session,
      'hook_execution',
      {
        event_name: 'user_prompt_submit',
        prompt_id: promptId,
        runs: [
          {
            name: b.hook,
            status: { status: 'failed', error: `blocked: ${b.reason}`, elapsed_ms: b.waitMs ?? 40, blocked: true },
          },
        ],
      },
      true,
    );
    update(session, 'hook_annotation', { message: `\u26a0 Prompt blocked by ${hookShown(b.hook)}: ${b.reason}` }, true);
    const cancel = {
      cancellationCategory: 'HookDenied',
      cancellationContext: { reason: b.reason, hook_name: b.hook },
    };
    update(session, 'turn_completed', { prompt_id: promptId, stop_reason: 'cancelled', elapsed_ms: 10 }, true, cancel);
    return reply(id, { stopReason: 'cancelled', _meta: { sessionId: session.id, ...cancel } });
  }
  // A built-in runs with no model call (`turn.rs`): the typed text stored as a `hostTurn` user chunk (not sent live),
  // its output a live and stored `hostTurn` agent chunk, then `end_turn`. A script whose `match` names it plays instead;
  // a `goal` script plays only `/goal <objective>`, as `/goal pause|resume|clear|status` are the agent's own.
  const builtin =
    index === -1 ||
    (scripts[index].match ?? '') === '' ||
    scripts[index].goal !== undefined ||
    text.trim().startsWith('/workflow read-pdf ')
      ? hostTurnCommand(session, text)
      : null;
  if (builtin !== null) {
    record({ kind: 'host-turn', name: builtin, text });
    const goalWord = builtin === 'goal' ? text.trim().slice('/goal'.length).trim().toLowerCase() : null;
    session.goal ??= null;
    if (goalWord === 'resume' && GOAL_PAUSED.includes(session.goal?.status)) return goalResume(id, session);
    // The state line is stored before the typed text, as the agent's store writes them.
    const goalSaid = goalWord === null ? null : goalCommand(session, goalWord);
    persistLine(session, 'session/update', {
      sessionId: session.id,
      update: {
        sessionUpdate: 'user_message_chunk',
        content: { type: 'text', text: text.trim() },
        _meta: { hostTurn: true },
      },
      _meta: { eventId: `${session.id}-${++seq}`, agentTimestampMs: Date.now() },
    });
    // The scenario's words stand for a built-in typed bare (`/goal`, its status); a goal command says the agent's own.
    const scripted = goalWord === null || goalWord === '' || goalWord === 'status';
    const managed = builtin === 'workflow' ? (readPdfStart(session, text) ?? manageRun(session, text)) : null;
    const said = managed ?? (scripted ? scenario.hostTurnText?.[builtin] : undefined) ?? goalSaid ?? `Ran /${builtin}.`;
    update(session, 'agent_message_chunk', { content: { type: 'text', text: said }, _meta: { hostTurn: true } });
    // v0.3.6 ends a `/workflow` turn with a live and stored `turn_completed`.
    if (builtin === 'workflow')
      update(session, 'turn_completed', { prompt_id: randomUUID(), stop_reason: 'end_turn', elapsed_ms: 28 }, true);
    persistSummary(session);
    return reply(id, { stopReason: 'end_turn', _meta: { sessionId: session.id } });
  }
  session.turn = { promptId: randomUUID(), cancelled: false, wake: null, rpcId: id, startedAt: Date.now() };
  // A typed prompt drops a recap being written (`invalidate_side_calls_for_new_prompt`).
  session.recapEpoch = (session.recapEpoch ?? 0) + 1;
  // The prompt's mode line is stored before its chunk, as the agent writes them (`turn.rs`).
  if (typeof params._meta?.mode === 'string' && scenario.noPlanMode !== true) promptMode(session, params._meta.mode);
  // `/goal <objective>` (`setup_goal`): the goal's first line (and the planner's, `planning`) before the prompt, which
  // the agent runs and stores as its goal rules. `budget` as `--budget`; `wakeRounds` rounds after the turn's end.
  if (script.goal !== undefined) {
    const gs = script.goal;
    session.goal = {
      id: randomUUID(),
      objective: gs.objective ?? text.trim().slice('/goal'.length).trim(),
      status: 'active',
      tokens: 0,
      elapsed: 0,
      rounds: 0,
      round: 0,
      wakeRounds: gs.wakeRounds ?? 0,
      ...(gs.budget === undefined ? {} : { budget: gs.budget }),
      ...(gs.wakeRoundMs === undefined ? {} : { wakeRoundMs: gs.wakeRoundMs }),
      ...(gs.wakeEndOn === undefined ? {} : { wakeEndOn: gs.wakeEndOn }),
      ...(gs.end === undefined ? {} : { end: gs.end }),
      ...(gs.endMessage === undefined ? {} : { endMessage: gs.endMessage }),
      ...(gs.gapMs === undefined ? {} : { gapMs: gs.gapMs }),
      ...(gs.after === undefined ? {} : { after: gs.after }),
    };
    goalLine(session, true, { last_event: 'goal_created' });
    if (gs.planning === true) {
      goalLine(session, true, { last_event: 'goal_created', planning: true });
      goalLine(session, true, { last_event: 'planning_completed' });
    }
  }
  session.messages += 1;
  session.promptTexts = [...(session.promptTexts ?? []).slice(0, session.messages - 1), text];
  persistLine(session, 'session/update', {
    sessionId: session.id,
    update: {
      sessionUpdate: 'user_message_chunk',
      content: { type: 'text', text: script.goal === undefined ? text : goalRules(session.goal.objective, false) },
      _meta: { modelId: session.model, promptIndex: session.messages - 1 },
    },
    _meta: { eventId: `${session.id}-${++seq}`, agentTimestampMs: Date.now() },
  });
  // The note of a PDF for the PDF tool, its own stored chunk, as the agent stores each block of a prompt.
  for (const block of Array.isArray(params.prompt) ? params.prompt : []) {
    if (promptPdf([block]) === null) continue;
    persistLine(session, 'session/update', {
      sessionId: session.id,
      update: {
        sessionUpdate: 'user_message_chunk',
        content: block,
        _meta: { modelId: session.model, promptIndex: session.messages - 1 },
      },
      _meta: { eventId: `${session.id}-${++seq}`, agentTimestampMs: Date.now() },
    });
  }
  // `promptPictures`: the agent's one-shots on the prompt's own pictures, made after its chunks (`turn.rs` writes the
  // chunks, then runs `normalize_images_with_notices`), sent and stored: `image_dropped` by default, or the
  // `sessionUpdate` named. The real store can put a later picture chunk of the prompt after them.
  for (const line of script.promptPictures ?? []) {
    const { sessionUpdate = 'image_dropped', ...fields } = line;
    update(session, sessionUpdate, fields, true);
  }
  if (!session.titled && scenario.noTitle !== true && script.noTitle !== true) {
    session.titled = true;
    session.title = scenario.blankTitle === true ? ' ' : text.slice(0, 50);
    send({
      method: 'session/update',
      params: { sessionId: session.id, update: { sessionUpdate: 'session_info_update', title: session.title } },
    });
    persistSummary(session);
  }
  // Step numbers run on across the session's turns, so tool call ids stay unique, as the agent's are.
  let n = session.steps ?? 0;
  const pdf = promptPdf(params.prompt);
  const pdfRunId = `wf_pdf_${n + 1}`;
  for (const scripted of script.steps) {
    const step = withPdf(scripted, pdf, pdfRunId);
    if (session.turn === null) return;
    if (session.turn.cancelled) return endCancelled(id, session, 'MidTurnAbort');
    n += 1;
    session.steps = n;
    if (step.sleepMs !== undefined) {
      await new Promise((resolve) => {
        const timer = setTimeout(resolve, step.sleepMs);
        session.turn.wake = () => {
          clearTimeout(timer);
          resolve();
        };
      });
      continue;
    }
    // `slowLoad`: the next `session/load` of this session, in a later agent process, answers `slowLoad` ms late (a
    // marker beside its updates.jsonl, used once).
    if (step.slowLoad !== undefined) {
      if (scenario.persist && grokHome !== '') {
        mkdirSync(persistDir(session), { recursive: true });
        writeFileSync(join(persistDir(session), 'slow_load.json'), JSON.stringify({ ms: step.slowLoad }));
      }
      continue;
    }
    // The user's skills or workflows changed mid-turn (a SKILL.md written): the session's list is pushed again.
    if (step.commands !== undefined) {
      session.commands = step.commands;
      commandsUpdate(session, step.totalTokens);
      continue;
    }
    // A test's gate: the turn goes on once the file `holdOn` exists (Stop still ends it at once).
    // A round of the goal inside the turn (the agent runs its rounds in the typed prompt), or a change of its state.
    if (step.goalRound !== undefined) {
      await goalRound(session, step.goalRound);
      if (step.goalRound.text !== undefined)
        update(session, 'agent_message_chunk', { content: { type: 'text', text: step.goalRound.text } });
      continue;
    }
    if (step.goalStatus !== undefined) {
      session.goal.status = step.goalStatus.status;
      if (step.goalStatus.pauseMessage !== undefined) session.goal.pauseMessage = step.goalStatus.pauseMessage;
      goalLine(session, true, step.goalStatus.extra ?? {});
      continue;
    }
    if (step.holdOn !== undefined) {
      await new Promise((resolve) => {
        const timer = setInterval(() => {
          if (existsSync(step.holdOn)) done();
        }, 10);
        const done = () => {
          clearInterval(timer);
          resolve();
        };
        session.turn.wake = done;
      });
      continue;
    }
    // As the agent's auto compaction (`run_compact_only`): started, then completed or failed after `ms` (`none`: no end
    // update, as when the agent dies); Stop during it sends cancelled and ends the turn.
    if (step.compact !== undefined) {
      const c = step.compact;
      const percentage = c.percentage ?? 85;
      update(
        session,
        'auto_compact_started',
        {
          tokens_used: c.tokensBefore ?? 222_822,
          context_window: c.window ?? 262_144,
          percentage,
          reason: `Context window ${percentage}% full`,
        },
        true,
      );
      await new Promise((resolve) => {
        const timer = setTimeout(resolve, c.ms ?? 0);
        session.turn.wake = () => {
          clearTimeout(timer);
          resolve();
        };
      });
      if (session.turn === null) return;
      if (session.turn.cancelled) {
        update(session, 'auto_compact_cancelled', { reason: 'user_cancelled' }, true);
        return endCancelled(id, session, 'MidTurnAbort');
      }
      if (c.outcome === 'none') continue;
      if (c.outcome === 'failed')
        update(
          session,
          'auto_compact_failed',
          { error: c.error ?? "it'll retry on the next turn, or start a new session using /new.\nstream idle timeout" },
          true,
        );
      else
        update(
          session,
          'auto_compact_completed',
          {
            tokens_before: c.tokensBefore ?? 222_822,
            tokens_after: c.tokensAfter ?? 31_204,
            elapsed_ms: c.ms ?? 0,
            summary_preview: null,
          },
          true,
        );
      continue;
    }
    if (step.exit !== undefined) process.exit(step.exit);
    if (step.thought !== undefined)
      await streamText(session, 'agent_thought_chunk', step.thought, step.chunk, step.delayMs, step.totalTokens);
    else if (step.text !== undefined)
      await streamText(session, 'agent_message_chunk', step.text, step.chunk, step.delayMs, step.totalTokens);
    else if (step.webCall !== undefined) {
      const call = step.webCall;
      const toolCallId = call.id ?? `toolu_${n}`;
      const direct = call.shape === 'direct';
      const name = direct ? call.tool_name : 'use_tool';
      const variant = direct ? 'MCPTool' : 'UseTool';
      const bare = direct ? call.tool_input : { tool_name: call.tool_name, tool_input: call.tool_input };
      const tagged = call.rawInput ?? { variant, tool_name: call.tool_name, tool_input: call.tool_input };
      const meta = { 'x.ai/tool': xaiTool(name) };
      callDeltas(session, toolCallId, name, bare);
      update(session, 'tool_call', { toolCallId, title: name, rawInput: bare, _meta: meta });
      xaiOnly(session, 'pending_interaction', { tool_call_id: toolCallId, kind: 'permission' });
      update(session, 'tool_call_update', {
        toolCallId,
        kind: 'other',
        title: call.tool_name,
        locations: [],
        rawInput: tagged,
        _meta: meta,
      });
      const options = call.offerEnableAlwaysApprove
        ? [
            ...PERMISSION_OPTIONS,
            { optionId: 'enable-always-approve', name: 'enable always approve', kind: 'allow_once' },
          ]
        : PERMISSION_OPTIONS;
      const allowed = await permissionStep(
        session,
        toolCallId,
        { kind: 'other', title: call.tool_name, rawInput: tagged, _meta: meta },
        options,
      );
      if (allowed === undefined) return;
      if (allowed === null) return rejected(id, session, toolCallId, call.tool_name);
      const webTool = call.tool_name === 'mlx_web__web_search' || call.tool_name === 'mlx_web__web_fetch';
      const failed = (text, message) =>
        update(session, 'tool_call_update', {
          toolCallId,
          status: 'failed',
          content: [{ type: 'content', content: { type: 'text', text } }],
          rawOutput: { error: 'tool_execution_failed', message },
        });
      if (step.onAllow?.fail !== undefined) {
        failed(`Tool \`${call.tool_name}\` failed: ${step.onAllow.fail}`, step.onAllow.fail);
      } else if (!webTool) {
        const qualified = call.tool_name.includes('__');
        const message = qualified
          ? `Tool not found: ${call.tool_name}`
          : `'${call.tool_name}' is not a valid MCP tool name. Tool names must be qualified as \`server__tool\` (e.g., \`linear__save_issue\`). Use \`search_tool\` to discover available tools.`;
        failed(`Tool \`${call.tool_name}\` failed via \`use_tool\`: ${message}`, message);
      } else {
        update(session, 'tool_call_update', {
          toolCallId,
          status: 'completed',
          rawOutput: {
            type: 'MCP',
            tool_name: call.tool_name.slice('mlx_web__'.length),
            server_name: 'mlx_web',
            output: { OkayOutput: step.onAllow?.okayOutput ?? '' },
          },
        });
      }
    } else if (step.parallel !== undefined) {
      // Shell requests sent at the same time, as parallel tool calls of one answer.
      await Promise.all(
        // Each entry is a command, or `{ command, complete: false }` for a call that has not ended when the step does.
        step.parallel.map(async (entry, k) => {
          const command = typeof entry === 'string' ? entry : entry.command;
          const toolCallId = `toolu_${n}_p${k}`;
          const meta = { 'x.ai/tool': xaiTool('run_terminal_command') };
          const rawInput = { variant: 'Bash', command, description: 'run', is_background: false };
          update(session, 'tool_call', { toolCallId, title: 'run_terminal_command', rawInput, _meta: meta });
          const toolCall = { kind: 'execute', title: 'run', rawInput, _meta: meta };
          const answer = await permissionStep(session, toolCallId, toolCall, PERMISSION_OPTIONS, {
            hookAsk: { hookName: session.guard },
          });
          record({ kind: 'parallel', toolCallId, answer: answer ?? null });
          if (answer !== undefined && answer !== null && entry.complete !== false) {
            update(session, 'tool_call_update', { toolCallId, status: 'completed' });
          }
        }),
      );
    } else if (step.permission !== undefined) {
      const p = step.permission;
      const toolCallId = p.id ?? `toolu_${n}`;
      const meta = { 'x.ai/tool': p.tool ?? xaiTool(p.toolName) };
      // `bare: true`: no `tool_call` update first, as a helper agent's request arrives.
      if (p.bare !== true)
        update(session, 'tool_call', { toolCallId, title: p.toolName, rawInput: p.rawInput, _meta: meta });
      const command = typeof p.rawInput?.command === 'string' ? p.rawInput.command : null;
      const grantedBy =
        command !== null && grants.has(grantKey(session, command))
          ? grants.get(grantKey(session, command))
          : EDIT_TOOLS.has(p.toolName) && session.editsAllowed === true
            ? session.editsGrant
            : null;
      let via = null;
      if (grantedBy === null) {
        const options = [
          ...(p.options ?? [
            ...PERMISSION_OPTIONS,
            { optionId: 'reject-always', name: 'reject always', kind: 'reject_always' },
          ]),
        ];
        if (p.offerEnableAlwaysApprove)
          options.push({ optionId: 'enable-always-approve', name: 'enable always approve', kind: 'allow_once' });
        // `hookAsk` in a step: 'guard' is this start's guard, another string names another hook, null sends none.
        const hookName =
          p.hookAsk === undefined
            ? session.guard !== null && p.toolName === 'run_terminal_command'
              ? session.guard
              : null
            : p.hookAsk === 'guard'
              ? session.guard
              : p.hookAsk;
        const guarded = hookName !== null;
        const title = p.title ?? p.toolName;
        const allowed = await permissionStep(
          session,
          toolCallId,
          {
            kind: p.kind ?? 'execute',
            title: guarded ? `${title} — hook '${hookName}' asks for confirmation` : title,
            rawInput: p.rawInput,
            _meta: meta,
          },
          options,
          guarded || p.meta !== undefined
            ? {
                ...p.meta,
                ...(guarded
                  ? { hookAsk: { hookName, ...(p.hookReason === undefined ? {} : { reason: p.hookReason }) } }
                  : {}),
              }
            : undefined,
        );
        if (allowed === undefined) return;
        if (allowed === null && session.turn.note) {
          noteDenied(session, toolCallId, p.toolName, session.turn.note);
          continue;
        }
        if (allowed === null) return rejected(id, session, toolCallId, p.toolName);
        via = session.lastAsk;
        if (allowed === 'always-allow' && command !== null) grants.set(grantKey(session, command), via);
        if (allowed === 'allow-edits-session') {
          session.editsAllowed = true;
          session.editsGrant = via;
        }
      }
      record({ kind: 'exec', toolCallId, via, grantedBy });
      const output =
        p.outputEnv !== undefined ? `${p.outputEnv}=${process.env[p.outputEnv] ?? ''}` : (p.output ?? 'done');
      // `complete: false`: a call that never reports its end. `rawOutput`: the tool's own result (a run's id).
      if (p.complete !== false) {
        update(session, 'tool_call_update', {
          toolCallId,
          status: 'completed',
          content: [{ type: 'content', content: { type: 'text', text: output } }],
          ...(p.rawOutput === undefined ? {} : { rawOutput: p.rawOutput }),
        });
      }
    } else if (step.planMode !== undefined) {
      // The model moved the mode itself (`enter_plan_mode`): the agent's bridge sends and stores the line.
      setMode(session, step.planMode);
    } else if (step.exitPlan !== undefined) {
      const toolCallId = step.exitPlan.id ?? `toolu_${n}`;
      const meta = { 'x.ai/tool': xaiTool('exit_plan_mode') };
      update(session, 'tool_call', { toolCallId, title: 'exit_plan_mode', rawInput: {}, _meta: meta });
      xaiOnly(session, 'pending_interaction', { tool_call_id: toolCallId, kind: 'plan_approval' });
      const planContent = 'planContent' in step.exitPlan ? step.exitPlan.planContent : null;
      const answer = await agentRequest('_x.ai/exit_plan_mode', {
        sessionId: session.id,
        toolCallId,
        ...(step.exitPlan.omitPlanContent === true ? {} : { planContent }),
      });
      xaiOnly(session, 'interaction_resolved', { tool_call_id: toolCallId });
      const approved = answer?.outcome === 'approved';
      // As the agent: the approved `exit_plan_mode` call leaves plan mode (the bridge's `PlanModeExited`).
      if (approved && session.planMode === 'plan') setMode(session, 'default');
      const feedback = typeof answer?.feedback === 'string' ? ` Feedback: ${answer.feedback}` : '';
      // As the agent: with no plan file (`planContent` null) a rejection's feedback is dropped.
      const noPlan = typeof planContent !== 'string';
      update(session, 'tool_call_update', {
        toolCallId,
        status: 'completed',
        content: [
          {
            type: 'content',
            content: {
              type: 'text',
              text: approved
                ? 'Your plan has been approved. You can now start coding.'
                : noPlan
                  ? 'The user does not want to exit plan mode. Continue planning and ask the user what they would like to do.'
                  : `The user did not approve the plan.${feedback}`,
            },
          },
        ],
      });
    } else if (step.question !== undefined) {
      const done = await questionStep(session, step.question, step.question.id ?? `toolu_${n}`);
      if (done === 'dropped') return;
    } else if (step.elicit !== undefined) {
      const done = await elicitStep(session, step.elicit);
      if (done === 'dropped') return;
    } else if (step.policyDenied !== undefined) {
      const d = step.policyDenied;
      const toolCallId = d.id ?? `toolu_${n}`;
      const meta = { 'x.ai/tool': xaiTool(d.toolName) };
      callDeltas(session, toolCallId, d.toolName, d.rawInput ?? {});
      update(session, 'tool_call', { toolCallId, title: d.toolName, rawInput: d.rawInput ?? {}, _meta: meta });
      xaiOnly(session, 'pending_interaction', { tool_call_id: toolCallId, kind: 'permission' });
      update(session, 'tool_call_update', {
        toolCallId,
        kind: 'other',
        title: d.title ?? d.toolName,
        locations: [],
        rawInput: d.taggedInput ?? d.rawInput ?? {},
        _meta: meta,
      });
      xaiOnly(session, 'interaction_resolved', { tool_call_id: toolCallId });
      update(session, 'tool_call_update', {
        toolCallId,
        status: 'failed',
        content: [
          {
            type: 'content',
            content: {
              type: 'text',
              text: `Tool \`${d.toolName}\` was not executed: Denied by permission policy: deny rule on ${d.rule}`,
            },
          },
        ],
      });
    } else if (step.giveUp !== undefined) {
      // A request the agent stops waiting on, as at its time limit (`ask_user_question`) or when an MCP server leaves:
      // `pending_interaction` from the asking session (a `requester`'s for a helper's ask), the request, then after the
      // gate (`on`, a file; else `afterMs`) `interaction_resolved` from that session (`pending_interaction.rs`); it
      // records the answer the host still sends. `background: true`: the script goes on at once; else it waits for the
      // resolve line.
      const g = step.giveUp;
      const toolCallId = g.id ?? `toolu_${n}`;
      const asker = g.requester?.sessionId ?? session.id;
      const meta = { 'x.ai/tool': xaiTool(g.kind === 'plan' ? 'exit_plan_mode' : 'run_terminal_command') };
      const rawInput = { variant: 'Bash', command: g.command ?? 'ls', description: 'run', is_background: false };
      if (g.requester === undefined) {
        update(session, 'tool_call', {
          toolCallId,
          title: g.kind === 'plan' ? 'exit_plan_mode' : 'run_terminal_command',
          rawInput: g.kind === 'plan' ? {} : rawInput,
          _meta: meta,
        });
      }
      const signal = (sessionUpdate, fields) =>
        send({
          method: '_x.ai/session_notification',
          params: { sessionId: asker, update: { sessionUpdate, tool_call_id: toolCallId, ...fields } },
        });
      signal('pending_interaction', { kind: g.kind === 'plan' ? 'plan_approval' : 'permission' });
      const pending =
        g.kind === 'plan'
          ? agentRequest('_x.ai/exit_plan_mode', { sessionId: session.id, toolCallId, planContent: g.plan ?? '# Plan' })
          : agentRequest('session/request_permission', {
              sessionId: session.id,
              toolCall: { toolCallId, kind: 'execute', title: 'run', rawInput, _meta: meta },
              options: PERMISSION_OPTIONS,
              _meta: {
                ...(session.guard === null ? {} : { hookAsk: { hookName: session.guard } }),
                ...(g.requester === undefined
                  ? {}
                  : {
                      'x.ai/requester': {
                        sessionId: g.requester.sessionId,
                        label: g.requester.label ?? 'helper',
                        runId: g.requester.runId ?? null,
                        cwd: g.requester.cwd ?? session.cwd,
                      },
                    }),
              },
            });
      void pending.then((answer) => record({ kind: 'given-up', toolCallId, answer: answer ?? null }));
      const resolved = (async () => {
        if (typeof g.on === 'string') while (!existsSync(g.on)) await sleep(10);
        else await sleep(g.afterMs ?? 0);
        signal('interaction_resolved', {});
      })();
      if (g.background !== true) await resolved;
    } else if (step.badPermission !== undefined) {
      // A permission request with no tool call and no options: the reader cannot parse it.
      const answer = await agentRequest('session/request_permission', { sessionId: session.id });
      record({ kind: 'bad-permission', answer: answer ?? null });
    } else if (step.notify !== undefined) {
      send({ method: step.notify.method, params: { sessionId: session.id, ...step.notify.params } });
    } else if (step.serverStatus !== undefined) {
      send({
        method: '_x.ai/mcp/server_status',
        params: { sessionId: session.id, name: 'mlx_web', source: 'local', tools: null, ...step.serverStatus },
      });
    } else if (step.progress !== undefined) {
      xaiOnly(session, 'stream_progress', { progress: step.progress });
    } else if (step.retry !== undefined) {
      // As the agent's retry one-shots (`extensions/notification.rs`): sent at once and stored. `retry_state` by
      // default (`type` retrying / exhausted / failed, snake_case fields); `sessionUpdate` names an auto_recovery_* one.
      const { sessionUpdate = 'retry_state', ...fields } = step.retry;
      update(session, sessionUpdate, fields, true);
    } else if (step.hookStarted !== undefined) {
      // A batch of hooks the turn waits on: live only. `prompt_id` true names the turn's own prompt.
      const { prompt_id: promptId, ...fields } = step.hookStarted;
      hookRunStarted(session, { ...fields, ...(promptId === true ? { prompt_id: session.turn.promptId } : {}) });
    } else if (step.hook !== undefined) {
      // A hook batch's outcome (`hook_dispatch.rs` `send_hook_execution`): sent at once and stored. `prompt_id` true
      // names the turn's own prompt; a string names another turn's (a late turn-end report).
      const { prompt_id: promptId, ...fields } = step.hook;
      const prompt =
        promptId === true
          ? { prompt_id: session.turn.promptId }
          : typeof promptId === 'string'
            ? { prompt_id: promptId }
            : {};
      update(session, 'hook_execution', { ...fields, ...prompt }, true);
    } else if (step.hookNote !== undefined) {
      // A hook's note (`hook_annotation`, kind `note` left out as the agent does): sent at once and stored.
      update(session, 'hook_annotation', step.hookNote, true);
    } else if (step.hookDenied !== undefined) {
      await hookDeniedStep(session, step.hookDenied, step.hookDenied.id ?? `toolu_${n}`);
    } else if (step.picture !== undefined) {
      // A picture of a tool result the agent dropped (`tool_calls.rs`) or a retry without the pictures
      // (`image_strip.rs`): `image_dropped` by default, sent at once and stored.
      const { sessionUpdate = 'image_dropped', ...fields } = step.picture;
      update(session, sessionUpdate, fields, true);
    } else if (step.pictureRead !== undefined) {
      pictureReadStep(session, step.pictureRead, step.pictureRead.id ?? `toolu_${n}`);
    } else if (step.deepRawInput !== undefined) {
      const depth = step.deepRawInput;
      const head = JSON.stringify({
        jsonrpc: '2.0',
        method: 'session/update',
        params: { sessionId: session.id, update: { sessionUpdate: 'tool_call_update', toolCallId: `toolu_${n}` } },
      });
      // Built as text: JSON.stringify of this value would itself overflow the stack.
      const nested = '['.repeat(depth) + ']'.repeat(depth);
      process.stdout.write(`${head.slice(0, -3)},"rawInput":${nested}}}}\n`);
    } else if (step.edit !== undefined) {
      const done = await editStep(id, session, step.edit, step.edit.id ?? `toolu_${n}`);
      if (done === 'dropped' || done === 'rejected') return;
    } else if (step.tool !== undefined) {
      const t = step.tool;
      const toolCallId = t.id ?? `toolu_${n}`;
      const meta = { 'x.ai/tool': xaiTool(t.toolName) };
      update(session, 'tool_call', {
        toolCallId,
        title: t.title ?? t.toolName,
        ...(t.kind === undefined ? {} : { kind: t.kind }),
        rawInput: t.rawInput ?? {},
        _meta: meta,
      });
      update(session, 'tool_call_update', {
        toolCallId,
        status: t.status ?? 'completed',
        content: [{ type: 'content', content: { type: 'text', text: t.output ?? '' } }],
      });
    } else if (step.background !== undefined) {
      backgroundStep(session, step.background, step.background.id ?? `toolu_${n}`);
    } else if (step.helper !== undefined) {
      helperStep(session, step.helper, step.helper.id ?? `toolu_${n}`);
    } else if (step.helperProgress !== undefined) {
      helperProgressStep(session, step.helperProgress);
    } else if (step.monitorEvents !== undefined) {
      monitorEventsStep(session, step.monitorEvents);
    } else if (step.drain !== undefined) {
      drainStep(session, step.drain);
    } else if (step.schedule !== undefined) {
      scheduleStep(session, step.schedule, step.schedule.id ?? `toolu_${n}`);
    } else if (step.shell !== undefined) {
      await shellStep(session, step.shell, step.shell.id ?? `toolu_${n}`);
    } else if (step.workflow !== undefined) {
      startWorkflow(session, step.workflow, n);
    } else if (step.readPdf !== undefined) {
      // With no PDF for the tool in the prompt, the model has none to read.
      if (step.readPdf.pdf !== '$PDF') readPdfRun(session, step.readPdf, n, step.readPdf.pdf, null);
    } else if (step.runControl !== undefined) {
      runControlStep(session, step.runControl, step.runControl.id ?? `toolu_${n}`);
    } else if (step.plan !== undefined) {
      update(session, 'plan', { entries: step.plan });
    } else if (step.modelChanged !== undefined) {
      update(session, 'model_changed', { model_id: step.modelChanged }, true);
    } else if (step.turnCompleted !== undefined) {
      // A turn the agent gave up on (after `retry_state` `failed`) stores `stop_reason` "error" with its words and kind.
      const failed = step.turnCompleted.agentResult;
      update(
        session,
        'turn_completed',
        {
          prompt_id: session.turn.promptId,
          stop_reason: failed === undefined ? 'end_turn' : 'error',
          ...(failed === undefined
            ? {}
            : {
                agent_result: failed,
                ...(step.turnCompleted.errorKind === undefined ? {} : { error_kind: step.turnCompleted.errorKind }),
              }),
          usage: {
            inputTokens: step.turnCompleted.inputTokens ?? 0,
            outputTokens: step.turnCompleted.outputTokens ?? 0,
            modelCalls: 1,
          },
          // `elapsed: 'turn'`: the time since the prompt came, as the agent counts it; else a fixed 10 ms.
          elapsed_ms: step.turnCompleted.elapsed === 'turn' ? Date.now() - session.turn.startedAt : 10,
        },
        true,
      );
    } else if (step.lengthError !== undefined) {
      // As the agent on a reasoning-only length stop: the prompt's error reply, then its durable turn end line.
      const l = step.lengthError;
      const promptId = session.turn.promptId;
      session.turn = null;
      replyError(id, -32603, 'Internal error', {
        message: 'response truncated by max_tokens',
        error_kind: 'max_tokens_truncation',
      });
      if (l.exitAfterMs !== undefined) setTimeout(() => process.exit(0), l.exitAfterMs);
      if (l.noEnd === true) return;
      if (l.endAfterMs !== undefined) await sleep(l.endAfterMs);
      update(
        session,
        'turn_completed',
        {
          prompt_id: promptId,
          stop_reason: 'error',
          agent_result: 'response truncated by max_tokens',
          error_kind: 'max_tokens_truncation',
          usage: { inputTokens: 100, outputTokens: l.outputTokens ?? 28384, modelCalls: 1 },
          elapsed_ms: 20,
        },
        true,
      );
      return;
    } else if (step.error !== undefined) {
      session.turn = null;
      return replyError(id, step.error.code, step.error.message ?? 'Internal error', step.error.data);
    } else if (step.readSteers !== undefined) {
      readSteers(session);
    } else if (step.end !== undefined) {
      const left = session.turn.steers ?? [];
      session.turn = null;
      commandsUpdate(session);
      persistSummary(session);
      if (step.end.wake?.together === true) {
        batch = [];
        update(
          session,
          'turn_completed',
          {
            prompt_id: `typed-${randomUUID()}`,
            stop_reason: step.end.stopReason,
            usage: { inputTokens: 1, outputTokens: 1, modelCalls: 1 },
            elapsed_ms: 10,
          },
          true,
        );
      }
      reply(id, { stopReason: step.end.stopReason, _meta: { sessionId: session.id, ...step.end.meta } });
      if (step.end.wake !== undefined) void wakeLater(session, step.end.wake);
      if (left.length > 0) void steerTurns(session, left);
      if (session.goal?.status === 'active' && (session.goal.wakeRounds > 0 || session.goal.after !== undefined))
        void goalWakes(session);
      return;
    }
  }
  if (session.turn?.cancelled) return endCancelled(id, session, 'MidTurnAbort');
  const left = session.turn?.steers ?? [];
  session.turn = null;
  reply(id, { stopReason: 'end_turn', _meta: { sessionId: session.id } });
  if (left.length > 0) void steerTurns(session, left);
}

// Background work that outlives the turn, as stored v0.3.4 sessions hold it (`tools/notification_bridge.rs`,
// `extensions/notification.rs`): a command or monitor, a subagent, a schedule. Each ends behind its gate: `endOn` (a
// file), else `endAfterMs`; with neither it runs until the process exits.
async function gateOpens(on, ms) {
  if (typeof on === 'string') {
    while (!existsSync(on)) await sleep(10);
    return true;
  }
  if (typeof ms === 'number') {
    await sleep(ms);
    return true;
  }
  return false;
}

/** An xAI update the agent sends live on a method of its own and stores as `_x.ai/session/update`. */
function xaiUpdate(session, method, sessionUpdate, fields) {
  const params = {
    sessionId: session.id,
    update: { sessionUpdate, ...fields },
    _meta: { eventId: `${session.id}-${++seq}`, agentTimestampMs: Date.now() },
  };
  send({ method, params });
  persistLine(session, '_x.ai/session/update', params);
}

/** `background_tasks`: the session's whole task list, last wins, after each start and end. */
function taskList(session) {
  update(session, 'background_tasks', { tasks: [...(session.tasks?.values() ?? [])].map((t) => ({ ...t })) }, true);
}

function systemTime(ms) {
  return { secs_since_epoch: Math.floor(ms / 1000), nanos_since_epoch: (ms % 1000) * 1_000_000 };
}

// `_x.ai/task/kill` (`extensions/task.rs`): `{result: {taskId, outcome}}`, `killed` for a running task of the session,
// `already_exited` for an ended one, `not_found` for any other; a session the process does not hold gets
// `{result: null, error}`. The end line follows the reply. `faults.killReply: 'error'` answers with the error envelope,
// `'rpc-error'` with a JSON-RPC error, `'already_exited'` or `'not_found'` with that outcome and leaves the task as it
// is (the races with its end); `faults.hangOn` never answers.
function killTask(id, params) {
  const session = sessions.get(params?.sessionId);
  const taskId = params?.taskId;
  const source = params?.source ?? 'clientUi';
  record({ kind: 'kill', taskId, source });
  if (faults.killReply === 'rpc-error') return replyError(id, -32603, 'Internal error');
  if (session === undefined || faults.killReply === 'error')
    return reply(id, { result: null, error: 'session not found' });
  if (faults.killReply === 'already_exited' || faults.killReply === 'not_found')
    return reply(id, { result: { taskId, outcome: faults.killReply } });
  const row = session.tasks?.get(taskId);
  const outcome = row === undefined ? 'not_found' : row.status === 'running' ? 'killed' : 'already_exited';
  reply(id, { result: { taskId, outcome } });
  if (outcome === 'killed') void session.taskKills?.get(taskId)?.(source);
}

// A `background` step: a command (or, with `monitor: true`, a monitor) the model starts in the background. The tool
// call, `task_backgrounded`, the list, then the tool result that names the task; at its gate `task_completed`
// (`will_wake` unless `wake: false`), the list again, and `wakeAfterMs` later the wake turn `task-completed-<id>`.
// `_x.ai/task/kill` ends it at once as the agent does: `explicitly_killed`, signal `killed`, and a wake turn (even with
// `wake: false`); a `teardown` kill counts as delivered and wakes nothing.
function backgroundStep(session, b, toolCallId) {
  const monitor = b.monitor === true;
  const taskId = b.taskId ?? `task-${toolCallId}`;
  const command = b.command ?? (monitor ? 'tail -f build.log' : 'npm run dev');
  const description = b.description ?? (monitor ? 'Watch the build log' : 'Start the dev server');
  const outputFile = join(persistDir(session), 'terminal', `${toolCallId}.log`);
  const name = monitor ? 'monitor' : 'run_terminal_command';
  if (monitor) {
    session.lastMonitor = taskId;
    session.monitors ??= new Map();
    session.monitors.set(taskId, { toolCallId, command, output: '' });
  }
  const meta = {
    'x.ai/tool': {
      version: 1,
      name,
      kind: 'execute',
      namespace: 'grok_build',
      label: monitor ? 'Monitor' : 'Run Command',
      read_only: false,
    },
  };
  const startedAt = Date.now();
  update(session, 'tool_call', {
    toolCallId,
    title: name,
    rawInput: monitor ? { command, description } : { command, description, is_background: true },
    _meta: meta,
  });
  xaiUpdate(session, '_x.ai/task_backgrounded', 'task_backgrounded', {
    tool_call_id: toolCallId,
    task_id: taskId,
    command,
    cwd: session.cwd,
    output_file: outputFile,
    ...(monitor ? { monitor_description: description } : { description }),
  });
  session.tasks ??= new Map();
  const row = {
    task_id: taskId,
    command,
    description,
    cwd: session.cwd,
    kind: monitor ? 'monitor' : 'bash',
    status: 'running',
    started_at: new Date(startedAt).toISOString(),
    output_file: outputFile,
  };
  session.tasks.set(taskId, row);
  taskList(session);
  update(session, 'tool_call_update', {
    toolCallId,
    status: 'completed',
    content: [{ type: 'content', content: { type: 'text', text: `Background task ${taskId} started` } }],
    rawOutput: monitor
      ? { type: 'Monitor', taskId, timeoutMs: 36_000_000, persistent: false }
      : {
          type: 'BackgroundTaskStarted',
          task_id: taskId,
          task_type: 'bash',
          output_file: outputFile,
          status: 'running',
          command,
          summary: `Background task ${taskId} started`,
        },
  });
  let ended = false;
  const end = async (killedBy) => {
    if (ended) return;
    ended = true;
    session.taskKills.delete(taskId);
    const endedAt = Date.now();
    const killed = killedBy !== null;
    // A Stop from the app is not delivered to the model, so it wakes it whatever the step says; a teardown never does.
    const wake = killed ? killedBy !== 'teardown' : b.wake !== false;
    const exitCode = killed ? null : (b.exitCode ?? 0);
    xaiUpdate(session, '_x.ai/task_completed', 'task_completed', {
      task_snapshot: {
        task_id: taskId,
        command,
        cwd: session.cwd,
        start_time: systemTime(startedAt),
        end_time: systemTime(endedAt),
        output: '',
        output_file: outputFile,
        truncated: false,
        output_total_bytes: 0,
        exit_code: exitCode,
        signal: killed ? 'killed' : null,
        completed: true,
        kind: monitor ? 'monitor' : 'bash',
        block_waited: false,
        explicitly_killed: killed,
        kill_result_delivered: killedBy === 'teardown',
        owner_session_id: session.id,
        description,
        is_backgrounded: true,
      },
      will_wake: wake,
    });
    session.tasks.set(taskId, {
      ...row,
      status: killed ? 'failed' : 'completed',
      ended_at: new Date(endedAt).toISOString(),
      ...(exitCode === null ? { signal: 'killed' } : { exit_code: exitCode }),
    });
    taskList(session);
    if (wake) {
      // The agent admits the wake prompt a moment after the end line.
      if (typeof b.wakeAfterMs === 'number') await sleep(b.wakeAfterMs);
      await wakeTurn(session, `task-completed-${taskId}`, {
        ...b,
        wakeText: killed ? `A background command was stopped: ${command}` : `A background command ended: ${command}`,
        wakeAnswer: 'The command ended.',
      });
    }
  };
  session.taskKills ??= new Map();
  session.taskKills.set(taskId, (source) => end(source));
  void (async () => {
    if (!(await gateOpens(b.endOn, b.endAfterMs))) return;
    await end(null);
  })();
}

/**
 * Helpers that run in the background, by child session id. `_x.ai/subagent/list_running` lists all but a run's (`run`),
 * as the agent's lists no workflow children.
 */
const helpers = new Map();
/** Helpers that ended, by child session id: their terminal status, for a cancel that comes after. */
const helpersEnded = new Map();

// A `helper` step: a subagent the model starts in the background (`spawn_subagent`, `run_in_background`). Its live
// `subagent_spawned` names its address (`agentAddress: addr-<child>`, none with `address: false`). At its gate it is
// no longer listed and sends `subagent_finished` (`will_wake` unless `wake: false`) and the wake turn
// `subagent-completed-<id>`; with `lostEnd: true` the end line never comes, as when it is lost on the way.
// `_x.ai/subagent/cancel` ends it at once as the agent ends a background one: `subagent_finished` `cancelled`, no wake.
// `lines` are the helper's own session lines, sent right after its start (`more` after the `moreOn` gate): see
// `childLines`. With `lines`, the helper's `turn_completed` comes before `subagent_finished`, stored `endLagMs` later.
function helperStep(session, h, toolCallId) {
  const child = h.child ?? `${session.id}-helper-${toolCallId}`;
  const description = h.description ?? 'Find the flaky test';
  const meta = {
    'x.ai/tool': {
      version: 1,
      name: 'spawn_subagent',
      kind: 'task',
      namespace: 'grok_build',
      label: 'Subagent',
      read_only: false,
    },
  };
  const startedAt = Date.now();
  update(session, 'tool_call', {
    toolCallId,
    title: 'spawn_subagent',
    rawInput: { description, prompt: description, subagent_type: 'general-purpose' },
    _meta: { ...meta, subagentBackground: true },
  });
  update(
    session,
    'subagent_spawned',
    {
      subagent_id: child,
      parent_session_id: session.id,
      ...(session.turn ? { parent_prompt_id: session.turn.promptId } : {}),
      child_session_id: child,
      subagent_type: 'general-purpose',
      description,
      effective_context_source: 'new',
      model: session.model,
      ...(h.address === false ? {} : { agentAddress: `addr-${child}` }),
    },
    true,
  );
  const finish = (status, wake) => {
    helpers.delete(child);
    helpersEnded.set(child, status);
    update(
      session,
      'subagent_finished',
      {
        subagent_id: child,
        child_session_id: child,
        status,
        tool_calls: 3,
        turns: 2,
        duration_ms: Date.now() - startedAt,
        tokens_used: 1200,
        ...(status === 'completed' ? { output: 'Found it.' } : {}),
        will_wake: wake,
      },
      true,
    );
  };
  const held = {
    session,
    description,
    startedAt,
    address: h.address === false ? null : `addr-${child}`,
    own: null,
    cancel: () => finish('cancelled', false),
  };
  helpers.set(child, held);
  session.lastHelper = child;
  update(session, 'tool_call_update', {
    toolCallId,
    status: 'completed',
    content: [{ type: 'content', content: { type: 'text', text: `Subagent ${child} started in the background` } }],
  });
  const own = Array.isArray(h.lines) ? childSession(session, child) : null;
  held.own = own;
  if (own !== null) {
    childLines(own, h.lines);
    if (Array.isArray(h.more)) void gateOpens(h.moreOn).then((open) => open && childLines(own, h.more));
  }
  void (async () => {
    if (!(await gateOpens(h.endOn, h.endAfterMs))) return;
    if (!helpers.has(child)) return;
    if (h.lostEnd === true) {
      helpers.delete(child);
      return;
    }
    const wake = h.wake !== false;
    if (own !== null) childLine(own, 'turn_completed', { stop_reason: 'end_turn' }, true, h.endLagMs ?? 0);
    finish('completed', wake);
    if (wake) {
      await wakeTurn(session, `subagent-completed-${child}`, {
        ...h,
        wakeText: `A subagent finished: ${description}`,
        wakeAnswer: 'The subagent is done.',
      });
    }
  })();
}

/**
 * A helper's own session, stored beside its parent's with a summary, so the reader finds its folder. The summary names
 * the parent, as the agent's does (`session_kind: subagent`, `parent_session_id`).
 */
function childSession(session, child) {
  const own = {
    id: child,
    cwd: session.cwd,
    title: 'Find the flaky test',
    createdAt: new Date().toISOString(),
    messages: 1,
    model: session.model,
    parentId: session.id,
  };
  persistSummary(own);
  return own;
}

/** One line of a helper's own session, live at once and stored `lagMs` later. */
function childLine(own, sessionUpdate, fields, xai = false, lagMs = 0) {
  const params = {
    sessionId: own.id,
    update: { sessionUpdate, ...fields },
    _meta: { eventId: `${own.id}-${++seq}`, agentTimestampMs: Date.now() },
  };
  send({ method: xai ? '_x.ai/session_notification' : 'session/update', params });
  const store = () => persistLine(own, xai ? '_x.ai/session/update' : 'session/update', params);
  if (lagMs > 0) setTimeout(store, lagMs);
  else store();
  return params;
}

// A helper's own lines: `{thought|message, chunks}` streams the text in that many live chunks and stores it as one line
// with the newest chunk's `_meta` (`persistence.rs` `maybe_merge_notification`), `everyMs` apart when set; `{tool,
// title}` is a call that completes (`openMs` after it opens, when set); `{prompt}` is its task; `{waitMs}` waits. Lines
// after a wait stop once the helper ended.
function childLines(own, lines, from = 0) {
  for (let n = from; n < lines.length; n += 1) {
    const l = lines[n];
    if (typeof l.waitMs === 'number') {
      setTimeout(() => helpers.has(own.id) && childLines(own, lines, n + 1), l.waitMs);
      return;
    }
    if (typeof l.tool === 'string' && typeof l.openMs === 'number') {
      childLine(own, 'tool_call', { toolCallId: l.tool, title: l.title ?? 'read', status: 'pending' });
      setTimeout(() => {
        if (!helpers.has(own.id)) return;
        childLine(own, 'tool_call_update', { toolCallId: l.tool, status: 'completed' });
        childLines(own, lines, n + 1);
      }, l.openMs);
      return;
    }
    if (typeof l.everyMs === 'number') {
      paceText(own, l, () => childLines(own, lines, n + 1));
      return;
    }
    if (typeof l.prompt === 'string') {
      childLine(own, 'user_message_chunk', { content: { type: 'text', text: l.prompt } });
    } else if (typeof l.tool === 'string') {
      childLine(own, 'tool_call', { toolCallId: l.tool, title: l.title ?? 'read', status: 'pending' });
      childLine(own, 'tool_call_update', { toolCallId: l.tool, status: 'completed' });
    } else {
      const sessionUpdate = typeof l.thought === 'string' ? 'agent_thought_chunk' : 'agent_message_chunk';
      const text = l.thought ?? l.message ?? '';
      const size = Math.max(1, Math.ceil(text.length / (l.chunks ?? 1)));
      let meta = null;
      for (let i = 0; i < text.length; i += size) {
        meta = {
          eventId: `${own.id}-${++seq}`,
          agentTimestampMs: Date.now(),
        };
        send({
          method: 'session/update',
          params: {
            sessionId: own.id,
            update: { sessionUpdate, content: { type: 'text', text: text.slice(i, i + size) } },
            _meta: meta,
          },
        });
      }
      persistLine(own, 'session/update', {
        sessionId: own.id,
        update: { sessionUpdate, content: { type: 'text', text } },
        _meta: meta,
      });
    }
  }
}

/** A thought or message of a helper streamed one chunk each `everyMs`, stored as one line once it ends. */
function paceText(own, l, then) {
  const sessionUpdate = typeof l.thought === 'string' ? 'agent_thought_chunk' : 'agent_message_chunk';
  const text = l.thought ?? l.message ?? '';
  const size = Math.max(1, Math.ceil(text.length / (l.chunks ?? 1)));
  let meta = null;
  const step = (i) => {
    if (!helpers.has(own.id)) return;
    if (i >= text.length) {
      persistLine(own, 'session/update', {
        sessionId: own.id,
        update: { sessionUpdate, content: { type: 'text', text } },
        _meta: meta,
      });
      then();
      return;
    }
    meta = { eventId: `${own.id}-${++seq}`, agentTimestampMs: Date.now() };
    send({
      method: 'session/update',
      params: {
        sessionId: own.id,
        update: { sessionUpdate, content: { type: 'text', text: text.slice(i, i + size) } },
        _meta: meta,
      },
    });
    setTimeout(() => step(i + size), l.everyMs);
  };
  step(0);
}

// A `helperProgress` step: `subagent_progress` ticks of a helper (`child`, else the last helper the session started), sent
// live only and never stored, as the agent's publisher sends them (`agent/subagent/mod.rs`, `meta: None`). After its
// gate (`on`, else `afterMs`, else at once) it sends one tick per entry of `ticks` (each a partial set of numbers over
// the last), `everyMs` apart. `sessionId` names another session than the parent's (a nested helper's tick).
function helperProgressStep(session, p) {
  const child = p.child ?? session.lastHelper ?? `${session.id}-helper`;
  const startedAt = Date.now();
  void (async () => {
    if ((p.on !== undefined || p.afterMs !== undefined) && !(await gateOpens(p.on, p.afterMs))) return;
    let last = { turn_count: 1, tool_call_count: 0, tokens_used: 1_000, context_usage_pct: 3, error_count: 0 };
    for (const [i, tick] of (p.ticks ?? [{}]).entries()) {
      if (i > 0 && typeof p.everyMs === 'number') await sleep(p.everyMs);
      last = { ...last, ...tick };
      send({
        method: '_x.ai/session_notification',
        params: {
          sessionId: p.sessionId ?? session.id,
          update: {
            sessionUpdate: 'subagent_progress',
            subagent_id: child,
            parent_session_id: p.sessionId ?? session.id,
            child_session_id: child,
            duration_ms: Date.now() - startedAt,
            context_window_tokens: 32_768,
            tools_used: ['read_file'],
            ...last,
          },
        },
      });
    }
  })();
}

// A `monitorEvents` step: a monitor's output lines (`task`, else the last monitor the session started) as the agent's
// bridge sends them (`tools/notification_bridge.rs` `MonitorEvent`): `_x.ai/monitor_event`, raw text, no `_meta`,
// never stored. Each comes after the output tick of the monitor's own tool call (`BashOutputChunk`: `tool_call_update`
// `in_progress` with the whole output so far, no `_meta`, never stored), which the agent sends between turns too.
// After its gate (`on`, else `afterMs`, else at once) one line per entry of `events`, `everyMs` apart; `sessionId` names
// another session than the monitor's. With `wake: true` the agent, idle, then drains them as one turn of its own
// (`notifications-<uuid>`, `notification_drain.rs`) after `wakeAfterMs`.
function monitorEventsStep(session, m) {
  const taskId = m.task ?? session.lastMonitor ?? `${session.id}-monitor`;
  const description = m.description ?? 'Watch the build log';
  void (async () => {
    if ((m.on !== undefined || m.afterMs !== undefined) && !(await gateOpens(m.on, m.afterMs))) return;
    for (const [i, text] of (m.events ?? []).entries()) {
      if (i > 0 && typeof m.everyMs === 'number') await sleep(m.everyMs);
      const own = m.sessionId === undefined ? session.monitors?.get(taskId) : undefined;
      if (own !== undefined) {
        const delta = `${text}\n`;
        own.output += delta;
        send({
          method: 'session/update',
          params: {
            sessionId: session.id,
            update: {
              sessionUpdate: 'tool_call_update',
              toolCallId: own.toolCallId,
              status: 'in_progress',
              content: [{ type: 'content', content: { type: 'text', text: own.output } }],
              rawOutput: {
                type: 'Bash',
                output: incrementalBashOutput ? [] : [...Buffer.from(own.output)],
                output_for_prompt: own.output,
                exit_code: 0,
                command: own.command,
                truncated: false,
                signal: null,
                timed_out: false,
                description: null,
                current_dir: session.cwd,
                output_file: '',
                total_bytes: Buffer.byteLength(own.output),
                ...(incrementalBashOutput ? { output_delta: [...Buffer.from(delta)] } : {}),
              },
            },
          },
        });
      }
      send({
        method: '_x.ai/monitor_event',
        params: {
          sessionId: m.sessionId ?? session.id,
          update: { sessionUpdate: 'monitor_event', task_id: taskId, description, event_text: text },
        },
      });
    }
    if (m.wake !== true) return;
    if (typeof m.wakeAfterMs === 'number') await sleep(m.wakeAfterMs);
    const wrapped = (m.events ?? []).map(
      (text) => `<monitor-event description="${description}" task_id="${taskId}">\n${text}\n</monitor-event>`,
    );
    await wakeTurn(session, `notifications-${randomUUID()}`, {
      ...m,
      wakeText: m.wakeText ?? wrapped.join('\n'),
      wakeAnswer: m.answer ?? 'The monitor reported.',
    });
  })();
}

// A `drain` step: at its gate (`on`, else `afterMs`) the agent runs what ended with no wake of its own (`wake: false`
// ends) as one turn of its own, `notifications-<uuid>` (grok `notification_drain.rs`).
function drainStep(session, d) {
  void (async () => {
    if (!(await gateOpens(d.on, d.afterMs))) return;
    await wakeTurn(session, `notifications-${randomUUID()}`, {
      ...d,
      wakeText: d.text ?? 'Background tasks ended.',
      wakeAnswer: d.answer ?? 'They are done.',
    });
  })();
}

// `_x.ai/subagent/cancel {subagentId}` (`extensions/task.rs`): `{result: {subagentId, cancelled, outcome: {kind}}}`,
// `cancelled` for a running helper (its `subagent_finished` follows the reply), `already_finished` with its status for
// one that ended, `not_found` for any other id. `faults.cancelReply`: `'error'` answers with the error envelope,
// `'rpc-error'` with a JSON-RPC error, `'already_finished'` or `'not_found'` with that outcome and leaves the helper
// running (the races with its end).
function cancelHelper(id, params) {
  const subagentId = params?.subagentId;
  record({ kind: 'subagent-cancel', subagentId });
  if (faults.cancelReply === 'rpc-error') return replyError(id, -32603, 'Internal error');
  if (faults.cancelReply === 'error') return reply(id, { result: null, error: 'cancel failed' });
  const outcome = (kind) => reply(id, { result: { subagentId, cancelled: kind === 'cancelled', outcome: { kind } } });
  if (faults.cancelReply === 'not_found') return outcome('not_found');
  if (faults.cancelReply === 'already_finished') {
    return reply(id, {
      result: { subagentId, cancelled: false, outcome: { kind: 'already_finished', status: 'completed' } },
    });
  }
  const helper = helpers.get(subagentId);
  if (helper?.endsFirst !== undefined) {
    helper.endsFirst();
    return reply(id, {
      result: { subagentId, cancelled: false, outcome: { kind: 'already_finished', status: 'completed' } },
    });
  }
  if (helper !== undefined) {
    outcome('cancelled');
    helper.cancel();
    return;
  }
  const ended = helpersEnded.get(subagentId);
  if (ended === undefined) return outcome('not_found');
  reply(id, { result: { subagentId, cancelled: false, outcome: { kind: 'already_finished', status: ended } } });
}

// `_x.ai/subagent/message {sessionId, agentAddress, content, queue?}` (`extensions/subagent_message.rs`), with
// `GROK_ACTIVE_AGENT_MESSAGES` on: `{result: {kind}}`. One text block of 1 to 32 KiB, else `unsupported_content` or
// `limit`; an address no running helper of that session holds is `rejected`; else `accepted` with a message id.
// Without the flag (`faults.messagesOff`, or the variable not `1`) the method is not found. `faults.messageReplies`
// answers the messages it would accept in order, one kind each (`'error'`: the error envelope), then `accepted`. An
// accepted text is stored in the helper's own session 800 ms later, as its next safe point does, with no live line
// (`parent_message.rs` sends it to the store alone).
function messageHelper(id, params) {
  const content = Array.isArray(params?.content) ? params.content : [];
  const text = content.length === 1 && content[0]?.type === 'text' ? content[0].text : null;
  record({
    kind: 'subagent-message',
    sessionId: params?.sessionId,
    agentAddress: params?.agentAddress,
    queue: params?.queue,
    text,
  });
  if (faults.messagesOff === true || process.env.GROK_ACTIVE_AGENT_MESSAGES !== '1')
    return replyError(id, -32601, 'Method not found');
  const kind = (k, fields = {}) => reply(id, { result: { kind: k, ...fields } });
  if (typeof text !== 'string') return kind('unsupported_content');
  const bytes = Buffer.byteLength(text, 'utf8');
  if (bytes === 0 || bytes > 32 * 1024) return kind('limit', { maxBytes: 32 * 1024, observedBytes: bytes });
  const owner = [...helpers.entries()].find(
    ([, h]) => h.session.id === params?.sessionId && h.address !== null && h.address === params?.agentAddress,
  );
  if (owner === undefined) return kind('rejected');
  const next = Array.isArray(faults.messageReplies) ? faults.messageReplies.shift() : undefined;
  if (next === 'error') return reply(id, { result: null, error: 'session not found' });
  if (typeof next === 'string') return kind(next);
  kind('accepted', { messageId: randomUUID() });
  const [child, held] = owner;
  if (held.own !== null && held.own !== undefined) {
    setTimeout(() => {
      if (!helpers.has(child)) return;
      persistLine(held.own, 'session/update', {
        sessionId: child,
        update: { sessionUpdate: 'user_message_chunk', content: { type: 'text', text } },
        _meta: { eventId: `${child}-${++seq}`, agentTimestampMs: Date.now() },
      });
    }, 800);
  }
}

/**
 * `_x.ai/subagent/list_running`: the session's background helpers that run now (`extensions/task.rs`). With
 * `faults.listRunningLagMs` the list is read when the request comes and the reply goes that much later: a helper
 * started meanwhile reaches the host before a reply that does not name it.
 */
async function listRunning(id, params) {
  const sessionId = params?.sessionId;
  const subagents = [...helpers]
    .filter(([, h]) => h.session.id === sessionId && h.run === undefined)
    .map(([child, h]) => ({
      subagentId: child,
      parentSessionId: sessionId,
      childSessionId: child,
      subagentType: 'general-purpose',
      description: h.description,
      startedAtEpochMs: h.startedAt,
      durationMs: Date.now() - h.startedAt,
      turnCount: 1,
      toolCallCount: 0,
      tokensUsed: 0,
      contextWindowTokens: 32768,
      contextUsagePct: 0,
      toolsUsed: [],
      errorCount: 0,
    }));
  record({ kind: 'list-running', sessionId, listed: subagents.map((h) => h.childSessionId) });
  if (typeof faults.listRunningLagMs === 'number') await sleep(faults.listRunningLagMs);
  reply(id, { result: { subagents } });
  record({ kind: 'list-running-replied', sessionId });
}

// A `schedule` step: `scheduler_create` makes a schedule (`scheduled_task_created` before the tool result); at its gate
// `scheduled_task_deleted` with `reason` (default `expired`). With `fireAfterMs` it fires once then (`scheduler/actor.rs`):
// its run is a background subagent (`subagent_spawned`, stored), then `scheduled_task_fired` names that run (live only,
// never stored), and the run ends after `runMs` (default 3 s) with `subagent_finished` and no wake.
function scheduleStep(session, s, toolCallId) {
  const taskId = s.taskId ?? `sched-${toolCallId}`;
  const prompt = s.prompt ?? 'Check the deploy';
  const every = s.every ?? 'every 5 minutes';
  update(session, 'tool_call', {
    toolCallId,
    title: 'scheduler_create',
    rawInput: { interval: '5m', prompt },
    _meta: {
      'x.ai/tool': {
        version: 1,
        name: 'scheduler_create',
        kind: 'other',
        namespace: 'grok_build',
        label: 'Scheduler',
        read_only: false,
      },
    },
  });
  session.schedules ??= new Set();
  session.schedules.add(taskId);
  xaiUpdate(session, '_x.ai/scheduled_task_created', 'scheduled_task_created', {
    task_id: taskId,
    prompt,
    human_schedule: every,
    next_fire_at: new Date(Date.now() + 300_000).toISOString(),
  });
  update(session, 'tool_call_update', {
    toolCallId,
    status: 'completed',
    rawOutput: { type: 'SchedulerCreate', id: taskId, humanSchedule: every, updated: false },
  });
  void (async () => {
    if (!(await gateOpens(s.endOn, s.endAfterMs))) return;
    if (session.schedules?.delete(taskId) !== true) return;
    xaiUpdate(session, '_x.ai/scheduled_task_deleted', 'scheduled_task_deleted', {
      task_id: taskId,
      reason: s.reason ?? 'expired',
    });
  })();
  if (typeof s.fireAfterMs === 'number') void fireSchedule(session, taskId, prompt, every, s);
}

async function fireSchedule(session, taskId, prompt, every, s) {
  await sleep(s.fireAfterMs);
  if (session.schedules?.has(taskId) !== true) return;
  const child = `${session.id}-fire-${taskId}`;
  const startedAt = Date.now();
  update(
    session,
    'subagent_spawned',
    {
      subagent_id: child,
      parent_session_id: session.id,
      child_session_id: child,
      subagent_type: 'general-purpose',
      description: prompt,
      effective_context_source: 'new',
      model: session.model,
    },
    true,
  );
  helpers.set(child, { session, description: prompt, startedAt });
  send({
    method: '_x.ai/scheduled_task_fired',
    params: {
      sessionId: session.id,
      update: {
        sessionUpdate: 'scheduled_task_fired',
        task_id: taskId,
        prompt,
        human_schedule: every,
        next_fire_at: new Date(Date.now() + 300_000).toISOString(),
        subagent_id: child,
      },
      _meta: { eventId: `${session.id}-${++seq}`, agentTimestampMs: Date.now() },
    },
  });
  record({ kind: 'schedule-fire', taskId, child });
  await sleep(s.runMs ?? 3_000);
  if (!helpers.delete(child)) return;
  update(
    session,
    'subagent_finished',
    {
      subagent_id: child,
      child_session_id: child,
      status: 'completed',
      tool_calls: 2,
      turns: 1,
      duration_ms: Date.now() - startedAt,
      tokens_used: 900,
      output: 'Checked.',
      will_wake: false,
    },
    true,
  );
}

/**
 * A shell command that prints `ticks` one after another (`notification_bridge.rs` BashOutputChunk). Every live tick's
 * text block is the whole output so far; its `rawOutput.output` is that output as bytes, or, when the client sent
 * `x.ai/incrementalBashOutput`, `[]` with the new bytes in `output_delta`. Ticks are live only; the end line is stored.
 */
async function shellStep(session, s, toolCallId) {
  const command = s.command ?? 'ls';
  const meta = {
    'x.ai/tool': {
      version: 1,
      name: 'run_terminal_command',
      kind: 'execute',
      namespace: 'grok_build',
      label: 'Run Command',
      read_only: false,
    },
  };
  update(session, 'tool_call', {
    toolCallId,
    title: 'run_terminal_command',
    rawInput: { command, description: 'run' },
    _meta: meta,
  });
  update(session, 'tool_call_update', {
    toolCallId,
    kind: 'execute',
    title: `Execute \`${command}\``,
    content: [{ type: 'content', content: { type: 'text', text: 'run' } }],
    locations: [],
    rawInput: { variant: 'Bash', command, description: 'run', is_background: false },
    _meta: meta,
  });
  const bash = (whole, fields) => ({
    type: 'Bash',
    output_for_prompt: stripVTControlCharacters(whole),
    exit_code: 0,
    command,
    truncated: false,
    signal: null,
    timed_out: false,
    description: null,
    current_dir: session.cwd,
    output_file: '',
    total_bytes: Buffer.byteLength(whole),
    ...fields,
  });
  let sent = Buffer.alloc(0);
  for (const tick of s.ticks ?? []) {
    if (session.turn === null || session.turn.cancelled) return;
    const whole = Buffer.concat([sent, Buffer.from(tick)]);
    const raw = incrementalBashOutput
      ? bash(whole.toString(), { output: [], output_delta: [...whole.subarray(sent.length)] })
      : bash(whole.toString(), { output: [...whole] });
    sent = whole;
    send({
      method: 'session/update',
      params: {
        sessionId: session.id,
        update: {
          sessionUpdate: 'tool_call_update',
          toolCallId,
          status: 'in_progress',
          content: [{ type: 'content', content: { type: 'text', text: whole.toString() } }],
          rawOutput: raw,
        },
      },
    });
    await sleep(s.tickMs ?? 20);
  }
  if (session.turn === null || session.turn.cancelled) return;
  const whole = sent.toString();
  const exitCode = s.exitCode ?? 0;
  update(session, 'tool_call_update', {
    toolCallId,
    status: 'completed',
    content: [{ type: 'content', content: { type: 'text', text: whole } }],
    rawOutput: bash(whole, {
      output: [...sent],
      output_for_prompt: `exit: ${exitCode}\n${stripVTControlCharacters(whole)}`,
      exit_code: exitCode,
      description: 'run',
      output_file: join(persistDir(session), 'terminal', `${toolCallId}.log`),
    }),
  });
}

// A loaded machine: SIGTERM lands late, and the agent keeps answering until then.
if (faults.exitAfterTermMs !== undefined && faults.exitAfterTermMs !== null)
  process.on('SIGTERM', () => setTimeout(() => process.exit(143), faults.exitAfterTermMs));
for (let i = 0; i < (faults.garbageLines ?? 0); i++) process.stdout.write(`not json ${i}\n`);
if (faults.stderr) process.stderr.write(faults.stderr);
if (faults.exitAfterMs !== undefined && faults.exitAfterMs !== null)
  setTimeout(() => process.exit(3), faults.exitAfterMs);

let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  buffer += chunk;
  for (;;) {
    const nl = buffer.indexOf('\n');
    if (nl === -1) break;
    const line = buffer.slice(0, nl);
    buffer = buffer.slice(nl + 1);
    if (line.trim() === '') continue;
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      continue;
    }
    record({ kind: 'in', msg: message });
    handle(message).catch(() => {});
  }
});
process.stdin.on('end', () => {
  if (faults.exitAfterEofMs === undefined) process.exit(0);
  setTimeout(() => process.exit(0), faults.exitAfterEofMs);
});
