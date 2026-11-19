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

function persistSummary(session) {
  if (!scenario.persist || grokHome === '') return;
  const dir = persistDir(session);
  mkdirSync(dir, { recursive: true });
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
    }),
  );
}

function update(session, sessionUpdate, fields, xai = false, extraMeta = {}) {
  const params = {
    sessionId: session.id,
    update: { sessionUpdate, ...fields },
    _meta: { eventId: `${session.id}-${++seq}`, agentTimestampMs: Date.now(), ...extraMeta },
  };
  send({ method: xai ? '_x.ai/session_notification' : 'session/update', params });
  persistLine(session, xai ? '_x.ai/session/update' : 'session/update', params);
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
function commandsUpdate(session) {
  if (!liveExtras) return;
  send({
    method: 'session/update',
    params: {
      sessionId: session.id,
      update: {
        sessionUpdate: 'available_commands_update',
        availableCommands: [
          { name: 'compact', description: 'Compress conversation history to save context window', input: null },
          {
            name: 'always-approve',
            description: 'Toggle always-approve mode (skip all permission prompts)',
            input: { hint: 'on|off' },
          },
        ],
      },
    },
  });
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
      context: {
        systemPromptTokens: 432,
        toolDefinitionsCount: sess.toolDefinitionsCount ?? 2,
        toolDefinitionsTokens: 261,
        usageCategories: categories,
      },
    },
  };
}

function modelsBlock(current) {
  return {
    currentModelId: current,
    availableModels: models.map((m) => ({
      modelId: m.modelId,
      name: m.modelId,
      _meta: { totalContextTokens: m.totalContextTokens ?? 32768 },
    })),
  };
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
    messages: 0,
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
      reply(id, {
        sessionId: session.id,
        models: modelsBlock(session.model),
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
      scheduleReady(session, params.mcpServers);
      return;
    }
    case 'session/load': {
      const session = newSession(params.sessionId, params, true);
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
      reply(id, {
        models: modelsBlock(session.model),
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
      scheduleReady(session, params.mcpServers);
      return;
    }
    case '_x.ai/session/info': {
      const session = sessions.get(params?.sessionId);
      if (sess.infoMissing || session === undefined) return replyError(id, -32601, 'Method not found');
      return reply(id, infoResult(session));
    }
    case 'session/set_model': {
      const session = sessions.get(params?.sessionId);
      if (session === undefined || !models.some((m) => m.modelId === params.modelId)) {
        return replyError(id, -32602, 'Invalid params', { message: 'unknown model id' });
      }
      session.model = params.modelId;
      persistSummary(session);
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
    case 'session/cancel': {
      if (faults.ignoreCancel) return;
      const session = sessions.get(params?.sessionId);
      // As the agent: the turn task is dropped at once, and a later answer to its open request goes nowhere.
      if (faults.abortOnCancel && session?.turn?.asking)
        return endCancelled(session.turn.rpcId, session, 'MidTurnAbort');
      if (session?.turn) session.turn.cancelled = true;
      else session?.wakeTurn?.cancel();
      session?.turn?.wake?.();
      return;
    }
    case '_x.ai/workflows/stop':
      return stopRun(id, params);
    default:
      if (id !== undefined) replyError(id, -32601, 'Method not found');
  }
}

function endCancelled(id, session, category) {
  update(session, 'turn_completed', { prompt_id: session.turn.promptId, stop_reason: 'cancelled' }, true, {
    cancellationCategory: category,
  });
  session.turn = null;
  reply(id, { stopReason: 'cancelled', _meta: { sessionId: session.id, cancellationCategory: category } });
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
// (`stopOn` holds the stop until its file exists; `stopRejects` answers every stop as rejected). `askAfterEnd`: the
// helper asks only after the terminal update, as a late ask of a run that has ended. A `complete` run reaches its last
// phase first; `failed` keeps its phase and sends `pause_message`.
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
    pauseMessage: null,
  };
  runs.set(run.id, run);
  workflowUpdate(run);
  const agent = {
    id: `${run.id}-a0`,
    child: w.childSessionId ?? `${session.id}-child-${n}`,
    label: w.label ?? 'researcher-0',
    phase: run.phases[1] ?? run.phases[0],
    state: 'running',
    tokens: 0,
    durationMs: 0,
  };
  run.agents.push(agent);
  update(
    session,
    'subagent_spawned',
    {
      subagent_id: agent.id,
      child_session_id: agent.child,
      parent_session_id: session.id,
      subagent_type: 'general-purpose',
      description: agent.label,
      workflow_run_id: run.id,
    },
    true,
  );
  run.phase = Math.min(1, run.phases.length - 1);
  run.lastEvent = 'agent_started';
  workflowUpdate(run);
  void runWorkflow(run, agent, w);
}

async function runWorkflow(run, agent, w) {
  const session = run.session;
  const child = { id: agent.child, cwd: w.childCwd ?? session.cwd };
  if (!(await runGate(run, w.childOn, w.childAfterMs))) return;
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
    record({ kind: 'child-answer', runId: run.id, toolCallId, answer: optionId });
    return optionId === 'allow-once' || optionId === 'always-allow';
  };
  let allowed = true;
  if (w.ask !== false && w.askAfterEnd !== true) {
    allowed = await ask();
    if (run.stopped) return;
  }
  if (w.askAfterEnd !== true)
    update(child, 'tool_call_update', { toolCallId, status: allowed ? 'completed' : 'failed' });
  if (!(await runGate(run, w.completeOn, w.completeAfterMs))) return;
  agent.state = 'done';
  agent.tokens = 1200;
  agent.durationMs = Date.now() - run.startedAt;
  update(
    session,
    'subagent_finished',
    {
      subagent_id: agent.id,
      child_session_id: agent.child,
      status: 'completed',
      duration_ms: agent.durationMs,
      tokens_used: agent.tokens,
    },
    true,
  );
  run.status = w.status ?? 'complete';
  run.lastEvent = 'run_finished';
  if (run.status === 'complete') run.phase = run.phases.length - 1;
  if (run.status === 'failed') run.pauseMessage = w.error ?? 'The script failed.';
  workflowUpdate(run);
  if (w.askAfterEnd === true) {
    const late = await ask();
    update(child, 'tool_call_update', { toolCallId, status: late ? 'completed' : 'failed' });
  }
  if (w.wake === false) return;
  await wakeTurn(session, `workflow-completed-${run.id}-${run.revision}`, w);
}

/** The agent's own turn after a run ended: a hidden prompt, an answer, then its end line (`session/cancel` ends it). */
async function wakeTurn(session, promptId, w) {
  let cancel = () => {};
  let wasCancelled = false;
  const cancelled = new Promise((resolve) => {
    cancel = () => {
      wasCancelled = true;
      resolve(true);
    };
  });
  session.wakeTurn = { cancel };
  wakePrompt(session, 'A workflow finished. Read its report.', w.wakeEcho === true);
  update(session, 'agent_message_chunk', { content: { type: 'text', text: 'The report is ready.' } });
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
}

/** `x.ai/workflows/stop`: an active run is cancelled with no wake; any other answer is a rejection. */
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
  if (run.status !== 'active' || run.stopRejects)
    return reply(id, { outcome: 'rejected', code: 'workflow_control_not_applicable', detail: run.status });
  run.stopped = true;
  run.status = 'cancelled';
  run.lastEvent = 'run_stopped';
  for (const a of run.agents) if (a.state === 'running') a.state = 'cancelled';
  reply(id, { outcome: 'stopped', runId: run.id, name: run.name });
  workflowUpdate(run);
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
  session.turn = { promptId: randomUUID(), cancelled: false, wake: null, rpcId: id };
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
  for (const step of script.steps) {
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
      // `complete: false`: a call that never reports its end.
      if (p.complete !== false) {
        update(session, 'tool_call_update', {
          toolCallId,
          status: 'completed',
          content: [{ type: 'content', content: { type: 'text', text: output } }],
        });
      }
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
    } else if (step.workflow !== undefined) {
      startWorkflow(session, step.workflow, n);
    } else if (step.plan !== undefined) {
      update(session, 'plan', { entries: step.plan });
    } else if (step.modelChanged !== undefined) {
      update(session, 'model_changed', { model_id: step.modelChanged }, true);
    } else if (step.turnCompleted !== undefined) {
      update(
        session,
        'turn_completed',
        {
          prompt_id: session.turn.promptId,
          stop_reason: 'end_turn',
          usage: {
            inputTokens: step.turnCompleted.inputTokens ?? 0,
            outputTokens: step.turnCompleted.outputTokens ?? 0,
            modelCalls: 1,
          },
          elapsed_ms: 10,
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
    } else if (step.end !== undefined) {
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
      return;
    }
  }
  if (session.turn?.cancelled) return endCancelled(id, session, 'MidTurnAbort');
  session.turn = null;
  reply(id, { stopReason: 'end_turn', _meta: { sessionId: session.id } });
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
