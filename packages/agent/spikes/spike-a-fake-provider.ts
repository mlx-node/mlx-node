/**
 * Spike A: prove a fake in-process `streamSimple` provider drives pi's full
 * coding-agent loop through `main()` (headless print mode, `--mode json`).
 *
 * NOT production code. Run from the repo root:
 *
 *   oxnode packages/agent/spikes/spike-a-fake-provider.ts
 *
 * The script runs in two modes:
 *  - orchestrator (no args): creates temp dirs, spawns one child per scenario
 *    (via `oxnode <this-file> --scenario <name>`), parses child stdout (pi's
 *    JSON event stream) + stderr (SPIKE_* markers), prints checklist verdicts.
 *  - child (`--scenario text|tools|malformed`): registers the fake `mlx`
 *    provider via an inline extension and calls pi's `main()` in-process.
 *
 * Child processes are used because pi's print mode takes over stdout, reads
 * piped stdin until EOF, and `main()`'s exit behavior is itself under test
 * (checklist item 8) — per-scenario isolation keeps every run observable.
 */
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import type {
  AssistantMessage,
  AssistantMessageEventStream,
  Context,
  Model,
  SimpleStreamOptions,
  ToolCall,
} from '@earendil-works/pi-ai';
import type { ExtensionAPI, InlineExtension } from '@earendil-works/pi-coding-agent';

const SELF = fileURLToPath(import.meta.url);
const TARGET_FILE = 'spike-target.txt';
const TARGET_CONTENT = 'MLX_SPIKE_FILE_CONTENT_42\n';
const TEXT_REPLY = 'Hello from the fake MLX provider.';
const TOOL_REPLY = 'The file says: MLX_SPIKE_FILE_CONTENT_42';
const MALFORMED_REPLY = 'Recovered after the malformed tool call.';

type Scenario = 'text' | 'tools' | 'malformed';
const SCENARIOS: Scenario[] = ['text', 'tools', 'malformed'];

// ---------------------------------------------------------------------------
// Child mode: run pi's main() with the fake provider
// ---------------------------------------------------------------------------

function mark(line: string): void {
  // stderr: pi's print mode takes over stdout (console.log is redirected to
  // stderr too, but be explicit so pre-takeover logs do not pollute stdout).
  process.stderr.write(`${line}\n`);
}

function zeroUsage(): AssistantMessage['usage'] {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

async function runChild(scenario: Scenario): Promise<void> {
  if (!process.env.PI_CODING_AGENT_DIR) {
    mark('SPIKE_FATAL PI_CODING_AGENT_DIR not set');
    process.exit(2);
  }
  process.env.PI_SKIP_VERSION_CHECK = '1';

  process.on('exit', (code) => {
    mark(`SPIKE_PROCESS_EXIT code=${code}`);
  });

  // File for the `read` tool to fetch (created in the scenario temp cwd).
  writeFileSync(join(process.cwd(), TARGET_FILE), TARGET_CONTENT);

  // Import AFTER env vars are set (checklist item 1).
  const piAi = await import('@earendil-works/pi-ai');
  const { createAssistantMessageEventStream } = piAi;

  let callIndex = 0;

  const pushStreamError = (
    stream: AssistantMessageEventStream,
    model: Model<string>,
    error: unknown,
  ): void => {
    const message: AssistantMessage = {
      role: 'assistant',
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: zeroUsage(),
      stopReason: 'error',
      errorMessage: error instanceof Error ? error.message : String(error),
      timestamp: Date.now(),
    };
    stream.push({ type: 'error', reason: 'error', error: message });
    stream.end();
  };

  const fakeStreamSimple = (
    model: Model<string>,
    context: Context,
    _options?: SimpleStreamOptions,
  ): AssistantMessageEventStream => {
    const stream = createAssistantMessageEventStream();
    const call = callIndex++;
    const toolNames = (context.tools ?? []).map((t) => t.name);
    mark(
      `SPIKE_STREAM_CALL n=${call} messages=${context.messages.length} ` +
        `hasReadTool=${toolNames.includes('read')} tools=${toolNames.join(',')}`,
    );

    (async () => {
      // Pattern copied from pi's own providers (packages/ai/src/api/google-generative-ai.ts):
      // one mutable `output` AssistantMessage carried as `partial` on every event.
      const output: AssistantMessage = {
        role: 'assistant',
        content: [],
        api: model.api,
        provider: model.provider,
        model: model.id,
        usage: zeroUsage(),
        stopReason: 'stop',
        timestamp: Date.now(),
      };
      stream.push({ type: 'start', partial: output });

      const emitText = (text: string) => {
        output.content.push({ type: 'text', text: '' });
        const idx = output.content.length - 1;
        stream.push({ type: 'text_start', contentIndex: idx, partial: output });
        for (const piece of text.split(/(?<= )/)) {
          (output.content[idx] as { type: 'text'; text: string }).text += piece;
          stream.push({ type: 'text_delta', contentIndex: idx, delta: piece, partial: output });
        }
        stream.push({ type: 'text_end', contentIndex: idx, content: text, partial: output });
      };

      const emitToolCall = (args: Record<string, unknown>) => {
        const toolCall: ToolCall = {
          type: 'toolCall',
          id: `spike_call_${call}`,
          name: 'read',
          arguments: {},
        };
        output.content.push(toolCall);
        const idx = output.content.length - 1;
        stream.push({ type: 'toolcall_start', contentIndex: idx, partial: output });
        const json = JSON.stringify(args);
        toolCall.arguments = args;
        stream.push({ type: 'toolcall_delta', contentIndex: idx, delta: json, partial: output });
        stream.push({ type: 'toolcall_end', contentIndex: idx, toolCall, partial: output });
        output.stopReason = 'toolUse';
      };

      const wantsToolCall = scenario !== 'text' && call === 0;
      if (wantsToolCall) {
        const target = join(process.cwd(), TARGET_FILE);
        // malformed: wrong property name (`file` instead of required `path`)
        // must FAIL the read tool's TypeBox schema (checklist item 7).
        emitToolCall(scenario === 'malformed' ? { file: target } : { path: target });
      } else {
        if (call > 0) {
          // Second call: prove pi fed the tool result back (checklist items 5/7).
          const roles = context.messages.map((m) => m.role);
          mark(`SPIKE_CTX_CALL1 roles=${roles.join(',')}`);
          const toolResult = context.messages.find((m) => m.role === 'toolResult');
          mark(`SPIKE_TOOLRESULT_FOUND=${toolResult !== undefined}`);
          if (toolResult) {
            mark(`SPIKE_TOOLRESULT_JSON=${JSON.stringify(toolResult)}`);
          }
        }
        emitText(scenario === 'text' ? TEXT_REPLY : scenario === 'tools' ? TOOL_REPLY : MALFORMED_REPLY);
      }

      // Usage plumbing (checklist item 6): non-zero tokens, zero cost.
      output.usage = {
        input: 7,
        output: 5,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 12,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      };
      stream.push({
        type: 'done',
        reason: output.stopReason as 'stop' | 'toolUse',
        message: output,
      });
      stream.end();
    })().catch((error: unknown) => {
      pushStreamError(stream, model, error);
    });

    return stream;
  };

  const fakeProviderExtension: InlineExtension = {
    name: 'mlx-fake-provider',
    factory: (pi: ExtensionAPI) => {
      mark('SPIKE_EXT_FACTORY_CALLED');
      try {
        pi.registerProvider('mlx', {
          name: 'MLX (fake)',
          baseUrl: 'mlx://local',
          apiKey: 'mlx-local',
          api: 'mlx',
          streamSimple: fakeStreamSimple,
          models: [
            {
              id: 'fake-model',
              name: 'Fake Model',
              reasoning: false,
              input: ['text'],
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
              contextWindow: 32768,
              maxTokens: 4096,
            },
          ],
        });
        mark('SPIKE_REGISTER_OK');
      } catch (error) {
        mark(`SPIKE_REGISTER_ERROR ${error instanceof Error ? error.message : String(error)}`);
        throw error;
      }
    },
  };

  const prompts: Record<Scenario, string> = {
    text: 'Say hello.',
    tools: `Read ${TARGET_FILE} and tell me what it says.`,
    malformed: `Read ${TARGET_FILE} (the provider will send a bad tool call).`,
  };

  const argv = [
    '--mode',
    'json',
    '--model',
    'mlx/fake-model',
    // `text` keeps session persistence ON to prove PI_CODING_AGENT_DIR
    // redirection gets populated (finding 1a); the rest stay ephemeral.
    ...(scenario === 'text' ? [] : ['--no-session']),
    '-p',
    prompts[scenario],
  ];

  const { main } = await import('@earendil-works/pi-coding-agent');
  await main(argv, { extensionFactories: [fakeProviderExtension] });
  mark(`SPIKE_MAIN_RETURNED exitCode=${process.exitCode ?? 0}`);
}

// ---------------------------------------------------------------------------
// Orchestrator mode: spawn one child per scenario and judge the checklist
// ---------------------------------------------------------------------------

interface ChildResult {
  scenario: Scenario;
  exitCode: number | null;
  timedOut: boolean;
  naturalExit: boolean;
  events: unknown[];
  stdout: string;
  stderr: string;
  agentDir: string;
  agentDirEntries: string[];
}

function runScenario(scenario: Scenario, root: string): Promise<ChildResult> {
  const cwd = join(root, scenario, 'cwd');
  const agentDir = join(root, scenario, 'agent');
  mkdirSync(cwd, { recursive: true });
  mkdirSync(agentDir, { recursive: true });

  return new Promise((resolve) => {
    const child = spawn('oxnode', [SELF, '--scenario', scenario], {
      cwd,
      env: {
        ...process.env,
        PI_CODING_AGENT_DIR: agentDir,
        PI_SKIP_VERSION_CHECK: '1',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    let timedOut = false;
    child.stdout.on('data', (chunk: Buffer) => (stdout += chunk.toString()));
    child.stderr.on('data', (chunk: Buffer) => (stderr += chunk.toString()));

    const killer = setTimeout(() => {
      timedOut = true;
      child.kill('SIGKILL');
    }, 120_000);

    child.on('close', (code) => {
      clearTimeout(killer);
      const events: unknown[] = [];
      for (const line of stdout.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('{')) continue;
        try {
          events.push(JSON.parse(trimmed));
        } catch {
          // non-JSON stdout line; ignore
        }
      }
      let agentDirEntries: string[] = [];
      try {
        agentDirEntries = readdirSync(agentDir);
      } catch {
        // leave empty
      }
      resolve({
        scenario,
        exitCode: code,
        timedOut,
        naturalExit: !timedOut,
        events,
        stdout,
        stderr,
        agentDir,
        agentDirEntries,
      });
    });
  });
}

/** Depth-first search for any node matching the predicate in parsed JSON. */
function deepFind(node: unknown, predicate: (o: Record<string, unknown>) => boolean): Record<string, unknown> | undefined {
  if (Array.isArray(node)) {
    for (const item of node) {
      const hit = deepFind(item, predicate);
      if (hit) return hit;
    }
    return undefined;
  }
  if (node !== null && typeof node === 'object') {
    const obj = node as Record<string, unknown>;
    if (predicate(obj)) return obj;
    for (const value of Object.values(obj)) {
      const hit = deepFind(value, predicate);
      if (hit) return hit;
    }
  }
  return undefined;
}

interface Verdict {
  item: string;
  status: 'PROVEN' | 'FAILED' | 'SURPRISE';
  evidence: string;
}

function judge(results: Record<Scenario, ChildResult>): Verdict[] {
  const verdicts: Verdict[] = [];
  const text = results.text;
  const tools = results.tools;
  const malformed = results.malformed;
  const push = (item: string, ok: boolean, evidence: string) =>
    verdicts.push({ item, status: ok ? 'PROVEN' : 'FAILED', evidence });

  // 1. Boot through main() with env redirect
  const booted = SCENARIOS.every((s) => results[s].events.length > 0);
  const dirPopulated = text.agentDirEntries.length > 0;
  push(
    '1. Boot (env vars -> dynamic import -> main(argv, {extensionFactories}))',
    booted && dirPopulated,
    `JSON events per scenario: ${SCENARIOS.map((s) => `${s}=${results[s].events.length}`).join(' ')}; ` +
      `temp PI_CODING_AGENT_DIR populated with: [${text.agentDirEntries.join(', ')}]`,
  );

  // 2. Provider registration + model resolution, no auth prompt
  const registered = SCENARIOS.every((s) => results[s].stderr.includes('SPIKE_REGISTER_OK'));
  const noAuthPrompt = SCENARIOS.every(
    (s) => !/\/login|No models available|not logged in/i.test(results[s].stderr),
  );
  const modelResolved = deepFind(text.events, (o) => o.model === 'fake-model' && o.provider === 'mlx') !== undefined;
  push(
    '2. registerProvider("mlx", {...streamSimple}) resolves --model mlx/fake-model without auth',
    registered && noAuthPrompt && modelResolved,
    `SPIKE_REGISTER_OK=${registered} noAuthPrompt=${noAuthPrompt} assistantEventWithFakeModel=${modelResolved}`,
  );

  // 3. Headless print mode
  push(
    '3. Headless print mode (-p, --mode json, no TTY)',
    text.exitCode === 0 && !text.timedOut,
    `text scenario exit=${text.exitCode} timedOut=${text.timedOut}`,
  );

  // 4. Text streaming through pi's output
  const sawDelta = deepFind(text.events, (o) => o.type === 'message_update') !== undefined;
  const finalText = deepFind(
    text.events,
    (o) => typeof o.text === 'string' && (o.text as string).includes(TEXT_REPLY),
  );
  push(
    '4. Text streaming (start -> text_start/delta/end -> done{stop}) reaches pi output',
    sawDelta && finalText !== undefined,
    `message_update seen=${sawDelta}; final text found=${finalText !== undefined}`,
  );

  // 5. Tool loop: toolcall -> pi executes read -> toolResult in second call context
  const secondCallSawResult = tools.stderr.includes('SPIKE_TOOLRESULT_FOUND=true');
  const twoCalls = tools.stderr.includes('SPIKE_STREAM_CALL n=1');
  const resultHasContent = tools.stderr.includes('MLX_SPIKE_FILE_CONTENT_42');
  const finalToolText = deepFind(
    tools.events,
    (o) => typeof o.text === 'string' && (o.text as string).includes(TOOL_REPLY),
  );
  push(
    '5. Tool loop (toolcall trio -> done{toolUse} -> pi runs read -> toolResult -> continuation)',
    secondCallSawResult && twoCalls && resultHasContent && finalToolText !== undefined && tools.exitCode === 0,
    `secondStreamCall=${twoCalls} toolResultInContext=${secondCallSawResult} ` +
      `fileContentInResult=${resultHasContent} finalTextDelivered=${finalToolText !== undefined} exit=${tools.exitCode}`,
  );

  // 6. Usage plumbing
  const usageEvent = deepFind(
    text.events,
    (o) =>
      typeof o.usage === 'object' &&
      o.usage !== null &&
      (o.usage as Record<string, unknown>).totalTokens === 12,
  );
  push(
    '6. Usage on final message (input=7 output=5 totalTokens=12, cost zeros) crashes nothing',
    usageEvent !== undefined && text.exitCode === 0,
    `usage(totalTokens=12) visible in session events=${usageEvent !== undefined}`,
  );

  // 7. Malformed tool call -> error tool result fed back
  const errorResultLine = malformed.stderr
    .split('\n')
    .find((l) => l.startsWith('SPIKE_TOOLRESULT_JSON='));
  let errorResultIsError = false;
  if (errorResultLine) {
    try {
      const parsed = JSON.parse(errorResultLine.slice('SPIKE_TOOLRESULT_JSON='.length)) as Record<string, unknown>;
      errorResultIsError = parsed.isError === true;
    } catch {
      // leave false
    }
  }
  push(
    '7. Malformed tool args (read {file} not {path}) -> pi feeds back error toolResult (pi owns retry loop)',
    errorResultIsError && malformed.exitCode === 0,
    errorResultLine
      ? `error toolResult: ${errorResultLine.slice(0, 400)}`
      : `NO SPIKE_TOOLRESULT_JSON line; stderr tail: ${malformed.stderr.slice(-300)}`,
  );

  // 8. Exit behavior of main() in print mode
  const mainReturned = SCENARIOS.every((s) => results[s].stderr.includes('SPIKE_MAIN_RETURNED'));
  const naturalExit = SCENARIOS.every((s) => results[s].naturalExit);
  push(
    '8. Exit behavior: main() RETURNS in print mode (no process.exit) and process exits naturally',
    mainReturned && naturalExit,
    `SPIKE_MAIN_RETURNED in all scenarios=${mainReturned}; natural process exit (no kill)=${naturalExit}; ` +
      SCENARIOS.map((s) => `${s}:exit=${results[s].exitCode}`).join(' '),
  );

  return verdicts;
}

async function runOrchestrator(): Promise<void> {
  const root = mkdtempSync(join(tmpdir(), 'spike-a-'));
  console.log(`Spike A temp root: ${root}`);

  const results = {} as Record<Scenario, ChildResult>;
  for (const scenario of SCENARIOS) {
    console.log(`\n--- running scenario: ${scenario} ---`);
    const result = await runScenario(scenario, root);
    results[scenario] = result;
    console.log(
      `scenario=${scenario} exit=${result.exitCode} timedOut=${result.timedOut} jsonEvents=${result.events.length}`,
    );
    for (const line of result.stderr.split('\n')) {
      if (line.startsWith('SPIKE_')) console.log(`  ${line.slice(0, 500)}`);
    }
  }

  console.log('\n=== Spike A checklist verdicts ===');
  let failed = 0;
  for (const v of judge(results)) {
    if (v.status === 'FAILED') failed++;
    console.log(`\n[${v.status}] ${v.item}`);
    console.log(`  evidence: ${v.evidence}`);
  }
  console.log(`\nArtifacts kept under ${root} (per-scenario cwd/ and agent/ dirs).`);
  process.exitCode = failed > 0 ? 1 : 0;
}

// ---------------------------------------------------------------------------

const scenarioArg = process.argv.indexOf('--scenario');
if (scenarioArg >= 0) {
  const scenario = process.argv[scenarioArg + 1] as Scenario;
  if (!SCENARIOS.includes(scenario)) {
    mark(`SPIKE_FATAL unknown scenario: ${scenario}`);
    process.exit(2);
  }
  await runChild(scenario);
} else {
  await runOrchestrator();
}
