#!/usr/bin/env oxnode
/// <reference types="node" />
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { realpathSync } from 'node:fs';
import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { cpus, totalmem } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import { runAppEval } from '../evals/agent-installed/app.js';
import { detectionCases } from '../evals/agent-installed/cases.js';
import {
  AgentCli,
  cliInvocation,
  evaluateAgentCase,
  type AgentSample,
  type AgentState,
} from '../evals/agent-installed/cli.js';
import { summarize } from '../evals/agent-installed/evaluate.js';
import { observeProcess } from '../evals/agent-installed/process-memory.js';
import { metricsTraceDir } from '../packages/agent/src/paths.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const sha256 = (value: string): string => createHash('sha256').update(value).digest('hex');

export function evalOptions(args: string[]) {
  const { values } = parseArgs({
    args,
    options: {
      entrypoint: { type: 'string', default: 'delegate' },
      output: { type: 'string' },
      split: { type: 'string', default: 'all' },
      repeat: { type: 'string', default: '1' },
      case: { type: 'string', multiple: true },
      seed: { type: 'string', default: '1' },
      help: { type: 'boolean' },
      list: { type: 'boolean' },
    },
  });
  const repeat = Number(values.repeat);
  if (!Number.isSafeInteger(repeat) || repeat < 1 || repeat > 20)
    throw new Error('--repeat must be an integer from 1 to 20.');
  if (!['all', 'development', 'holdout'].includes(values.split)) throw new Error('Invalid --split.');
  if (values.entrypoint !== 'agent' && values.entrypoint !== 'delegate' && values.entrypoint !== 'app')
    throw new Error('--entrypoint must be agent, delegate or app.');
  for (const id of values.case ?? [])
    if (!detectionCases.some((c) => c.id === id)) throw new Error(`Unknown case: ${id}`);
  const cases = detectionCases.filter(
    (c) => (values.split === 'all' || c.split === values.split) && (!values.case || values.case.includes(c.id)),
  );
  if (!cases.length) throw new Error('No cases selected.');
  const entrypoint: 'agent' | 'delegate' | 'app' = values.entrypoint;
  return { ...values, entrypoint, repeat, cases };
}

export async function main(args = process.argv.slice(2)): Promise<number> {
  const options = evalOptions(args);
  if (options.help) {
    console.log(
      'Usage: vp exec oxnode scripts/eval-agent-installed.ts [--entrypoint agent|delegate|app] [--case ID] [--split all|development|holdout] [--repeat 1..20] [--seed VALUE] [--output NEW_DIR] [--list]\nRuns the installed mlx-agent binary headless (one -p run per case) against a spawned `mlx serve`, mirroring `mlx agent`/`mlx delegate`, or the App detector and production desktop sidecar. Uses entrypoint defaults, without inference tuning overrides.',
    );
    return 0;
  }
  if (options.list) {
    for (const c of options.cases) console.log(`${c.id}\t${c.split}\t${c.expected}\t${c.category}`);
    return 0;
  }
  if (options.entrypoint === 'app') return runAppEval(options, root);
  const output = resolve(
    options.output ?? join(root, '.cache/agent-installed-eval', new Date().toISOString().replaceAll(':', '-')),
  );
  await mkdir(dirname(output), { recursive: true });
  await mkdir(output);
  const workspace = join(output, 'evidence');
  await mkdir(workspace);
  const sources = Object.fromEntries(
    await Promise.all(
      [
        'packages/dashboard/src/coding-agent-detection.ts',
        'evals/agent-installed/cases.ts',
        'evals/agent-installed/cli.ts',
        'evals/agent-installed/evaluate.ts',
        'evals/agent-installed/process-memory.ts',
        'scripts/eval-agent-installed.ts',
        'packages/cli/dist/cli.js',
        'packages/cli/dist/commands/serve.js',
        'packages/cli/src/commands/agent/index.ts',
        'packages/cli/src/commands/agent/grok-build.ts',
        'packages/cli/src/commands/delegate.ts',
        'packages/agent/src/delegate.ts',
        'packages/agent/src/paths.ts',
        'packages/lm/src/model-discovery.ts',
        'packages/server/src/host/paths.ts',
      ].map(async (path) => [path, sha256(await readFile(join(root, path), 'utf8'))]),
    ),
  );
  const metadata = {
    schemaVersion: 2,
    startedAt: new Date().toISOString(),
    commit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
    dirty: execFileSync('git', ['status', '--porcelain'], { cwd: root, encoding: 'utf8' }).length > 0,
    entrypoint: options.entrypoint,
    executable: process.execPath,
    argv: cliInvocation(root, options.entrypoint),
    cwd: workspace,
    sources,
    corpusSha256: sha256(JSON.stringify(detectionCases)),
    cpu: cpus()[0]?.model,
    totalMemoryBytes: totalmem(),
    seed: options.seed,
    split: options.split,
    plannedSamples: options.cases.length * options.repeat,
    inheritedInferenceEnvironment: Object.fromEntries(
      Object.entries(process.env).filter(([key]) =>
        /^(MLX_(PAGED_|CACHE_LIMIT|IDLE_CLEAR|CONTINUOUS_BATCHING|SERVE_FORCE_SERIAL)|GROK_HOME$)/.test(key),
      ),
    ),
    note: "Installed mlx-agent binary run headless (`-p`, streaming-messages-json) against a spawned `mlx serve`, the same process/env pairing `mlx agent` and `mlx delegate` perform; the serve is shared across cases so the model stays loaded, and each case is a fresh session. Default model follows the CLI's own selection (persisted pick, else first discovered); tools, system prompt, permissions and session persistence use the entrypoint's normal defaults. The detector rubric is a user task, not a system-prompt override. This measures agent installation classification and read-file/session behavior, not the dashboard HTTP completion path or live Codex instruction loading. File-layout/verdict-cache behavior is covered separately by service unit tests. Synthetic evidence and labels are fixed before this run.",
  };
  await writeFile(join(output, 'metadata.json'), JSON.stringify(metadata, null, 2) + '\n');
  await writeFile(join(output, 'cases.json'), JSON.stringify(options.cases, null, 2) + '\n');
  console.log(
    `Starting mlx ${options.entrypoint} eval with its normal defaults. ${metadata.plannedSamples} cases. ${output}`,
  );
  const cli = new AgentCli(root, options.entrypoint, workspace, output);
  const samples: AgentSample[] = [];
  let fatal: string | undefined;
  let initialState: AgentState | undefined;
  let closing: Promise<void> | undefined;
  const close = (): Promise<void> => (closing ??= cli.close());
  const stop = (): void => {
    fatal ??= 'Evaluation interrupted.';
    void close();
  };
  process.once('SIGINT', stop);
  process.once('SIGTERM', stop);
  let memory: ReturnType<typeof observeProcess> | undefined;
  let timeout = setTimeout(() => {
    fatal = 'Runtime startup timed out.';
    void close();
  }, 60_000);
  try {
    // Probe the binary + default model and boot `mlx serve` (the process the
    // memory observer and metrics trace follow; per-case agent children are
    // short-lived).
    initialState = await cli.start();
    memory = observeProcess(cli.child.pid!, join(output, 'memory.jsonl'));
    clearTimeout(timeout);
    if (initialState.model?.provider !== 'mlx')
      throw new Error('No local model selected. Install/select one in mlx agent first.');
    console.log(`Selected ${initialState.model.id}. No settings overridden.`);
    await writeFile(join(output, 'initial-state.json'), JSON.stringify(initialState, null, 2) + '\n');
    for (let repeat = 1; repeat <= options.repeat; repeat++) {
      const ordered = [...options.cases].sort((a, b) =>
        sha256(`${options.seed}/${repeat}/${a.id}`).localeCompare(sha256(`${options.seed}/${repeat}/${b.id}`)),
      );
      for (const fixture of ordered) {
        if (fatal) break;
        const runId = `${repeat}-${fixture.id}`;
        memory?.phase(runId);
        timeout = setTimeout(() => {
          fatal = `Case ${runId} timed out after 5 minutes.`;
          void close();
        }, 300_000);
        const sample = await evaluateAgentCase(cli, fixture, workspace, output, runId);
        clearTimeout(timeout);
        sample.checks.defaultModel = sample.session?.model?.id === initialState.model.id;
        sample.checks.defaultThinking = sample.session?.thinkingLevel === initialState.thinkingLevel;
        sample.passed &&= sample.checks.defaultModel && sample.checks.defaultThinking;
        samples.push(sample);
        await appendFile(join(output, 'samples.jsonl'), JSON.stringify({ ...sample, repeat }) + '\n');
        console.log(
          `${sample.passed ? 'PASS' : 'FAIL'} ${runId}: ${sample.actual}, ${(sample.elapsedMs / 1000).toFixed(1)}s${sample.error ? `, ${sample.error}` : ''}`,
        );
      }
      if (fatal) break;
    }
  } catch (error) {
    fatal ??= String(error);
  } finally {
    clearTimeout(timeout);
    await memory?.stop();
    await close();
    process.removeListener('SIGINT', stop);
    process.removeListener('SIGTERM', stop);
  }
  const { modelCalls: taskRequests, ...scores } = summarize(samples);
  const memorySummary = memory?.summary ?? { peakRssBytes: 0, peakPhysicalBytes: undefined };
  const summary = {
    ...scores,
    taskRequests,
    recordedInferenceTurns: 0,
    complete: !fatal && samples.length === metadata.plannedSamples,
    fatal,
    initialState,
    memory: memorySummary,
  };
  for (const date of new Set([metadata.startedAt.slice(0, 10), new Date().toISOString().slice(0, 10)])) {
    // The metrics trace is keyed by the serve child pid — inference happens there.
    if (cli.child?.pid === undefined) break;
    const trace = join(metricsTraceDir(), `${date}-${cli.child.pid}.jsonl`);
    try {
      const records = await readFile(trace, 'utf8');
      await appendFile(join(output, 'metrics.jsonl'), records);
      summary.recordedInferenceTurns += records.trim().split('\n').filter(Boolean).length;
    } catch (error) {
      await appendFile(join(output, 'metrics-status.jsonl'), JSON.stringify({ trace, error: String(error) }) + '\n');
    }
  }
  await writeFile(join(output, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
  await writeFile(
    join(output, 'report.md'),
    [
      '# Agent installation CLI eval',
      '',
      `Command: mlx-agent ${cliInvocation(root, options.entrypoint).join(' ')} (against spawned mlx serve; mirrors mlx ${options.entrypoint}).`,
      `Model: ${initialState?.model?.id ?? 'unavailable'} (normal defaults).`,
      `Result: ${summary.passed}/${summary.total} passed; ${metadata.plannedSamples} planned. Complete: ${summary.complete}.`,
      `False installed: ${summary.falseInstalled}; errors: ${summary.errors}.`,
      `Observed physical footprint peak: ${memorySummary.peakPhysicalBytes === undefined ? 'unavailable' : (memorySummary.peakPhysicalBytes / 2 ** 30).toFixed(2) + ' GiB'}. Sampled RSS peak: ${(memorySummary.peakRssBytes / 2 ** 30).toFixed(2)} GiB. External observation of the mlx serve process only; no allocator/cache override.`,
      `Task latency p50/p95: ${summary.latencyMs.p50?.toFixed(0) ?? 'n/a'} / ${summary.latencyMs.p95?.toFixed(0) ?? 'n/a'} ms. First task includes lazy model load; tasks include tool calls and natural cache reuse.`,
      '',
      '| Split/category | Passed | Total |',
      '| --- | ---: | ---: |',
      ...Object.entries(summary.categories).map(
        ([category, score]) => `| ${category} | ${score.passed} | ${score.total} |`,
      ),
      '',
      ...summary.failures.map(
        (f) => `- ${f.id}: expected ${f.expected}, got ${f.actual}; ${f.failedChecks.join(', ')} ${f.error ?? ''}`,
      ),
      ...(fatal ? ['', `Incomplete: ${fatal}`] : []),
      '',
      metadata.note,
      '',
      'Agent events: events.jsonl. Inference metrics: metrics.jsonl. Diagnostics: serve.log / stderr.log. Saved sessions and copies: samples.jsonl / sessions/. Memory: memory.jsonl. Exact fixtures and source hashes: cases.json / metadata.json.',
      '',
    ].join('\n'),
  );
  console.log(`${summary.passed}/${summary.total} passed. ${join(output, 'report.md')}`);
  return !summary.complete ? 2 : summary.passed === summary.total ? 0 : 1;
}

if (process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().then(
    (code) => {
      process.exitCode = code;
    },
    (error) => {
      console.error(String(error));
      process.exitCode = 2;
    },
  );
}
