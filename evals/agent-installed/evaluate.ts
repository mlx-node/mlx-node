import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { setTimeout } from 'node:timers/promises';

import {
  INSTALL_CHECK_TIMEOUT_MS,
  parseLocalJson,
  type localCompletion,
  type LocalInferenceConnection,
} from '../../packages/agent/src/delegate.js';
import { detectionResult } from '../../packages/dashboard/src/coding-agent-detection.js';
import { CodingAgentsService, type CodingAgentRow } from '../../packages/dashboard/src/coding-agents.js';
import { selectedText, type DetectionCase, type SourceRange, type Verdict } from './cases.js';

export interface ModelCall {
  model: string;
  system: string;
  input: string;
  maxTokens: number | undefined;
  elapsedMs: number;
  answer?: string;
  error?: string;
}

export interface Sample {
  id: string;
  split: DetectionCase['split'];
  category: string;
  expected: Verdict;
  actual: CodingAgentRow['status'];
  source?: SourceRange;
  expectedRanges: readonly SourceRange[];
  elapsedMs: number;
  calls: ModelCall[];
  checks: Record<string, boolean>;
  passed: boolean;
  error?: string;
}

/** An independent exact-label/source oracle. A plausible JSON answer alone cannot pass. */
export function grade(fixture: DetectionCase, actual: string, source?: SourceRange): Record<string, boolean> {
  return {
    verdict: actual === fixture.expected,
    evidence:
      fixture.expected === 'not-installed'
        ? source === undefined
        : fixture.ranges.some(([start, end]) => source?.[0] === start && source[1] === end),
  };
}

async function settled(
  service: CodingAgentsService,
  agent: DetectionCase['agent'],
  signal?: AbortSignal,
): Promise<CodingAgentRow> {
  const deadline = performance.now() + INSTALL_CHECK_TIMEOUT_MS + 10_000;
  while (true) {
    signal?.throwIfAborted();
    const row = (await service.state()).agents.find((row) => row.id === agent)!;
    if (!['waiting', 'checking', 'installing'].includes(row.status)) return row;
    if (performance.now() > deadline) throw new Error('The setup service did not settle after its request timeout.');
    await setTimeout(20, undefined, { signal });
  }
}

/** Exercise the actual App detection and cache path, with real localCompletion in the App eval. */
export async function evaluateCase(
  fixture: DetectionCase,
  connection: LocalInferenceConnection,
  complete: typeof localCompletion,
  signal?: AbortSignal,
  exerciseContentChange = false,
): Promise<Sample> {
  const home = await mkdtemp(join(tmpdir(), 'mlx-agent-installed-eval-'));
  const started = performance.now();
  const calls: ModelCall[] = [];
  let service: CodingAgentsService | undefined;
  const sample: Sample = {
    id: fixture.id,
    split: fixture.split,
    category: fixture.category,
    expected: fixture.expected,
    expectedRanges: fixture.ranges,
    actual: 'error',
    elapsedMs: 0,
    calls,
    checks: {},
    passed: false,
  };
  try {
    const folder = fixture.agent === 'claude' ? '.claude' : fixture.agent === 'codex' ? '.codex' : '.grok';
    const base = join(home, folder, fixture.agent === 'claude' ? 'CLAUDE.md' : 'AGENTS.md');
    const override = join(home, folder, 'AGENTS.override.md');
    await mkdir(dirname(base), { recursive: true });
    if (!fixture.missing) await writeFile(base, fixture.text);
    if (fixture.override !== undefined) await writeFile(override, fixture.override);
    await mkdir(join(home, '.mlx-node', 'agent'), { recursive: true });
    await writeFile(
      join(home, '.mlx-node', 'agent', 'settings.json'),
      JSON.stringify({ defaultProvider: 'mlx', defaultModel: connection.model }),
    );
    const recorded: typeof localCompletion = async (target, system, messages, requestSignal, maxTokens) => {
      const call: ModelCall = {
        model: target.model,
        system,
        input: messages[0]?.content ?? '',
        maxTokens,
        elapsedMs: 0,
      };
      calls.push(call);
      const start = performance.now();
      try {
        call.answer = await complete(
          target,
          system,
          messages,
          signal ? AbortSignal.any([signal, ...(requestSignal ? [requestSignal] : [])]) : requestSignal,
          maxTokens,
        );
        return call.answer;
      } catch (error) {
        call.error = String(error);
        throw error;
      } finally {
        call.elapsedMs = performance.now() - start;
      }
    };
    const create = (): CodingAgentsService =>
      new CodingAgentsService({
        home,
        env: {},
        listModels: async () => [connection.model],
        connect: async () => connection,
        prepareCommand: async () => fixture.command,
        complete: recorded,
      });
    service = create();
    await service.start('detect', fixture.agent);
    const row = await settled(service, fixture.agent, signal);
    sample.actual = row.status;
    sample.error = row.status === 'error' ? (row.detail ?? undefined) : undefined;
    const text = selectedText(fixture);
    if (calls[0]?.answer !== undefined) {
      const parsed = detectionResult(parseLocalJson(calls[0].answer), text);
      if (parsed.source) sample.source = [parsed.source.startLine, parsed.source.endLine];
    }
    const target = fixture.agent === 'codex' && fixture.override?.trim() ? override : base;
    const beforeCache = calls.length;
    sample.checks = {
      ...grade(fixture, sample.actual, sample.source),
      selectedFile: row.path === target,
      modelCalled: beforeCache === (text.trim() && !fixture.missing ? 1 : 0),
      defaultModel: calls.every((call) => call.model === connection.model),
    };
    // Neither a repeated status check nor a service restart may re-run inference.
    if (row.status !== 'error') {
      await service.start('detect', fixture.agent);
      sample.checks.memoryCache =
        (await settled(service, fixture.agent, signal)).status === row.status && calls.length === beforeCache;
      await service.close();
      service = create();
      await service.start('detect', fixture.agent);
      sample.checks.diskCache =
        (await settled(service, fixture.agent, signal)).status === row.status && calls.length === beforeCache;
    }
    sample.checks.preservedFiles =
      (fixture.missing
        ? await readFile(base).then(
            () => false,
            (error: NodeJS.ErrnoException) => error.code === 'ENOENT',
          )
        : (await readFile(base, 'utf8')) === fixture.text) &&
      (fixture.override === undefined || (await readFile(override, 'utf8')) === fixture.override);
    if (exerciseContentChange) {
      const changed = 'Use gh directly for GitHub tasks. No delegation rule is configured.';
      await writeFile(target, changed);
      await service.start('detect', fixture.agent);
      sample.checks.contentInvalidation =
        (await settled(service, fixture.agent, signal)).status === 'not-installed' && calls.length === beforeCache + 1;
      const beforeForced = calls.length;
      await service.start('detect', fixture.agent, true);
      sample.checks.forcedRecheck =
        (await settled(service, fixture.agent, signal)).status === 'not-installed' && calls.length === beforeForced + 1;
      sample.checks.preservedChangedFile = (await readFile(target, 'utf8')) === changed;
    }
    sample.passed = Object.values(sample.checks).every(Boolean);
  } catch (error) {
    sample.error = String(error);
    sample.passed = false;
  } finally {
    await service?.close();
    await rm(home, { recursive: true, force: true });
    sample.elapsedMs = performance.now() - started;
  }
  return sample;
}

export function summarize(samples: readonly Sample[]) {
  const elapsed = samples.flatMap((sample) => sample.calls.map((call) => call.elapsedMs)).sort((a, b) => a - b);
  const confusion: Record<string, Record<string, number>> = {};
  const categories: Record<string, { total: number; passed: number }> = {};
  for (const sample of samples) {
    const row = (confusion[sample.expected] ??= {});
    row[sample.actual] = (row[sample.actual] ?? 0) + 1;
    const group = (categories[`${sample.split}/${sample.category}`] ??= { total: 0, passed: 0 });
    group.total++;
    if (sample.passed) group.passed++;
  }
  return {
    total: samples.length,
    passed: samples.filter((s) => s.passed).length,
    failures: samples
      .filter((s) => !s.passed)
      .map((s) => ({
        id: s.id,
        expected: s.expected,
        actual: s.actual,
        source: s.source,
        failedChecks: Object.keys(s.checks).filter((key) => !s.checks[key]),
        error: s.error,
      })),
    falseInstalled: samples.filter((s) => s.actual === 'installed' && s.expected !== 'installed').length,
    errors: samples.filter((s) => s.actual === 'error').length,
    modelCalls: elapsed.length,
    latencyMs: {
      p50: elapsed[Math.ceil(elapsed.length * 0.5) - 1] ?? null,
      p95: elapsed[Math.ceil(elapsed.length * 0.95) - 1] ?? null,
    },
    confusion,
    categories,
  };
}
