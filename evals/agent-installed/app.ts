import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createWriteStream } from 'node:fs';
import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { cpus, homedir, totalmem } from 'node:os';
import { dirname, join, resolve } from 'node:path';

import { INSTALL_CHECK_GENERATION, localCompletion, preferredLocalModel } from '../../packages/agent/src/delegate.js';
import { sidecarEnvOverrides } from '../../packages/desktop/src/main/child-env.js';
import { resolveAppPaths } from '../../packages/desktop/src/main/paths.js';
import { normalizeSettings } from '../../packages/desktop/src/main/settings.js';
import { nodeChildTransport } from '../../packages/desktop/src/main/supervisor/child-node.js';
import { createSupervisor } from '../../packages/desktop/src/main/supervisor/index.js';
import { engineEnvFor, LAUNCHER_ENGINE_POLICY } from '../../packages/server/src/host/env-policy.js';
import { detectionCases, type DetectionCase } from './cases.js';
import { evaluateCase, summarize, type Sample } from './evaluate.js';
import { observeProcess } from './process-memory.js';

const sha256 = (text: string): string => createHash('sha256').update(text).digest('hex');

/** Production App sidecar + detector. The Node transport is the desktop's existing fallback. */
export async function runAppEval(
  options: { output?: string; cases: DetectionCase[]; repeat: number; seed: string; split: string },
  root: string,
): Promise<number> {
  const output = resolve(
    options.output ?? join(root, '.cache/agent-installed-eval', `app-${new Date().toISOString().replaceAll(':', '-')}`),
  );
  await mkdir(dirname(output), { recursive: true });
  await mkdir(output);
  const paths = resolveAppPaths({
    packaged: false,
    appPath: join(root, 'packages/desktop'),
    resourcesPath: '',
    userData: join(
      process.platform === 'darwin'
        ? join(homedir(), 'Library/Application Support')
        : (process.env.XDG_CONFIG_HOME ?? join(homedir(), '.config')),
      'mlx-node',
    ),
  });
  const settingsText = await readFile(paths.settingsFile, 'utf8').catch((error: NodeJS.ErrnoException) => {
    if (error.code !== 'ENOENT') throw error;
    return '{}';
  });
  const { settings } = normalizeSettings(JSON.parse(settingsText));
  const sources = Object.fromEntries(
    await Promise.all(
      [
        'packages/agent/src/delegate.ts',
        'packages/agent/dist/delegate.js',
        'packages/dashboard/src/coding-agents.ts',
        'packages/dashboard/src/coding-agent-detection.ts',
        'packages/desktop/dist/inference/index.js',
        'packages/desktop/dist/inference/sidecar.js',
        'packages/server/dist/host/index.js',
        'evals/agent-installed/app.ts',
        'evals/agent-installed/evaluate.ts',
        'evals/agent-installed/cases.ts',
      ].map(async (path) => [path, sha256(await readFile(join(root, path), 'utf8'))]),
    ),
  );
  const metadata = {
    startedAt: new Date().toISOString(),
    commit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
    dirty: execFileSync('git', ['status', '--porcelain'], { cwd: root, encoding: 'utf8' }).length > 0,
    entrypoint: 'app',
    executable: process.execPath,
    sidecarEntry: paths.sidecarEntry,
    transport: 'production nodeChildTransport fallback (not Electron utilityProcess)',
    modelsDirSetting: settings.modelsDir,
    generation: INSTALL_CHECK_GENERATION,
    enginePolicy: LAUNCHER_ENGINE_POLICY,
    inheritedInferenceEnvironment: Object.fromEntries(
      Object.entries(process.env).filter(([key]) =>
        /^(MLX_(PAGED_|CACHE_LIMIT|IDLE_CLEAR|CONTINUOUS_BATCHING|SERVE_FORCE_SERIAL)|PI_CODING_AGENT_DIR$)/.test(key),
      ),
    ),
    sources,
    corpusSha256: sha256(JSON.stringify(detectionCases)),
    cpu: cpus()[0]?.model,
    totalMemoryBytes: totalmem(),
    seed: options.seed,
    split: options.split,
    plannedSamples: options.cases.length * options.repeat,
    note: 'Real CodingAgentsService, localCompletion and built desktop inference sidecar under the production supervisor and launcher policy. No model, allocator or cache tuning. Synthetic disposable instruction homes and executable paths; installed model inventory/default are real. Tests App classification, file selection and verdict caches, not Electron UI/IPC, installation writes or live coding-agent instruction loading. Reused synthetic labels are regression coverage, not an unseen holdout.',
  };
  await writeFile(join(output, 'metadata.json'), JSON.stringify(metadata, null, 2) + '\n');
  await writeFile(join(output, 'cases.json'), JSON.stringify(options.cases, null, 2) + '\n');
  const diagnostics = createWriteStream(join(output, 'sidecar.log'));
  let memory: ReturnType<typeof observeProcess> | undefined;
  const observers: ReturnType<typeof observeProcess>[] = [];
  const supervisor = createSupervisor({
    entry: paths.sidecarEntry,
    cwd: root,
    traceDir: join(output, 'traces'),
    transport: (spec, events) => {
      const child = nodeChildTransport(spec, events);
      if (child.pid) {
        memory = observeProcess(child.pid, join(output, 'memory.jsonl'));
        observers.push(memory);
      }
      return child;
    },
    enginePolicyEnv: engineEnvFor(LAUNCHER_ENGINE_POLICY),
    env: sidecarEnvOverrides({ nativeAddon: paths.nativeAddon, modelsDir: settings.modelsDir }),
  });
  const samples: Sample[] = [];
  let fatal: string | undefined;
  let model: string | undefined;
  const abort = new AbortController();
  const stop = (): void => {
    fatal ??= 'Evaluation interrupted.';
    abort.abort(new Error(fatal));
  };
  const unsubscribe = supervisor.on((event) => {
    if (event.type === 'log') diagnostics.write(`[${event.stream}] ${event.line}\n`);
    if (event.type === 'crashed' || event.type === 'gave-up') {
      fatal = 'The App inference sidecar exited unexpectedly.';
      stop();
    }
  });
  process.once('SIGINT', stop);
  process.once('SIGTERM', stop);
  const originalFetch = globalThis.fetch;
  // Observe the unmodified wire request and response. Never record credential headers.
  globalThis.fetch = async (input, init) => {
    const url = new URL(input instanceof Request ? input.url : input.toString());
    if (url.pathname !== '/v1/responses') return originalFetch(input, init);
    const started = performance.now();
    const request = typeof init?.body === 'string' ? JSON.parse(init.body) : undefined;
    try {
      const response = await originalFetch(input, init);
      const body = await response.clone().text();
      await appendFile(
        join(output, 'http.jsonl'),
        JSON.stringify({ request, response: body, status: response.status, elapsedMs: performance.now() - started }) +
          '\n',
      );
      return response;
    } catch (error) {
      await appendFile(join(output, 'http.jsonl'), JSON.stringify({ request, error: String(error) }) + '\n');
      throw error;
    }
  };
  try {
    console.log(`Starting App detection eval: ${metadata.plannedSamples} cases. ${output}`);
    await supervisor.start();
    const models = await supervisor.request<{ name: string }[]>({ op: 'models' });
    model = (await preferredLocalModel()) ?? process.env.ANTHROPIC_MODEL ?? models.map((m) => m.name).sort()[0];
    if (!model || !models.some((m) => m.name === model))
      throw new Error('Install/select the default local model before checking agent installation.');
    const info = await supervisor.request<{ url: string; modelsDir: string }>({ op: 'info' });
    const token = supervisor.connectionToken();
    if (!token) throw new Error('The App inference service did not provide a connection token.');
    const connection = { url: info.url, token, model };
    await writeFile(
      join(output, 'runtime.json'),
      JSON.stringify({ model, models, info, pid: supervisor.snapshot().pid }, null, 2) + '\n',
    );
    console.log(
      `Default ${model}; medium thinking, ${INSTALL_CHECK_GENERATION.max_output_tokens} output tokens. One request at a time.`,
    );
    for (let repeat = 1; repeat <= options.repeat; repeat++) {
      const ordered = [...options.cases].sort((a, b) =>
        sha256(`${options.seed}/${repeat}/${a.id}`).localeCompare(sha256(`${options.seed}/${repeat}/${b.id}`)),
      );
      for (const fixture of ordered) {
        if (abort.signal.aborted) break;
        memory?.phase(`${repeat}-${fixture.id}`);
        const sample = await evaluateCase(
          fixture,
          connection,
          localCompletion,
          abort.signal,
          fixture.id === 'generated-prompt',
        );
        samples.push(sample);
        await appendFile(join(output, 'samples.jsonl'), JSON.stringify({ ...sample, repeat }) + '\n');
        console.log(
          `${sample.passed ? 'PASS' : 'FAIL'} ${repeat}-${fixture.id}: ${sample.actual}, ${(sample.elapsedMs / 1000).toFixed(1)}s${sample.error ? `, ${sample.error}` : ''}`,
        );
      }
      if (abort.signal.aborted) break;
    }
  } catch (error) {
    fatal ??= String(error);
  } finally {
    globalThis.fetch = originalFetch;
    await Promise.all(observers.map((observer) => observer.stop()));
    await supervisor.dispose();
    unsubscribe();
    await new Promise<void>((resolve) => diagnostics.end(resolve));
    process.removeListener('SIGINT', stop);
    process.removeListener('SIGTERM', stop);
  }
  const summary = {
    ...summarize(samples),
    complete: !fatal && samples.length === metadata.plannedSamples,
    fatal,
    model,
    generation: INSTALL_CHECK_GENERATION,
    memory: memory?.summary,
  };
  await writeFile(join(output, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
  console.log(
    `${summary.passed}/${summary.total} passed. Complete: ${summary.complete}. ${join(output, 'summary.json')}`,
  );
  return !summary.complete ? 2 : summary.passed === summary.total ? 0 : 1;
}
