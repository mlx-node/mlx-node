/** Real `mlx agent` read/edit/test task, isolated from the working checkout.
 * Run with native ARM64 Node on macOS after yarn build:native && yarn build:ts:
 * oxnode scripts/benchmark-qwen35-agent-task.ts MODEL RESULT_DIRECTORY
 * Uses high thinking and production sampling, paging, SSD, and MTP defaults. Task success is
 * independently checked after exit; timings alone are never a passing result.
 */
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, readdir, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const [modelArg, directoryArg] = process.argv.slice(2);
if (!modelArg || !directoryArg) throw new Error('Expected MODEL RESULT_DIRECTORY');
if (process.platform === 'darwin' && process.arch !== 'arm64') throw new Error('Native ARM64 Node is required');
const directory = resolve(directoryArg);
await mkdir(directory, { recursive: true });
const fixture = await mkdtemp(join(directory, 'task-'));
const model = resolve(modelArg);
const cli = fileURLToPath(new URL('../packages/cli/dist/cli.js', import.meta.url));
const addon = fileURLToPath(new URL('../packages/core/mlx-core.darwin-arm64.node', import.meta.url));
const addonSha256 = createHash('sha256')
  .update(await readFile(addon))
  .digest('hex');
const startedAt = new Date().toISOString();
await writeFile(
  join(fixture, 'queue.mjs'),
  `export class TaskQueue {
  #tail = Promise.resolve();
  run(callback, signal) {
    const result = this.#tail.then(() => callback());
    this.#tail = result.catch(() => {});
    return result;
  }
}
`,
);
const tests = `import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TaskQueue } from './queue.mjs';
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return {promise, resolve}; };
test('FIFO, one callback at a time', async () => {
  const q = new TaskQueue(), gate = deferred(), order = [];
  const a = q.run(async () => { order.push('a'); await gate.promise; order.push('b'); return 7; });
  const b = q.run(() => { order.push('c'); return 9; });
  await Promise.resolve(); assert.deepEqual(order, ['a']); gate.resolve();
  assert.equal(await a, 7); assert.equal(await b, 9); assert.deepEqual(order, ['a','b','c']);
});
test('a failed callback does not poison later work', async () => {
  const q = new TaskQueue();
  await assert.rejects(q.run(() => { throw new Error('failed'); }), /failed/);
  assert.equal(await q.run(() => 42), 42);
});
test('an already aborted task never calls its callback', async () => {
  const q = new TaskQueue(), c = new AbortController(); c.abort(new Error('cancelled'));
  let called = false;
  await assert.rejects(q.run(() => { called = true; }, c.signal), /cancelled/);
  assert.equal(called, false);
});
test('a queued abort rejects promptly and never executes', async () => {
  const q = new TaskQueue(), gate = deferred(), c = new AbortController();
  const active = q.run(() => gate.promise);
  let called = false;
  const queued = q.run(() => { called = true; }, c.signal);
  const rejected = assert.rejects(queued, /queued cancellation/);
  c.abort(new Error('queued cancellation'));
  await Promise.race([rejected, new Promise((_, reject) => setTimeout(() => reject(new Error('abort did not reject promptly')), 100))]);
  gate.resolve(); await active; await q.run(() => {}); assert.equal(called, false);
});
test('abort after callback starts does not release queue ownership early', async () => {
  const q = new TaskQueue(), gate = deferred(), started = deferred(), c = new AbortController();
  const active = q.run(async () => { started.resolve(); await gate.promise; return 17; }, c.signal);
  await started.promise; c.abort(new Error('too late'));
  let nextRan = false; const next = q.run(() => { nextRan = true; });
  await Promise.resolve(); assert.equal(nextRan, false); gate.resolve();
  assert.equal(await active, 17); await next; assert.equal(nextRan, true);
});
`;
await writeFile(join(fixture, 'queue.test.mjs'), tests);
await writeFile(
  join(fixture, 'README.md'),
  'TaskQueue must serialize asynchronous callbacks. Queued or already-aborted work rejects with signal.reason and never runs. Once a callback starts it owns the queue until settlement; cancellation after start is its own responsibility. Rejections must not poison subsequent work. Remove abort listeners after cancellation or callback start.\n',
);
await writeFile(
  join(fixture, 'AGENTS.md'),
  'Work only in this fixture. Read README.md, queue.mjs and queue.test.mjs. Use Node built-ins; do not install dependencies or delegate. Run node --test queue.test.mjs before finishing.\n',
);
function run(args: string[], env: NodeJS.ProcessEnv, timeoutMs: number) {
  return new Promise<{ code: number | null; pid: number | undefined; ms: number; stdout: string; stderr: string }>(
    (done, reject) => {
      const started = performance.now();
      const child = spawn(process.execPath, args, {
        cwd: fixture,
        env,
        stdio: ['ignore', 'pipe', 'pipe'],
        detached: process.platform !== 'win32',
      });
      let stdout = '',
        stderr = '';
      const timer = setTimeout(() => {
        // Include any fixture test/tool subprocesses holding inherited pipes.
        // This process group belongs exclusively to this benchmark invocation.
        if (process.platform === 'win32') child.kill('SIGKILL');
        else if (child.pid !== undefined) {
          try {
            process.kill(-child.pid, 'SIGKILL');
          } catch (error) {
            if ((error as NodeJS.ErrnoException).code !== 'ESRCH') throw error;
          }
        }
      }, timeoutMs);
      child.stdout.on('data', (value) => {
        stdout += value;
      });
      child.stderr.on('data', (value) => {
        stderr += value;
      });
      child.on('error', (error) => {
        clearTimeout(timer);
        reject(error);
      });
      child.on('close', (code) => {
        clearTimeout(timer);
        done({ code, pid: child.pid, ms: performance.now() - started, stdout, stderr });
      });
    },
  );
}
const env = {
  ...process.env,
  PI_CODING_AGENT_DIR: join(fixture, '.pi'),
  MLX_AGENT_AUTO_APPROVE: '1',
  MLX_COLD_CACHE_DIR: process.env.MLX_COLD_CACHE_DIR ?? join(fixture, '.cold'),
};
const baseline = await run(['--test', 'queue.test.mjs'], env, 10_000);
if (baseline.code === 0) throw new Error('Invalid fixture: tests should fail before the fix');
const agent = await run(
  [
    cli,
    'agent',
    '--models-dir',
    dirname(model),
    '--model',
    `mlx/${basename(model)}`,
    '--no-session',
    '--thinking',
    'high',
    '--mode',
    'json',
    '-p',
    'Fix TaskQueue cancellation according to README.md. Read the source and tests, make a small correct change, add a regression test that cancellation of one queued task does not prevent the next queued task from running, and run the tests. Do not delegate.',
  ],
  env,
  600_000,
);
// Restore the original acceptance tests after the agent finishes. The agent's
// own extra tests remain in its saved transcript/edited-tests artifact, while
// independent verification cannot be passed by deleting an assertion.
await writeFile(join(directory, 'agent-tests.mjs'), await readFile(join(fixture, 'queue.test.mjs')));
await writeFile(
  join(fixture, 'acceptance.test.mjs'),
  tests +
    `test('cancelling the middle task preserves an already queued follower', async () => {
  const q = new TaskQueue(), gate = deferred(), c = new AbortController();
  const active = q.run(() => gate.promise);
  let cancelledRan = false;
  const cancelled = q.run(() => { cancelledRan = true; }, c.signal);
  let followerStarted = false;
  const follower = q.run(() => { followerStarted = true; return 29; });
  const rejected = assert.rejects(cancelled, /middle task/);
  c.abort(new Error('middle task'));
  await Promise.race([rejected, new Promise((_, reject) => setTimeout(() => reject(new Error('abort did not reject promptly')), 100))]);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(followerStarted, false, 'queued cancellation must not release the active task');
  gate.resolve(); await active;
  assert.equal(await follower, 29); assert.equal(cancelledRan, false);
});\n`,
);
const verification = await run(['--test', 'acceptance.test.mjs'], env, 10_000);
const traceDir = join(homedir(), '.mlx-node', 'metrics', 'traces');
const traces = [];
for (const name of await readdir(traceDir).catch(() => [])) {
  if (name.endsWith(`-${agent.pid}.jsonl`)) {
    for (const line of (await readFile(join(traceDir, name), 'utf8')).trim().split('\n')) {
      if (!line) continue;
      const trace = JSON.parse(line);
      // macOS reuses PIDs; a trace from an older process is not this task.
      if (trace.ts >= Date.parse(startedAt)) traces.push(trace);
    }
  }
}
await writeFile(
  join(directory, 'result.json'),
  JSON.stringify(
    {
      startedAt,
      model,
      fixture,
      addonSha256,
      runtime: { node: process.version, arch: process.arch },
      env: Object.fromEntries(Object.entries(env).filter(([key]) => key.startsWith('MLX_'))),
      baseline,
      agent,
      verification,
      traces,
    },
    null,
    2,
  ),
);
console.log(
  JSON.stringify({
    fixture,
    agentMs: agent.ms,
    agentExit: agent.code,
    verificationExit: verification.code,
    turns: traces.length,
  }),
);
if (agent.code !== 0 || verification.code !== 0 || !traces.length) process.exitCode = 1;
