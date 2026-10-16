/// <reference types="node" />

import { spawn, type ChildProcess } from 'node:child_process';
import { closeSync, openSync, writeSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import { request } from 'node:http';
import { join } from 'node:path';
import { createInterface } from 'node:readline';

import { ROOT, llamaServerPath, sleep, writeJson, type FixtureCase } from '../common.js';
import { descendants, processTable } from './admission.js';
import { HERE, MODEL_PATH, PORT, type Performance, type Runtime } from './suite.js';

const RESPONSE_TIMEOUT_MS = 1_200_000;

export interface Result {
  event: 'result';
  runtime: Runtime;
  name: string;
  run: number | string;
  pid: number;
  wallMs: number;
  firstEventMs: number | null;
  numTokens: number;
  promptTokens: number;
  cachedTokens: number;
  performance: Performance;
  [field: string]: unknown;
}

export interface Runner {
  sample(item: FixtureCase, run: number | string, maxNewTokens: number): Promise<Result>;
  close(): Promise<void>;
}

function waitExit(child: ChildProcess, ms: number): Promise<boolean> {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve(true);
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      child.off('exit', done);
      resolve(false);
    }, ms);
    function done() {
      clearTimeout(timer);
      resolve(true);
    }
    child.once('exit', done);
  });
}

function signalTree(pid: number, signal: NodeJS.Signals): void {
  for (const member of descendants(processTable(), pid)) {
    try {
      process.kill(member, signal);
    } catch {
      // Already gone.
    }
  }
}

// Mailbox for the worker's stdout events; `eof` marks a closed stdout.
class Inbox {
  private readonly items: Record<string, unknown>[] = [];
  private readonly waiters: ((item: Record<string, unknown>) => void)[] = [];

  push(item: Record<string, unknown>): void {
    const waiter = this.waiters.shift();
    if (waiter) waiter(item);
    else this.items.push(item);
  }

  async next(): Promise<Record<string, unknown>> {
    const event =
      this.items.shift() ??
      (await new Promise<Record<string, unknown>>((resolve, reject) => {
        const waiter = (value: Record<string, unknown>) => {
          clearTimeout(timer);
          resolve(value);
        };
        const timer = setTimeout(() => {
          this.waiters.splice(this.waiters.indexOf(waiter), 1);
          reject(new Error(`No mlx worker event within ${RESPONSE_TIMEOUT_MS / 1000} s`));
        }, RESPONSE_TIMEOUT_MS);
        this.waiters.push(waiter);
      }));
    if (event.event === 'eof') throw new Error('mlx worker exited; inspect its log');
    return event;
  }
}

export class MlxWorker implements Runner {
  loaded: Record<string, unknown> = {};
  private readonly inbox = new Inbox();
  private readonly child: ChildProcess;
  private readonly log: number;

  private constructor(block: string, out: string, env: Record<string, string>) {
    this.log = openSync(join(out, 'raw', `mlx-${block}.log`), 'w');
    this.child = spawn('oxnode', [join(HERE, 'mlx-worker.ts'), out], {
      cwd: ROOT,
      env: { ...process.env, ...env },
      stdio: ['pipe', 'pipe', this.log],
    });
    this.child.stdin!.on('error', () => {
      // A dead worker surfaces through its closed stdout.
    });
    const lines = createInterface({ input: this.child.stdout! });
    lines.on('line', (line) => {
      writeSync(this.log, `${line}\n`);
      let event: unknown;
      try {
        event = JSON.parse(line);
      } catch {
        return;
      }
      const name = (event as { event?: unknown } | null)?.event;
      if (name === 'loaded' || name === 'result') this.inbox.push(event as Record<string, unknown>);
    });
    lines.on('close', () => this.inbox.push({ event: 'eof' }));
  }

  static async start(block: string, out: string, env: Record<string, string>): Promise<MlxWorker> {
    const worker = new MlxWorker(block, out, env);
    try {
      worker.loaded = await worker.inbox.next();
      if (worker.loaded.event !== 'loaded')
        throw new Error(`Expected the worker's loaded event: ${String(worker.loaded.event)}`);
      return worker;
    } catch (error) {
      await worker.close();
      throw error;
    }
  }

  async sample(item: FixtureCase, run: number | string, maxNewTokens: number): Promise<Result> {
    this.child.stdin!.write(`${JSON.stringify({ name: item.name, run, maxNewTokens })}\n`);
    return (await this.inbox.next()) as Result;
  }

  async close(): Promise<void> {
    const { child } = this;
    if (!(await waitExit(child, 0))) {
      child.stdin!.write('{"action":"exit"}\n');
      // The spawned PID is the oxnode launcher; signal the model process too.
      if (!(await waitExit(child, 30_000))) {
        signalTree(child.pid!, 'SIGTERM');
        if (!(await waitExit(child, 30_000))) signalTree(child.pid!, 'SIGKILL');
        await waitExit(child, 30_000);
      }
    }
    closeSync(this.log);
  }
}

export class LlamaServer implements Runner {
  loadMs = 0;
  readonly args: string[];
  private readonly child: ChildProcess;

  private constructor(
    block: string,
    private readonly out: string,
    protocolArgs: readonly string[],
  ) {
    // --no-webui is not in the fixture's argument list but was on the measured command line.
    const args = ['-m', MODEL_PATH, '--host', '127.0.0.1', '--port', String(PORT), ...protocolArgs, '--no-webui'];
    const binary = llamaServerPath();
    this.args = [binary, ...args];
    const log = openSync(join(out, 'raw', `llama-${block}.log`), 'w');
    this.child = spawn(binary, args, { cwd: ROOT, stdio: ['ignore', log, log] });
    closeSync(log);
  }

  static async start(block: string, out: string, protocolArgs: readonly string[]): Promise<LlamaServer> {
    const started = performance.now();
    const server = new LlamaServer(block, out, protocolArgs);
    try {
      let ready = false;
      for (let attempt = 0; attempt < 600 && !ready; attempt++) {
        if (await waitExit(server.child, 0)) throw new Error('llama exited; inspect its log');
        try {
          const health = await fetch(`http://127.0.0.1:${PORT}/health`, { signal: AbortSignal.timeout(2000) });
          ready = ((await health.json()) as { status?: unknown }).status === 'ok';
        } catch {
          // Still loading.
        }
        if (!ready) await sleep(1000);
      }
      if (!ready) throw new Error('llama health timeout');
      const props = await fetch(`http://127.0.0.1:${PORT}/props`, { signal: AbortSignal.timeout(900_000) });
      await writeFile(join(out, 'raw', `llama-${block}-props.json`), JSON.stringify(await props.json(), null, 2));
      server.loadMs = performance.now() - started;
      return server;
    } catch (error) {
      await server.close();
      throw error;
    }
  }

  async sample(item: FixtureCase, run: number | string, maxNewTokens: number): Promise<Result> {
    // Integer IDs bypass llama.cpp's own tokenizer and chat template.
    const body = JSON.stringify({
      prompt: item.tokenIds,
      n_predict: maxNewTokens,
      temperature: 0,
      top_k: 0,
      top_p: 1,
      min_p: 0,
      repeat_penalty: 1,
      presence_penalty: 0,
      frequency_penalty: 0,
      seed: 42,
      cache_prompt: false,
      return_tokens: true,
      stream: true,
      timings_per_token: true,
      ignore_eos: false,
    });
    const start = performance.now();
    let first: number | undefined;
    let last: number | undefined;
    let final: Record<string, unknown> | undefined;
    const tokens: number[] = [];
    const text: string[] = [];
    const onLine = (line: string) => {
      if (!line.startsWith('data: ')) return;
      const data = line.slice(6).trim();
      if (data === '[DONE]') return;
      const event = JSON.parse(data) as { tokens?: number[]; content?: string; stop?: boolean };
      if ('error' in event) throw new Error(`llama.cpp error: ${data}`);
      const now = performance.now();
      if (event.tokens?.length || event.content) {
        first ??= now;
        last = now;
        tokens.push(...(event.tokens ?? []));
        text.push(event.content ?? '');
      }
      if (event.stop) final = event;
    };
    await new Promise<void>((resolve, reject) => {
      const req = request(
        {
          host: '127.0.0.1',
          port: PORT,
          path: '/completion',
          method: 'POST',
          agent: false,
          headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
          timeout: RESPONSE_TIMEOUT_MS,
        },
        (response) => {
          response.setEncoding('utf8');
          let buffered = '';
          if (response.statusCode !== 200) {
            response.on('data', (chunk: string) => (buffered += chunk));
            response.on('end', () =>
              reject(new Error(`llama.cpp /completion HTTP ${response.statusCode}: ${buffered}`)),
            );
            return;
          }
          response.on('data', (chunk: string) => {
            buffered += chunk;
            try {
              for (let end = buffered.indexOf('\n'); end >= 0; end = buffered.indexOf('\n')) {
                const line = buffered.slice(0, end);
                buffered = buffered.slice(end + 1);
                onLine(line);
              }
            } catch (error) {
              req.destroy(error as Error);
            }
          });
          response.on('end', () => {
            try {
              onLine(buffered);
              resolve();
            } catch (error) {
              reject(error as Error);
            }
          });
          response.on('error', reject);
        },
      );
      req.on('timeout', () => req.destroy(new Error('llama.cpp /completion timed out')));
      req.on('error', reject);
      req.end(body);
    });
    if (!final) throw new Error('No llama final');
    const timings = final.timings as Record<string, number>;
    const cached = timings.cache_n ?? 0;
    const result: Result = {
      event: 'result',
      runtime: 'llama.cpp',
      name: item.name,
      run,
      pid: this.child.pid!,
      wallMs: performance.now() - start,
      firstEventMs: first === undefined ? null : first - start,
      streamDecodeTps:
        first !== undefined && last !== undefined && last > first
          ? (tokens.length - 1) / ((last - first) / 1000)
          : null,
      numTokens: final.tokens_predicted as number,
      promptTokens: final.tokens_evaluated as number,
      cachedTokens: cached,
      performance: {
        ttftMs: timings.prompt_ms,
        prefillTokensPerSecond: timings.prompt_per_second,
        decodeTokensPerSecond: timings.predicted_per_second,
      },
      timings,
      final,
      tokens,
      text: text.join(''),
    };
    await writeJson(join(this.out, 'raw', `llama-${run}-${item.name}.json`), result);
    if (result.promptTokens !== item.tokenIds.length || cached !== 0) {
      throw new Error(
        `llama.cpp prompt/cache accounting: expected ${item.tokenIds.length}/0, received ${result.promptTokens}/${cached}`,
      );
    }
    return result;
  }

  async close(): Promise<void> {
    if (await waitExit(this.child, 0)) return;
    this.child.kill('SIGTERM');
    if (!(await waitExit(this.child, 30_000))) {
      this.child.kill('SIGKILL');
      await waitExit(this.child, 30_000);
    }
  }
}
