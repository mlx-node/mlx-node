import { spawn, type ChildProcess } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

import type { SharedEndpoint } from '../src/provider/shared-protocol.js';

describe('shared inference across OS processes', () => {
  it('elects one worker for simultaneous launches and recovers after its exit', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'mlx-shared-process-'));
    const children: ChildProcess[] = [];
    const workerPids: number[] = [];
    const workers = new Map<number, ChildProcess>();
    const launch = async (): Promise<string> => {
      const child = spawn(
        fileURLToPath(new URL('../../../node_modules/.bin/oxnode', import.meta.url)),
        [fileURLToPath(new URL('./fixtures/shared-service-worker.ts', import.meta.url)), directory, '0'],
        { stdio: ['ignore', 'pipe', 'pipe'] },
      );
      children.push(child);
      let stderr = '';
      child.stderr!.setEncoding('utf8').on('data', (value: string) => {
        stderr += value;
      });
      return new Promise((resolve, reject) => {
        child.once('error', reject);
        child.once('exit', (code) => {
          if (code) reject(new Error(stderr || `worker exited ${code}`));
        });
        let output = '';
        child.stdout!.setEncoding('utf8').on('data', (value: string) => {
          output += value;
          if (output.includes('\n')) {
            const line = output.trim();
            if (line.startsWith('ready:')) {
              const pid = Number(line.slice(6));
              workerPids.push(pid);
              workers.set(pid, child);
            }
            resolve(line);
          }
        });
      });
    };
    try {
      const launches = await Promise.all([launch(), launch(), launch()]);
      expect(launches.filter((line) => line.startsWith('ready:'))).toHaveLength(1);
      expect(launches.filter((line) => line === 'contender')).toHaveLength(2);
      const endpoint = JSON.parse(await readFile(join(directory, 'endpoint.json'), 'utf8')) as SharedEndpoint;
      const send = async (connection: SharedEndpoint, sessionId: string) => {
        const response = await fetch(`http://127.0.0.1:${connection.port}/stream`, {
          method: 'POST',
          headers: { authorization: `Bearer ${connection.token}` },
          body: JSON.stringify({
            clientId: 'fixture',
            profile: { discovered: { path: '/fixture', name: 'test' } },
            model: { id: 'test' },
            context: { messages: [] },
            options: { sessionId },
          }),
        });
        return JSON.parse(await response.text()) as { error: string };
      };
      expect(await Promise.all([send(endpoint, 'one'), send(endpoint, 'two')])).toEqual([
        { error: `${endpoint.pid}:one` },
        { error: `${endpoint.pid}:two` },
      ]);
      expect(await send(endpoint, 'three')).toEqual({ error: `${endpoint.pid}:three` });
      expect(await readFile(join(directory, 'loads'), 'utf8')).toBe(`${endpoint.pid}\n`);

      const winner = workers.get(endpoint.pid)!;
      // oxnode may wrap the actual worker; its published PID owns the listener.
      const exited = once(winner, 'exit');
      process.kill(endpoint.pid, 'SIGTERM');
      await exited;
      const restarted = await launch();
      expect(restarted).toMatch(/^ready:/);
      const replacement = JSON.parse(await readFile(join(directory, 'endpoint.json'), 'utf8')) as SharedEndpoint;
      expect(replacement.pid).not.toBe(endpoint.pid);
      expect(replacement.token).not.toBe(endpoint.token);
      expect(await send(replacement, 'after restart')).toEqual({ error: `${replacement.pid}:after restart` });
      expect(await readFile(join(directory, 'loads'), 'utf8')).toBe(`${endpoint.pid}\n${replacement.pid}\n`);

      // A killed worker cannot release its claim. Election must recognize the
      // dead process without deleting a slot a delayed contender could reuse.
      const crashed = once(workers.get(replacement.pid)!, 'exit');
      process.kill(replacement.pid, 'SIGKILL');
      await crashed;
      expect(await launch()).toMatch(/^ready:/);
      const recovered = JSON.parse(await readFile(join(directory, 'endpoint.json'), 'utf8')) as SharedEndpoint;
      expect(recovered.pid).not.toBe(replacement.pid);
      expect(await send(recovered, 'after crash')).toEqual({ error: `${recovered.pid}:after crash` });
      expect(await readFile(join(directory, 'loads'), 'utf8')).toBe(
        `${endpoint.pid}\n${replacement.pid}\n${recovered.pid}\n`,
      );
    } finally {
      for (const pid of workerPids) {
        try {
          process.kill(pid, 'SIGTERM');
        } catch {
          /* already exited */
        }
      }
      await Promise.all(
        children.map((child) =>
          child.exitCode !== null || child.signalCode !== null ? Promise.resolve() : once(child, 'exit'),
        ),
      );
      await rm(directory, { recursive: true, force: true });
    }
  }, 20_000);
});
