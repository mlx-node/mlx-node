import { describe, expect, it } from 'vite-plus/test';

import { createRunQueue } from '../../demo/jspace/useLensRun';

/** Let every already-resolved microtask settle. */
const flush = () => new Promise<void>((r) => setTimeout(r, 0));

describe('single-flight run queue', () => {
  it('runs one at a time, coalesces to the newest pending, and settles the superseded one', async () => {
    const dispatched: string[] = [];
    const releases: ((v: string) => void)[] = [];
    const q = createRunQueue<string, string>((req) => {
      dispatched.push(req);
      return new Promise<string>((res) => releases.push(res));
    });

    const pA = q.run('A');
    const pB = q.run('B'); // pending
    const pC = q.run('C'); // supersedes B before B ever dispatches

    expect(dispatched).toEqual(['A']);
    expect(q.inFlight()).toBe(true);

    // B is superseded: it must SETTLE, as null. Not reject, not hang.
    await expect(pB).resolves.toBeNull();

    releases[0]!('rA');
    await expect(pA).resolves.toBe('rA');
    await flush();

    expect(dispatched).toEqual(['A', 'C']); // B never dispatched
    releases[1]!('rC');
    await expect(pC).resolves.toBe('rC');
    await flush();
    expect(q.inFlight()).toBe(false);
  });

  it('reports not-in-flight once a lone run resolves', async () => {
    let n = 0;
    const q = createRunQueue<string, string>(async (r) => `${r}:${++n}`);
    await expect(q.run('A')).resolves.toBe('A:1');
    await flush();
    expect(q.inFlight()).toBe(false);
  });

  it('propagates a dispatch failure to its own caller only', async () => {
    const q = createRunQueue<string, string>(async () => {
      throw new Error('boom');
    });
    await expect(q.run('A')).rejects.toThrow('boom');
    await flush();
    expect(q.inFlight()).toBe(false);
  });

  // The recovery guarantee the /jspace worker-generation fix relies on: when an
  // in-flight dispatch REJECTS (e.g. its worker was torn down and the readout
  // aborted), the queue must drain and promote the PENDING run immediately — not
  // strand it behind the dead dispatch until the client timeout.
  it('a rejected in-flight dispatch drains and dispatches the pending run', async () => {
    const dispatched: string[] = [];
    const resolves: ((v: string) => void)[] = [];
    const rejects: ((e: unknown) => void)[] = [];
    const q = createRunQueue<string, string>((req) => {
      dispatched.push(req);
      return new Promise<string>((res, rej) => {
        resolves.push(res);
        rejects.push(rej);
      });
    });

    const pA = q.run('A'); // in flight
    const pB = q.run('B'); // pending behind A
    expect(dispatched).toEqual(['A']);
    // Swallow A's rejection so the test failure surface stays clean.
    pA.catch(() => {});

    rejects[0]!(new Error('worker torn down')); // A's worker died mid-dispatch
    await expect(pA).rejects.toThrow('worker torn down');
    await flush();

    expect(dispatched).toEqual(['A', 'B']); // B promoted the instant A settled
    resolves[1]!('rB');
    await expect(pB).resolves.toBe('rB');
    await flush();
    expect(q.inFlight()).toBe(false);
  });
});
