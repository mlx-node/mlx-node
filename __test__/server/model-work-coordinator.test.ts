import { describe, expect, it } from 'vitest';

import { ModelWorkCoordinator } from '../../packages/server/src/model-work-coordinator.js';

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

const tick = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

describe('ModelWorkCoordinator', () => {
  it('lets inference readers overlap', async () => {
    const coordinator = new ModelWorkCoordinator();
    const release = deferred();
    const events: string[] = [];

    const a = coordinator.withInference(async () => {
      events.push('a:start');
      await release.promise;
      events.push('a:end');
    });
    const b = coordinator.withInference(async () => {
      events.push('b:start');
      await release.promise;
      events.push('b:end');
    });

    await tick();
    expect(events).toEqual(['a:start', 'b:start']);

    release.resolve();
    await Promise.all([a, b]);
    expect(events).toEqual(['a:start', 'b:start', 'a:end', 'b:end']);
  });

  it('flags the first writer as load_owner=true and a follower as load_owner=false', async () => {
    // Observability split: request A drives a cold load while request B
    // arrives microseconds later. Without the split, B's `resolve_ms`
    // looks like the full load latency even though A actually paid the
    // cost. `withModelLoadInstrumented` returns `owner=true` only when
    // the lock was free at sync-time; the follower observes `owner=false`
    // because A's `acquireWrite` already incremented `waitingWriters`.
    const coordinator = new ModelWorkCoordinator();
    const releaseA = deferred();
    const order: string[] = [];

    const aPromise = coordinator.withModelLoadInstrumented(async () => {
      order.push('a:start');
      await releaseA.promise;
      order.push('a:end');
      return 'A';
    });

    // Let A acquire the writer slot before B arrives. After this tick,
    // A is parked inside `fn` (awaiting `releaseA.promise`); the writer
    // lock is held synchronously so B will observe contention.
    await tick();

    const bPromise = coordinator.withModelLoadInstrumented(async () => {
      order.push('b:start');
      return 'B';
    });

    await tick();
    expect(order).toEqual(['a:start']);

    releaseA.resolve();
    const [a, b] = await Promise.all([aPromise, bPromise]);

    expect(a.owner).toBe(true);
    expect(a.result).toBe('A');
    expect(b.owner).toBe(false);
    expect(b.result).toBe('B');
    expect(order).toEqual(['a:start', 'a:end', 'b:start']);
  });

  it('partitions wait vs own-execution time in the instrumented outcome', async () => {
    // Regression: `serverLoadWaitMs` and `serverModelResolveMs` used to
    // both subtract from the same end timestamp, so a follower request
    // saw both fields report the full cold-load duration. The
    // coordinator now measures the two phases internally and exposes
    // them as `waitMs` (time blocked in `acquireWrite`) and `ownMs`
    // (time inside `fn` with the writer lock held). Together they
    // partition the call without overlap.
    const coordinator = new ModelWorkCoordinator();
    const aStarted = deferred();
    const releaseA = deferred();
    const OWN_DURATION_MS = 50;

    const aPromise = coordinator.withModelLoadInstrumented(async () => {
      aStarted.resolve();
      // Sleep inside the writer-held region so A's `ownMs` measures
      // the load-like work and B's `waitMs` measures the same span
      // from the other side of the lock.
      await new Promise<void>((resolve) => setTimeout(resolve, OWN_DURATION_MS));
      await releaseA.promise;
      return 'A';
    });

    // Wait until A has the writer slot and is parked inside `fn`.
    // Otherwise B might arrive before A's synchronous `acquireWrite`
    // increments `waitingWriters`, and the owner check would mis-fire.
    await aStarted.promise;

    const bPromise = coordinator.withModelLoadInstrumented(async () => 'B');

    releaseA.resolve();
    const [a, b] = await Promise.all([aPromise, bPromise]);

    expect(a.owner).toBe(true);
    expect(a.waitMs).toBeLessThan(20);
    expect(a.ownMs).toBeGreaterThanOrEqual(OWN_DURATION_MS - 5);

    expect(b.owner).toBe(false);
    expect(b.waitMs).toBeGreaterThanOrEqual(OWN_DURATION_MS - 5);
    expect(b.ownMs).toBeLessThan(20);
  });

  it('flags load_owner=true when a writer arrives during active inference reads', async () => {
    // Regression: the owner-decision predicate used to AND in
    // `activeReaders === 0`, so a writer that arrived during a live
    // inference read was mislabeled as `owner=false` and its
    // own load latency landed in `load_wait_ms` instead of
    // `resolve_ms`. The contract per `ModelLoadOutcome` and the
    // surrounding doc-comment is: a caller owns the load iff no
    // other writer is active and no writer is queued ahead — active
    // readers MUST NOT demote the arriving writer because, once
    // those reads drain, this writer is the one that performs the
    // load.
    const coordinator = new ModelWorkCoordinator();
    const readerRelease = deferred();
    const READER_HOLD_MS = 40;

    const readerPromise = coordinator.withInference(async () => {
      // Hold the reader long enough that B observes a measurable
      // `waitMs` after the reads drain.
      await new Promise<void>((resolve) => setTimeout(resolve, READER_HOLD_MS));
      await readerRelease.promise;
    });

    // Let the reader synchronously increment `activeReaders` before
    // the writer arrives, so the bug branch (the `activeReaders === 0`
    // term) would fire if reintroduced.
    await tick();

    const writerPromise = coordinator.withModelLoadInstrumented(async () => 'W');

    readerRelease.resolve();
    const [, w] = await Promise.all([readerPromise, writerPromise]);

    expect(w.owner).toBe(true);
    expect(w.result).toBe('W');
    expect(w.waitMs).toBeGreaterThanOrEqual(READER_HOLD_MS - 5);
  });

  it('gives pending model loads priority over new inference readers', async () => {
    const coordinator = new ModelWorkCoordinator();
    const releaseRead = deferred();
    const releaseWrite = deferred();
    const events: string[] = [];

    const read1 = coordinator.withInference(async () => {
      events.push('read1:start');
      await releaseRead.promise;
      events.push('read1:end');
    });
    await tick();

    const write = coordinator.withModelLoad(async () => {
      events.push('write:start');
      await releaseWrite.promise;
      events.push('write:end');
    });
    const read2 = coordinator.withInference(async () => {
      events.push('read2:start');
    });

    await tick();
    expect(events).toEqual(['read1:start']);

    releaseRead.resolve();
    await tick();
    expect(events).toEqual(['read1:start', 'read1:end', 'write:start']);

    releaseWrite.resolve();
    await Promise.all([read1, write, read2]);
    expect(events).toEqual(['read1:start', 'read1:end', 'write:start', 'write:end', 'read2:start']);
  });
});
