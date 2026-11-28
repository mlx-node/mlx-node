import { describe, expect, it } from 'vite-plus/test';

import { ModelWorkCoordinator } from '../../packages/server/src/model-work-coordinator.js';

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

const tick = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

function withDeadline<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Timed out after ${ms}ms waiting for ${label}`)), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      },
    );
  });
}

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

  it('wakes parked readers immediately when the only queued writer aborts behind an active reader', async () => {
    const coordinator = new ModelWorkCoordinator();
    const releaseRead1 = deferred();
    const events: string[] = [];

    const read1 = coordinator.withInference(async () => {
      events.push('read1:start');
      await releaseRead1.promise;
      events.push('read1:end');
    });
    try {
      await tick();

      const abortWriter = new AbortController();
      const write = coordinator.withModelLoad(
        async () => {
          events.push('write:start');
        },
        'aborted-model',
        abortWriter.signal,
      );
      const read2 = coordinator.withInference(async () => {
        events.push('read2:start');
      });

      await tick();
      expect(coordinator.waitingWriters).toBe(1);
      expect(events).toEqual(['read1:start']);

      abortWriter.abort();
      await withDeadline(expect(write).rejects.toThrow(), 2_000, 'aborted writer rejection');
      await withDeadline(read2, 2_000, 'parked reader wake-up');

      // read2 must have run while read1 is still active.
      expect(coordinator.waitingWriters).toBe(0);
      expect(coordinator.readerCount).toBe(1);
      expect(coordinator.lastLoad).toBeNull();
      expect(events).toEqual(['read1:start', 'read2:start']);
    } finally {
      releaseRead1.resolve();
      await withDeadline(read1, 2_000, 'read1 completion');
    }
    expect(events).toEqual(['read1:start', 'read2:start', 'read1:end']);
  });

  it('keeps parked readers waiting until every queued writer ahead of them aborts or completes', async () => {
    const coordinator = new ModelWorkCoordinator();
    const releaseRead1 = deferred();
    const events: string[] = [];

    const read1 = coordinator.withInference(async () => {
      events.push('read1:start');
      await releaseRead1.promise;
      events.push('read1:end');
    });
    try {
      await tick();

      const abortW1 = new AbortController();
      const abortW2 = new AbortController();
      const w1 = coordinator.withModelLoad(
        async () => {
          events.push('w1:start');
        },
        'm1',
        abortW1.signal,
      );
      const w2 = coordinator.withModelLoad(
        async () => {
          events.push('w2:start');
        },
        'm2',
        abortW2.signal,
      );
      const read2 = coordinator.withInference(async () => {
        events.push('read2:start');
      });

      await tick();
      expect(coordinator.waitingWriters).toBe(2);
      expect(events).toEqual(['read1:start']);

      abortW1.abort();
      await withDeadline(expect(w1).rejects.toThrow(), 2_000, 'w1 abort');
      await tick();
      // w2 is still queued, so writer-preference fairness keeps read2 parked.
      expect(coordinator.waitingWriters).toBe(1);
      expect(events).toEqual(['read1:start']);

      abortW2.abort();
      await withDeadline(expect(w2).rejects.toThrow(), 2_000, 'w2 abort');
      await withDeadline(read2, 2_000, 'read2 wake-up after w2 abort');
      expect(coordinator.waitingWriters).toBe(0);
      expect(events).toEqual(['read1:start', 'read2:start']);
    } finally {
      releaseRead1.resolve();
      await withDeadline(read1, 2_000, 'read1 completion');
    }
  });

  it('rejects a parked reader sharing the aborted signal with the last queued writer while waking live readers', async () => {
    const coordinator = new ModelWorkCoordinator();
    const releaseRead1 = deferred();
    const releaseLiveRead = deferred();
    const events: string[] = [];

    const read1 = coordinator.withInference(async () => {
      events.push('read1:start');
      await releaseRead1.promise;
      events.push('read1:end');
    });
    let liveReader: Promise<void> | undefined;
    try {
      await tick();

      const sharedAbort = new AbortController();
      const writer = coordinator.withModelLoad(
        async () => {
          events.push('writer:start');
        },
        'm1',
        sharedAbort.signal,
      );
      const cancelledReader = coordinator.withInference(async () => {
        events.push('cancelledReader:start');
      }, sharedAbort.signal);
      liveReader = coordinator.withInference(async () => {
        events.push('liveReader:start');
        await releaseLiveRead.promise;
        events.push('liveReader:end');
      });

      await tick();
      expect(coordinator.waitingWriters).toBe(1);
      expect(events).toEqual(['read1:start']);

      sharedAbort.abort();
      await withDeadline(expect(writer).rejects.toThrow(), 2_000, 'shared writer abort');
      await withDeadline(expect(cancelledReader).rejects.toThrow(), 2_000, 'shared reader abort');
      await tick();

      // Only liveReader woke; cancelledReader never ran and did not inflate readerCount.
      expect(coordinator.waitingWriters).toBe(0);
      expect(coordinator.readerCount).toBe(2);
      expect(events).toEqual(['read1:start', 'liveReader:start']);
    } finally {
      releaseLiveRead.resolve();
      releaseRead1.resolve();
      const pending = liveReader ? [liveReader, read1] : [read1];
      await withDeadline(Promise.allSettled(pending), 2_000, 'reader cleanup').catch(() => {});
    }
    expect(coordinator.readerCount).toBe(0);
    expect(events).toEqual(['read1:start', 'liveReader:start', 'liveReader:end', 'read1:end']);
  });

  it('aborts without running fn or writing lastLoad when signal aborts after writer handoff before microtask resumes', async () => {
    for (const instrumented of [false, true]) {
      const coordinator = new ModelWorkCoordinator();
      const releaseFirst = deferred();
      const abortSecond = new AbortController();
      const events: string[] = [];
      let sawGrantedBeforeAbort = false;

      const first = coordinator.withModelLoad(async () => {
        events.push('first:start');
        await releaseFirst.promise;
        events.push('first:end');
        // Queue a 2-hop microtask: hop 1 runs just before `withModelLoad`'s
        // `finally -> releaseWrite() -> drain()` grants `second`, so hop 2
        // lands strictly AFTER `grant()` removed `onAbort` and BEFORE
        // `second`'s `await acquireWrite(signal)` continuation resumes.
        queueMicrotask(() => {
          queueMicrotask(() => {
            sawGrantedBeforeAbort = coordinator.writerActive && coordinator.waitingWriters === 1;
            abortSecond.abort();
          });
        });
      }, 'first-model');
      try {
        await tick();

        const second = instrumented
          ? coordinator.withModelLoadInstrumented(
              async () => {
                events.push('second:start');
              },
              'second-model',
              abortSecond.signal,
            )
          : coordinator.withModelLoad(
              async () => {
                events.push('second:start');
              },
              'second-model',
              abortSecond.signal,
            );
        let lastLoadWhenThirdStarted: ModelWorkCoordinator['lastLoad'] = null;
        const third = coordinator.withModelLoad(async () => {
          lastLoadWhenThirdStarted = coordinator.lastLoad;
          events.push('third:start');
        }, 'third-model');
        await tick();

        releaseFirst.resolve();

        await withDeadline(expect(second).rejects.toThrow(), 2_000, 'post-handoff second abort');
        await withDeadline(Promise.all([first, third]), 2_000, 'first and third writers');

        expect(sawGrantedBeforeAbort).toBe(true);
        expect(lastLoadWhenThirdStarted).toMatchObject({ label: 'first-model', ok: true, error: null });
        expect(events).toEqual(['first:start', 'first:end', 'third:start']);
        expect(coordinator.writerActive).toBe(false);
        expect(coordinator.waitingWriters).toBe(0);
        expect(coordinator.lastLoad).toMatchObject({ label: 'third-model', ok: true });
      } finally {
        releaseFirst.resolve();
        await withDeadline(first, 2_000, 'first writer cleanup');
      }
    }
  });

  it('aborts withInference without running fn if signal aborts after reader grant before microtask resumes', async () => {
    const coordinator = new ModelWorkCoordinator();
    const releaseWrite = deferred();
    const abortRead = new AbortController();
    const events: string[] = [];
    let sawReaderGrantedBeforeAbort = false;

    const write = coordinator.withModelLoad(async () => {
      events.push('write:start');
      await releaseWrite.promise;
      events.push('write:end');
      queueMicrotask(() => {
        queueMicrotask(() => {
          sawReaderGrantedBeforeAbort = coordinator.readerCount === 1;
          abortRead.abort();
        });
      });
    }, 'm1');
    try {
      await tick();

      const cancelledRead = coordinator.withInference(async () => {
        events.push('cancelledRead:start');
      }, abortRead.signal);
      await tick();

      releaseWrite.resolve();

      await withDeadline(expect(cancelledRead).rejects.toThrow(), 2_000, 'post-grant reader abort');
      await withDeadline(write, 2_000, 'write completion');

      expect(sawReaderGrantedBeforeAbort).toBe(true);
      expect(events).toEqual(['write:start', 'write:end']);
      expect(coordinator.readerCount).toBe(0);
    } finally {
      releaseWrite.resolve();
      await withDeadline(write, 2_000, 'write cleanup');
    }
  });
});
