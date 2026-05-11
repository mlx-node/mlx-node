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
