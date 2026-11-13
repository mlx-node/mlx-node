/**
 * The model select call against a fake engine on 127.0.0.1: the status map, the request shape, the loopback rule, the
 * name check, the ceiling, and the abort that closes the socket.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { answerOfStatus, selectModel, selectUrl, type SelectTarget } from '../src/chat/agent/select-model.js';
import { until } from './helpers/chat-fake.js';
import { gate, SELECT_PATH, startFakeEngine, type FakeEngine } from './helpers/fake-engine.js';

const TOKEN = 'tok-SELECT-0123456789';
const MODEL = 'qwen3.5-4b';

let engine: FakeEngine;
let release: Array<() => void>;

function target(url = engine.url): SelectTarget {
  return { url, token: TOKEN, models: [{ name: MODEL }, { name: 'qwen3.5-9b' }] };
}

beforeEach(async () => {
  engine = await startFakeEngine();
  release = [];
});

afterEach(async () => {
  for (const open of release) open();
  await engine.close();
});

describe('selectModel', () => {
  it.each([
    [200, 'selected'],
    [501, 'selected'],
    [400, 'selected'],
    [404, 'unknown'],
    [401, 'auth'],
    [403, 'auth'],
    [429, 'failed'],
    [500, 'failed'],
    [418, 'failed'],
  ] as const)('engine status %i is %s', async (status, answer) => {
    expect(answerOfStatus(status)).toBe(answer);
    engine.script(() => ({ status, json: {} }));
    expect(await selectModel(target(), MODEL, { maxMs: 5_000 })).toBe(answer);
    expect(engine.selects()).toHaveLength(1);
  });

  it('posts the model and one tiny message with the token in the header only', async () => {
    expect(await selectModel(target(), MODEL, { maxMs: 5_000 })).toBe('selected');
    const [request] = engine.requests;
    expect(request).toMatchObject({ method: 'POST', path: SELECT_PATH });
    expect(request!.headers['x-api-key']).toBe(TOKEN);
    expect(request!.headers['content-type']).toBe('application/json');
    expect(JSON.parse(request!.body)).toEqual({ model: MODEL, messages: [{ role: 'user', content: '.' }] });
    expect(request!.body).not.toContain(TOKEN);
  });

  it('keeps a path prefix of the engine URL', () => {
    expect(selectUrl('http://127.0.0.1:4000/')?.pathname).toBe(SELECT_PATH);
    expect(selectUrl('http://127.0.0.1:4000/base')?.pathname).toBe(`/base${SELECT_PATH}`);
  });

  it('sends nothing to an address that is not loopback http', async () => {
    const port = new URL(engine.url).port;
    for (const url of [`http://localhost:${port}`, `https://127.0.0.1:${port}`, 'http://10.0.0.1:1', 'not a url']) {
      expect(await selectModel(target(url), MODEL, { maxMs: 5_000 })).toBe('socket');
    }
    expect(engine.requests).toHaveLength(0);
  });

  it('a name the endpoint file does not list is unknown, with no request', async () => {
    expect(await selectModel(target(), 'mlx-local', { maxMs: 5_000 })).toBe('unknown');
    expect(engine.requests).toHaveLength(0);
  });

  it('a load longer than the ceiling fails, and the socket is closed', async () => {
    const hold = gate();
    release.push(hold.open);
    engine.script(() => ({ until: hold.promise, json: {} }));
    expect(await selectModel(target(), MODEL, { maxMs: 300 })).toBe('failed');
    await until(() => engine.selects()[0]?.closedEarly === true);
  });

  it('an abort closes the socket at once', async () => {
    const hold = gate();
    release.push(hold.open);
    engine.script(() => ({ until: hold.promise, json: {} }));
    const abort = new AbortController();
    const pending = selectModel(target(), MODEL, { maxMs: 60_000, signal: abort.signal });
    await until(() => engine.selects().length === 1);
    abort.abort();
    expect(await pending).toBe('aborted');
    await until(() => engine.selects()[0]?.closedEarly === true);
  });

  it('a dropped socket and a closed port are socket failures', async () => {
    engine.script(() => ({ drop: true }));
    expect(await selectModel(target(), MODEL, { maxMs: 5_000 })).toBe('socket');
    const url = engine.url;
    await engine.close();
    engine = await startFakeEngine();
    expect(await selectModel(target(url), MODEL, { maxMs: 5_000 })).toBe('socket');
  });
});
