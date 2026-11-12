/** @vitest-environment happy-dom */

import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  assertHarnessChat,
  attachHarness,
  BRIDGE_FILE,
  DEV_CLOSE,
  DEV_FOLDER,
  DEV_OPEN,
  DEV_PICK_FOLDER,
  DEV_RPC,
  DEV_RPC_REPLY,
  devHarness,
  GALLERY_PATH,
  galleryHtml,
  guardRuntime,
  HARNESS_LONG_MODEL,
  harnessChatOptions,
  type HarnessChannel,
  type HarnessClient,
} from '../scripts/dev/harness-plugin.js';
import type { ApiResponse } from '../src/api/errors.js';
import { KeyringSecretStore } from '../src/chat/providers/secrets.js';
import type { LiveEvent } from '../src/live.js';
import type { RpcRuntime } from '../src/rpc/host.js';
import { DASHBOARD_PORT_MESSAGE } from '../src/rpc/protocol.js';
import type { ApiCall } from '../src/runtime.js';

type Listener = (data: unknown, client: HarnessClient) => void;

/** Base URLs as the harness gets them from its fake providers. */
const LOOPBACK = {
  exa: 'http://127.0.0.1:41001',
  deepseek: 'http://127.0.0.1:41002',
  brave: 'http://127.0.0.1:41003',
  firecrawl: 'http://127.0.0.1:41004',
};

/** Vite's HMR socket as the plugin sees it: custom events in, `client.send` out. */
class FakeSocket implements HarnessChannel {
  readonly listeners = new Map<string, Set<Listener>>();
  on(event: string, listener: Listener): void {
    const set = this.listeners.get(event) ?? new Set<Listener>();
    set.add(listener);
    this.listeners.set(event, set);
  }
  off(event: string, listener: Listener): void {
    this.listeners.get(event)?.delete(listener);
  }
  emit(event: string, data: unknown, client: HarnessClient): void {
    for (const listener of this.listeners.get(event) ?? []) listener(data, client);
  }
}

class FakeClient implements HarnessClient {
  readonly sent: Array<{ event: string; payload: unknown }> = [];
  send(event: string, payload?: unknown): void {
    // The real socket serializes: what crosses must survive JSON.
    this.sent.push({ event, payload: JSON.parse(JSON.stringify(payload)) as unknown });
  }
  replies(channel: string): unknown[] {
    return this.sent
      .filter((entry) => entry.event === DEV_RPC_REPLY && (entry.payload as { channel: string }).channel === channel)
      .map((entry) => (entry.payload as { message: unknown }).message);
  }
}

function fakeRuntime(): RpcRuntime & { calls: ApiCall[]; watchers: Set<(event: LiveEvent) => void> } {
  const calls: ApiCall[] = [];
  const watchers = new Set<(event: LiveEvent) => void>();
  return {
    calls,
    watchers,
    call(call: ApiCall): Promise<ApiResponse> {
      calls.push(call);
      return Promise.resolve({ ok: true, status: 200, body: { path: call.path } });
    },
    subscribe: () => () => {},
    watchLive(listener: (event: LiveEvent) => void): () => void {
      watchers.add(listener);
      return () => {
        watchers.delete(listener);
      };
    },
  };
}

const HARNESS_HOME = join(tmpdir(), 'mlx-dev-harness-test-home');

async function settle(): Promise<void> {
  for (let i = 0; i < 5; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
}

describe('dev harness plugin', () => {
  it('opens a channel, answers a call over the socket, and closes it', async () => {
    const socket = new FakeSocket();
    const client = new FakeClient();
    const runtime = fakeRuntime();
    const detach = attachHarness(socket, runtime);
    socket.emit(DEV_OPEN, { channel: 'tab-1' }, client);
    socket.emit(
      DEV_RPC,
      { channel: 'tab-1', message: { kind: 'call', id: 1, call: { method: 'GET', path: '/api/models' } } },
      client,
    );
    await settle();
    expect(runtime.calls.map((call) => call.path)).toEqual(['/api/models']);
    expect(client.replies('tab-1')).toEqual([
      { kind: 'response', id: 1, response: { ok: true, status: 200, body: { path: '/api/models' } } },
    ]);

    socket.emit(DEV_CLOSE, { channel: 'tab-1' }, client);
    socket.emit(
      DEV_RPC,
      { channel: 'tab-1', message: { kind: 'call', id: 2, call: { method: 'GET', path: '/api/health' } } },
      client,
    );
    await settle();
    expect(runtime.calls).toHaveLength(1);
    detach();
    expect([...socket.listeners.values()].every((set) => set.size === 0)).toBe(true);
  });

  it('answers the two network routes locally unless the network is allowed', async () => {
    const runtime = fakeRuntime();
    const guarded = guardRuntime(runtime, false);
    const download = await guarded.call({ method: 'POST', path: '/api/downloads', body: { repo: 'x/y' } });
    const updates = await guarded.call({ method: 'get', path: '/api/catalog/updates?force=1' });
    for (const response of [download, updates]) {
      expect(response).toMatchObject({ ok: false, code: 'E_UNAVAILABLE', message: 'Disabled in the dev harness' });
    }
    expect(runtime.calls).toEqual([]);
    expect((await guarded.call({ method: 'GET', path: '/api/downloads' })).ok).toBe(true);
    const open = guardRuntime(runtime, true);
    expect((await open.call({ method: 'POST', path: '/api/downloads' })).ok).toBe(true);
    expect(runtime.calls.map((call) => `${call.method} ${call.path}`)).toEqual([
      'GET /api/downloads',
      'POST /api/downloads',
    ]);
  });

  it('answers a folder request with the fixture work folder, to the asking tab only', () => {
    const socket = new FakeSocket();
    const client = new FakeClient();
    const other = new FakeClient();
    const detach = attachHarness(socket, fakeRuntime(), '/fixture/work');
    socket.emit(DEV_PICK_FOLDER, { requestId: 'r-1', path: '/etc' }, client);
    for (const requestId of ['', 'x'.repeat(65), 7, undefined]) socket.emit(DEV_PICK_FOLDER, { requestId }, client);
    socket.emit(DEV_PICK_FOLDER, null, client);
    expect(client.sent).toEqual([{ event: DEV_FOLDER, payload: { requestId: 'r-1', path: '/fixture/work' } }]);
    expect(other.sent).toEqual([]);
    detach();
    socket.emit(DEV_PICK_FOLDER, { requestId: 'r-2' }, client);
    expect(client.sent).toHaveLength(1);

    const bare = new FakeSocket();
    attachHarness(bare, fakeRuntime());
    bare.emit(DEV_PICK_FOLDER, { requestId: 'r-3' }, other);
    expect(other.sent).toEqual([{ event: DEV_FOLDER, payload: { requestId: 'r-3', path: null } }]);
  });

  it('releases a live watch when the tab socket disconnects', async () => {
    const socket = new FakeSocket();
    const client = new FakeClient();
    const other = new FakeClient();
    const runtime = fakeRuntime();
    attachHarness(socket, runtime);
    socket.emit(DEV_OPEN, { channel: 'tab-1' }, client);
    socket.emit(DEV_OPEN, { channel: 'tab-2' }, other);
    socket.emit(DEV_RPC, { channel: 'tab-1', message: { kind: 'watch', id: 7, topic: 'live' } }, client);
    socket.emit(DEV_RPC, { channel: 'tab-2', message: { kind: 'watch', id: 7, topic: 'live' } }, other);
    await settle();
    expect(runtime.watchers.size).toBe(2);
    for (const watcher of runtime.watchers) watcher({ type: 'data', version: 1, traces: true, runtime: false });
    expect(client.replies('tab-1')).toEqual([
      { kind: 'live', id: 7, event: { type: 'data', version: 1, traces: true, runtime: false } },
    ]);

    socket.emit('vite:client:disconnect', undefined, client);
    await settle();
    expect(runtime.watchers.size).toBe(1);
  });

  it('is a serve-only plugin that injects the bridge after the entry script', () => {
    const plugin = devHarness(fakeRuntime(), { allowNetwork: false, chat: harnessChatOptions(HARNESS_HOME, LOOPBACK) });
    expect(plugin.apply).toBe('serve');
    const hook = plugin.transformIndexHtml as {
      order: string;
      handler: (html: string, context: { path: string; originalUrl?: string }) => unknown;
    };
    expect(hook.order).toBe('post');
    expect(hook.handler('', { path: '/index.html', originalUrl: '/engine' })).toEqual([
      { tag: 'script', attrs: { type: 'module', src: `/@fs${BRIDGE_FILE}` }, injectTo: 'body' },
    ]);
    expect(BRIDGE_FILE.endsWith('/scripts/dev/bridge.ts')).toBe(true);
    expect(hook.handler('', { path: GALLERY_PATH, originalUrl: `${GALLERY_PATH}?x=1` })).toEqual([]);
  });

  it('serves the gallery page from scripts/, its entry by absolute /@fs/ path', () => {
    const html = galleryHtml();
    const marker = 'src="/@fs';
    const start = html.indexOf(marker);
    expect(start).toBeGreaterThan(-1);
    const src = html.slice(start + marker.length, html.indexOf('"', start + marker.length));
    expect(src.endsWith('/scripts/dev/gallery/gallery.tsx')).toBe(true);
    expect(existsSync(src)).toBe(true);
    expect(html).toContain('<div id="root"></div>');
  });
});

describe('dev harness bridge', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('waits for DOMContentLoaded, so a late main.tsx listener still gets the port', async () => {
    Object.defineProperty(document, 'readyState', { configurable: true, get: () => 'interactive' });
    const post = vi.spyOn(window, 'postMessage');
    const received: unknown[] = [];
    try {
      await import('../scripts/dev/bridge.js');
      expect(post).not.toHaveBeenCalled();

      await new Promise((resolve) => setTimeout(resolve, 50));
      window.addEventListener('message', (event: MessageEvent) => received.push(event.data));
      expect(post).not.toHaveBeenCalled();

      document.dispatchEvent(new Event('DOMContentLoaded'));
      expect(post).toHaveBeenCalledTimes(1);
      const [message, origin, transfer] = post.mock.calls[0]! as unknown as [unknown, string, unknown[]];
      expect(message).toEqual({ type: DASHBOARD_PORT_MESSAGE, generation: 1 });
      expect(origin).toBe('*');
      expect(transfer).toHaveLength(1);
      await new Promise((resolve) => setTimeout(resolve, 20));
      expect(received).toEqual([{ type: DASHBOARD_PORT_MESSAGE, generation: 1 }]);

      document.dispatchEvent(new Event('DOMContentLoaded'));
      expect(post).toHaveBeenCalledTimes(1);
    } finally {
      delete (document as { readyState?: unknown }).readyState;
    }
  });
});

describe('dev harness Chat', () => {
  it('runs Chat only on the fake agent, with keys in memory and every provider on loopback', () => {
    const chat = harnessChatOptions(HARNESS_HOME, LOOPBACK);
    expect(() => assertHarnessChat(chat)).not.toThrow();
    expect(chat.root).toBe(join(HARNESS_HOME, 'chat'));
    expect(chat.homeDir).toBe(HARNESS_HOME);
    expect(() => assertHarnessChat(undefined)).toThrow('Chat options are required');
    expect(() => assertHarnessChat({ ...chat, agentCommand: undefined })).toThrow('fake agent');
    expect(() =>
      assertHarnessChat({ ...chat, agentCommand: { command: '/usr/local/bin/mlx', args: ['agent'] } }),
    ).toThrow('fake agent');
    expect(chat.providerBaseUrls).toEqual(LOOPBACK);
    expect(() => assertHarnessChat({ ...chat, secrets: new KeyringSecretStore() })).toThrow('kept in memory');
    expect(() => assertHarnessChat({ ...chat, secrets: undefined })).toThrow('kept in memory');
    expect(() => assertHarnessChat({ ...chat, providerBaseUrls: undefined })).toThrow('loopback');
    expect(() =>
      assertHarnessChat({ ...chat, providerBaseUrls: { ...LOOPBACK, brave: 'https://api.search.brave.com' } }),
    ).toThrow('loopback');
    expect(() =>
      assertHarnessChat({ ...chat, providerBaseUrls: { ...LOOPBACK, exa: 'http://192.168.1.2:80' } }),
    ).toThrow('loopback');
    expect(chat.baseEnv?.HOME).toBe(HARNESS_HOME);
    expect(() => assertHarnessChat({ ...chat, baseEnv: { ...chat.baseEnv, HOME: '/Users/someone' } })).toThrow(
      'fixture home',
    );
    expect(() => assertHarnessChat({ ...chat, baseEnv: undefined })).toThrow('fixture home');
    const { firecrawl: _missing, ...three } = LOOPBACK;
    expect(() => assertHarnessChat({ ...chat, providerBaseUrls: three })).toThrow('loopback');
    expect(() =>
      devHarness(fakeRuntime(), { allowNetwork: false, chat: { ...chat, agentCommand: undefined } }),
    ).toThrow('fake agent');
  });

  it('gives the endpoint a fresh throwaway token each time', async () => {
    const a = await harnessChatOptions(HARNESS_HOME, LOOPBACK).readEndpoint!();
    const b = await harnessChatOptions(HARNESS_HOME, LOOPBACK).readEndpoint!();
    expect(a!.token).not.toBe(b!.token);
    expect(a!.models.map((m) => m.name)).toEqual(['qwen3.5-4b', 'qwen3.5-9b', HARNESS_LONG_MODEL]);
    const engine = await harnessChatOptions(HARNESS_HOME, LOOPBACK, 'http://127.0.0.1:41010').readEndpoint!();
    expect(engine!.url).toBe('http://127.0.0.1:41010');
  });

  it('lets the provider routes through, so a key is checked against the fake providers', async () => {
    const seen: string[] = [];
    const runtime: RpcRuntime = {
      ...fakeRuntime(),
      call: (call) => {
        seen.push(`${call.method} ${call.path}`);
        return Promise.resolve({ ok: true, body: {} } as ApiResponse);
      },
    };
    const guarded = guardRuntime(runtime, false);
    for (const [method, path] of [
      ['GET', '/api/providers'],
      ['PUT', '/api/providers/exa/key'],
      ['DELETE', '/api/providers/exa/key'],
      ['POST', '/api/providers/exa/validate'],
      ['PUT', '/api/providers/selection'],
    ] as const) {
      expect((await guarded.call({ method, path })).ok).toBe(true);
    }
    expect(seen).toHaveLength(5);
  });

  it('forwards the Chat streams of the served runtime', () => {
    const calls: string[] = [];
    const runtime: RpcRuntime = {
      ...fakeRuntime(),
      watchChat: (scope) => {
        calls.push(scope.scope);
        return () => {};
      },
    };
    const guarded = guardRuntime(runtime, false);
    guarded.watchChat!({ scope: 'host' }, () => {})();
    expect(calls).toEqual(['host']);
  });
});
