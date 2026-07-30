/**
 * The INFERENCE sidecar's rules.
 *
 * The one that matters most is the handshake: the supervisor takes the URL that
 * arrives in `mlx:ready` as the authority on where `/health` lives and polls
 * nothing else, so a sidecar that announces a URL it is not listening on parks
 * the app in `starting` for a whole readiness budget and then reports a failure
 * that names the wrong cause. Every readiness test here therefore CONNECTS to
 * the announced URL rather than string-matching it, and the server behind it
 * answers with a per-test nonce so that reaching *something* is not mistaken for
 * reaching the host.
 */

import { randomUUID } from 'node:crypto';
import { createServer, type Server } from 'node:http';
import { AddressInfo } from 'node:net';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import {
  EXIT_STARTUP_FAILED,
  handleSidecarRequest,
  NoParentChannelError,
  resolveParentChannel,
  runSidecar,
  SIDECAR_HOST_OPTIONS,
  SIDECAR_HOST_CLOSE_TIMEOUT_MS,
  sidecarHostOptions,
  type ParentChannel,
  type SidecarHost,
} from '../src/inference/sidecar.js';

/** Stand-in for the per-process CSPRNG token the real entry mints. */
const TEST_AUTH_TOKEN = 'test-sidecar-token';

const servers: Server[] = [];

afterEach(async () => {
  for (const server of servers.splice(0)) {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

/**
 * A `SidecarHost` backed by a REAL listening socket, exactly as
 * `createInferenceHost` is: `port: 0`, and the URL built from what the kernel
 * assigned rather than from what was asked for.
 */
async function realHost(overrides: Partial<SidecarHost> = {}): Promise<SidecarHost & { nonce: string }> {
  const nonce = randomUUID();
  const server = createServer((req, res) => {
    if (req.url === '/health') {
      res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify({ status: 'ok', nonce }));
      return;
    }
    res.writeHead(404).end();
  });
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = (server.address() as AddressInfo).port;
  return {
    nonce,
    url: `http://127.0.0.1:${port}`,
    port,
    modelsDir: '/models',
    boundModel: 'a-model',
    models: [{ name: 'a-model', path: '/models/a-model', modelType: 'qwen3' }],
    logDir: null,
    health: () => ({ status: 'ok', nonce }),
    loadModel: async () => {},
    close: async () => {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    },
    ...overrides,
  };
}

interface Harness {
  channel: ParentChannel;
  /** Everything posted to the supervisor, in order. */
  sent: Record<string, unknown>[];
  /** Deliver one supervisor → sidecar message. */
  deliver(message: unknown): void;
  stderr: string[];
  exits: number[];
  signal(): void;
  deps: Parameters<typeof runSidecar>[0];
}

function harness(createHost: () => Promise<SidecarHost>): Harness {
  const sent: Record<string, unknown>[] = [];
  const stderr: string[] = [];
  const exits: number[] = [];
  let inbound: ((message: unknown) => void) | null = null;
  let stopHandler: (() => void) | null = null;

  const channel: ParentChannel = {
    post: (message) => sent.push(message as Record<string, unknown>),
    onMessage: (listener) => {
      inbound = listener;
    },
  };
  const h: Harness = {
    channel,
    sent,
    stderr,
    exits,
    deliver: (message) => inbound?.(message),
    signal: () => stopHandler?.(),
    deps: {
      channel,
      createHost,
      authToken: TEST_AUTH_TOKEN,
      onStopSignal: (handler) => {
        stopHandler = handler;
      },
      logError: (line) => stderr.push(line),
      exit: (code) => exits.push(code),
    },
  };
  return h;
}

/** A promise the test releases by hand, to hold `createHost` open. */
function deferred(): { promise: Promise<void>; release: () => void } {
  let release!: () => void;
  const promise = new Promise<void>((resolve) => {
    release = resolve;
  });
  return { promise, release };
}

const readyUrl = (sent: Record<string, unknown>[]): string => {
  const ready = sent.find((message) => message.kind === 'mlx:ready');
  expect(ready, `no mlx:ready in ${JSON.stringify(sent)}`).toBeDefined();
  return ready?.url as string;
};

describe('the readiness handshake', () => {
  it('announces a URL that is actually listening, and is actually the host', async () => {
    const host = await realHost();
    const h = harness(async () => host);
    await runSidecar(h.deps);

    // Connect. A string comparison would pass just as happily for a URL built by
    // arithmetic on the right port, which is the failure this exists to catch.
    const body = (await (await fetch(new URL('/health', readyUrl(h.sent)))).json()) as { nonce: string };
    expect(body.nonce).toBe(host.nonce);
  });

  it('binds an ephemeral port instead of reserving one first', () => {
    // `createInferenceHost` treats an omitted port as "pick one for me" and calls
    // `pickFreePort()` — bind, read, close, then listen again on the same number.
    // Losing that race is an EADDRINUSE at launch with no user-actionable cause.
    // `0` lets the kernel assign and the host reads the socket back.
    expect(SIDECAR_HOST_OPTIONS.port).toBe(0);
    // Loopback is what makes the supervisor's `isLoopbackHttpUrl` guard pass by
    // construction rather than by luck.
    expect(SIDECAR_HOST_OPTIONS.host).toBe('127.0.0.1');
  });

  it('gates the port with a secret, because loopback is not a closed door', () => {
    // Any local process can reach a loopback port, and so can JavaScript on any
    // page the user's browser is showing — `fetch('http://127.0.0.1:<port>')`
    // is an ordinary cross-origin request, and the server's pre-auth default of
    // `Access-Control-Allow-Origin: *` made the reply readable to it. Without a
    // token here, `/v1/messages` is open to every one of them.
    const options = sidecarHostOptions();
    expect(typeof options.authToken).toBe('string');
    // 32 CSPRNG bytes, base64url. A short or low-entropy value is worse than
    // none: it also switches CORS off, so the surface LOOKS protected.
    expect(options.authToken.length).toBeGreaterThanOrEqual(40);
  });

  it('mints a fresh token per process, never a build-time constant', () => {
    // A module-level constant would be identical in every install of the app,
    // which makes it a published password.
    const tokens = new Set(Array.from({ length: 8 }, () => sidecarHostOptions().authToken));
    expect(tokens.size).toBe(8);
  });

  it('keeps the bind that the token protects', () => {
    const options = sidecarHostOptions();
    expect(options.port).toBe(SIDECAR_HOST_OPTIONS.port);
    expect(options.host).toBe(SIDECAR_HOST_OPTIONS.host);
  });

  it('says nothing at all when the host cannot start', async () => {
    const h = harness(() => Promise.reject(new Error('No models discovered under /models.')));
    await runSidecar(h.deps);

    expect(h.sent).toEqual([]);
    // Exits immediately rather than waiting out the supervisor's 60 s readiness
    // budget: no amount of waiting discovers a model that is not there, and a
    // fast exit turns it into a visible `failed` with the reason attached.
    expect(h.exits).toEqual([EXIT_STARTUP_FAILED]);
    expect(h.stderr.join('\n')).toContain('No models discovered');
  });
});

describe('requests', () => {
  it('answers every request with the matching id', async () => {
    const host = await realHost();
    const h = harness(async () => host);
    await runSidecar(h.deps);

    h.deliver({ kind: 'mlx:request', id: 7, payload: { op: 'info' } });
    h.deliver({ kind: 'mlx:request', id: 8, payload: { op: 'models' } });
    await tick();

    const replies = h.sent.filter((message) => message.kind === 'mlx:response');
    expect(replies.map((r) => r.id)).toEqual([7, 8]);
    expect(replies.every((r) => r.ok === true)).toBe(true);
    expect((replies[0].value as { url: string }).url).toBe(host.url);
    expect(replies[1].value).toEqual([{ name: 'a-model', path: '/models/a-model', modelType: 'qwen3' }]);
  });

  it('reports a failure rather than dropping a request it cannot answer', async () => {
    const h = harness(() => realHost());
    await runSidecar(h.deps);

    h.deliver({ kind: 'mlx:request', id: 1, payload: { op: 'nope' } });
    h.deliver({ kind: 'mlx:request', id: 2, payload: 'not an object' });
    await tick();

    const replies = h.sent.filter((message) => message.kind === 'mlx:response');
    // A dropped request is worse than a failed one: the supervisor's promise
    // hangs until its own timeout and the reason is lost.
    expect(replies.map((r) => [r.id, r.ok])).toEqual([
      [1, false],
      [2, false],
    ]);
    expect(String(replies[0].error)).toContain('unknown op');
  });

  it('answers a request that arrives before the host is up', async () => {
    const slow = deferred();
    const h = harness(async () => {
      await slow.promise;
      return realHost();
    });
    const running = runSidecar(h.deps);

    h.deliver({ kind: 'mlx:request', id: 4, payload: { op: 'health' } });
    await tick();

    // The supervisor only sends a request once it has seen a serving `/health`,
    // so this should not happen — but if it does, a dropped request leaves its
    // promise hanging for the full 60 s request timeout with no reason attached,
    // and the tray reports a healthy sidecar throughout.
    expect(h.sent).toEqual([{ kind: 'mlx:response', id: 4, ok: false, error: 'inference host is not up yet' }]);

    slow.release();
    await running;
  });

  it('ignores anything that is not an mlx:request', async () => {
    const h = harness(() => realHost());
    await runSidecar(h.deps);
    const before = h.sent.length;

    for (const junk of [null, 'hello', 42, { kind: 'mlx:response', id: 1 }, { kind: 'mlx:request' }]) {
      h.deliver(junk);
    }
    await tick();

    expect(h.sent.length).toBe(before);
  });

  it('propagates a load failure as a failed response, not a crash', async () => {
    const host = await realHost({
      loadModel: () => Promise.reject(new Error('Model "ghost" not found under /models.')),
    });
    const h = harness(async () => host);
    await runSidecar(h.deps);

    h.deliver({ kind: 'mlx:request', id: 3, payload: { op: 'load', name: 'ghost' } });
    await tick();

    const reply = h.sent.find((message) => message.kind === 'mlx:response');
    expect(reply?.ok).toBe(false);
    expect(String(reply?.error)).toContain('not found');
    expect(h.exits).toEqual([]);
  });
});

describe('handleSidecarRequest', () => {
  it('projects models field by field', async () => {
    const host = await realHost({
      // `DiscoveredModel` also carries a `preset`, which is the server package's
      // business and crosses a structured clone. Passing the object through would
      // break this channel the day a non-clonable member is added there.
      models: [{ name: 'm', path: '/p', modelType: 'gemma4', preset: { sampling: () => 1 } } as never],
    });
    const value = (await handleSidecarRequest(host, { op: 'models' })) as unknown[];
    expect(value).toEqual([{ name: 'm', path: '/p', modelType: 'gemma4' }]);
    expect(() => structuredClone(value)).not.toThrow();
  });

  it('refuses a load with no name', async () => {
    const host = await realHost();
    await expect(handleSidecarRequest(host, { op: 'load' })).rejects.toThrow(/requires a model name/);
  });
});

describe('shutdown', () => {
  it('closes the host and exits 0 on the stop signal', async () => {
    const host = await realHost();
    let closed = 0;
    let closeOptions: { timeoutMs?: number } | undefined;
    const h = harness(async () => ({
      ...host,
      close: async (options) => {
        closed += 1;
        closeOptions = options;
        await host.close();
      },
    }));
    await runSidecar(h.deps);

    h.signal();
    await tick();

    expect(closed).toBe(1);
    // The parent reserves a cleanup margin AFTER this bounded HTTP drain. If
    // this silently falls back to the server default, the two values can drift
    // back to equality and SIGKILL can interrupt logger/temp-root disposal.
    expect(closeOptions).toEqual({ timeoutMs: SIDECAR_HOST_CLOSE_TIMEOUT_MS });
    // 0, and the supervisor does NOT read it as success: it decides clean-vs-crash
    // from the intent it recorded before the kill, because `utilityProcess.kill()`
    // reports 0 either way.
    expect(h.exits).toEqual([0]);
  });

  it('still exits when close() throws', async () => {
    const host = await realHost({ close: () => Promise.reject(new Error('socket refused to die')) });
    const h = harness(async () => host);
    await runSidecar(h.deps);

    h.signal();
    await tick();

    // A host that will not close must not become a menubar app that will not
    // quit — the user's only remaining option is Force Quit.
    expect(h.exits).toEqual([0]);
    expect(h.stderr.join('\n')).toContain('socket refused to die');
  });

  it('is idempotent', async () => {
    const h = harness(() => realHost());
    await runSidecar(h.deps);
    h.signal();
    h.signal();
    await tick();
    expect(h.exits).toEqual([0]);
  });

  it('does not announce readiness for a host that is already stopping', async () => {
    const slow = deferred();
    const host = await realHost();
    let closed = 0;
    const h = harness(async () => {
      await slow.promise;
      return { ...host, close: async () => void (closed += 1) };
    });

    const running = runSidecar(h.deps);
    h.signal();
    slow.release();
    await running;

    expect(h.sent.filter((message) => message.kind === 'mlx:ready')).toEqual([]);
    // The host appeared after the decision to stop. Leaving it would strand a
    // listening socket nobody can reach.
    expect(closed).toBe(1);
  });
});

describe('resolveParentChannel', () => {
  it('prefers parentPort, whose message events carry { data }', () => {
    const posted: unknown[] = [];
    const seen: unknown[] = [];
    const inbound: ((event: { data: unknown }) => void)[] = [];
    const channel = resolveParentChannel({
      parentPort: {
        postMessage: (message) => posted.push(message),
        on: (_event, listener) => {
          inbound.push(listener);
        },
      },
      // Present too. `parentPort` exists ONLY in a utilityProcess, so preferring
      // it can never mis-detect; the other order would rest on `process.send`
      // being absent under Electron, which nothing documents.
      send: () => posted.push('WRONG CHANNEL'),
      on: () => {},
    });
    channel.onMessage((message) => seen.push(message));
    channel.post({ kind: 'mlx:ready' });
    for (const listener of inbound) listener({ data: { kind: 'mlx:request' } });

    expect(posted).toEqual([{ kind: 'mlx:ready' }]);
    // Unwrapped: `.on('message')` under `child_process` hands over the raw value
    // and a utilityProcess hands over a MessageEvent. The sidecar sees one shape.
    expect(seen).toEqual([{ kind: 'mlx:request' }]);
  });

  it('falls back to process.send, whose message events carry the raw value', () => {
    const posted: unknown[] = [];
    const seen: unknown[] = [];
    const inbound: ((message: unknown) => void)[] = [];
    const proc = {
      send(message: unknown) {
        // `process.send` reads `this` for the IPC channel; a detached reference
        // throws.
        expect(this).toBe(proc);
        posted.push(message);
      },
      on: (_event: 'message', listener: (message: unknown) => void) => {
        inbound.push(listener);
      },
    };
    const channel = resolveParentChannel(proc);
    channel.onMessage((message) => seen.push(message));
    channel.post({ kind: 'mlx:ready' });
    for (const listener of inbound) listener({ kind: 'mlx:request' });

    expect(posted).toEqual([{ kind: 'mlx:ready' }]);
    expect(seen).toEqual([{ kind: 'mlx:request' }]);
  });

  it('refuses to run with no parent at all', () => {
    // Running the entry by hand is a legitimate way to reproduce a wedge in a
    // terminal, and it must say so rather than throw a TypeError on the first
    // postMessage.
    expect(() => resolveParentChannel({ on: () => {} })).toThrow(NoParentChannelError);
  });
});

const tick = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));
