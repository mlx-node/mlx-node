/**
 * The 2026-11-13 22:13 case end to end, on the scripted fake agent: a page call gets no reply for 10 minutes while the
 * Control Panel answers every ping. The call fails alone with `E_TIMEOUT`, MAIN records it and restarts nothing, and
 * the Work turn that runs meanwhile completes and reaches the page.
 *
 * MAIN's clock is injected; the page's deadlines are short and nothing waits on them but their own reply.
 */

import { mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { MessageChannel } from 'node:worker_threads';

import { describe, expect, it } from 'vite-plus/test';

import { fakeService, until } from '../../dashboard/__test__/helpers/chat-fake.js';
import type { ApiResponse } from '../../dashboard/src/api/errors.js';
import type { ChatHostEvent, ConversationEvent } from '../../dashboard/src/chat/types.js';
import { createRpcClient } from '../../dashboard/src/rpc/client.js';
import { createOpenCalls, serveRuntimeOverPort, type RpcRuntime } from '../../dashboard/src/rpc/host.js';
import { bindEventTargetPort } from '../../dashboard/src/rpc/port.js';
import { splitTurns, turnOutcome } from '../../dashboard/ui/src/lib/chat-turns.js';
import { closingOf, pongOf } from '../src/control-panel/lifecycle.js';
import { PING_INTERVAL_MS } from '../src/control-panel/liveness.js';
import { createControlPanelBroker, type BrokerPort } from '../src/main/broker.js';
import { createPanelLedger, readAppEnds, type LedgerRow } from '../src/panel-ledger.js';

const STUCK = '/api/chat/conversations?mode=work';

describe('a page call with no reply for 10 minutes while pings are answered', () => {
  it('fails alone; no recovery; the running Work turn completes and reaches the page', async () => {
    const fake = fakeService('work-permission');
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    const window = fake.service.watch({ scope: 'host' }, () => {});
    const clock = { now: 1_000 };
    const calls = createOpenCalls(() => clock.now);

    const runtime: RpcRuntime = {
      call: (call) =>
        call.method === 'GET' && call.path === STUCK
          ? new Promise<ApiResponse>(() => {})
          : Promise.resolve({ ok: true, status: 200, body: null }),
      subscribe: () => () => {},
      watchLive: () => () => {},
      watchChat: (scope, send) => fake.service.watch(scope, send),
    };
    const { port1, port2 } = new MessageChannel();
    port1.unref();
    port2.unref();
    const dispose = serveRuntimeOverPort(runtime, bindEventTargetPort(port2), { calls });
    const client = createRpcClient(bindEventTargetPort(port1), { timeoutMs: 20, cancellationGraceMs: 20 });

    // MAIN, with the Control Panel's real pong over the same open calls and the same turns.
    const rows: LedgerRow[] = [];
    const timers: Array<{ fn: () => void; ms: number; cancelled: boolean }> = [];
    let kills = 0;
    let pings = 0;
    const broker = createControlPanelBroker<object>({
      spawn: (events) => ({
        sendPort: () => {},
        kill: () => {
          kills += 1;
          // What the panel's teardown does: every running turn is cancelled.
          void fake.service.close();
        },
        forceKill: () => {},
        ping: (n) => {
          pings += 1;
          events.onMessage(
            pongOf(
              n,
              {
                busyConversations: () => fake.service.busyConversations(),
                closingConversations: () => fake.service.closingConversations(),
                workerQueue: () => null,
              },
              calls,
              clock.now,
            ),
          );
        },
        pid: () => 1,
      }),
      createChannel: () => ({ port1: { close: () => {} } as BrokerPort, port2: { close: () => {} } as BrokerPort }),
      sendToRenderer: () => {},
      isRendererAlive: () => true,
      report: () => {},
      record: (row) => {
        rows.push(row);
        return null;
      },
      now: () => clock.now,
      delay: (fn, ms) => {
        const timer = { fn, ms, cancelled: false };
        timers.push(timer);
        return () => {
          timer.cancelled = true;
        };
      },
    });
    broker.attach({});

    try {
      const { conversation } = await fake.service.create({ mode: 'work', text: 'ls', cwd: project });
      const seen: Array<ChatHostEvent | ConversationEvent> = [];
      client.watchChat({ scope: 'conversation', conversationId: conversation.id }, (event) => seen.push(event));
      const card = (): string | null => {
        for (const event of seen) {
          if (event.type === 'interaction') return event.interaction.id;
          if (event.type === 'sync' && event.interactions.length > 0) return event.interactions[0]!.id;
        }
        return null;
      };
      await until(() => card() !== null, 5_000);
      expect(fake.service.busyConversations()).toEqual([conversation.id]);

      const stuck = await client.call({ method: 'GET', path: STUCK });
      expect(stuck).toMatchObject({ ok: false, code: 'E_TIMEOUT', message: 'The app did not answer in 0 s.' });

      // Ten minutes of MAIN ticks, each on time, each answered.
      for (let tick = 0; tick < 120; tick += 1) {
        clock.now += PING_INTERVAL_MS;
        const due = timers.filter((t) => t.ms === PING_INTERVAL_MS && !t.cancelled);
        for (const t of due) timers.splice(timers.indexOf(t), 1);
        for (const t of due) t.fn();
      }
      expect([pings, kills]).toEqual([120, 0]);
      expect(rows.some((row) => row.k === 'intent')).toBe(false);
      expect(rows.filter((row) => row.k === 'slow')).toEqual([
        expect.objectContaining({ op: 'rpc', route: 'GET /api/chat/conversations', thread: 'worker' }),
      ]);

      await fake.service.answer(conversation.id, card()!, { optionId: 'allow-once' });
      const ended = (): string | undefined => {
        for (const event of seen) if (event.type === 'turn' && event.turn === null) return event.outcome;
        return undefined;
      };
      await until(() => ended() !== undefined, 5_000);
      expect(ended()).toBe('completed');
      expect(fake.service.busyConversations()).toEqual([]);
    } finally {
      window();
      client.close();
      dispose();
      const disposed = broker.dispose();
      for (const timer of timers.splice(0)) if (!timer.cancelled) timer.fn();
      await disposed;
      await fake.close();
    }
  }, 30_000);
});

describe('a Control Panel that stops answering while a Work turn runs', () => {
  it('is restarted after 120 s; the record names the trigger and the rebuilt thread says why the turn stopped', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-ledger-turn-'));
    const fake = fakeService('work-permission', { appEnds: () => readAppEnds(dir) });
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    const window = fake.service.watch({ scope: 'host' }, () => {});
    // MAIN's clock ends its 26 ticks at about the real time of the stop, so the agent's own stored times line up.
    const clock = { now: Date.now() - 26 * PING_INTERVAL_MS };
    const ledger = createPanelLedger({
      dir,
      pid: 4_000,
      ppid: 1,
      ver: '1.0.0',
      startMs: clock.now,
      now: () => clock.now,
      mono: () => 0,
    });
    const timers: Array<{ fn: () => void; ms: number; cancelled: boolean }> = [];
    let answering = true;
    let kills = 0;
    const panel = {
      busyConversations: () => fake.service.busyConversations(),
      closingConversations: () => fake.service.closingConversations(),
      workerQueue: () => null,
    };
    const broker = createControlPanelBroker<object>({
      spawn: (events) => ({
        sendPort: () => {},
        kill: () => {
          kills += 1;
          // The panel's teardown: the closing note first, then every running turn is cancelled, then exit 0.
          events.onMessage(closingOf(panel));
          void fake.service.close().then(() => {
            clock.now = Date.now();
            events.onExit(0);
          });
        },
        forceKill: () => {},
        ping: (n) => {
          if (answering) events.onMessage(pongOf(n, panel, createOpenCalls(), clock.now));
        },
        pid: () => 4_001,
      }),
      createChannel: () => ({ port1: { close: () => {} } as BrokerPort, port2: { close: () => {} } as BrokerPort }),
      sendToRenderer: () => {},
      isRendererAlive: () => true,
      report: () => {},
      record: (row) => ledger.record(row),
      now: () => clock.now,
      delay: (fn, ms) => {
        const timer = { fn, ms, cancelled: false };
        timers.push(timer);
        return () => {
          timer.cancelled = true;
        };
      },
    });
    const tick = (): void => {
      clock.now += PING_INTERVAL_MS;
      const due = timers.filter((t) => t.ms === PING_INTERVAL_MS && !t.cancelled);
      for (const t of due) timers.splice(timers.indexOf(t), 1);
      for (const t of due) t.fn();
    };
    broker.attach({});

    try {
      const { conversation } = await fake.service.create({ mode: 'work', text: 'ls', cwd: project });
      const id = conversation.id;
      const seen: ConversationEvent[] = [];
      const off = fake.service.watchConversation(id, (event) => seen.push(event));
      await until(
        () => seen.some((e) => e.type === 'interaction' || (e.type === 'sync' && e.interactions.length > 0)),
        5_000,
      );
      off();

      tick();
      answering = false;
      // This tick still sees the answer to the last ping.
      tick();
      for (let silent = 0; silent < 23; silent += 1) tick();
      expect(kills).toBe(0);
      tick();
      expect(kills).toBe(1);
      await until(() => !broker.running(), 5_000);
      ledger.close();

      const ends = await readAppEnds(dir);
      expect(ends).toEqual([
        expect.objectContaining({ trigger: 'no-ping', silentMs: 120_000, code: 0, convs: [id], cancelled: [id] }),
      ]);

      // The next Control Panel: the same folders, the ledger read at start, the thread rebuilt from the agent's files.
      const next = fake.restart();
      expect((await next.state()).appEnds).toEqual(ends);
      const rebuilt: ConversationEvent[] = [];
      const offNext = next.watchConversation(id, (event) => rebuilt.push(event));
      await until(() => rebuilt.some((e) => e.type === 'sync'), 5_000);
      offNext();
      const sync = rebuilt.find((e) => e.type === 'sync');
      const entries = sync?.type === 'sync' ? sync.entries : [];
      const turns = splitTurns(entries, []);
      const last = turns.at(-1)!;
      const line = turnOutcome(last, {
        last: true,
        running: false,
        live: null,
        mode: 'work',
        app: { ends, conversationId: id, next: null },
      });
      expect(line?.text).toBe(
        'The app restarted its Control Panel because it did not answer for 120 s. The turn was stopped by the app, not by you.',
      );
    } finally {
      window();
      const disposed = broker.dispose();
      for (const timer of timers.splice(0)) if (!timer.cancelled) timer.fn();
      await disposed;
      await fake.close();
      rmSync(dir, { recursive: true, force: true });
    }
  }, 30_000);
});

describe('a signal shortly after the user pressed Stop', () => {
  it('leaves the stopped turn out of the closing note while its cancel reply is still on the way', async () => {
    const fake = fakeService('work-permission');
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    const panel = {
      busyConversations: () => fake.service.busyConversations(),
      closingConversations: () => fake.service.closingConversations(),
      workerQueue: () => null,
    };
    try {
      const { conversation } = await fake.service.create({ mode: 'work', text: 'ls', cwd: project });
      const id = conversation.id;
      const seen: ConversationEvent[] = [];
      const off = fake.service.watchConversation(id, (event) => seen.push(event));
      await until(
        () => seen.some((e) => e.type === 'interaction' || (e.type === 'sync' && e.interactions.length > 0)),
        5_000,
      );
      off();
      expect(closingOf(panel)).toEqual({ type: 'mlx-closing', convs: [id] });
      expect(await fake.service.cancel(id)).toEqual({ cancelled: true });
      expect(fake.service.busyConversations()).toEqual([id]);
      expect(closingOf(panel)).toEqual({ type: 'mlx-closing', convs: [] });
    } finally {
      await fake.close();
    }
  }, 30_000);
});
