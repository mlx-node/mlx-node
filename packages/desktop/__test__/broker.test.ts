/**
 * MAIN's broker: one channel per handshake, and what happens when CONTROL PANEL dies.
 *
 * Crash isolation is only real if the crash is handled. "CONTROL PANEL is a separate
 * process so a dashboard bug cannot take down the tray" is worth nothing if the
 * result is a tray that survives next to a window that is permanently blank —
 * so the restart and the re-broker are as much a part of the design as the
 * split itself, and they are what most of this file is about.
 */

import { mkdtempSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { PING_INTERVAL_MS } from '../src/control-panel/liveness.js';
import { CONTROL_PANEL_BROKER_KILL_GRACE_MS } from '../src/control-panel/shutdown-timings.js';
import { EXIT_STARTUP_FAILED } from '../src/inference/exit-codes.js';
import {
  createControlPanelBroker,
  type ControlPanelChild,
  type BrokerDeps,
  type BrokerEvent,
  type BrokerPort,
} from '../src/main/broker.js';
import { createPanelLedger, readAppEnds, type LedgerRow } from '../src/panel-ledger.js';

/** A renderer, as far as the broker is concerned. */
interface Renderer {
  id: string;
  alive: boolean;
  /** Ports it was handed, in order. */
  received: TestPort[];
  refuse?: boolean;
}

interface TestPort extends BrokerPort {
  id: string;
  closed: boolean;
  /** Generation MAIN sent beside this renderer-facing port. */
  generation: number | null;
  /** Set once it has been transferred; a transferred port is no longer ours. */
  handedTo: 'control-panel' | 'renderer' | null;
}

interface Spawn {
  child: ControlPanelChild;
  killed: number;
  forceKilled: number;
  ports: TestPort[];
  /** Pings MAIN sent, by number. */
  pings: number[];
  exit(code: number | null): void;
  /** A message on the parent port, from the child. */
  say(message: unknown): void;
  /** Make the next `sendPort` throw, as a dead channel does. */
  refuse: boolean;
}

interface Harness {
  deps: BrokerDeps<Renderer>;
  spawns: Spawn[];
  events: BrokerEvent[];
  ports: TestPort[];
  /** Pending `delay` callbacks, oldest first. */
  timers: { fn: () => void; ms: number; cancelled: boolean }[];
  /** Run every pending timer whose delay is <= `ms`. */
  fire(): void;
  /** Move the clock by `ms`, then run the pending ping ticks. */
  advance(ms: number): void;
  /** Kills and ledger rows, in the order the broker made them. */
  order: string[];
  /** Ledger rows, in order, with the row ids the broker got back. */
  rows: Array<LedgerRow & { id: string }>;
  clock: { now: number };
}

function harness(
  options: { spawnThrows?: boolean | 'ENOENT'; record?: (row: LedgerRow) => string | null } = {},
): Harness {
  const spawns: Spawn[] = [];
  const events: BrokerEvent[] = [];
  const ports: TestPort[] = [];
  const timers: { fn: () => void; ms: number; cancelled: boolean }[] = [];
  const rows: Array<LedgerRow & { id: string }> = [];
  const order: string[] = [];
  const clock = { now: 1_000 };
  let nextPort = 0;

  const makePort = (): TestPort => {
    const port: TestPort = {
      id: `p${(nextPort += 1)}`,
      closed: false,
      generation: null,
      handedTo: null,
      close(): void {
        port.closed = true;
      },
    };
    ports.push(port);
    return port;
  };

  const h: Harness = {
    spawns,
    events,
    ports,
    timers,
    clock,
    rows,
    order,
    fire(): void {
      for (const timer of timers.splice(0)) {
        if (!timer.cancelled) timer.fn();
      }
    },
    advance(ms: number): void {
      clock.now += ms;
      const due = timers.filter((timer) => timer.ms === PING_INTERVAL_MS && !timer.cancelled);
      for (const timer of due) timers.splice(timers.indexOf(timer), 1);
      for (const timer of due) timer.fn();
    },
    deps: {
      spawn(handlers): ControlPanelChild {
        if (options.spawnThrows === true) throw new Error('fork failed');
        if (options.spawnThrows === 'ENOENT') {
          throw Object.assign(new Error('spawn ENOENT'), { code: 'ENOENT' });
        }
        const record: Spawn = {
          killed: 0,
          forceKilled: 0,
          ports: [],
          pings: [],
          refuse: false,
          exit: (code) => handlers.onExit(code),
          say: (message) => handlers.onMessage(message),
          child: {
            sendPort(port: BrokerPort): void {
              if (record.refuse) throw new Error('channel closed');
              (port as TestPort).handedTo = 'control-panel';
              record.ports.push(port as TestPort);
            },
            kill(): void {
              record.killed += 1;
              order.push('kill');
            },
            forceKill(): void {
              record.forceKilled += 1;
            },
            ping(n: number): void {
              record.pings.push(n);
            },
            pid: () => 4_242,
          },
        };
        spawns.push(record);
        return record.child;
      },
      createChannel: () => ({ port1: makePort(), port2: makePort() }),
      sendToRenderer(target: Renderer, port: BrokerPort, generation: number): void {
        if (target.refuse === true) throw new Error('WebContents destroyed');
        (port as TestPort).handedTo = 'renderer';
        (port as TestPort).generation = generation;
        target.received.push(port as TestPort);
      },
      isRendererAlive: (target) => target.alive,
      report: (event) => events.push(event),
      record(row: LedgerRow): string | null {
        order.push(`row:${row.k}${'what' in row ? `:${row.what}` : ''}`);
        if (options.record !== undefined) return options.record(row);
        const id = `1.1000#${String(rows.length + 1)}`;
        rows.push({ ...row, id });
        return id;
      },
      now: () => clock.now,
      delay(fn, ms): () => void {
        const timer = { fn, ms, cancelled: false };
        timers.push(timer);
        return () => {
          timer.cancelled = true;
        };
      },
    },
  };
  return h;
}

const renderer = (id = 'win'): Renderer => ({ id, alive: true, received: [] });

describe('brokering', () => {
  it('spawns CONTROL PANEL lazily, on the first handshake', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps);
    expect(h.spawns).toHaveLength(0);
    // A menubar app's resting state is a tray icon. Forking a process that owns a
    // SQLite worker thread and a 30 s rescan timer before anyone has opened the
    // window is a cost paid by every user who never opens it.
    broker.attach(renderer());
    expect(h.spawns).toHaveLength(1);
  });

  it('hands the two ends of one channel to CONTROL PANEL and the renderer', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps);
    const win = renderer();
    broker.attach(win);

    expect(h.ports).toHaveLength(2);
    expect(h.spawns[0].ports).toEqual([h.ports[0]]);
    expect(win.received).toEqual([h.ports[1]]);
    // MAIN keeps neither end. It is not on the data path — a relay would put
    // every dashboard call through the process that owns the tray.
    expect(h.ports.every((port) => port.handedTo !== null)).toBe(true);
    expect(h.events).toEqual([{ type: 'brokered', generation: 1 }]);
  });

  it('mints a FRESH channel for every handshake', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps);
    const win = renderer();

    broker.attach(win);
    broker.attach(win); // the reload

    // A transferred port is consumed on delivery: it cannot be re-sent, only
    // replaced. Re-using the first channel would hand the reloaded page a port
    // whose peer end is already gone.
    expect(h.ports).toHaveLength(4);
    expect(win.received.map((p) => p.id)).toEqual(['p2', 'p4']);
    expect(h.spawns[0].ports.map((p) => p.id)).toEqual(['p1', 'p3']);
    expect(new Set(win.received.map((p) => p.id)).size).toBe(2);
    expect(broker.generation()).toBe(2);
    // One process across both. A reload must not restart the runtime.
    expect(h.spawns).toHaveLength(1);
  });

  it('reuses the running child across handshakes', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps);
    broker.attach(renderer('a'));
    broker.attach(renderer('b'));
    expect(h.spawns).toHaveLength(1);
  });
});

describe('a handshake that cannot be completed', () => {
  it('closes both ends and kills the child when CONTROL PANEL refuses the port', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps);
    h.spawns.length = 0;
    broker.attach(renderer());
    // Second attempt, with the channel gone underneath us.
    h.spawns[0].refuse = true;
    const win = renderer('b');
    broker.attach(win);

    const [, , port1, port2] = h.ports;
    expect([port1.closed, port2.closed]).toEqual([true, true]);
    expect(win.received).toEqual([]);
    // It could not take a port, so it will not answer one. Killing it routes the
    // recovery through the one restart budget instead of inventing a second path.
    expect(h.spawns[0].killed).toBe(1);
    expect(h.events.at(-1)).toMatchObject({ type: 'broker-failed', generation: 2 });
  });

  it('closes only the renderer’s end when the renderer refuses it', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps);
    const win = renderer();
    win.refuse = true;
    broker.attach(win);

    const [port1, port2] = h.ports;
    // port1 is already transferred and no longer ours to touch; closing port2 is
    // what tells CONTROL PANEL, through its own `close` event, to release the
    // subscriptions this handshake opened.
    expect(port1.closed).toBe(false);
    expect(port1.handedTo).toBe('control-panel');
    expect(port2.closed).toBe(true);
    expect(h.spawns[0].killed).toBe(0);
    expect(h.events.at(-1)).toMatchObject({ type: 'broker-failed' });
  });

  it('sends to CONTROL PANEL first, so a dead renderer never leaves a live port to nothing', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps);
    const win = renderer();
    win.refuse = true;
    broker.attach(win);

    // The order is the decision. Renderer-first would leave the page holding a
    // port whose peer never arrived: every call it makes hangs until its own
    // deadline, with nothing on either side able to say why.
    expect(h.spawns[0].ports).toHaveLength(1);
    expect(win.received).toEqual([]);
  });

  it('never throws out of attach, even when the fork itself fails', () => {
    const h = harness({ spawnThrows: true });
    const broker = createControlPanelBroker(h.deps, { restart: { backoffBaseMs: 1, probationMs: 1 } });
    const win = renderer();

    // `attach` runs inside an `ipcMain` handler. A throw there is an uncaught
    // exception in MAIN, which takes the tray and the INFERENCE supervisor with
    // it — the app would die of the failure it exists to survive.
    expect(() => broker.attach(win)).not.toThrow();

    expect(win.received).toEqual([]);
    expect(h.events.at(-1)).toMatchObject({ type: 'broker-failed', reason: expect.stringContaining('failed to fork') });
    // A fork that fails produces no `exit`, so the budget has to be driven from
    // the failure itself or nothing would ever retry — or give up.
    expect(h.events.some((e) => e.type === 'control-panel-exited')).toBe(true);
    h.fire();
    expect(h.events.some((e) => e.type === 'control-panel-gave-up')).toBe(true);
  });

  // Splash `EngineLaunchFailed`: a missing entry stays missing.
  it('stops at once when the fork finds nothing to run', () => {
    const h = harness({ spawnThrows: 'ENOENT' });
    const broker = createControlPanelBroker(h.deps);
    broker.attach(renderer());

    expect(h.events.filter((e) => e.type === 'control-panel-gave-up')).toEqual([
      {
        type: 'control-panel-gave-up',
        message: expect.stringContaining('control panel failed to fork: spawn ENOENT'),
      },
    ]);
    // No restart is scheduled.
    expect(h.timers).toEqual([]);
  });
});

describe('when CONTROL PANEL dies', () => {
  it('reports it, restarts, and re-brokers to the window that is still open', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps, { restart: { backoffBaseMs: 10 } });
    const win = renderer();
    broker.attach(win);
    expect(win.received).toHaveLength(1);

    h.clock.now += 60_000;
    h.spawns[0].exit(1);

    expect(h.events.at(-1)).toMatchObject({ type: 'control-panel-exited', code: 1, streak: 1 });
    expect(broker.running()).toBe(false);

    h.fire();

    // The renderer's port died with the old process. It is holding a channel to
    // nothing until MAIN hands it a new one, and a dashboard that stays blank
    // until the user thinks to reload is the failure the restart is for.
    expect(h.spawns).toHaveLength(2);
    expect(win.received).toHaveLength(2);
    expect(h.spawns[1].ports).toHaveLength(1);
  });

  it('does not re-broker to a window that has since been destroyed', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps, { restart: { backoffBaseMs: 10 } });
    const win = renderer();
    broker.attach(win);

    win.alive = false;
    h.spawns[0].exit(1);
    h.fire();

    // Still restarted — the next window open must not pay for a spawn — but the
    // dead renderer is not posted to. `webContents.postMessage` on a destroyed
    // contents throws.
    expect(h.spawns).toHaveLength(2);
    expect(win.received).toHaveLength(1);
  });

  it('gives up at the short failure whose backoff would reach the cap', () => {
    const h = harness();
    // Splash CRASH_LOOP_FAILURES = 2 + ceil(log2(probation / base)) = 3.
    const broker = createControlPanelBroker(h.deps, { restart: { backoffBaseMs: 1, probationMs: 2 } });
    broker.attach(renderer());

    for (let crash = 0; crash < 3; crash += 1) {
      h.spawns.at(-1)?.exit(1);
      h.fire();
    }

    expect(h.events.filter((e) => e.type === 'control-panel-gave-up')).toEqual([
      {
        type: 'control-panel-gave-up',
        message: expect.stringContaining('it failed 3 times in a row, each within 0.002 s of starting (exited 1)'),
      },
    ]);
    expect(h.spawns).toHaveLength(3);
    expect(broker.running()).toBe(false);
  });

  // `EXIT_STARTUP_FAILED`: CONTROL PANEL not forked as a utilityProcess. A
  // relaunch meets it again (Splash `permanent()`).
  it('reports a permanent give-up to a later handshake instead of silently doing nothing', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps);
    broker.attach(renderer());
    h.spawns[0].exit(EXIT_STARTUP_FAILED);
    h.fire();
    expect(h.spawns).toHaveLength(1);

    const win = renderer('later');
    broker.attach(win);

    expect(win.received).toEqual([]);
    expect(h.events.at(-1)).toMatchObject({
      type: 'broker-failed',
      reason: expect.stringContaining('stopped restarting the control panel: exited 78'),
    });
  });

  it('starts the count anew after a full probation of service', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps, {
      restart: { backoffBaseMs: 1, probationMs: 10_000 },
    });
    broker.attach(renderer());

    h.spawns[0].exit(1);
    h.fire();
    h.spawns[1].exit(1);
    h.fire();
    expect(h.events.at(-2)).toMatchObject({ streak: 2 });

    // This one ran long enough to prove it can. Measured from the fork, which is
    // all the broker knows — CONTROL PANEL has no readiness handshake to measure from.
    h.clock.now += 20_000;
    h.spawns[2].exit(1);
    expect(h.events.at(-1)).toMatchObject({ type: 'control-panel-exited', streak: 1 });
  });

  it('treats exit 0 as a crash', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps, { restart: { backoffBaseMs: 1 } });
    broker.attach(renderer());
    h.spawns[0].exit(0);

    // Nothing asks CONTROL PANEL to stop except `dispose()`, so an exit of 0 means it
    // fell off the end of its entry — not that everything is fine.
    expect(h.events.at(-1)).toMatchObject({ type: 'control-panel-exited', code: 0, streak: 1 });
    h.fire();
    expect(h.spawns).toHaveLength(2);
  });
});

describe('dispose', () => {
  it('uses a default grace longer than the CONTROL PANEL process shutdown cap', async () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps);
    broker.attach(renderer());

    const done = broker.dispose();
    expect(h.timers.filter((timer) => !timer.cancelled).map((timer) => timer.ms)).toEqual([
      CONTROL_PANEL_BROKER_KILL_GRACE_MS,
      CONTROL_PANEL_BROKER_KILL_GRACE_MS * 2,
    ]);

    h.spawns[0].exit(0);
    await done;
  });

  it('SIGTERMs and waits for the child to go', async () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps);
    broker.attach(renderer());

    const done = broker.dispose();
    expect(h.spawns[0].killed).toBe(1);
    // Waiting is the point: CONTROL PANEL's shutdown drains in-flight downloads so a
    // partial multi-GB `.staging` tree is not orphaned.
    let settled = false;
    void done.then(() => (settled = true));
    await tick();
    expect(settled).toBe(false);

    h.spawns[0].exit(0);
    await done;
    expect(h.spawns[0].forceKilled).toBe(0);
  });

  it('escalates to SIGKILL after the grace period', async () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps);
    broker.attach(renderer());

    const done = broker.dispose();
    h.fire(); // grace expiry, then the cap
    expect(h.spawns[0].forceKilled).toBe(1);
    await done;
  });

  it('resolves even if the exit event is never delivered', async () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps);
    broker.attach(renderer());
    const done = broker.dispose();
    h.fire();
    // A quit that hangs is the worst failure a menubar app has: with no Dock
    // icon the user's only remaining option is Force Quit.
    await done;
  });

  it('does not respawn after disposal', async () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps, { restart: { backoffBaseMs: 1 } });
    broker.attach(renderer());
    const done = broker.dispose();
    h.spawns[0].exit(0);
    await done;
    h.fire();

    expect(h.spawns).toHaveLength(1);
    expect(h.events.some((e) => e.type === 'control-panel-exited')).toBe(false);
  });

  it('is idempotent and returns immediately with no child', async () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps);
    await broker.dispose();
    await broker.dispose();
    expect(h.spawns).toHaveLength(0);
  });

  it('ignores a handshake that arrives after disposal', async () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps);
    await broker.dispose();
    const win = renderer();
    broker.attach(win);
    expect(h.spawns).toHaveLength(0);
    expect(win.received).toEqual([]);
  });
});

const tick = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

const CONV = '019a1f2e-7b3c-7d4e-8f50-61728394a5b6';
const pong = (open: number, convs: string[] = []): unknown => ({ type: 'mlx-pong', n: 1, open, convs, slow: null });

/** One ping tick on time, `times` over. */
function ticks(h: Harness, times: number): void {
  for (let i = 0; i < times; i += 1) h.advance(PING_INTERVAL_MS);
}

/** An attached child that has spoken once (open turns per `open`); the next tick sees that message. */
function spoken(
  open: number,
  options: Parameters<typeof harness>[0] = {},
): { h: Harness; broker: ReturnType<typeof createControlPanelBroker<Renderer>> } {
  const h = harness(options);
  const broker = createControlPanelBroker(h.deps, { restart: { backoffBaseMs: 10 } });
  broker.attach(renderer());
  h.clock.now += PING_INTERVAL_MS;
  h.spawns[0].say(pong(open, open > 0 ? [CONV] : []));
  h.advance(0);
  return { h, broker };
}

describe('liveness: MAIN is the only judge', () => {
  it('pings every 5 s on the parent port while the child lives', () => {
    const h = harness();
    createControlPanelBroker(h.deps).attach(renderer());
    ticks(h, 3);
    expect(h.spawns[0].pings).toEqual([1, 2, 3]);
  });

  it('idle child, no message: no kill after 5 silent on-time ticks, kill at the 6th', () => {
    const { h } = spoken(0);
    ticks(h, 5);
    expect(h.spawns[0].killed).toBe(0);
    ticks(h, 1);
    expect(h.spawns[0].killed).toBe(1);
    expect(h.rows.find((r) => r.k === 'intent')).toMatchObject({
      what: 'panel.stop',
      cause: 'main.panel_no_pong',
      stage: 'idle',
      ageMs: 30_000,
    });
  });

  it('a turn in the last pong: no kill at silent tick 23, kill at 24', () => {
    const { h } = spoken(1);
    ticks(h, 23);
    expect(h.spawns[0].killed).toBe(0);
    ticks(h, 1);
    expect(h.spawns[0].killed).toBe(1);
    expect(h.rows.find((r) => r.k === 'intent')).toMatchObject({
      stage: 'turn',
      ageMs: 120_000,
      open: 1,
      convs: [CONV],
    });
  });

  it('a clock jump of 60 s between ticks is not counted: the kill waits for the 6th on-time tick', () => {
    const { h } = spoken(0);
    ticks(h, 5);
    h.advance(PING_INTERVAL_MS + 60_000);
    expect(h.spawns[0].killed).toBe(0);
    expect(h.rows.some((r) => r.k === 'intent')).toBe(false);
    ticks(h, 1);
    expect(h.spawns[0].killed).toBe(1);
  });

  it('a late tick now and then keeps the on-time silence: a dead child is still restarted', () => {
    const { h } = spoken(0);
    for (let round = 0; round < 3; round += 1) {
      ticks(h, 2);
      h.advance(PING_INTERVAL_MS + 2_000);
    }
    expect(h.spawns[0].killed).toBe(1);
    expect(h.rows.find((r) => r.k === 'intent')).toMatchObject({ cause: 'main.panel_no_pong', ageMs: 30_000 });
  });

  it('a message from a child that already exited changes nothing', () => {
    const { h } = spoken(0);
    const old = h.spawns[0];
    old.exit(1);
    h.fire();
    const rows = h.rows.length;
    old.say({ type: 'mlx-worker', reason: 'exit', route: null, ageMs: null, attempt: 1, open: 0 });
    old.say(pong(1, [CONV]));
    expect(h.rows).toHaveLength(rows);
    ticks(h, 6);
    expect(h.spawns).toHaveLength(2);
    expect(h.spawns[1].killed).toBe(0);
  });

  it('a worker note with no pong is life too: it restarts the count', () => {
    const { h } = spoken(0);
    ticks(h, 5);
    h.spawns[0].say({
      type: 'mlx-worker',
      reason: 'no-ack',
      route: 'GET /api/sessions/:id',
      ageMs: 64_000,
      attempt: 1,
      open: 0,
    });
    ticks(h, 6);
    expect(h.spawns[0].killed).toBe(0);
    expect(h.rows.find((r) => r.k === 'intent')).toMatchObject({
      what: 'worker.restart',
      cause: 'panel.worker_down',
      reason: 'no-ack',
      route: 'GET /api/sessions/:id',
    });
  });

  it('a child that never speaks gets the long bound: no kill before tick 24', () => {
    const h = harness();
    createControlPanelBroker(h.deps).attach(renderer());
    ticks(h, 23);
    expect(h.spawns[0].killed).toBe(0);
    ticks(h, 1);
    expect(h.spawns[0].killed).toBe(1);
    expect(h.rows.find((r) => r.k === 'intent')).toMatchObject({ stage: 'start' });
  });

  it('records a slow loop at 2 silent ticks and closes it on the next message', () => {
    const { h } = spoken(0);
    ticks(h, 2);
    expect(h.rows.filter((r) => r.k === 'slow')).toMatchObject([{ op: 'loop', state: 'open', ageMs: 10_000 }]);
    h.clock.now += 1_000;
    h.spawns[0].say(pong(0));
    expect(h.rows.filter((r) => r.k === 'slow')).toMatchObject([
      { op: 'loop', state: 'open' },
      { op: 'loop', state: 'done', ageMs: 11_000 },
    ]);
  });

  it('records a page call open 30 s once, by route pattern', () => {
    const { h } = spoken(0);
    const slow = {
      call: 7,
      route: 'GET /api/chat/conversations',
      thread: 'worker',
      ageMs: 30_000,
      wq: 3,
      wbusyMs: 31_000,
      wroute: 'GET /api/sessions/:id',
    };
    h.spawns[0].say({ type: 'mlx-pong', n: 2, open: 0, convs: [], slow });
    h.spawns[0].say({ type: 'mlx-pong', n: 3, open: 0, convs: [], slow: { ...slow, ageMs: 35_000 } });
    expect(h.rows.filter((r) => r.k === 'slow')).toEqual([
      expect.objectContaining({
        op: 'rpc',
        state: 'open',
        route: 'GET /api/chat/conversations',
        thread: 'worker',
        wq: 3,
      }),
    ]);
  });
});

describe('every stop is written before the signal, and every exit after', () => {
  it('intent, then SIGTERM; the exit row names the intent and the time since', () => {
    const { h } = spoken(1);
    ticks(h, 24);
    expect(h.order.filter((o) => o === 'kill' || o.startsWith('row:intent'))).toEqual([
      'row:intent:panel.stop',
      'kill',
    ]);
    const intent = h.rows.find((r) => r.k === 'intent')!;
    h.clock.now += 1_500;
    h.spawns[0].say({ type: 'mlx-closing', convs: [CONV] });
    h.spawns[0].exit(0);
    expect(h.rows.find((r) => r.k === 'child.exit')).toMatchObject({
      role: 'panel',
      asked: true,
      cause: 'main.panel_no_pong',
      by: intent.id,
      sinceIntentMs: 1_500,
      code: 0,
      open: 1,
      convs: [CONV],
      cancelled: [CONV],
      pid: 4_242,
    });
    expect(h.rows.filter((r) => r.k === 'intent' || r.k === 'child.exit').map((r) => r.k)).toEqual([
      'intent',
      'child.exit',
    ]);
  });

  it('escalates with its own intent row', () => {
    const h = harness();
    createControlPanelBroker(h.deps, { killGraceMs: 25 }).attach(renderer());
    ticks(h, 24);
    const grace = h.timers.find((t) => t.ms === 25)!;
    grace.fn();
    expect(h.spawns[0].forceKilled).toBe(1);
    expect(h.rows.filter((r) => r.k === 'intent').map((r) => ('what' in r ? r.what : null))).toEqual([
      'panel.stop',
      'panel.kill',
    ]);
  });

  it('an unasked exit 0 is a signal; any other code is a crash', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps, { restart: { backoffBaseMs: 1 } });
    broker.attach(renderer());
    h.spawns[0].exit(0);
    h.fire();
    h.spawns[1].exit(1);
    expect(h.rows.filter((r) => r.k === 'child.exit').map((r) => ('cause' in r ? [r.cause, r.asked] : null))).toEqual([
      ['os.signal_term', false],
      ['panel.exit_unasked', false],
    ]);
  });

  it('dispose writes the quit intent before the kill', async () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps);
    broker.attach(renderer());
    const done = broker.dispose();
    expect(h.order.filter((o) => o === 'kill' || o.startsWith('row:intent'))).toEqual([
      'row:intent:panel.stop',
      'kill',
    ]);
    h.spawns[0].exit(0);
    await done;
    expect(h.rows.find((r) => r.k === 'child.exit')).toMatchObject({ cause: 'user.app_quit', asked: true });
  });

  it('a quit during the kill grace keeps the no-ping cause', async () => {
    const { h, broker } = spoken(1);
    ticks(h, 24);
    const intent = h.rows.find((r) => r.k === 'intent')!;
    h.clock.now += 3_000;
    const done = broker.dispose();
    h.spawns[0].exit(0);
    await done;
    expect(h.rows.filter((r) => r.k === 'intent')).toHaveLength(1);
    expect(h.rows.find((r) => r.k === 'child.exit')).toMatchObject({
      cause: 'main.panel_no_pong',
      by: intent.id,
      sinceIntentMs: 3_000,
    });
  });

  it('a refused port stops the child once, with its own cause, and escalates; no tick overwrites it', () => {
    const { h, broker } = spoken(1);
    h.spawns[0].refuse = true;
    broker.attach(renderer('b'));
    ticks(h, 30);
    expect(h.spawns[0].killed).toBe(1);
    expect(h.rows.filter((r) => r.k === 'intent').map((r) => ('cause' in r ? r.cause : null))).toEqual([
      'main.port_refused',
    ]);
    h.timers.find((t) => t.ms === CONTROL_PANEL_BROKER_KILL_GRACE_MS && !t.cancelled)!.fn();
    expect(h.spawns[0].forceKilled).toBe(1);
    h.spawns[0].exit(null);
    expect(h.rows.find((r) => r.k === 'child.exit')).toMatchObject({ cause: 'main.port_refused', asked: true });
  });

  it('a ledger that throws changes nothing', () => {
    const h = harness({
      record: () => {
        throw new Error('EACCES');
      },
    });
    createControlPanelBroker(h.deps).attach(renderer());
    ticks(h, 24);
    expect(h.spawns[0].killed).toBe(1);
  });
});

describe('from the record to the next panel', () => {
  it('a real ledger and readAppEnds give the no-ping end with its silence and turns', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-ledger-broker-'));
    try {
      let clockRef: { now: number } = { now: 0 };
      const ledger = createPanelLedger({
        dir,
        pid: 77,
        ppid: 1,
        ver: '1.0.0',
        startMs: 1_000,
        now: () => clockRef.now,
        mono: () => clockRef.now - 1_000,
      });
      const { h } = spoken(1, { record: (row) => ledger.record(row) });
      clockRef = h.clock;
      ticks(h, 24);
      h.spawns[0].say({ type: 'mlx-closing', convs: [CONV] });
      h.clock.now += 2_000;
      h.spawns[0].exit(0);
      ledger.close();
      expect(readdirSync(dir)).toEqual(['main-1000-77.1.jsonl']);
      const kinds = readFileSync(join(dir, 'main-1000-77.1.jsonl'), 'utf8')
        .split('\n')
        .filter((l) => l !== '')
        .map((l) => (JSON.parse(l) as { k: string }).k);
      expect(kinds).toEqual(['proc', 'child.spawn', 'slow', 'intent', 'slow', 'child.exit']);
      const ends = await readAppEnds(dir);
      expect(ends).toEqual([
        {
          trigger: 'no-ping',
          decidedAt: h.clock.now - 2_000,
          exitAt: h.clock.now,
          silentMs: 120_000,
          code: 0,
          convs: [CONV],
          cancelled: [CONV],
        },
      ]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
