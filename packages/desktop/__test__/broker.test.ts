/**
 * MAIN's broker: one channel per handshake, and what happens when CONTROL PANEL dies.
 *
 * Crash isolation is only real if the crash is handled. "CONTROL PANEL is a separate
 * process so a dashboard bug cannot take down the tray" is worth nothing if the
 * result is a tray that survives next to a window that is permanently blank —
 * so the restart and the re-broker are as much a part of the design as the
 * split itself, and they are what most of this file is about.
 */

import { describe, expect, it } from 'vite-plus/test';

import { CONTROL_PANEL_BROKER_KILL_GRACE_MS } from '../src/control-panel/shutdown-timings.js';
import {
  createControlPanelBroker,
  type ControlPanelChild,
  type BrokerDeps,
  type BrokerEvent,
  type BrokerPort,
} from '../src/main/broker.js';

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
  exit(code: number | null): void;
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
  clock: { now: number };
}

function harness(options: { spawnThrows?: boolean } = {}): Harness {
  const spawns: Spawn[] = [];
  const events: BrokerEvent[] = [];
  const ports: TestPort[] = [];
  const timers: { fn: () => void; ms: number; cancelled: boolean }[] = [];
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
    fire(): void {
      for (const timer of timers.splice(0)) {
        if (!timer.cancelled) timer.fn();
      }
    },
    deps: {
      spawn(handlers): ControlPanelChild {
        if (options.spawnThrows === true) throw new Error('fork failed');
        const record: Spawn = {
          killed: 0,
          forceKilled: 0,
          ports: [],
          refuse: false,
          exit: (code) => handlers.onExit(code),
          child: {
            sendPort(port: BrokerPort): void {
              if (record.refuse) throw new Error('channel closed');
              (port as TestPort).handedTo = 'control-panel';
              record.ports.push(port as TestPort);
            },
            kill(): void {
              record.killed += 1;
            },
            forceKill(): void {
              record.forceKilled += 1;
            },
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
    const broker = createControlPanelBroker(h.deps, { restart: { maxConsecutiveCrashes: 2, baseDelayMs: 1 } });
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
});

describe('when CONTROL PANEL dies', () => {
  it('kills, escalates, restarts, and re-brokers a child whose transport wedged', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps, { restart: { baseDelayMs: 10 }, killGraceMs: 25 });
    const win = renderer();
    broker.attach(win);
    const generation = broker.generation();

    broker.recover(win, generation);
    broker.recover(win, generation);
    // Repeated reports from calls sharing one bad transport are coalesced.
    expect(h.spawns[0].killed).toBe(1);
    expect(h.timers.some((timer) => timer.ms === 25)).toBe(true);

    h.fire();
    expect(h.spawns[0].forceKilled).toBe(1);

    h.spawns[0].exit(null);
    h.fire();
    expect(h.spawns).toHaveLength(2);
    expect(win.received).toHaveLength(2);
    expect(broker.running()).toBe(true);
  });

  it('cancels recovery escalation when SIGTERM succeeds', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps, { restart: { baseDelayMs: 10 }, killGraceMs: 25 });
    const win = renderer();
    broker.attach(win);

    broker.recover(win, broker.generation());
    h.spawns[0].exit(null);
    h.fire();

    expect(h.spawns[0].forceKilled).toBe(0);
    expect(h.spawns).toHaveLength(2);
    expect(win.received).toHaveLength(2);
  });

  it('ignores an unresponsive report from a renderer that is already gone', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps);
    const win = renderer();
    broker.attach(win);
    win.alive = false;

    broker.recover(win, broker.generation());

    expect(h.spawns[0].killed).toBe(0);
  });

  it('ignores a queued recovery report after a replacement generation is brokered', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps, { restart: { baseDelayMs: 10 } });
    const win = renderer();
    broker.attach(win);
    const staleGeneration = broker.generation();
    expect(win.received[0].generation).toBe(staleGeneration);

    // Generation N's report is queued outside MAIN while that child exits and
    // the ordinary restart path brokers N+1.
    h.spawns[0].exit(1);
    h.fire();
    const replacementGeneration = broker.generation();
    expect(replacementGeneration).toBeGreaterThan(staleGeneration);
    expect(win.received.at(-1)?.generation).toBe(replacementGeneration);
    const exitsBeforeStaleDelivery = h.events.filter((event) => event.type === 'control-panel-exited').length;

    // Deliver the old report only now. Resolving it against the current `child`
    // would kill the healthy replacement and consume another crash-budget slot.
    broker.recover(win, staleGeneration);

    expect(h.spawns[1].killed).toBe(0);
    expect(h.spawns[1].forceKilled).toBe(0);
    expect(h.events.filter((event) => event.type === 'control-panel-exited')).toHaveLength(exitsBeforeStaleDelivery);
    expect(broker.running()).toBe(true);
  });

  it('reports it, restarts, and re-brokers to the window that is still open', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps, { restart: { baseDelayMs: 10 } });
    const win = renderer();
    broker.attach(win);
    expect(win.received).toHaveLength(1);

    h.clock.now += 60_000;
    h.spawns[0].exit(1);

    expect(h.events.at(-1)).toMatchObject({ type: 'control-panel-exited', code: 1, consecutiveCrashes: 1 });
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
    const broker = createControlPanelBroker(h.deps, { restart: { baseDelayMs: 10 } });
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

  it('gives up after the budget and stops respawning', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps, { restart: { maxConsecutiveCrashes: 3, baseDelayMs: 1 } });
    broker.attach(renderer());

    for (let crash = 0; crash < 3; crash += 1) {
      h.spawns.at(-1)?.exit(1);
      h.fire();
    }

    expect(h.events.filter((e) => e.type === 'control-panel-gave-up')).toEqual([
      { type: 'control-panel-gave-up', consecutiveCrashes: 3 },
    ]);
    expect(h.spawns).toHaveLength(3);
    expect(broker.running()).toBe(false);
  });

  it('reports the give-up to a later handshake instead of silently doing nothing', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps, { restart: { maxConsecutiveCrashes: 1, baseDelayMs: 1 } });
    broker.attach(renderer());
    h.spawns[0].exit(1);
    h.fire();

    const win = renderer('later');
    broker.attach(win);

    expect(win.received).toEqual([]);
    expect(h.events.at(-1)).toMatchObject({ type: 'broker-failed', reason: expect.stringContaining('gave up') });
  });

  it('earns a fresh budget after staying up', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps, {
      restart: { maxConsecutiveCrashes: 3, baseDelayMs: 1, healthyForMs: 10_000 },
    });
    broker.attach(renderer());

    h.spawns[0].exit(1);
    h.fire();
    h.spawns[1].exit(1);
    h.fire();
    expect(h.events.at(-2)).toMatchObject({ consecutiveCrashes: 2 });

    // This one ran long enough to prove it can. Measured from the fork, which is
    // all the broker knows — CONTROL PANEL has no readiness handshake to measure from.
    h.clock.now += 20_000;
    h.spawns[2].exit(1);
    expect(h.events.at(-1)).toMatchObject({ type: 'control-panel-exited', consecutiveCrashes: 1 });
  });

  it('treats exit 0 as a crash', () => {
    const h = harness();
    const broker = createControlPanelBroker(h.deps, { restart: { baseDelayMs: 1 } });
    broker.attach(renderer());
    h.spawns[0].exit(0);

    // Nothing asks CONTROL PANEL to stop except `dispose()`, so an exit of 0 means it
    // fell off the end of its entry — not that everything is fine.
    expect(h.events.at(-1)).toMatchObject({ type: 'control-panel-exited', code: 0, consecutiveCrashes: 1 });
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
    expect(h.timers.map((timer) => timer.ms)).toEqual([
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
    const broker = createControlPanelBroker(h.deps, { restart: { baseDelayMs: 1 } });
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
