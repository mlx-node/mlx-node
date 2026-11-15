/**
 * MAIN's half of the CONTROL PANEL connection: keep the CONTROL PANEL utilityProcess alive, and
 * hand the renderer a port straight to it.
 *
 * ## MAIN brokers, then leaves
 *
 * One `MessageChannelMain` per handshake: `port1` is transferred to the CONTROL PANEL
 * process, `port2` to the renderer. After that MAIN is not on the data path at
 * all — measured earlier, RPC kept flowing while MAIN was blocked for 3 s. A
 * relay would put every dashboard call through the process that owns the tray,
 * which is the one thing `index.ts` exists to keep unblocked.
 *
 * ## Why a channel per handshake, and not one for the app's lifetime
 *
 * A transferred port is consumed on delivery. It cannot be re-sent — only
 * replaced — so a reload, a renderer crash, or a CONTROL PANEL restart all need a fresh
 * channel. `attach` is therefore called by the *renderer* asking (via
 * `CONTROL_PANEL_READY_CHANNEL`), and mints a new one every time.
 *
 * ## Why not `createSupervisor`
 *
 * Its shape is INFERENCE's: a readiness handshake, `/health` polling, a
 * per-generation trace file for class-(b) errors, and an RPC map. CONTROL PANEL has none
 * of those and needs one thing that supervisor's `ChildHandle` deliberately
 * cannot do — transfer a port. The restart arithmetic is shared instead, from
 * `supervisor/state.ts` (Splash `RestartPolicy.after`), so the two agree on
 * what a crash loop is. Service is counted from the fork: CONTROL PANEL has
 * no Ready.
 *
 * ## The broker is the only liveness judge
 *
 * It pings the child every 5 s on the parent port and counts ticks with no
 * message of any kind from it. A tick more than 1 s off its schedule proves
 * MAIN itself did not run on time (a stall, a sleep, a clock jump), so that
 * tick is not counted. The child is restarted after 6 silent on-time ticks with no
 * turn in its last pong, or 24 while a turn runs or before its first message.
 * No page message can stop or restart it. Every stop is written to the ledger
 * before the signal, and every exit gets one `child.exit` row.
 *
 * Electron-free by construction: `MessageChannelMain`, `utilityProcess` and
 * `WebContents` all arrive through {@link BrokerDeps}, so every rule below is
 * reachable from a plain Node test. `control-panel-child.ts` is the Electron half and
 * makes no decisions.
 */

import {
  IDLE_SILENT_TICKS,
  PING_INTERVAL_MS,
  readPanelMessage,
  SLOW_CALL_RECORD_MS,
  SLOW_LOOP_TICKS,
  TICK_LATE_TOLERANCE_MS,
  TURN_SILENT_TICKS,
  type PanelMessage,
  type SlowCall,
} from '../control-panel/liveness.js';
import { CONTROL_PANEL_BROKER_KILL_GRACE_MS } from '../control-panel/shutdown-timings.js';
import { EXIT_STARTUP_FAILED } from '../inference/exit-codes.js';
import type { LedgerRow, StopCause } from '../panel-ledger.js';
import {
  backoffDoublings,
  DEFAULT_RESTART_POLICY,
  FRESH_RESTART_STATE,
  restartAfter,
  stopCause,
  type RestartPolicy,
  type RestartState,
} from './supervisor/state.js';

/** One end of a `MessageChannelMain`, as much of it as the broker touches. */
export interface BrokerPort {
  close(): void;
}

/** The live CONTROL PANEL child. Not an EventEmitter — see `ChildEvents` in `supervisor/types.ts`. */
export interface ControlPanelChild {
  /**
   * Transfer one port to the CONTROL PANEL process. Throws if the channel is already
   * gone, which is the only way the broker learns about a child that died
   * between `spawn` and here.
   */
  sendPort(port: BrokerPort): void;
  /** SIGTERM. */
  kill(): void;
  /** After the grace period. `utilityProcess.kill()` cannot escalate on its own. */
  forceKill(): void;
  /** Post `{type: 'mlx-ping', n}` on the parent port. Throws if the channel is gone. */
  ping(n: number): void;
  /** The OS pid once the process started, else null. */
  pid(): number | null;
}

export type BrokerEvent =
  /** A renderer is now talking to CONTROL PANEL directly. */
  | { type: 'brokered'; generation: number }
  /** The handshake could not be completed. The renderer is left with no port. */
  | { type: 'broker-failed'; generation: number; reason: string }
  | { type: 'control-panel-exited'; code: number | null; streak: number }
  /** Out of restart budget. The Control Panel window stays empty until the app restarts. */
  | { type: 'control-panel-gave-up'; message: string };

/**
 * Everything that touches Electron. `Target` is the renderer handle — a
 * `WebContents` in production, anything at all in a test.
 */
export interface BrokerDeps<Target> {
  /** Fork the CONTROL PANEL entry. `onExit` fires exactly once per child; `onMessage` for each parent-port message. */
  spawn(events: { onExit(code: number | null): void; onMessage(message: unknown): void }): ControlPanelChild;
  createChannel(): { port1: BrokerPort; port2: BrokerPort };
  /**
   * Hand `port` and its broker generation to `target`. Throws if the renderer
   * is gone.
   */
  sendToRenderer(target: Target, port: BrokerPort, generation: number): void;
  /**
   * Is this renderer still there? Read before re-brokering after a restart: a
   * `WebContents` that has been destroyed since it asked must not be handed a
   * port, and `postMessage` on one throws rather than returning.
   */
  isRendererAlive(target: Target): boolean;
  report(event: BrokerEvent): void;
  /** Append one ledger row and return its row id. Never throws; a failed write changes nothing here. */
  record(row: LedgerRow): string | null;
  now(): number;
  /** `setTimeout`, returning its cancel. */
  delay(fn: () => void, ms: number): () => void;
}

export interface ControlPanelBroker<Target> {
  /**
   * The renderer says it is listening. Spawns CONTROL PANEL if it is not up, mints a
   * fresh channel, and hands out both ends.
   *
   * Never throws: a failed handshake is reported and leaves the window empty,
   * which is a visible failure, whereas a throw out of an `ipcMain` handler is
   * an unhandled rejection in MAIN.
   */
  attach(target: Target): void;
  /** How many channels have been minted. 1 after the first successful attach. */
  generation(): number;
  /** Whether a CONTROL PANEL child is currently up. */
  running(): boolean;
  /**
   * SIGTERM the child and resolve when it is gone, escalating after
   * `killGraceMs`. Awaited during quit: CONTROL PANEL owns the download manager, whose
   * shutdown aborts in-flight transfers so a partial multi-GB `.staging` tree is
   * not orphaned. Idempotent.
   */
  dispose(): Promise<void>;
}

export interface BrokerOptions {
  restart?: Partial<RestartPolicy>;
  /** SIGTERM → SIGKILL window during {@link ControlPanelBroker.dispose}. */
  killGraceMs?: number;
}

export function createControlPanelBroker<Target>(
  deps: BrokerDeps<Target>,
  options: BrokerOptions = {},
): ControlPanelBroker<Target> {
  const policy: RestartPolicy = { ...DEFAULT_RESTART_POLICY, ...options.restart };
  backoffDoublings(policy);
  const killGraceMs = options.killGraceMs ?? CONTROL_PANEL_BROKER_KILL_GRACE_MS;

  let child: ControlPanelChild | null = null;
  let generation = 0;
  let disposed = false;
  /** Splash's stop message once restarting stopped. */
  let gaveUp: string | null = null;
  let restartState: RestartState = FRESH_RESTART_STATE;
  /** When the current child was forked: its service, for the restart policy. */
  let spawnedAtMs = 0;
  /** The last fork threw ENOENT: a missing entry stays missing (Splash `EngineLaunchFailed`). */
  let forkMissing = false;
  let cancelRestart: (() => void) | null = null;
  let recovery: { target: ControlPanelChild; cancelEscalation: () => void } | null = null;
  /** What the broker knows of the live child: its liveness count, its last pong, and why it was asked to stop. */
  let watch: Watch | null = null;
  let forks = 0;
  /**
   * The renderer that last asked. Kept so a child that comes back after a crash
   * can be re-brokered without the page having to notice and ask again — its
   * port simply died under it, and a dashboard that stays blank until the user
   * thinks to reload is the failure this whole restart path is for.
   */
  let lastTarget: Target | null = null;
  let exitWaiters: (() => void)[] = [];
  /** Why the last `ensureChild` came back empty, for the one report `attach` makes. */
  let spawnFailure: string | null = null;

  function closeQuietly(port: BrokerPort | null): void {
    if (port === null) return;
    try {
      port.close();
    } catch {
      // Closing a port whose peer is already gone is the normal shutdown race.
    }
  }

  function record(row: LedgerRow): string | null {
    try {
      return deps.record(row);
    } catch {
      return null;
    }
  }

  function heard(w: Watch, value: unknown): void {
    if (watch !== w) return;
    const now = deps.now();
    w.heard = true;
    w.everHeard = true;
    w.lastHeardAt = now;
    if (w.slowLoopSince !== null) {
      record({ k: 'slow', op: 'loop', target: 'panel', state: 'done', ageMs: now - w.slowLoopSince });
      w.slowLoopSince = null;
    }
    const message = readPanelMessage(value);
    if (message !== null) onPanelMessage(w, message);
  }

  function onPanelMessage(w: Watch, message: PanelMessage): void {
    if (message.type === 'mlx-closing') {
      w.cancelled = message.convs;
      return;
    }
    if (message.type === 'mlx-worker') {
      record({
        k: 'intent',
        what: 'worker.restart',
        cause: 'panel.worker_down',
        reason: message.reason,
        ...(message.route === null ? {} : { route: message.route }),
        ...(message.ageMs === null ? {} : { ageMs: message.ageMs }),
        attempt: message.attempt,
        open: message.open,
        ...targetOf(w),
      });
      return;
    }
    w.open = message.open;
    w.convs = message.convs;
    w.slow = message.slow;
    const slow = message.slow;
    if (slow !== null && slow.ageMs >= SLOW_CALL_RECORD_MS && !w.slowCalls.has(slow.call)) {
      w.slowCalls.add(slow.call);
      record({
        k: 'slow',
        op: 'rpc',
        state: 'open',
        ...(slow.route === null ? {} : { route: slow.route }),
        ...(slow.thread === null ? {} : { thread: slow.thread }),
        ageMs: slow.ageMs,
        wq: slow.wq,
        wbusyMs: slow.wbusyMs,
        ...(slow.wroute === null ? {} : { wroute: slow.wroute }),
      });
    }
  }

  function targetOf(w: Watch): { target?: string } {
    return w.spawnRow === null ? {} : { target: w.spawnRow };
  }

  function armTick(w: Watch): void {
    w.expectedAt = deps.now() + PING_INTERVAL_MS;
    w.cancelTick = deps.delay(() => tick(w), PING_INTERVAL_MS);
  }

  function tick(w: Watch): void {
    w.cancelTick = null;
    if (watch !== w || child !== w.child || recovery !== null || w.intent !== null || disposed) return;
    const now = deps.now();
    // A late tick is not counted: a stalled MAIN, a sleep or a clock jump is not the child's silence. Resetting
    // instead would let one late tick per limit keep a dead child forever.
    if (w.heard) w.silentTicks = 0;
    else if (Math.abs(now - w.expectedAt) <= TICK_LATE_TOLERANCE_MS) w.silentTicks += 1;
    w.heard = false;
    const limit = !w.everHeard || w.open > 0 ? TURN_SILENT_TICKS : IDLE_SILENT_TICKS;
    if (w.silentTicks === SLOW_LOOP_TICKS && w.slowLoopSince === null) {
      w.slowLoopSince = w.lastHeardAt;
      record({ k: 'slow', op: 'loop', target: 'panel', state: 'open', ageMs: now - w.lastHeardAt });
    }
    if (w.silentTicks >= limit) {
      stopUnanswering(w, now);
      return;
    }
    w.n += 1;
    try {
      w.child.ping(w.n);
    } catch {
      // A channel that is gone means an exit is on its way; its handler owns what follows.
    }
    armTick(w);
  }

  /** The child said nothing for the whole limit while MAIN ran on time: restart it, and write why first. */
  function stopUnanswering(w: Watch, now: number): void {
    const stage = !w.everHeard ? 'start' : w.open > 0 ? 'turn' : 'idle';
    const row = record({
      k: 'intent',
      what: 'panel.stop',
      cause: 'main.panel_no_pong',
      stage,
      // The judged silence: this many pings in a row, each on time, got no message back.
      ageMs: w.silentTicks * PING_INTERVAL_MS,
      open: w.open,
      convs: w.convs,
      ...(w.slow?.route == null ? {} : { route: w.slow.route }),
      ...(w.slow?.thread == null ? {} : { thread: w.slow.thread }),
      ...targetOf(w),
    });
    stopChild(w, { row, at: now, cause: 'main.panel_no_pong' });
  }

  /** SIGTERM `w`'s child with the intent already recorded, and SIGKILL it after the grace. */
  function stopChild(w: Watch, intent: Intent): void {
    w.intent = intent;
    if (w.cancelTick !== null) {
      w.cancelTick();
      w.cancelTick = null;
    }
    const live = w.child;
    // Install the escalation before SIGTERM: a test double (and some process
    // wrappers) may report exit synchronously from `kill()`.
    const cancelEscalation = deps.delay(() => {
      if (child !== live) return;
      record({
        k: 'intent',
        what: 'panel.kill',
        cause: 'main.kill_escalation',
        ...(intent.row === null ? {} : { by: intent.row }),
        ...targetOf(w),
      });
      try {
        live.forceKill();
      } catch {
        // A raced exit is success; its onExit path owns the restart.
      }
    }, killGraceMs);
    recovery = { target: live, cancelEscalation };
    try {
      live.kill();
    } catch {
      // The scheduled escalation remains armed and makes a second, uncatchable attempt via SIGKILL.
    }
  }

  /** One `child.exit` row per exit of a forked child. */
  function recordExit(w: Watch, code: number | null, crashes: number): void {
    const now = deps.now();
    const intent = w.intent;
    const cause: StopCause = intent?.cause ?? (code === 0 ? 'os.signal_term' : 'panel.exit_unasked');
    const pid = w.child.pid();
    record({
      k: 'child.exit',
      role: 'panel',
      ...(w.spawnRow === null ? {} : { child: w.spawnRow }),
      ...(pid === null ? {} : { pid }),
      ...(code === null ? {} : { code }),
      asked: intent !== null,
      cause,
      ...(intent?.row == null ? {} : { by: intent.row }),
      ...(intent === null ? {} : { sinceIntentMs: now - intent.at }),
      uptimeMs: now - w.spawnedAt,
      crashes,
      open: w.open,
      convs: w.convs,
      ...(w.cancelled === null ? {} : { cancelled: w.cancelled }),
    });
  }

  function ensureChild(): ControlPanelChild | null {
    if (child !== null) return child;
    if (disposed || gaveUp !== null) return null;
    spawnedAtMs = deps.now();
    spawnFailure = null;
    forkMissing = false;
    let w: Watch | null = null;
    try {
      child = deps.spawn({
        onExit: (code) => onExit(code),
        onMessage: (message) => {
          if (w !== null) heard(w, message);
        },
      });
    } catch (error) {
      // A fork that fails outright — a missing `dist/control-panel/index.js`, a bad
      // executable — never produces an `exit`, so nothing else would run the
      // restart budget and the broker would sit with nothing scheduled and
      // nothing to report. Synthesise the exit, exactly as the INFERENCE
      // supervisor does for a transport that reports `error` and no `exit`.
      spawnFailure = `control panel failed to fork: ${describe(error)}`;
      forkMissing = isMissingFile(error);
      onExit(null);
      return null;
    }
    forks += 1;
    const pid = child.pid();
    const spawnRow = record({ k: 'child.spawn', role: 'panel', gen: forks, ...(pid === null ? {} : { pid }) });
    w = {
      child,
      spawnRow,
      spawnedAt: spawnedAtMs,
      heard: false,
      everHeard: false,
      lastHeardAt: spawnedAtMs,
      silentTicks: 0,
      expectedAt: 0,
      cancelTick: null,
      n: 0,
      open: 0,
      convs: [],
      slow: null,
      slowCalls: new Set(),
      slowLoopSince: null,
      cancelled: null,
      intent: null,
    };
    watch = w;
    armTick(w);
    return child;
  }

  function onExit(code: number | null): void {
    const exited = child;
    child = null;
    const exitedWatch = watch !== null && watch.child === exited ? watch : null;
    if (exitedWatch !== null) {
      watch = null;
      exitedWatch.cancelTick?.();
      exitedWatch.cancelTick = null;
    }
    if (recovery?.target === exited) {
      recovery.cancelEscalation();
      recovery = null;
    }
    const waiters = exitWaiters;
    exitWaiters = [];
    for (const waiter of waiters) waiter();
    if (disposed) {
      if (exitedWatch !== null) recordExit(exitedWatch, code, restartState.streak);
      return;
    }

    // Every non-disposal CONTROL PANEL exit consumes the crash budget. That
    // includes a liveness restart: repeatedly wedging is a crash loop even
    // though MAIN delivered the terminating signal. An exit code of 0 still
    // does not make it healthy.
    const permanent = forkMissing || code === EXIT_STARTUP_FAILED;
    const after = restartAfter(policy, restartState, { servedMs: deps.now() - spawnedAtMs, permanent });
    restartState = after.state;
    const decision = after.decision;
    if (exitedWatch !== null) recordExit(exitedWatch, code, restartState.streak);
    deps.report({ type: 'control-panel-exited', code, streak: restartState.streak });

    if (decision.action === 'give-up') {
      const cause = stopCause(policy, decision.permanent, spawnFailure ?? `exited ${code ?? 'with no status'}`);
      gaveUp = `stopped restarting the control panel: ${cause}. Quit and reopen the app after fixing the cause.`;
      deps.report({ type: 'control-panel-gave-up', message: gaveUp });
      return;
    }
    cancelRestart = deps.delay(() => {
      cancelRestart = null;
      if (disposed) return;
      // Re-broker on the way back up. The renderer's old port died with the old
      // process; it is holding a channel to nothing until it gets a new one.
      const target = lastTarget;
      if (target !== null && deps.isRendererAlive(target)) attach(target);
      else ensureChild();
    }, decision.delayMs);
  }

  function attach(target: Target): void {
    if (disposed) return;
    lastTarget = target;
    const live = ensureChild();
    if (live === null) {
      deps.report({
        type: 'broker-failed',
        generation,
        reason: spawnFailure ?? gaveUp ?? 'broker disposed',
      });
      return;
    }

    const { port1, port2 } = deps.createChannel();
    generation += 1;

    // CONTROL PANEL first, renderer second, and the order is the decision.
    //
    // If the renderer post fails after CONTROL PANEL has port1, closing port2 tells CONTROL PANEL
    // through its own `close` event and it releases the subscriptions that
    // handshake opened. The other order leaves the renderer holding a live port
    // to a process that never got its end — every call it makes then hangs until
    // its own deadline, with nothing on either side able to say why.
    try {
      live.sendPort(port1);
    } catch (error) {
      closeQuietly(port1);
      closeQuietly(port2);
      deps.report({ type: 'broker-failed', generation, reason: `control panel refused the port: ${describe(error)}` });
      // It could not take a port, so it is not going to answer one either. Kill
      // it and let `onExit` run the restart budget rather than inventing a
      // second recovery path here.
      const w = watch;
      if (w !== null && w.child === live && w.intent === null) {
        const row = record({
          k: 'intent',
          what: 'panel.stop',
          cause: 'main.port_refused',
          open: w.open,
          convs: w.convs,
          ...targetOf(w),
        });
        stopChild(w, { row, at: deps.now(), cause: 'main.port_refused' });
        return;
      }
      if (w?.intent != null) return;
      try {
        live.kill();
      } catch {
        // Never throw out of an ipcMain handler; the exit path owns what follows.
      }
      return;
    }

    try {
      deps.sendToRenderer(target, port2, generation);
    } catch (error) {
      // port1 is transferred and no longer ours to touch; port2 never left.
      closeQuietly(port2);
      lastTarget = null;
      deps.report({ type: 'broker-failed', generation, reason: `renderer refused the port: ${describe(error)}` });
      return;
    }

    deps.report({ type: 'brokered', generation });
  }

  return {
    attach,
    generation: () => generation,
    running: () => child !== null,
    async dispose(): Promise<void> {
      if (disposed) return;
      disposed = true;
      lastTarget = null;
      if (cancelRestart !== null) {
        cancelRestart();
        cancelRestart = null;
      }
      if (recovery !== null) {
        recovery.cancelEscalation();
        recovery = null;
      }
      const live = child;
      if (live === null) return;
      const w = watch;
      if (w !== null && w.child === live) {
        w.cancelTick?.();
        w.cancelTick = null;
      }
      // A stop already decided keeps its cause: a quit during its kill grace did not cause it.
      if (w !== null && w.child === live && w.intent === null) {
        const row = record({
          k: 'intent',
          what: 'panel.stop',
          cause: 'user.app_quit',
          open: w.open,
          convs: w.convs,
          ...targetOf(w),
        });
        w.intent = { row, at: deps.now(), cause: 'user.app_quit' };
      }

      // Registered BEFORE the kill, so a transport that reports the exit
      // synchronously cannot land between the two.
      const gone = new Promise<void>((resolve) => {
        exitWaiters.push(resolve);
      });
      live.kill();
      const cancelEscalation = deps.delay(() => {
        // `kill()` flips a flag the moment the signal is SENT, so only the exit
        // handler can answer "is it gone yet" — and `child` is nulled there.
        if (child !== null) live.forceKill();
      }, killGraceMs);
      // SIGKILL is not refusable, so an exit always follows. The cap is for a
      // transport that loses the event: app quit must not be held hostage by
      // CONTROL PANEL, and MAIN's own deadline would then have to kill the app outright.
      const cancelCap = deps.delay(() => {
        const waiters = exitWaiters;
        exitWaiters = [];
        for (const waiter of waiters) waiter();
      }, killGraceMs * 2);
      await gone;
      cancelEscalation();
      cancelCap();
    },
  };
}

interface Intent {
  /** The intent row, or null when the ledger did not take it. */
  row: string | null;
  at: number;
  cause: StopCause;
}

interface Watch {
  child: ControlPanelChild;
  spawnRow: string | null;
  spawnedAt: number;
  /** A message arrived since the last tick. */
  heard: boolean;
  everHeard: boolean;
  lastHeardAt: number;
  silentTicks: number;
  expectedAt: number;
  cancelTick: (() => void) | null;
  n: number;
  /** From the last pong: running turns and their conversations. */
  open: number;
  convs: string[];
  slow: SlowCall | null;
  /** Page calls already recorded as slow, by the host's sequence number. */
  slowCalls: Set<number>;
  /** Set while a `slow loop` row is open: when the silence began. */
  slowLoopSince: number | null;
  /** The conversations the dying panel said it cancels; null until it says so. */
  cancelled: string[] | null;
  intent: Intent | null;
}

function isMissingFile(error: unknown): boolean {
  return typeof error === 'object' && error !== null && (error as { code?: unknown }).code === 'ENOENT';
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
