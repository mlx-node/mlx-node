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
 * `supervisor/state.ts`, so the two agree on what a crash loop is.
 *
 * Electron-free by construction: `MessageChannelMain`, `utilityProcess` and
 * `WebContents` all arrive through {@link BrokerDeps}, so every rule below is
 * reachable from a plain Node test. `control-panel-child.ts` is the Electron half and
 * makes no decisions.
 */

import { CONTROL_PANEL_BROKER_KILL_GRACE_MS } from '../control-panel/shutdown-timings.js';
import { DEFAULT_RESTART_POLICY, nextCrashCount, planRestart, type RestartPolicy } from './supervisor/state.js';

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
}

export type BrokerEvent =
  /** A renderer is now talking to CONTROL PANEL directly. */
  | { type: 'brokered'; generation: number }
  /** The handshake could not be completed. The renderer is left with no port. */
  | { type: 'broker-failed'; generation: number; reason: string }
  | { type: 'control-panel-exited'; code: number | null; consecutiveCrashes: number }
  /** Out of restart budget. The Control Panel window stays empty until the app restarts. */
  | { type: 'control-panel-gave-up'; consecutiveCrashes: number };

/**
 * Everything that touches Electron. `Target` is the renderer handle — a
 * `WebContents` in production, anything at all in a test.
 */
export interface BrokerDeps<Target> {
  /** Fork the CONTROL PANEL entry. `onExit` fires exactly once per child. */
  spawn(events: { onExit(code: number | null): void }): ControlPanelChild;
  createChannel(): { port1: BrokerPort; port2: BrokerPort };
  /**
   * Hand `port` and its broker generation to `target`. Throws if the renderer
   * is gone. The generation returns with any recovery report so a delayed
   * report cannot terminate the child serving a newer channel.
   */
  sendToRenderer(target: Target, port: BrokerPort, generation: number): void;
  /**
   * Is this renderer still there? Read before re-brokering after a restart: a
   * `WebContents` that has been destroyed since it asked must not be handed a
   * port, and `postMessage` on one throws rather than returning.
   */
  isRendererAlive(target: Target): boolean;
  report(event: BrokerEvent): void;
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
  /**
   * The renderer proved that the live child's transport stopped making
   * progress. SIGTERM it, escalate to SIGKILL after `killGraceMs`, then let the
   * ordinary exit path restart and re-broker. Idempotent per child.
   */
  recover(target: Target, expectedGeneration: number): void;
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
  const killGraceMs = options.killGraceMs ?? CONTROL_PANEL_BROKER_KILL_GRACE_MS;

  let child: ControlPanelChild | null = null;
  let generation = 0;
  let disposed = false;
  let gaveUp = false;
  let consecutiveCrashes = 0;
  /** When the current child was forked, for the crash-budget reset. */
  let spawnedAtMs = 0;
  let cancelRestart: (() => void) | null = null;
  let recovery: { target: ControlPanelChild; cancelEscalation: () => void } | null = null;
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

  function ensureChild(): ControlPanelChild | null {
    if (child !== null) return child;
    if (disposed || gaveUp) return null;
    spawnedAtMs = deps.now();
    spawnFailure = null;
    try {
      child = deps.spawn({ onExit: (code) => onExit(code) });
    } catch (error) {
      // A fork that fails outright — a missing `dist/control-panel/index.js`, a bad
      // executable — never produces an `exit`, so nothing else would run the
      // restart budget and the broker would sit with nothing scheduled and
      // nothing to report. Synthesise the exit, exactly as the INFERENCE
      // supervisor does for a transport that reports `error` and no `exit`.
      spawnFailure = `control panel failed to fork: ${describe(error)}`;
      onExit(null);
      return null;
    }
    return child;
  }

  function onExit(code: number | null): void {
    const exited = child;
    child = null;
    if (recovery?.target === exited) {
      recovery.cancelEscalation();
      recovery = null;
    }
    const waiters = exitWaiters;
    exitWaiters = [];
    for (const waiter of waiters) waiter();
    if (disposed) return;

    // Every non-disposal CONTROL PANEL exit consumes the crash budget. That
    // includes a watchdog recovery: repeatedly wedging is a crash loop even
    // though MAIN delivered the terminating signal. An exit code of 0 still
    // does not make it healthy.
    consecutiveCrashes = nextCrashCount(policy, consecutiveCrashes, deps.now() - spawnedAtMs);
    deps.report({ type: 'control-panel-exited', code, consecutiveCrashes });

    const decision = planRestart(policy, consecutiveCrashes);
    if (decision.action === 'give-up') {
      gaveUp = true;
      deps.report({ type: 'control-panel-gave-up', consecutiveCrashes });
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
        reason:
          spawnFailure ?? (gaveUp ? `control panel gave up after ${consecutiveCrashes} crashes` : 'broker disposed'),
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
      live.kill();
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

  function recover(target: Target, expectedGeneration: number): void {
    if (disposed || !deps.isRendererAlive(target)) return;
    // The report names the exact channel that observed the wedge. MAIN may have
    // replaced that child/channel while the page → preload → IPC signal was
    // queued; never resolve a stale report against whichever child is current.
    if (expectedGeneration !== generation) return;
    lastTarget = target;
    const live = child;
    if (live === null) {
      // An exit may already have scheduled the replacement. If it did not,
      // attach is the one path that knows how to spawn and broker safely.
      if (cancelRestart === null) attach(target);
      return;
    }
    if (recovery?.target === live) return;

    // Install the escalation before SIGTERM: a test double (and some process
    // wrappers) may report exit synchronously from `kill()`.
    const cancelEscalation = deps.delay(() => {
      if (child !== live) return;
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
      // Never throw out of an ipcMain handler. The scheduled escalation remains
      // armed and will make a second, uncatchable attempt via SIGKILL.
    }
  }

  return {
    attach,
    recover,
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
      const target = child;
      if (target === null) return;

      // Registered BEFORE the kill, so a transport that reports the exit
      // synchronously cannot land between the two.
      const gone = new Promise<void>((resolve) => {
        exitWaiters.push(resolve);
      });
      target.kill();
      const cancelEscalation = deps.delay(() => {
        // `kill()` flips a flag the moment the signal is SENT, so only the exit
        // handler can answer "is it gone yet" — and `child` is nulled there.
        if (child !== null) target.forceKill();
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

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
