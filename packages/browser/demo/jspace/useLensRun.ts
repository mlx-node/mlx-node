// useLensRun.ts — the /jspace execution model.
//
// The MLX worker runs one command at a time on a serial loop (model.rs:754) and
// there is NO cancel message for a queued `lensReadout` (lens-client.ts:124): an
// `AbortSignal` rejects the JS promise but cannot un-queue the Rust command.
// Overlapping readouts therefore pile up on that serial loop — a LATENCY problem,
// not corruption. So the app is single-flight: never dispatch while one is in
// flight, coalesce to the newest pending request, and drop stale results by
// generation.
//
// `createRunQueue` is the pure, React-free core (unit-tested in
// src/__tests__/jspace-single-flight.test.ts). `useLensRun` wraps it with a
// monotonic generation guard + a state machine so a superseded-but-completed run
// never paints over a newer one.

import * as React from 'react';

import type { LensReadoutRun } from '../../src/inspector-types';

// ---------------------------------------------------------------------------
// Pure single-flight core
// ---------------------------------------------------------------------------

export type RunQueue<Req, Res> = {
  /** Resolves with the result, or `null` if a newer request superseded this one. */
  run(req: Req): Promise<Res | null>;
  inFlight(): boolean;
};

/**
 * A single-flight queue over an async `dispatch`. At most ONE dispatch is in
 * flight at a time and at most ONE request is pending behind it.
 *
 * Semantics (pinned by the Step-1 test):
 *   - `run()` while idle dispatches immediately.
 *   - `run()` while a dispatch is in flight parks the request as `pending`.
 *   - a second `run()` while something is already pending SUPERSEDES the old
 *     pending: the old one **settles as `null`** (never rejects, never dangles —
 *     a rejected fire-and-forget run would be an unhandled rejection at every
 *     call site, a never-settling one would leak) and is replaced.
 *   - when the in-flight dispatch settles, the pending request (if any) is
 *     promoted and dispatched.
 *   - a dispatch that throws rejects ONLY its own caller and still drains the
 *     queue (the pending request is promoted regardless).
 */
export function createRunQueue<Req, Res>(dispatch: (req: Req) => Promise<Res>): RunQueue<Req, Res> {
  type Waiter = { req: Req; resolve: (value: Res | null) => void; reject: (err: unknown) => void };

  let running = false;
  let pending: Waiter | null = null;

  function start(waiter: Waiter): void {
    running = true;
    void (async () => {
      try {
        const res = await dispatch(waiter.req);
        waiter.resolve(res);
      } catch (err) {
        // A dispatch failure rejects ONLY this caller; the queue still drains.
        waiter.reject(err);
      } finally {
        running = false;
        if (pending) {
          const next = pending;
          pending = null;
          start(next);
        }
      }
    })();
  }

  return {
    run(req: Req): Promise<Res | null> {
      return new Promise<Res | null>((resolve, reject) => {
        const waiter: Waiter = { req, resolve, reject };
        if (!running) {
          start(waiter);
        } else {
          // Coalesce: the previous pending never gets to run — settle it as null.
          if (pending) pending.resolve(null);
          pending = waiter;
        }
      });
    },
    inFlight(): boolean {
      return running;
    },
  };
}

// ---------------------------------------------------------------------------
// React binding
// ---------------------------------------------------------------------------

/** The five fields a `/jspace` readout needs — a subset of `LensReadoutRequest`
 *  (no `type`/`id`, no diagnostic `noTile`). */
export type LensRunArgs = {
  promptIds: number[];
  layers: number[];
  topK: number;
  pinnedIds: number[];
  useJacobian: boolean;
};

export type LensRunState =
  | { status: 'idle' }
  | { status: 'running' }
  | { status: 'done'; result: LensReadoutRun }
  | { status: 'error'; message: string };

export type UseLensRun = {
  state: LensRunState;
  /** Bumped on each completed run so remount-keyed children (TopKBars) re-animate. */
  runKey: number;
  /** Fire a run. Resolves to the result, or `null` if it was superseded or
   *  became stale (a newer run started before it finished) or errored. */
  run: (args: LensRunArgs) => Promise<LensReadoutRun | null>;
  reset: () => void;
};

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

/**
 * Bind {@link createRunQueue} to React state with a monotonic generation guard.
 *
 * The queue guarantees only ONE dispatch runs at a time and resolves a
 * superseded *pending* request to `null`. But the currently in-flight request
 * always completes with a REAL result — so if the user fires run A then run B
 * while A is still on the wire, A resolves with a genuine (now stale) result.
 * The generation guard is what drops it: every `run()` claims `myGen`, and the
 * completed result is applied to state only while `runGenRef.current === myGen`.
 * Together, no superseded run ever paints over a newer one.
 *
 * `dispatch` receives an `AbortSignal` tied to the component's lifetime; it is
 * aborted on unmount so an in-flight `lensReadout` rejects instead of hanging on
 * a torn-down worker. The queue never overlaps dispatches, so a single
 * lifetime-scoped controller is sufficient (no per-run cancellation is possible
 * anyway — see the module header).
 */
export function useLensRun(
  dispatch: (args: LensRunArgs, signal: AbortSignal) => Promise<LensReadoutRun>,
): UseLensRun {
  const [state, setState] = React.useState<LensRunState>({ status: 'idle' });
  const [runKey, setRunKey] = React.useState(0);

  const runGenRef = React.useRef(0);
  const dispatchRef = React.useRef(dispatch);
  dispatchRef.current = dispatch;

  // One lifetime-scoped controller: aborted on unmount so a hung readout on a
  // dead worker rejects rather than waiting out its 60s timeout.
  const abortRef = React.useRef<AbortController | null>(null);
  if (abortRef.current === null) abortRef.current = new AbortController();

  const queueRef = React.useRef<RunQueue<LensRunArgs, LensReadoutRun> | null>(null);
  if (queueRef.current === null) {
    queueRef.current = createRunQueue<LensRunArgs, LensReadoutRun>((args) =>
      dispatchRef.current(args, abortRef.current!.signal),
    );
  }

  React.useEffect(
    () => () => {
      runGenRef.current++;
      abortRef.current?.abort();
    },
    [],
  );

  const run = React.useCallback(async (args: LensRunArgs): Promise<LensReadoutRun | null> => {
    const myGen = ++runGenRef.current;
    setState({ status: 'running' });
    let result: LensReadoutRun | null;
    try {
      result = await queueRef.current!.run(args);
    } catch (err) {
      if (runGenRef.current === myGen) {
        if (isAbortError(err)) setState({ status: 'idle' });
        else setState({ status: 'error', message: err instanceof Error ? err.message : String(err) });
      }
      return null;
    }
    // `null` ⇒ this request was superseded while pending and never dispatched.
    if (result === null) return null;
    // A newer run has started since; this completed result is stale — drop it.
    if (runGenRef.current !== myGen) return null;
    setState({ status: 'done', result });
    setRunKey((k) => k + 1);
    return result;
  }, []);

  const reset = React.useCallback(() => {
    runGenRef.current++;
    setState({ status: 'idle' });
  }, []);

  return { state, runKey, run, reset };
}
