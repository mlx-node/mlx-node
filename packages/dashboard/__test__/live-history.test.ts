/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { bumpConnectionGeneration } from '../ui/src/lib/connection.js';
import { nowItems } from '../ui/src/lib/engine-view.js';
import {
  LIVE_HISTORY_MAX,
  recordLiveSnapshot,
  resetLiveHistory,
  useLiveHistory,
  type LiveSample,
} from '../ui/src/lib/live-history.js';
import type { RuntimeLive } from '../ui/src/lib/types.js';
import { live, process as fixtureProcess } from './helpers/observability-fixtures.js';

interface Proc {
  pid: number;
  status: 'running' | 'dead' | 'unresponsive';
  outputTps: number | null;
  inFlight?: number;
  queued?: number;
}

const WORK = { loading: 0, waiting: 0, prepare: 10, prefill: 20, decode: 30, running: 0 };

function snapshot(now: number, processes: Proc[]): RuntimeLive {
  return {
    now,
    status: processes.some((p) => p.status === 'running') ? 'running' : 'stopped',
    supervisor: null,
    processes: processes.map((p) => ({
      pid: p.pid,
      status: p.status,
      ts: now - 100,
      rates: { windowMs: 10_000, outputTps: p.outputTps, prefillTps: null },
      work: { ms: WORK },
      mem: { active: 100, cache: 10 },
      models: [{ inFlight: p.inFlight ?? 0, queued: p.queued ?? 0 }],
    })),
  } as unknown as RuntimeLive;
}

let latest: readonly LiveSample[] = [];

function Probe(): ReactElement | null {
  latest = useLiveHistory();
  return null;
}

beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  resetLiveHistory();
});

afterEach(() => {
  document.body.innerHTML = '';
});

function mountProbe(): () => void {
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(createElement(Probe)));
  return () => act(() => root.unmount());
}

describe('live history', () => {
  it('appends one sample per running process, each with its own values', () => {
    const unmount = mountProbe();
    act(() => {
      recordLiveSnapshot(
        snapshot(1_000, [
          { pid: 7, status: 'running', outputTps: 30, inFlight: 2, queued: 1 },
          { pid: 8, status: 'running', outputTps: 12, inFlight: 1 },
          { pid: 9, status: 'dead', outputTps: 99, inFlight: 5 },
        ]),
      );
      recordLiveSnapshot(snapshot(2_000, [{ pid: 7, status: 'running', outputTps: 31 }]));
    });
    expect(latest).toHaveLength(3);
    const own = { t: 1_000, at: 900, prefillTps: null, work: WORK, memActive: 100, memCache: 10 };
    expect(latest[0]).toEqual({ ...own, pid: 7, outputTps: 30, inFlight: 2, queued: 1 });
    expect(latest[1]).toEqual({ ...own, pid: 8, outputTps: 12, inFlight: 1, queued: 0 });
    expect(latest[2].pid).toBe(7);
    unmount();
  });

  it('drops a snapshot under 900 ms after the last sample', () => {
    const unmount = mountProbe();
    act(() => {
      recordLiveSnapshot(snapshot(10_000, [{ pid: 1, status: 'running', outputTps: 1 }]));
      recordLiveSnapshot(snapshot(10_899, [{ pid: 1, status: 'running', outputTps: 2 }]));
      recordLiveSnapshot(snapshot(10_900, [{ pid: 1, status: 'running', outputTps: 3 }]));
    });
    expect(latest.map((sample) => sample.outputTps)).toEqual([1, 3]);
    unmount();
  });

  it('caps at 600 snapshots, oldest dropped, whatever the number of processes', () => {
    const unmount = mountProbe();
    act(() => {
      for (let i = 0; i < LIVE_HISTORY_MAX + 5; i += 1) {
        recordLiveSnapshot(snapshot(i * 1_000, [{ pid: 1, status: 'running', outputTps: i }]));
      }
    });
    expect(latest).toHaveLength(LIVE_HISTORY_MAX);
    expect(latest[0].outputTps).toBe(5);
    act(() => {
      recordLiveSnapshot(
        snapshot(LIVE_HISTORY_MAX * 2_000, [
          { pid: 1, status: 'running', outputTps: 1 },
          { pid: 2, status: 'running', outputTps: 2 },
        ]),
      );
    });
    expect(latest).toHaveLength(LIVE_HISTORY_MAX + 1);
    expect(latest[0].outputTps).toBe(6);
    unmount();
  });

  it('gives the second of two running processes its own ticker delta and spark, not the sum', () => {
    const unmount = mountProbe();
    const now = 200_000;
    const rates = (outputTps: number) => ({ windowMs: 10_000, outputTps, prefillTps: null });
    const two = (t: number, a: number, b: number): RuntimeLive =>
      live({
        now: t,
        processes: [
          fixtureProcess({ pid: 812, rates: rates(a), startedAt: 0 }),
          fixtureProcess({ pid: 900, rates: rates(b), startedAt: 0 }),
        ],
      });
    act(() => {
      for (let t = now - 70_000; t <= now; t += 1_000) recordLiveSnapshot(two(t, 40, t < now - 30_000 ? 20 : 30));
    });
    const second = nowItems(fixtureProcess({ pid: 900, rates: rates(30), startedAt: 0 }), latest, now);
    const spark = second[0].spark?.data.filter((value): value is number => value !== null) ?? [];
    expect(spark.length).toBeGreaterThan(0);
    expect(Math.max(...spark)).toBe(30);
    expect(second[0].delta?.text).toBe('+10.0');
    const first = nowItems(fixtureProcess({ pid: 812, rates: rates(40), startedAt: 0 }), latest, now);
    expect(Math.max(...(first[0].spark?.data.filter((value): value is number => value !== null) ?? []))).toBe(40);
    expect(first[0].delta?.text).toBe('0');
    unmount();
  });

  it('a reconnect empties the buffer', () => {
    const unmount = mountProbe();
    act(() => {
      recordLiveSnapshot(snapshot(1_000, [{ pid: 1, status: 'running', outputTps: 1 }]));
      recordLiveSnapshot(snapshot(2_000, [{ pid: 1, status: 'running', outputTps: 2 }]));
    });
    expect(latest).toHaveLength(2);
    act(() => bumpConnectionGeneration());
    expect(latest).toHaveLength(0);
    act(() => recordLiveSnapshot(snapshot(3_000, [{ pid: 2, status: 'running', outputTps: 4 }])));
    expect(latest.map((sample) => sample.pid)).toEqual([2]);
    unmount();
  });
});
