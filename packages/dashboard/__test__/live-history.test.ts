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
  decodeTps: number | null;
  inFlight?: number;
  queued?: number;
}

function snapshot(now: number, processes: Proc[]): RuntimeLive {
  return {
    now,
    status: processes.some((p) => p.status === 'running') ? 'running' : 'stopped',
    supervisor: null,
    processes: processes.map((p) => ({
      pid: p.pid,
      status: p.status,
      decodeTps: p.decodeTps,
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
          { pid: 7, status: 'running', decodeTps: 30, inFlight: 2, queued: 1 },
          { pid: 8, status: 'running', decodeTps: 12, inFlight: 1 },
          { pid: 9, status: 'dead', decodeTps: 99, inFlight: 5 },
        ]),
      );
      recordLiveSnapshot(snapshot(2_000, [{ pid: 7, status: 'running', decodeTps: 31 }]));
    });
    expect(latest).toHaveLength(3);
    expect(latest[0]).toEqual({ t: 1_000, pid: 7, outTps: 30, memActive: 100, memCache: 10, inFlight: 2, queued: 1 });
    expect(latest[1]).toEqual({ t: 1_000, pid: 8, outTps: 12, memActive: 100, memCache: 10, inFlight: 1, queued: 0 });
    expect(latest[2].pid).toBe(7);
    unmount();
  });

  it('drops a snapshot under 900 ms after the last sample', () => {
    const unmount = mountProbe();
    act(() => {
      recordLiveSnapshot(snapshot(10_000, [{ pid: 1, status: 'running', decodeTps: 1 }]));
      recordLiveSnapshot(snapshot(10_899, [{ pid: 1, status: 'running', decodeTps: 2 }]));
      recordLiveSnapshot(snapshot(10_900, [{ pid: 1, status: 'running', decodeTps: 3 }]));
    });
    expect(latest.map((sample) => sample.outTps)).toEqual([1, 3]);
    unmount();
  });

  it('caps at 600 snapshots, oldest dropped, whatever the number of processes', () => {
    const unmount = mountProbe();
    act(() => {
      for (let i = 0; i < LIVE_HISTORY_MAX + 5; i += 1) {
        recordLiveSnapshot(snapshot(i * 1_000, [{ pid: 1, status: 'running', decodeTps: i }]));
      }
    });
    expect(latest).toHaveLength(LIVE_HISTORY_MAX);
    expect(latest[0].outTps).toBe(5);
    act(() => {
      recordLiveSnapshot(
        snapshot(LIVE_HISTORY_MAX * 2_000, [
          { pid: 1, status: 'running', decodeTps: 1 },
          { pid: 2, status: 'running', decodeTps: 2 },
        ]),
      );
    });
    expect(latest).toHaveLength(LIVE_HISTORY_MAX + 1);
    expect(latest[0].outTps).toBe(6);
    unmount();
  });

  it('gives the second of two running processes its own ticker delta and spark, not the sum', () => {
    const unmount = mountProbe();
    const now = 200_000;
    const two = (t: number, a: number, b: number): RuntimeLive =>
      live({
        now: t,
        processes: [
          fixtureProcess({ pid: 812, decodeTps: a, startedAt: 0 }),
          fixtureProcess({ pid: 900, decodeTps: b, startedAt: 0 }),
        ],
      });
    act(() => {
      for (let t = now - 70_000; t <= now; t += 1_000) recordLiveSnapshot(two(t, 40, t < now - 30_000 ? 20 : 30));
    });
    const second = nowItems(fixtureProcess({ pid: 900, decodeTps: 30, startedAt: 0 }), latest, now);
    const spark = second[0].spark?.data.filter((value): value is number => value !== null) ?? [];
    expect(spark.length).toBeGreaterThan(0);
    expect(Math.max(...spark)).toBe(30);
    expect(second[0].delta?.text).toBe('+10.0');
    const first = nowItems(fixtureProcess({ pid: 812, decodeTps: 40, startedAt: 0 }), latest, now);
    expect(Math.max(...(first[0].spark?.data.filter((value): value is number => value !== null) ?? []))).toBe(40);
    expect(first[0].delta?.text).toBe('0');
    unmount();
  });

  it('an idle running process is 0; no running process is null', () => {
    const unmount = mountProbe();
    act(() => {
      recordLiveSnapshot(snapshot(1_000, [{ pid: 1, status: 'running', decodeTps: null }]));
      recordLiveSnapshot(snapshot(2_000, [{ pid: 1, status: 'unresponsive', decodeTps: 5 }]));
      recordLiveSnapshot(snapshot(3_000, []));
    });
    expect(latest[0].outTps).toBe(0);
    expect(latest[0].inFlight).toBe(0);
    expect(latest[1]).toEqual({
      t: 2_000,
      pid: null,
      outTps: null,
      memActive: null,
      memCache: null,
      inFlight: null,
      queued: null,
    });
    expect(latest[2].outTps).toBeNull();
    unmount();
  });

  it('a reconnect empties the buffer', () => {
    const unmount = mountProbe();
    act(() => {
      recordLiveSnapshot(snapshot(1_000, [{ pid: 1, status: 'running', decodeTps: 1 }]));
      recordLiveSnapshot(snapshot(2_000, [{ pid: 1, status: 'running', decodeTps: 2 }]));
    });
    expect(latest).toHaveLength(2);
    act(() => bumpConnectionGeneration());
    expect(latest).toHaveLength(0);
    act(() => recordLiveSnapshot(snapshot(3_000, [{ pid: 2, status: 'running', decodeTps: 4 }])));
    expect(latest.map((sample) => sample.pid)).toEqual([2]);
    unmount();
  });
});
