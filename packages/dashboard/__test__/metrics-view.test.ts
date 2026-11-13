import { describe, expect, it } from 'vite-plus/test';

import {
  axisNumber,
  byteAxis,
  cacheReuse,
  failureLabel,
  isKpiKey,
  isMetricsTab,
  isOutcomeFilter,
  kpiTiles,
  modelShareLabel,
  phaseMeans,
  phaseSegments,
  previousRange,
  speculationView,
  stopSentence,
  timeAxis,
  toBins,
  turnPhases,
  turnTimeline,
  versusText,
} from '../ui/src/lib/metrics-view.js';
import { alignedRange } from '../ui/src/lib/range.js';
import { busyTotals, emptySummary, quantiles, summary, tsPoint, turnRow } from './helpers/observability-fixtures.js';

describe('turnPhases', () => {
  it('sums to pre-dispatch plus duration', () => {
    const phases = turnPhases(
      turnRow('a', {
        preDispatchMs: 100,
        queueMs: 30,
        loadWaitMs: 20,
        resolveMs: 10,
        ttftMs: 200,
        prefillMs: 150,
        decodeMs: 700,
        durationMs: 1000,
      }),
    );
    expect(phases).toEqual({
      handlingMs: 40,
      resolveMs: 10,
      loadWaitMs: 20,
      queueMs: 30,
      waitMs: 0,
      slotMs: 0,
      prepareMs: 0,
      firstTokenMs: 0,
      setupMs: 50,
      prefillMs: 150,
      decodeMs: 700,
      tailMs: 100,
      totalMs: 1100,
    });
  });

  it('keeps the whole time to first token unsplit for a turn without a prefill window', () => {
    const phases = turnPhases(
      turnRow('u', {
        preDispatchMs: 0,
        ttftMs: 200,
        prefillMs: null,
        prefillTokens: null,
        decodeMs: 700,
        durationMs: 1000,
      }),
    );
    expect(phases).toMatchObject({ firstTokenMs: 200, setupMs: 0, prefillMs: 0, decodeMs: 700, tailMs: 100 });
  });

  it('caps a prefill window longer than the time to first token, and uses the window when ttft is missing', () => {
    const over = turnPhases(turnRow('o', { preDispatchMs: 0, ttftMs: 200, prefillMs: 260, durationMs: 1000 }));
    expect(over).toMatchObject({ firstTokenMs: 0, setupMs: 0, prefillMs: 200 });
    const noTtft = turnPhases(turnRow('n', { preDispatchMs: 0, ttftMs: null, prefillMs: 260, durationMs: 1000 }));
    expect(noTtft).toMatchObject({ firstTokenMs: 0, setupMs: 0, prefillMs: 260 });
    for (const phases of [over, noTtft]) {
      const parts =
        phases.handlingMs +
        phases.resolveMs +
        phases.loadWaitMs +
        phases.queueMs +
        phases.firstTokenMs +
        phases.setupMs +
        phases.prefillMs +
        phases.decodeMs +
        phases.tailMs;
      expect(parts).toBe(phases.totalMs);
    }
  });

  it('clamps parts that overrun their whole and treats bad inputs as 0', () => {
    const phases = turnPhases(
      turnRow('b', {
        preDispatchMs: 50,
        queueMs: 80,
        loadWaitMs: -5,
        resolveMs: Number.NaN,
        ttftMs: 900,
        prefillMs: null,
        decodeMs: 900,
        durationMs: 600,
      }),
    );
    expect(phases.queueMs).toBe(50);
    expect(phases.loadWaitMs).toBe(0);
    expect(phases.resolveMs).toBe(0);
    expect(phases.firstTokenMs).toBe(600);
    expect(phases.decodeMs).toBe(0);
    const parts =
      phases.handlingMs +
      phases.resolveMs +
      phases.loadWaitMs +
      phases.queueMs +
      phases.firstTokenMs +
      phases.setupMs +
      phases.prefillMs +
      phases.decodeMs +
      phases.tailMs;
    expect(parts).toBe(phases.totalMs);
  });

  it('puts the time of a load this turn owned in "Loading the model", not in Other', () => {
    const segments = phaseSegments(
      turnPhases(turnRow('c', { preDispatchMs: 9000, queueMs: 0, resolveMs: 8500, loadWaitMs: 0 })),
    );
    const byKey = new Map(segments.map((segment) => [segment.key, segment]));
    expect(byKey.get('resolve')?.label).toBe('Loading the model');
    expect(byKey.get('resolve')?.value).toBe(8500);
    expect(byKey.get('other')?.value).toBe(500 + (9400 - 300 - 9000));
  });

  it('reads totals with the same keys', () => {
    const segments = phaseSegments(busyTotals());
    expect(segments.map((segment) => segment.key)).toEqual([
      'queue',
      'loadWait',
      'resolve',
      'setup',
      'prefill',
      'decode',
      'other',
    ]);
    expect(segments.map((segment) => segment.value)).toEqual([2000, 1000, 8000, 4000, 16_000, 60_000, 1000]);
  });

  it('drops the phases a bar cannot show: 7 measured, 6 old, 8 mixed', () => {
    const measured = phaseSegments(busyTotals());
    const old = phaseSegments(busyTotals({ firstTokenWallMs: 20_000, setupWallMs: 0, prefillWallMs: 0 }));
    const mixed = phaseSegments(busyTotals({ firstTokenWallMs: 5_000 }));
    expect(measured).toHaveLength(7);
    expect(old.map((segment) => segment.label)).toEqual([
      'Queue',
      'Waiting for a model load',
      'Loading the model',
      'To first token',
      'Decode',
      'Other',
    ]);
    expect(mixed).toHaveLength(8);
    expect(mixed.find((segment) => segment.key === 'setup')?.color).toBe('var(--phase-setup)');
  });
});

describe('turnPhases with the native timers', () => {
  const sum = (p: ReturnType<typeof turnPhases>): number =>
    p.handlingMs +
    p.resolveMs +
    p.loadWaitMs +
    p.queueMs +
    p.waitMs +
    p.slotMs +
    p.prepareMs +
    p.firstTokenMs +
    p.setupMs +
    p.prefillMs +
    p.decodeMs +
    p.tailMs;

  it('a turn that waited 75 s for the model has a wait part, and the parts sum to the total', () => {
    const row = turnRow('b', {
      preDispatchMs: 0,
      queueMs: 0,
      waitMs: 75_070,
      slotMs: 0,
      prepareMs: 1_200,
      ttftMs: 2_600,
      prefillMs: 500,
      decodeMs: 2_000,
      durationMs: 79_700,
    });
    const phases = turnPhases(row);
    expect(phases).toMatchObject({ waitMs: 75_070, slotMs: 0, prepareMs: 1_200, setupMs: 900, decodeMs: 2_000 });
    expect(sum(phases)).toBe(phases.totalMs);
    expect(phaseSegments(phases).find((segment) => segment.key === 'wait')).toMatchObject({
      label: 'Model busy',
      value: 75_070,
    });
  });

  it('clamps timers that overrun their window and never passes the total', () => {
    const phases = turnPhases(
      turnRow('o', {
        preDispatchMs: 0,
        waitMs: 90_000,
        slotMs: 5_000,
        prepareMs: 5_000,
        ttftMs: 2_600,
        prefillMs: 500,
        decodeMs: 9_000,
        durationMs: 10_000,
      }),
    );
    expect(phases).toMatchObject({ waitMs: 7_400, decodeMs: 0, slotMs: 2_100, prepareMs: 0, setupMs: 0 });
    expect(sum(phases)).toBe(phases.totalMs);
  });

  it('a turn cut before its first token keeps its slot and prepare time', () => {
    const phases = turnPhases(
      turnRow('c', {
        outcome: 'cancelled',
        failPhase: 'prepare',
        preDispatchMs: 0,
        waitMs: 1_000,
        slotMs: 3_000,
        prepareMs: 9_000,
        ttftMs: null,
        prefillMs: null,
        decodeMs: null,
        durationMs: 5_000,
      }),
    );
    expect(phases).toMatchObject({ waitMs: 1_000, slotMs: 3_000, prepareMs: 1_000, firstTokenMs: 0, tailMs: 0 });
    expect(sum(phases)).toBe(phases.totalMs);
  });

  it('a record without the timers shows no empty part', () => {
    const keys = phaseSegments(turnPhases(turnRow('old'))).map((segment) => segment.key);
    expect(keys).not.toContain('wait');
    expect(keys).not.toContain('slot');
    expect(keys).not.toContain('prepare');
  });
});

describe('turnTimeline', () => {
  it('draws model busy, slot and prepare before the setup, the first-token marker after them', () => {
    const timeline = turnTimeline(
      turnRow('t', {
        preDispatchMs: 0,
        queueMs: 0,
        waitMs: 75_070,
        slotMs: 300,
        prepareMs: 1_200,
        ttftMs: 2_600,
        prefillMs: 500,
        decodeMs: 2_000,
        durationMs: 79_700,
      }),
    );
    expect(timeline.spans.map((span) => [span.kind, span.label, span.startMs, span.endMs])).toEqual([
      ['busy', 'Model busy', 0, 75_070],
      ['slot', 'Waiting for a slot', 75_070, 75_370],
      ['prepare', 'Preparing', 75_370, 76_570],
      ['setup', 'Setup', 76_570, 77_170],
      ['prefill', 'Prefill', 77_170, 77_670],
      ['decode', 'Decode', 77_670, 79_670],
      ['other', 'Other', 79_670, 79_700],
    ]);
    expect(timeline.markers[0]?.atMs).toBe(77_670);
  });

  it('names the prefix work in the setup of a turn whose prepare was not timed', () => {
    const before = (prepareMs: number | null): string | undefined =>
      turnTimeline(turnRow('p', { prepareMs })).spans.find((span) => span.kind === 'setup')?.detail?.[0]?.value;
    expect(before(null)).toBe('prefix lookup, cache restore, allocation');
    expect(before(0)).toBe('prefix lookup, cache restore, allocation');
    expect(before(20)).toBe('other requests on this model');
  });

  it('a turn stopped while waiting or preparing ends at that phase', () => {
    for (const failPhase of ['queued', 'slot', 'prepare'] as const) {
      const timeline = turnTimeline(
        turnRow('s', {
          outcome: 'cancelled',
          failPhase,
          preDispatchMs: 0,
          waitMs: 1_000,
          slotMs: 3_000,
          prepareMs: 500,
          ttftMs: null,
          prefillMs: null,
          decodeMs: null,
          durationMs: 5_000,
        }),
      );
      expect(timeline.spans.map((span) => span.id)).toEqual(['wait', 'slot', 'prepare']);
      expect(timeline.spans.at(-1)?.status).toBe('cancelled');
      expect(timeline.markers).toEqual([]);
    }
    expect(stopSentence({ outcome: 'cancelled', failPhase: 'prepare' })).toBe('Stopped while preparing.');
    expect(stopSentence({ outcome: 'error', failPhase: 'slot' })).toBe('Stopped while waiting for a slot.');
    expect(stopSentence({ outcome: 'cancelled', failPhase: 'queued' })).toBe('Stopped while waiting.');
  });

  it('marks derived spans estimated and measured ones not', () => {
    const timeline = turnTimeline(turnRow('d', { preDispatchMs: 100, queueMs: 40 }));
    const byId = new Map(timeline.spans.map((span) => [span.id, span]));
    expect(byId.get('queue')?.estimated).toBe(true);
    expect(byId.get('handling')?.estimated).toBe(true);
    expect(byId.get('decode')?.estimated).toBe(true);
    expect(byId.get('prefill')?.estimated).toBe(false);
    expect(timeline.markers.map((marker) => marker.label)).toEqual(['First token']);
    expect(timeline.domainMs).toEqual([0, 9500]);
  });

  it('draws Setup (a difference) then Prefill for a measured turn, the marker at their end', () => {
    const timeline = turnTimeline(turnRow('m', { preDispatchMs: 100, queueMs: 40 }));
    const kinds = timeline.spans.map((span) => [span.kind, span.label, span.startMs, span.endMs, span.estimated]);
    expect(kinds.slice(2, 4)).toEqual([
      ['setup', 'Setup', 100, 150, true],
      ['prefill', 'Prefill', 150, 400, false],
    ]);
    const prefill = timeline.spans.find((span) => span.kind === 'prefill');
    expect(prefill?.detail).toEqual([
      { label: 'Prefill tokens', value: '2,500' },
      { label: 'Prefill speed', value: '10K tok/s' },
    ]);
    expect(timeline.markers[0]?.atMs).toBe(400);
  });

  it('draws one To first token span for a turn without a prefill window', () => {
    const timeline = turnTimeline(
      turnRow('w', {
        preDispatchMs: 100,
        queueMs: 40,
        prefillMs: null,
        prefillTokens: null,
        setupMs: null,
        prefillTps: null,
      }),
    );
    const first = timeline.spans.filter(
      (span) => span.kind === 'first' || span.kind === 'setup' || span.kind === 'prefill',
    );
    expect(first.map((span) => [span.kind, span.label, span.startMs, span.endMs])).toEqual([
      ['first', 'To first token', 100, 400],
    ]);
    expect(first[0]?.detail?.[1]).toEqual({ label: 'Prefill', value: 'not measured for this turn' });
    expect(timeline.markers[0]?.atMs).toBe(400);
  });

  it('stops at the fail phase and gives the last span the outcome', () => {
    const prefill = turnTimeline(turnRow('e', { outcome: 'error', failPhase: 'prefill' }));
    expect(prefill.spans.map((span) => span.id)).not.toContain('decode');
    expect(prefill.spans.at(-1)?.status).toBe('error');
    const queued = turnTimeline(turnRow('f', { outcome: 'cancelled', failPhase: 'queued' }));
    expect(queued.spans.map((span) => span.id)).not.toContain('prefill');
    expect(queued.markers).toEqual([]);
    expect(queued.spans.at(-1)?.status).toBe('cancelled');
  });

  it('draws the whole turn when a failed turn has no fail phase', () => {
    const timeline = turnTimeline(turnRow('g', { outcome: 'error', failPhase: null }));
    expect(timeline.spans.map((span) => span.id)).toContain('decode');
    expect(stopSentence({ outcome: 'error', failPhase: null })).toBeNull();
    expect(stopSentence({ outcome: 'cancelled', failPhase: 'decode' })).toBe('Stopped during decode.');
  });
});

describe('guards', () => {
  it('accept their own values and reject others', () => {
    for (const tab of ['summary', 'speed', 'usage', 'speculation', 'failures', 'turns'])
      expect(isMetricsTab(tab)).toBe(true);
    for (const key of ['decode', 'ttft', 'turns', 'tokens', 'failed']) expect(isKpiKey(key)).toBe(true);
    for (const value of ['all', 'ok', 'error', 'cancelled']) expect(isOutcomeFilter(value)).toBe(true);
    for (const bad of ['', 'Summary', 'overview', 'decode ', 'none']) {
      expect(isMetricsTab(bad)).toBe(false);
      expect(isKpiKey(bad)).toBe(false);
      expect(isOutcomeFilter(bad)).toBe(false);
    }
  });
});

describe('units and bins', () => {
  it('prints an axis-unit value without a unit, 0 as 0', () => {
    expect([0, 5, 2.5, 1.234, 12.34, 1234.5, 0.0004, -0.0001].map(axisNumber)).toEqual([
      '0',
      '5',
      '2.5',
      '1.23',
      '12.3',
      '1,235',
      '0',
      '0',
    ]);
  });

  it('opens the last bin', () => {
    const bins = toBins([0, 10, 100], { n: 6, counts: [1, 2, 3] });
    expect(bins).toEqual([
      { from: 0, to: 10, count: 1 },
      { from: 10, to: 100, count: 2 },
      { from: 100, to: Infinity, count: 3 },
    ]);
    expect(toBins([0, 10], undefined).map((bin) => bin.count)).toEqual([0, 0]);
  });

  it('switches time and byte units at their edges', () => {
    expect(timeAxis(1999)).toEqual({ unit: 'ms', div: 1 });
    expect(timeAxis(2000)).toEqual({ unit: 's', div: 1000 });
    expect(timeAxis(120_000)).toEqual({ unit: 'min', div: 60_000 });
    expect(timeAxis(Number.NaN).unit).toBe('ms');
    expect(byteAxis(2 * 1024 ** 3 - 1).unit).toBe('MB');
    expect(byteAxis(2 * 1024 ** 3).unit).toBe('GB');
  });

  it('reads 0 prompt tokens as unknown, not 0 %', () => {
    expect(cacheReuse(0, 0)).toEqual({ text: '—', fraction: null });
    expect(cacheReuse(90, 10).fraction).toBe(0.9);
  });

  it('never labels a share 100 % while another model still holds turns', () => {
    expect(modelShareLabel(9999, 10_000)).toBe('9,999 turns · >99%');
    expect(modelShareLabel(1, 10_000)).toBe('1 turn · <1%');
    expect(modelShareLabel(10_000, 10_000)).toBe('10,000 turns · 100%');
    expect(modelShareLabel(0, 10_000)).toBe('0 turns · 0%');
  });
});

describe('kpiTiles', () => {
  it('keeps a null median as no data and never as 0', () => {
    const tiles = kpiTiles(emptySummary(), versusText('7d'), 'tails');
    const byKey = new Map(tiles.map((tile) => [tile.key, tile]));
    expect(byKey.get('decode')?.value).toBeNull();
    expect(byKey.get('ttft')?.value).toBeNull();
    expect(byKey.get('turns')?.value).toBe('0');
    expect(byKey.get('failed')?.delta.current).toBeNull();
  });

  it('prints tails from 20 turns and the sample below', () => {
    const many = new Map(kpiTiles(summary(), 'vs previous 7 days', 'tails').map((tile) => [tile.key, tile]));
    expect(many.get('decode')?.hint).toBe('p05 21.0 · p95 63.0');
    const few = summary({ latency: { ...summary().latency, decodeTps: quantiles(40, 3) } });
    const tiles = new Map(kpiTiles(few, 'vs previous 7 days', 'tails').map((tile) => [tile.key, tile]));
    expect(tiles.get('decode')?.hint).toBe('median of 3 turns');
    expect(tiles.get('decode')?.sampleCount).toBe(3);
    expect(tiles.get('turns')?.hint).toBe('95 ok · 3 failed · 2 cancelled');
    expect(tiles.get('failed')?.hint).toBe('3% of turns');
  });
});

describe('phaseMeans and previousRange', () => {
  it('gives a bucket with no turn null means', () => {
    const range = alignedRange('7d', new Date(2026, 10, 11, 12).getTime());
    const data = phaseMeans([tsPoint(range.starts[6], 'all')], range);
    expect(data[0].values.decode).toBeNull();
    expect(data[6].values.decode).toBe(6000);
    expect(data[6].samples?.decode).toBe(10);
  });

  it('lays the previous window bucket for bucket before the range', () => {
    const range = alignedRange('7d', new Date(2026, 10, 11, 12).getTime());
    const before = previousRange(range);
    expect(before.to).toBe(range.from);
    expect(before.starts).toHaveLength(7);
    expect(before.starts[6]).toBe(new Date(2026, 10, 4).getTime());
  });
});

describe('labels', () => {
  it('leaves null parts out of a failure label', () => {
    expect(failureLabel({ outcome: 'cancelled', errorKind: null, failPhase: 'decode', turns: 1 })).toBe(
      'Cancelled · during decode',
    );
    expect(failureLabel({ outcome: 'error', errorKind: 'finish_error', failPhase: null, turns: 1 })).toBe(
      'Failed · model reported an error',
    );
  });
});

describe('speculationView', () => {
  it('weights means by cycles over models that used MTP', () => {
    const view = speculationView([
      {
        model: 'a',
        turns: 10,
        mtpTurns: 10,
        cycles: 100,
        meanAccepted: 2,
        meanDrafts: 1,
        meanDepth: 3,
        acceptByPos: [
          { pos: 0, rate: 0.8, cycles: 100 },
          { pos: 1, rate: 0.5, cycles: 100 },
        ],
        decodeTpsWithMtp: quantiles(50, 10),
        decodeTpsWithoutMtp: quantiles(40, 0),
      },
      {
        model: 'b',
        turns: 10,
        mtpTurns: 5,
        cycles: 300,
        meanAccepted: 3,
        meanDrafts: 2,
        meanDepth: 4,
        acceptByPos: [{ pos: 0, rate: 0.4, cycles: 300 }],
        decodeTpsWithMtp: quantiles(50, 10),
        decodeTpsWithoutMtp: quantiles(40, 0),
      },
      {
        model: 'c',
        turns: 5,
        mtpTurns: 0,
        cycles: 0,
        meanAccepted: null,
        meanDrafts: null,
        meanDepth: null,
        acceptByPos: [],
        decodeTpsWithMtp: quantiles(null, 0),
        decodeTpsWithoutMtp: quantiles(40, 5),
      },
    ]);
    expect(view.models.map((row) => row.model)).toEqual(['a', 'b']);
    expect(view.tokensPerCycle).toBe(2.75);
    expect(view.rates).toEqual([0.5, 0.5]);
    expect(view.samples).toEqual([400, 100]);
    expect(view.mtpTurns).toBe(15);
    expect(view.turns).toBe(20);
  });
});
