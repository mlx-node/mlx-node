import * as React from 'react';

import { MathDisplay } from '../scaffolding/MathDisplay';

/**
 * Chapter 14 (Scaling) supplement — why learning-rate warmup matters, drawn as
 * a training-loss-vs-step curve. Companion to LrScheduleViz (which draws the LR
 * schedule itself); this widget draws the *consequence* of skipping warmup.
 *
 * Two scripted curves on the same axes (loss vs step), switched by a toggle:
 *
 *   • with warmup: a clean decaying curve — high start, exponential-ish decay
 *     to a low plateau, plus a little deterministic noise so it reads like a
 *     real run rather than a textbook exponential.
 *   • no warmup: at an aggressively high LR, a full-size first step on
 *     essentially random weights blows the loss up. It spikes off the top of the
 *     plot within a few steps and flat-lines at a "diverged (NaN)" ceiling — the
 *     shape behind many a "loss exploded on step 1" story (gentler runs may
 *     wobble and recover instead).
 *
 * Plot scaffold (W/H/PAD, xFor/yFor, pathD, axes/ticks, shaded region) is the
 * same line-chart skeleton as LrScheduleViz. No model, no worker: every point
 * is formula-derived. The y-axis is loss; x is training step over the same
 * TOTAL_STEPS=10,000 horizon the LR schedule uses.
 */

const TOTAL_STEPS = 10_000;
// Loss-axis ceiling. The diverged run is clamped here and labelled NaN; the
// healthy run lives well below it so the contrast is unmistakable.
const LOSS_CEIL = 9;
const LOSS_START = 6.5; // illustrative starting loss, compressed onto a legible 0–9 axis (real init CE for a large vocab is higher).
const LOSS_PLATEAU = 2.0; // a plausible converged training loss for a small model.
// Step at which the no-warmup run is already pinned to the ceiling (NaN). The
// blow-up is effectively instantaneous; we annotate this step in the readout.
const DIVERGE_STEP = 3;

// Healthy run: exponential decay from LOSS_START toward LOSS_PLATEAU, with a
// small deterministic ripple so the line looks like a real loss curve.
function lossWithWarmup(step: number): number {
  const decay = (LOSS_START - LOSS_PLATEAU) * Math.exp(-step / 1800);
  const ripple = 0.12 * Math.sin(step / 240) * Math.exp(-step / 5000);
  return LOSS_PLATEAU + decay + ripple;
}

// Diverged run: the first full-size step on random weights overshoots, and the
// loss runs away to the ceiling within a few steps, then flat-lines (NaN). We
// model the blow-up as a steep exponential rise clamped at LOSS_CEIL.
function lossNoWarmup(step: number): number {
  const blown = LOSS_START + (LOSS_CEIL - LOSS_START + 2) * (1 - Math.exp(-step / 1.4));
  return Math.min(LOSS_CEIL, blown);
}

type Mode = 'warmup' | 'none';

export function WarmupLossCurve() {
  const [mode, setMode] = React.useState<Mode>('warmup');

  const W = 540;
  const H = 200;
  const PAD = 30;
  const plotW = W - PAD * 2;
  const plotH = H - PAD * 2;

  const xFor = React.useCallback((step: number) => PAD + (step / TOTAL_STEPS) * plotW, [plotW]);
  const yFor = React.useCallback((loss: number) => PAD + plotH - (loss / LOSS_CEIL) * plotH, [plotH]);

  // Sample both curves once. 200 points is smooth at this width.
  const { warmupPts, nonePts } = React.useMemo(() => {
    const n = 200;
    const warmup: { x: number; y: number }[] = [];
    const none: { x: number; y: number }[] = [];
    for (let i = 0; i <= n; i++) {
      const step = (i / n) * TOTAL_STEPS;
      warmup.push({ x: xFor(step), y: yFor(lossWithWarmup(step)) });
      none.push({ x: xFor(step), y: yFor(lossNoWarmup(step)) });
    }
    return { warmupPts: warmup, nonePts: none };
  }, [xFor, yFor]);

  const toD = (pts: { x: number; y: number }[]) =>
    pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');

  const warmupD = toD(warmupPts);
  const noneD = toD(nonePts);

  const isWarmup = mode === 'warmup';
  // Dominant curve (selected) drawn bold; the other shown faintly for contrast.
  const dominantD = isWarmup ? warmupD : noneD;
  const ghostD = isWarmup ? noneD : warmupD;
  const dominantColor = isWarmup ? 'oklch(0.65 0.13 250)' : 'oklch(0.62 0.2 25)';
  const ghostColor = isWarmup ? 'oklch(0.62 0.2 25)' : 'oklch(0.65 0.13 250)';

  const divergeX = xFor(DIVERGE_STEP);
  const ceilY = yFor(LOSS_CEIL);

  const ariaLabel = isWarmup
    ? 'Training loss vs step with learning-rate warmup: a smooth decay from about 6.5 down to a low plateau near 2.0.'
    : `Training loss vs step with no warmup: the loss spikes to the diverged (NaN) ceiling by step ${DIVERGE_STEP} and flat-lines there.`;

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Warmup vs no warmup — the loss curve
        </div>
        <div className="text-[11px] text-muted-foreground">
          aggressive peak LR 1e-3 (stress test) · {TOTAL_STEPS.toLocaleString()} steps
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">
        A large enough learning rate applied to near-random initial weights can produce an enormous first step that
        sends the loss up instead of down. <strong>Warmup</strong> ramps the LR from <code>0</code> up to{' '}
        <code>lr_max</code> over the first ~1–10% of training so those early steps are gentle; cosine decay then takes
        over. The no-warmup curve below is a deliberately aggressive case — peak LR <code>1e-3</code>, well above the{' '}
        <code>3e-4</code> the schedule above uses — to make the blow-up visible; gentler runs may wobble and recover
        rather than diverge outright. It is the shape behind many a &ldquo;loss exploded on step 1&rdquo; story.
      </p>

      <MathDisplay latex={String.raw`\text{lr}(t)=\text{lr}_{\max}\cdot \frac{t}{W}\quad (t<W)`} />

      {/* Toggle: with warmup <-> no warmup. */}
      <div
        className="inline-flex flex-wrap gap-1 rounded-md border border-border p-0.5"
        role="group"
        aria-label="Warmup toggle"
      >
        {(
          [
            { id: 'warmup', label: 'with warmup' },
            { id: 'none', label: 'no warmup' },
          ] as const
        ).map((opt) => {
          const active = mode === opt.id;
          return (
            <button
              key={opt.id}
              type="button"
              aria-pressed={active}
              onClick={() => setMode(opt.id)}
              className={[
                'rounded px-2.5 py-1 text-xs font-medium transition-colors',
                active ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
              ].join(' ')}
            >
              {opt.label}
            </button>
          );
        })}
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label={ariaLabel}>
        {/* axes */}
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="currentColor" strokeOpacity={0.4} />
        <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="currentColor" strokeOpacity={0.4} />

        {/* x ticks */}
        {[0, 0.25, 0.5, 0.75, 1].map((f, i) => (
          <g key={`xt-${i}`}>
            <line
              x1={PAD + f * plotW}
              y1={H - PAD}
              x2={PAD + f * plotW}
              y2={H - PAD + 4}
              stroke="currentColor"
              strokeOpacity={0.4}
            />
            <text
              x={PAD + f * plotW}
              y={H - PAD + 14}
              fontSize={9}
              textAnchor="middle"
              fill="currentColor"
              fillOpacity={0.55}
            >
              {Math.round(f * TOTAL_STEPS).toLocaleString()}
            </text>
          </g>
        ))}
        <text x={PAD + plotW / 2} y={H - 4} fontSize={9} textAnchor="middle" fill="currentColor" fillOpacity={0.55}>
          training step
        </text>

        {/* y ticks at 0 (bottom) and ceiling (top) */}
        <text x={PAD - 4} y={PAD + plotH + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.55}>
          0
        </text>
        <text x={PAD - 4} y={PAD + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.55}>
          {LOSS_CEIL}
        </text>
        <text
          x={10}
          y={PAD + plotH / 2 + 3}
          fontSize={9}
          fill="currentColor"
          fillOpacity={0.55}
          transform={`rotate(-90, 10, ${PAD + plotH / 2 + 3})`}
        >
          loss
        </text>

        {/* diverged (NaN) ceiling line — the magnitude is meaningless past here */}
        <line
          x1={PAD}
          y1={ceilY}
          x2={W - PAD}
          y2={ceilY}
          stroke="oklch(0.62 0.2 25)"
          strokeOpacity={isWarmup ? 0.25 : 0.6}
          strokeDasharray="4 3"
        />
        <text
          x={W - PAD}
          y={ceilY + 11}
          fontSize={9}
          textAnchor="end"
          fill="oklch(0.62 0.2 25)"
          fillOpacity={isWarmup ? 0.4 : 0.9}
        >
          diverged (NaN)
        </text>

        {/* ghost (non-selected) curve, faint */}
        <path d={ghostD} fill="none" stroke={ghostColor} strokeWidth={1.5} strokeOpacity={0.22} />

        {/* dominant (selected) curve */}
        <path d={dominantD} fill="none" stroke={dominantColor} strokeWidth={2} />

        {/* divergence marker — only when the diverged curve is dominant */}
        {!isWarmup ? (
          <g>
            <line
              x1={divergeX}
              y1={PAD}
              x2={divergeX}
              y2={H - PAD}
              stroke="oklch(0.62 0.2 25)"
              strokeOpacity={0.5}
              strokeDasharray="3 3"
            />
            <circle cx={divergeX} cy={ceilY} r={4} fill="oklch(0.62 0.2 25)" />
            <text
              x={divergeX + 4}
              y={PAD + 12}
              fontSize={9}
              textAnchor="start"
              fill="oklch(0.62 0.2 25)"
              fillOpacity={0.95}
            >
              step ~{DIVERGE_STEP}: loss → NaN
            </text>
          </g>
        ) : null}
      </svg>

      {/* readout */}
      <div
        className={[
          'rounded-md border p-2 text-[12px]',
          isWarmup
            ? 'border-primary/40 bg-primary/5 text-foreground/90'
            : 'border-destructive/40 bg-destructive/5 text-foreground/90',
        ].join(' ')}
      >
        {isWarmup ? (
          <>
            <span className="font-mono font-semibold">with warmup</span> — loss decays smoothly from{' '}
            <span className="font-mono">{LOSS_START.toFixed(1)}</span> toward a plateau near{' '}
            <span className="font-mono">{LOSS_PLATEAU.toFixed(1)}</span>. Early steps are tiny while AdamW&apos;s moving
            averages fill in, so nothing blows up.
          </>
        ) : (
          <>
            <span className="font-mono font-semibold">no warmup</span> — the first full-size step on random weights
            overshoots and the loss runs away.{' '}
            <span className="font-mono text-destructive">step ~{DIVERGE_STEP}: loss → NaN</span>; from there it is
            flat-lined at the ceiling (the magnitude past divergence is meaningless).
          </>
        )}
      </div>

      <p className="text-[10px] text-muted-foreground">
        Illustrative — both curves are scripted formulas (exponential decay vs a clamped blow-up), not live training
        output.
      </p>
    </div>
  );
}
