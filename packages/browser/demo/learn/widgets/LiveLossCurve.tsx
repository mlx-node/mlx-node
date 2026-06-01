import * as React from 'react';

/**
 * Chapter 13 (Training) — live training-loss curve.
 *
 * Unlike WarmupLossCurve (which plots two scripted formula curves), this widget
 * is driven entirely by live data: the `losses` array is the real per-step
 * cross-entropy loss streamed back from a `TinyTrainer` training loop running
 * in the WASM/WebGPU worker. It reuses WarmupLossCurve's line-chart skeleton
 * (viewBox W=540/H=200/PAD=30, xFor/yFor, the `toD` path helper) so the two
 * curves read as siblings, but the y-axis autoscales to the data range and the
 * x-axis spans the configured `maxStep` horizon.
 *
 * Respects `prefers-reduced-motion`: the latest-point pulse is only rendered
 * when motion is allowed.
 */

const W = 540;
const H = 200;
const PAD = 30;
const PLOT_W = W - PAD * 2;
const PLOT_H = H - PAD * 2;

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = React.useState(false);
  React.useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduced(mq.matches);
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);
  return reduced;
}

export function LiveLossCurve({
  losses,
  maxStep,
}: {
  /** Per-step loss, in training order. Index 0 is step 1. */
  losses: number[];
  /** Horizon for the x-axis. Defaults to the number of points seen so far
   *  (clamped to a small minimum so the first points aren't crammed at x=0). */
  maxStep?: number;
}) {
  const reducedMotion = usePrefersReducedMotion();

  const finiteLosses = React.useMemo(() => losses.filter((v) => Number.isFinite(v)), [losses]);

  // Autoscale the y-axis to the data. Pad the range by 5% on each side so the
  // top/bottom points aren't glued to the axes; guard the degenerate
  // all-equal case (e.g. a single point) with a unit window.
  const { yMin, yMax } = React.useMemo(() => {
    if (finiteLosses.length === 0) return { yMin: 0, yMax: 1 };
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of finiteLosses) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (lo === hi) {
      return { yMin: lo - 0.5, yMax: hi + 0.5 };
    }
    const pad = (hi - lo) * 0.05;
    return { yMin: lo - pad, yMax: hi + pad };
  }, [finiteLosses]);

  const horizon = Math.max(maxStep ?? losses.length, 10, losses.length);

  const xFor = React.useCallback(
    (step: number) => PAD + (horizon <= 1 ? 0 : (step / (horizon - 1)) * PLOT_W),
    [horizon],
  );
  const yFor = React.useCallback(
    (loss: number) => {
      const span = yMax - yMin || 1;
      return PAD + PLOT_H - ((loss - yMin) / span) * PLOT_H;
    },
    [yMin, yMax],
  );

  const points = React.useMemo(
    () =>
      losses.map((loss, i) => ({
        x: xFor(i),
        // Clamp non-finite (NaN/Inf) losses to the top of the plot so a blow-up
        // is visible rather than breaking the path.
        y: yFor(Number.isFinite(loss) ? loss : yMax),
      })),
    [losses, xFor, yFor, yMax],
  );

  const toD = (pts: { x: number; y: number }[]) =>
    pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');

  const lineD = toD(points);
  const last = points.length > 0 ? points[points.length - 1]! : null;
  const lastLoss = finiteLosses.length > 0 ? finiteLosses[finiteLosses.length - 1]! : null;
  const firstLoss = finiteLosses.length > 0 ? finiteLosses[0]! : null;

  const curveColor = 'oklch(0.65 0.13 250)';

  const ariaLabel =
    lastLoss != null
      ? `Live training loss over ${losses.length} steps, currently ${lastLoss.toFixed(3)}.`
      : 'Live training loss curve — no steps recorded yet.';

  return (
    <div className="not-prose space-y-2">
      <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label={ariaLabel}>
        {/* axes */}
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="currentColor" strokeOpacity={0.4} />
        <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="currentColor" strokeOpacity={0.4} />

        {/* x label */}
        <text x={PAD + PLOT_W / 2} y={H - 4} fontSize={9} textAnchor="middle" fill="currentColor" fillOpacity={0.55}>
          training step
        </text>

        {/* y ticks: top = yMax, bottom = yMin (autoscaled) */}
        <text x={PAD - 4} y={PAD + PLOT_H + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.55}>
          {yMin.toFixed(2)}
        </text>
        <text x={PAD - 4} y={PAD + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.55}>
          {yMax.toFixed(2)}
        </text>
        <text
          x={10}
          y={PAD + PLOT_H / 2 + 3}
          fontSize={9}
          fill="currentColor"
          fillOpacity={0.55}
          transform={`rotate(-90, 10, ${PAD + PLOT_H / 2 + 3})`}
        >
          loss
        </text>

        {/* the live loss curve */}
        {points.length >= 2 ? <path d={lineD} fill="none" stroke={curveColor} strokeWidth={2} /> : null}

        {/* latest point marker — pulses only when motion is allowed */}
        {last ? (
          <circle cx={last.x} cy={last.y} r={3.5} fill={curveColor}>
            {!reducedMotion ? (
              <animate attributeName="r" values="3.5;5.5;3.5" dur="1.2s" repeatCount="indefinite" />
            ) : null}
          </circle>
        ) : null}

        {points.length === 0 ? (
          <text
            x={PAD + PLOT_W / 2}
            y={PAD + PLOT_H / 2}
            fontSize={11}
            textAnchor="middle"
            fill="currentColor"
            fillOpacity={0.5}
          >
            Press Train to start the loss curve
          </text>
        ) : null}
      </svg>

      {firstLoss != null && lastLoss != null ? (
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
          <span>
            step <span className="font-mono text-foreground/80">{losses.length}</span>
          </span>
          <span>
            loss <span className="font-mono text-foreground/80">{lastLoss.toFixed(3)}</span>
          </span>
          <span>
            initial <span className="font-mono text-foreground/70">{firstLoss.toFixed(3)}</span>
          </span>
        </div>
      ) : null}
    </div>
  );
}
