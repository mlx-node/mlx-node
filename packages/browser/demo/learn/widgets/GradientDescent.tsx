import { cn } from '@/lib/utils';
import * as React from 'react';

/**
 * Chapter 13 (Training) supplement — gradient descent on ONE weight.
 *
 * The chapter's prose says the optimizer "takes a small step in the direction
 * that lowers the loss." This widget makes that single sentence physical on a
 * 1-D loss curve, so a beginner can see the three things that matter:
 *
 *   1. GRADIENT = uphill direction (steepest increase). At the current weight
 *      `w` we draw the tangent to L(w); its slope is L'(w). The step always
 *      moves `w` the opposite way (downhill): w ← w − lr·L'(w).
 *   2. LEARNING RATE = step size, with a sweet spot. Tiny lr crawls; a good lr
 *      converges fast; too-large lr OVERSHOOTS the minimum and zig-zags, and
 *      past a threshold it diverges out of the bowl entirely.
 *   3. The same rule runs per-parameter on every weight at once — which is what
 *      AdamW automates for real training.
 *
 * Math (analytic gradient — nothing is numerically differentiated):
 *   L(w)  = a·(w − wStar)²          a convex parabola, single minimum at wStar
 *   L'(w) = 2a·(w − wStar)
 *   step  : w ← w − lr·L'(w) = wStar + (1 − 2a·lr)(w − wStar)
 *
 * The per-step multiplier on the distance to the minimum is m = 1 − 2a·lr, so:
 *   • |m| < 1  ⇔  0 < lr < 1/a   → converges
 *   • m = 0    ⇔  lr = 1/(2a)    → lands on the minimum in ONE step (critical)
 *   • m < 0  (|m| < 1)            → overshoots and zig-zags but still converges
 *   • |m| = 1  ⇔  lr = 1/a        → oscillates forever at constant amplitude (never settles)
 *   • |m| > 1  ⇔  lr > 1/a        → diverges out of the bowl
 *
 * With a = 0.5, wStar = 0, w0 = 4 the stability boundary is lr = 1/a = 2.0
 * (where w flips between ±w0 forever) and the critical lr is 1.0. The slider
 * runs lr ∈ [0, 2.6] so the learner can comfortably reach the zig-zag band
 * (1 < lr < 2, still converging), the knife-edge (lr = 2), and outright
 * divergence (lr > 2). A diverging `w` is clamped to the plot bounds so the marker never
 * draws off-canvas, and a "diverging" note appears once it leaves the bowl.
 *
 * Pure presentational — no model, no worker, no WASM, no network. Fully
 * keyboard-operable (native slider + buttons); auto-run is disabled under
 * prefers-reduced-motion.
 */

// Loss L(w) = A·(w − W_STAR)² with analytic gradient 2A·(w − W_STAR).
const A = 0.5;
const W_STAR = 0;
const W_START = 4;

// Divergence threshold (|1 − 2A·lr| = 1) and the one-step "critical" lr.
const LR_DIVERGE = 1 / A; // 2.0
const LR_CRITICAL = 1 / (2 * A); // 1.0
const LR_DEFAULT = 0.5; // m = 0.5 — a clean, fast-converging case
const LR_MAX = 2.6; // comfortably past LR_DIVERGE so divergence is reachable
const LR_MIN = 0;

// Plot domain. The starting weight (4) sits inside, and L(±5) = 12.5 so the
// bowl fills the frame. Anything outside this window counts as "left the bowl".
const W_MIN = -5;
const W_MAX = 5;
const L_MIN = 0;
const L_MAX = A * W_MAX * W_MAX; // 12.5 — the curve's value at the plot edges.

// SVG plot geometry — mirrors the WarmupLossCurve / LiveLossCurve skeleton.
const W = 540;
const H = 240;
const PAD_L = 34;
const PAD_R = 16;
const PAD_T = 18;
const PAD_B = 30;
const PLOT_W = W - PAD_L - PAD_R;
const PLOT_H = H - PAD_T - PAD_B;

// How many steps a single "Run" click iterates (motion allowed only).
const RUN_STEPS = 8;
const RUN_MS = 600;

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = React.useState<boolean>(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });
  React.useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, []);
  return reduced;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

const loss = (w: number) => A * (w - W_STAR) * (w - W_STAR);
const grad = (w: number) => 2 * A * (w - W_STAR);

// Math coords → SVG pixel coords. x maps weight, y maps loss (inverted).
function xPx(w: number): number {
  return PAD_L + ((w - W_MIN) / (W_MAX - W_MIN)) * PLOT_W;
}
function yPx(l: number): number {
  return PAD_T + PLOT_H - ((l - L_MIN) / (L_MAX - L_MIN)) * PLOT_H;
}

// Precompute the bowl path once — it never changes.
const BOWL_D = (() => {
  const n = 120;
  let d = '';
  for (let i = 0; i <= n; i++) {
    const w = W_MIN + (i / n) * (W_MAX - W_MIN);
    d += `${i === 0 ? 'M' : 'L'} ${xPx(w).toFixed(1)} ${yPx(loss(w)).toFixed(1)} `;
  }
  return d.trim();
})();

export function GradientDescent() {
  const reducedMotion = usePrefersReducedMotion();

  const [lr, setLr] = React.useState(LR_DEFAULT);
  const [w, setW] = React.useState(W_START);
  const [steps, setSteps] = React.useState(0);
  // Remaining auto-run iterations; an interval drains this to 0.
  const [runRemaining, setRunRemaining] = React.useState(0);

  // One gradient-descent update. `lr` is read from the latest state via a
  // functional updater + ref so Step/Run never apply a stale learning rate.
  const lrRef = React.useRef(lr);
  lrRef.current = lr;

  const stepOnce = React.useCallback(() => {
    setW((prev) => prev - lrRef.current * grad(prev));
    setSteps((s) => s + 1);
  }, []);

  const reset = React.useCallback(() => {
    setRunRemaining(0);
    setW(W_START);
    setSteps(0);
  }, []);

  // Auto-run: drain `runRemaining` one step per tick. Disabled entirely under
  // reduced motion (the Run button is hidden there, but guard anyway). The
  // interval is torn down on unmount and whenever the run finishes.
  React.useEffect(() => {
    if (reducedMotion || runRemaining <= 0) return;
    const id = window.setInterval(() => {
      stepOnce();
      setRunRemaining((r) => r - 1);
    }, RUN_MS);
    return () => window.clearInterval(id);
  }, [reducedMotion, runRemaining, stepOnce]);

  const running = runRemaining > 0;

  const lVal = loss(w);
  const gVal = grad(w);
  const m = 1 - 2 * A * lr;
  // "Diverging" the moment the weight has escaped the plotted bowl. (At the
  // exact threshold lr = 2 the weight oscillates on the rim forever; treat the
  // genuinely runaway case — outside the window — as diverging.)
  const outOfBowl = w < W_MIN || w > W_MAX;

  // Clamp the drawn marker so a runaway `w` can never paint off-canvas / NaN.
  const wDraw = clamp(w, W_MIN, W_MAX);
  const markerX = xPx(wDraw);
  const markerY = yPx(loss(wDraw));

  // Tangent segment at the current (clamped) weight: slope dL/dw in data units,
  // drawn as a short line centered on the marker. Convert the data-space slope
  // into pixel space (note y is inverted) so the drawn tangent is geometrically
  // honest against the bowl.
  const gDraw = grad(wDraw);
  const tangentHalfW = 1.1; // half-width of the tangent, in weight units
  const tx0 = wDraw - tangentHalfW;
  const tx1 = wDraw + tangentHalfW;
  const tl0 = loss(wDraw) + gDraw * (tx0 - wDraw);
  const tl1 = loss(wDraw) + gDraw * (tx1 - wDraw);
  // Clamp the tangent endpoints' loss to the plot so a steep slope stays inside.
  const tangent = {
    x1: xPx(tx0),
    y1: yPx(clamp(tl0, L_MIN, L_MAX)),
    x2: xPx(tx1),
    y2: yPx(clamp(tl1, L_MIN, L_MAX)),
  };

  // Preview of the NEXT step as a downhill arrow from the current marker to
  // where w would land (both clamped into the plot). Only meaningful while the
  // weight is still in the bowl.
  const wNext = w - lr * grad(w);
  const wNextDraw = clamp(wNext, W_MIN, W_MAX);
  const nextX = xPx(wNextDraw);
  const nextY = yPx(loss(wNextDraw));
  const showStepArrow = !outOfBowl && Math.abs(wNextDraw - wDraw) > 0.02;

  // Regime label for the readout — purely a function of lr (and thus m).
  // Boundary cases matter pedagogically: at lr = LR_DIVERGE (m = −1) the weight
  // does NOT diverge — it flips ±w0 forever (constant amplitude); only
  // lr > LR_DIVERGE genuinely blows up. lr = 0 takes no step at all.
  const regime: { label: string; tone: string } =
    lr <= LR_MIN
      ? { label: '', tone: 'text-muted-foreground' }
      : lr > LR_DIVERGE + 1e-9
        ? { label: 'too large → diverges out of the bowl', tone: 'text-amber-400' }
        : Math.abs(lr - LR_DIVERGE) <= 1e-9
          ? { label: 'knife-edge (lr = 2) → oscillates forever, never settles', tone: 'text-amber-400' }
          : m < 0
            ? { label: 'large → overshoots & zig-zags (still converges)', tone: 'text-foreground/80' }
            : Math.abs(m) > 0.9
              ? { label: 'tiny → crawls toward the bottom', tone: 'text-muted-foreground' }
              : { label: 'good → converges quickly', tone: 'text-emerald-400' };

  const minimumX = xPx(W_STAR);
  const minimumY = yPx(loss(W_STAR));

  const ariaLabel =
    `Loss bowl L(w) = ${A}·w². The weight is at ${w.toFixed(2)} after ${steps} ` +
    `step${steps === 1 ? '' : 's'}, where the loss is ${lVal.toFixed(2)} and the gradient is ` +
    `${gVal.toFixed(2)}. Learning rate ${lr.toFixed(2)}. ` +
    (outOfBowl ? 'The weight has diverged out of the bowl.' : `The minimum is at w = ${W_STAR}.`);

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">Gradient descent on one weight</div>
        <div className="text-[11px] text-muted-foreground">
          Step downhill: <span className="font-mono">w ← w − lr·L&apos;(w)</span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-[minmax(0,1fr)_auto]">
        {/* ---- The loss bowl ---- */}
        <svg
          role="img"
          aria-label={ariaLabel}
          viewBox={`0 0 ${W} ${H}`}
          className="block h-auto w-full rounded-md bg-muted/20"
        >
          <defs>
            <marker
              id="gd-step-head"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto"
            >
              <path d="M 0 0 L 10 5 L 0 10 Z" fill="oklch(0.72 0.18 350)" />
            </marker>
          </defs>

          {/* axes */}
          <line x1={PAD_L} y1={H - PAD_B} x2={W - PAD_R} y2={H - PAD_B} stroke="currentColor" strokeOpacity={0.4} />
          <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={H - PAD_B} stroke="currentColor" strokeOpacity={0.4} />

          {/* x ticks: w = −5 … 5 */}
          {[W_MIN, W_MIN / 2, W_STAR, W_MAX / 2, W_MAX].map((wt) => (
            <g key={`xt-${wt}`}>
              <line
                x1={xPx(wt)}
                y1={H - PAD_B}
                x2={xPx(wt)}
                y2={H - PAD_B + 4}
                stroke="currentColor"
                strokeOpacity={0.4}
              />
              <text
                x={xPx(wt)}
                y={H - PAD_B + 14}
                fontSize={9}
                textAnchor="middle"
                fill="currentColor"
                fillOpacity={0.55}
              >
                {wt}
              </text>
            </g>
          ))}
          <text
            x={PAD_L + PLOT_W / 2}
            y={H - 3}
            fontSize={9}
            textAnchor="middle"
            fill="currentColor"
            fillOpacity={0.55}
          >
            weight w
          </text>

          {/* y axis label (loss) */}
          <text
            x={PAD_L - 6}
            y={PAD_T + PLOT_H + 3}
            fontSize={9}
            textAnchor="end"
            fill="currentColor"
            fillOpacity={0.55}
          >
            0
          </text>
          <text x={PAD_L - 6} y={PAD_T + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.55}>
            {L_MAX}
          </text>
          <text
            x={11}
            y={PAD_T + PLOT_H / 2 + 3}
            fontSize={9}
            fill="currentColor"
            fillOpacity={0.55}
            transform={`rotate(-90, 11, ${PAD_T + PLOT_H / 2 + 3})`}
          >
            loss L(w)
          </text>

          {/* the convex loss bowl */}
          <path d={BOWL_D} fill="none" stroke="oklch(0.65 0.13 250)" strokeWidth={2} />

          {/* minimum marker at w = wStar */}
          <line
            x1={minimumX}
            y1={minimumY}
            x2={minimumX}
            y2={H - PAD_B}
            stroke="currentColor"
            strokeOpacity={0.18}
            strokeDasharray="3 3"
          />
          <circle cx={minimumX} cy={minimumY} r={3} fill="currentColor" fillOpacity={0.35} />
          <text x={minimumX} y={minimumY - 7} fontSize={9} textAnchor="middle" fill="currentColor" fillOpacity={0.5}>
            minimum
          </text>

          {/* tangent at the current weight — its slope IS the gradient */}
          <line
            x1={tangent.x1}
            y1={tangent.y1}
            x2={tangent.x2}
            y2={tangent.y2}
            stroke="oklch(0.80 0.13 80)"
            strokeWidth={2}
            strokeLinecap="round"
            className={cn(!reducedMotion && 'transition-all duration-150')}
          />

          {/* next-step arrow: from current w toward where it will land */}
          {showStepArrow ? (
            <line
              x1={markerX}
              y1={markerY}
              x2={nextX}
              y2={nextY}
              stroke="oklch(0.72 0.18 350)"
              strokeWidth={2}
              strokeLinecap="round"
              markerEnd="url(#gd-step-head)"
              className={cn(!reducedMotion && 'transition-all duration-150')}
            />
          ) : null}

          {/* current-weight marker, on top */}
          <circle
            cx={markerX}
            cy={markerY}
            r={5}
            fill="oklch(0.72 0.18 350)"
            stroke="var(--background)"
            strokeWidth={1.5}
            className={cn(!reducedMotion && 'transition-all duration-150')}
          />

          {outOfBowl ? (
            <text
              x={PAD_L + PLOT_W / 2}
              y={PAD_T + 14}
              fontSize={11}
              textAnchor="middle"
              fill="oklch(0.78 0.16 40)"
              fillOpacity={0.95}
            >
              diverging — w has left the bowl
            </text>
          ) : null}
        </svg>

        {/* ---- Controls + readouts ---- */}
        <div className="flex min-w-[220px] flex-col gap-3">
          <div className="space-y-1">
            <label htmlFor="gd-lr" className="block text-xs text-muted-foreground">
              learning rate <span className="font-mono text-foreground/85">lr = {lr.toFixed(2)}</span>
            </label>
            <input
              id="gd-lr"
              type="range"
              min={LR_MIN}
              max={LR_MAX}
              step={0.01}
              value={lr}
              onChange={(e) => setLr(Number(e.target.value))}
              className="w-full accent-primary"
              aria-valuetext={`learning rate ${lr.toFixed(2)}`}
            />
            <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
              <span>0</span>
              <span title="lands on the minimum in one step">{LR_CRITICAL.toFixed(1)} (1-step)</span>
              <span title="at lr = 2 the weight oscillates forever; past it the weight diverges">
                {LR_DIVERGE.toFixed(1)} (oscillate)
              </span>
            </div>
          </div>

          <div className="flex flex-wrap gap-1.5">
            <button
              type="button"
              onClick={stepOnce}
              disabled={running}
              className={cn(
                'rounded border border-border bg-muted/40 px-2.5 py-1 text-xs font-medium text-foreground/90',
                'transition-colors hover:bg-muted/70 disabled:opacity-40',
              )}
            >
              Step
            </button>
            {!reducedMotion ? (
              <button
                type="button"
                onClick={() => setRunRemaining(RUN_STEPS)}
                disabled={running}
                aria-pressed={running}
                className={cn(
                  'rounded border border-border bg-muted/40 px-2.5 py-1 text-xs font-medium text-foreground/90',
                  'transition-colors hover:bg-muted/70 disabled:opacity-40',
                )}
              >
                {running ? `Running… ${runRemaining}` : `Run ${RUN_STEPS}`}
              </button>
            ) : null}
            <button
              type="button"
              onClick={reset}
              className={cn(
                'rounded border border-border px-2.5 py-1 text-xs font-medium text-muted-foreground',
                'transition-colors hover:text-foreground',
              )}
            >
              Reset
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2" aria-live="polite">
            <div className="rounded-md border border-border/60 bg-muted/20 p-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">weight w</div>
              <div className="font-mono text-sm text-foreground/90">{w.toFixed(2)}</div>
            </div>
            <div className="rounded-md border border-border/60 bg-muted/20 p-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">loss L(w)</div>
              <div className="font-mono text-sm text-foreground/90">
                {Number.isFinite(lVal) ? lVal.toFixed(2) : '∞'}
              </div>
            </div>
            <div className="rounded-md border border-border/60 bg-muted/20 p-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">gradient dL/dw</div>
              <div className="font-mono text-sm text-foreground/90">
                {Number.isFinite(gVal) ? gVal.toFixed(2) : '∞'}
              </div>
            </div>
            <div className="rounded-md border border-border/60 bg-muted/20 p-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">steps taken</div>
              <div className="font-mono text-sm text-foreground/90">{steps}</div>
            </div>
          </div>

          {regime.label ? (
            <div className={cn('text-[12px] font-medium', regime.tone)} aria-live="polite">
              {regime.label}
            </div>
          ) : null}

          {outOfBowl ? (
            <div className="text-[12px] font-medium text-amber-400" aria-live="polite">
              diverging — lower the learning rate
            </div>
          ) : null}
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">
        The{' '}
        <span className="font-medium" style={{ color: 'oklch(0.80 0.13 80)' }}>
          gold tangent
        </span>{' '}
        is the gradient <span className="font-mono">L&apos;(w)</span> — its sign says which way is uphill, so the step
        (the{' '}
        <span className="font-medium" style={{ color: 'oklch(0.72 0.18 350)' }}>
          pink arrow
        </span>
        ) moves <span className="font-mono">w</span> the opposite way, downhill. A <strong>tiny</strong>{' '}
        <span className="font-mono">lr</span> barely moves and crawls; a <strong>good</strong> one drops to the bottom
        in a few steps; push <span className="font-mono">lr</span> past{' '}
        <span className="font-mono">{LR_DIVERGE.toFixed(1)}</span> and each step{' '}
        <strong>overshoots farther than the last</strong> — the weight bounces out of the bowl and the loss explodes.
      </p>

      <p className="text-[10px] text-muted-foreground">
        Illustrative — a 1-parameter cartoon. A real model has ~800M weights, and the same rule (step each weight
        downhill by its own gradient) runs on all of them at once — which is exactly what AdamW automates per parameter.
      </p>
    </div>
  );
}
