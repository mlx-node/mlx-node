import { cn } from '@/lib/utils';
import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

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

// Per-locale copy — every user-visible English string moved here verbatim.
// Math identifiers (w, lr, L(w), dL/dw) and numbers stay as-is in both locales.
const COPY = {
  en: {
    title: 'Gradient descent on one weight',
    stepDownhill: 'Step downhill:',
    regimeDiverge: 'too large → diverges out of the bowl',
    regimeKnifeEdge: 'knife-edge (lr = 2) → oscillates forever, never settles',
    regimeOvershoot: 'large → overshoots & zig-zags (still converges)',
    regimeTiny: 'tiny → crawls toward the bottom',
    regimeGood: 'good → converges quickly',
    ariaLabel: (w: number, steps: number, lVal: number, gVal: number, lr: number, outOfBowl: boolean) =>
      `Loss bowl L(w) = ${A}·w². The weight is at ${w.toFixed(2)} after ${steps} ` +
      `step${steps === 1 ? '' : 's'}, where the loss is ${lVal.toFixed(2)} and the gradient is ` +
      `${gVal.toFixed(2)}. Learning rate ${lr.toFixed(2)}. ` +
      (outOfBowl ? 'The weight has diverged out of the bowl.' : `The minimum is at w = ${W_STAR}.`),
    xAxis: 'weight w',
    yAxis: 'loss L(w)',
    minimum: 'minimum',
    divergingPlot: 'diverging — w has left the bowl',
    lrLabel: 'learning rate',
    lrValueText: (v: string) => `learning rate ${v}`,
    criticalTitle: 'lands on the minimum in one step',
    criticalLabel: (v: string) => `${v} (1-step)`,
    divergeTitle: 'at lr = 2 the weight oscillates forever; past it the weight diverges',
    divergeLabel: (v: string) => `${v} (oscillate)`,
    stepBtn: 'Step',
    runBtn: (n: number) => `Run ${n}`,
    runningBtn: (r: number) => `Running… ${r}`,
    resetBtn: 'Reset',
    readoutWeight: 'weight w',
    readoutLoss: 'loss L(w)',
    readoutGrad: 'gradient dL/dw',
    readoutSteps: 'steps taken',
    divergingNote: 'diverging — lower the learning rate',
    mainNote: (
      <>
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
      </>
    ),
    illustrative: (
      <>
        Illustrative — a 1-parameter cartoon. A real model has ~800M weights, and the same rule (step each weight
        downhill by its own gradient) runs on all of them at once — which is exactly what AdamW automates per parameter.
      </>
    ),
  },
  zh: {
    title: '单个权重上的梯度下降',
    stepDownhill: '朝下坡迈步：',
    regimeDiverge: '太大 → 冲出碗外发散',
    regimeKnifeEdge: '临界刀锋（lr = 2）→ 永远振荡，永不安顿',
    regimeOvershoot: '偏大 → 越过头、来回拉锯（仍会收敛）',
    regimeTiny: '太小 → 朝谷底缓慢爬行',
    regimeGood: '合适 → 快速收敛',
    ariaLabel: (w: number, steps: number, lVal: number, gVal: number, lr: number, outOfBowl: boolean) =>
      `损失碗 L(w) = ${A}·w²。走了 ${steps} 步后，权重位于 ${w.toFixed(2)}，此处损失为 ${lVal.toFixed(2)}，梯度为 ${gVal.toFixed(2)}。学习率 ${lr.toFixed(2)}。` +
      (outOfBowl ? '权重已发散出碗外。' : `极小值在 w = ${W_STAR} 处。`),
    xAxis: '权重 w',
    yAxis: '损失 L(w)',
    minimum: '极小值',
    divergingPlot: '发散中——w 已离开碗外',
    lrLabel: '学习率',
    lrValueText: (v: string) => `学习率 ${v}`,
    criticalTitle: '一步正好落在极小值上',
    criticalLabel: (v: string) => `${v}（一步）`,
    divergeTitle: 'lr = 2 时权重永远振荡；超过它权重就发散',
    divergeLabel: (v: string) => `${v}（振荡）`,
    stepBtn: '单步',
    runBtn: (n: number) => `运行 ${n} 步`,
    runningBtn: (r: number) => `运行中… ${r}`,
    resetBtn: '重置',
    readoutWeight: '权重 w',
    readoutLoss: '损失 L(w)',
    readoutGrad: '梯度 dL/dw',
    readoutSteps: '已走步数',
    divergingNote: '发散中——请调低学习率',
    mainNote: (
      <>
        <span className="font-medium" style={{ color: 'oklch(0.80 0.13 80)' }}>
          金色切线
        </span>
        就是梯度 <span className="font-mono">L&apos;(w)</span>
        ——它的符号指出哪边是上坡，所以这一步（
        <span className="font-medium" style={{ color: 'oklch(0.72 0.18 350)' }}>
          粉色箭头
        </span>
        ）让 <span className="font-mono">w</span> 朝相反方向、也就是下坡移动。<strong>太小</strong>的{' '}
        <span className="font-mono">lr</span> 几乎不动、缓慢爬行；<strong>合适</strong>的几步就落到谷底；把{' '}
        <span className="font-mono">lr</span> 推过 <span className="font-mono">{LR_DIVERGE.toFixed(1)}</span>
        ，每一步都会<strong>比上一步越得更远</strong>——权重弹出碗外，损失爆炸。
      </>
    ),
    illustrative: (
      <>
        仅为示意——一幅只有 1 个参数的卡通。真实模型约有 800M
        个权重，同一条规则（每个权重沿自己的梯度往下坡迈一步）在所有权重上同时运行——这正是 AdamW
        逐参数自动化的事情。
      </>
    ),
  },
} as const;

export function GradientDescent() {
  const copy = COPY[useLocale()];
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
        ? { label: copy.regimeDiverge, tone: 'text-amber-400' }
        : Math.abs(lr - LR_DIVERGE) <= 1e-9
          ? { label: copy.regimeKnifeEdge, tone: 'text-amber-400' }
          : m < 0
            ? { label: copy.regimeOvershoot, tone: 'text-foreground/80' }
            : Math.abs(m) > 0.9
              ? { label: copy.regimeTiny, tone: 'text-muted-foreground' }
              : { label: copy.regimeGood, tone: 'text-emerald-400' };

  const minimumX = xPx(W_STAR);
  const minimumY = yPx(loss(W_STAR));

  const ariaLabel = copy.ariaLabel(w, steps, lVal, gVal, lr, outOfBowl);

  return (
    <div className="@container not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="text-[11px] text-muted-foreground">
          {copy.stepDownhill} <span className="font-mono">w ← w − lr·L&apos;(w)</span>
        </div>
      </div>

      {/*
        `@2xl`, not `md:` — and the `@container` that answers it is on the shell
        div above. This widget renders in chapters that have a live right-hand
        pane, so its column is 425px wide even at a 1280px viewport: a VIEWPORT
        breakpoint says "wide" while the CONTAINER is not. With `md:` the
        220px-min controls took 232px and squeezed the plot to 181px — viewBox
        540 at scale 0.335, every axis label at 3.0 CSS px. `@2xl` (42rem) is
        measured against the shell, so the two columns split only when the
        column itself has room for them.
      */}
      <div className="grid grid-cols-1 gap-3 @2xl:grid-cols-[minmax(0,1fr)_auto]">
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
            {copy.xAxis}
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
            {copy.yAxis}
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
            {copy.minimum}
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
              {copy.divergingPlot}
            </text>
          ) : null}
        </svg>

        {/* ---- Controls + readouts ---- */}
        <div className="flex min-w-[220px] flex-col gap-3">
          <div className="space-y-1">
            <label htmlFor="gd-lr" className="block text-xs text-muted-foreground">
              {copy.lrLabel} <span className="font-mono text-foreground/85">lr = {lr.toFixed(2)}</span>
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
              aria-valuetext={copy.lrValueText(lr.toFixed(2))}
            />
            <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
              <span>0</span>
              <span title={copy.criticalTitle}>{copy.criticalLabel(LR_CRITICAL.toFixed(1))}</span>
              <span title={copy.divergeTitle}>{copy.divergeLabel(LR_DIVERGE.toFixed(1))}</span>
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
              {copy.stepBtn}
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
                {running ? copy.runningBtn(runRemaining) : copy.runBtn(RUN_STEPS)}
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
              {copy.resetBtn}
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2" aria-live="polite">
            <div className="rounded-md border border-border/60 bg-muted/20 p-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.readoutWeight}</div>
              <div className="font-mono text-sm text-foreground/90">{w.toFixed(2)}</div>
            </div>
            <div className="rounded-md border border-border/60 bg-muted/20 p-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.readoutLoss}</div>
              <div className="font-mono text-sm text-foreground/90">
                {Number.isFinite(lVal) ? lVal.toFixed(2) : '∞'}
              </div>
            </div>
            <div className="rounded-md border border-border/60 bg-muted/20 p-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.readoutGrad}</div>
              <div className="font-mono text-sm text-foreground/90">
                {Number.isFinite(gVal) ? gVal.toFixed(2) : '∞'}
              </div>
            </div>
            <div className="rounded-md border border-border/60 bg-muted/20 p-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.readoutSteps}</div>
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
              {copy.divergingNote}
            </div>
          ) : null}
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">{copy.mainNote}</p>

      <p className="text-[10px] text-muted-foreground">{copy.illustrative}</p>
    </div>
  );
}
