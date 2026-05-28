import * as React from 'react';

/**
 * Chapter 13 (Scaling) supplement — visualize a learning-rate schedule.
 *
 * Almost every LLM pretraining recipe uses linear warmup followed by cosine
 * decay. The shapes are:
 *
 *   step t in [0, W]:    lr(t) = lr_max * t / W                    (warmup)
 *   step t in [W, T]:    lr(t) = lr_min + 0.5 * (lr_max - lr_min)
 *                                       * (1 + cos(pi * (t-W)/(T-W)))   (cosine)
 *
 * Sliders for warmup steps and peak LR. Total steps fixed at 10,000 for
 * tractable on-screen scale. The plot animates a cursor along the curve so
 * the user can watch the current LR change with the slider.
 */

const TOTAL_STEPS = 10_000;
const LR_MIN = 1e-5;

export function LrScheduleViz() {
  const [lrMax, setLrMax] = React.useState(3e-4);
  const [warmupSteps, setWarmupSteps] = React.useState(2_000);
  const [cursor, setCursor] = React.useState(2_500);

  function lrAt(step: number): number {
    if (step <= warmupSteps) {
      return (lrMax * step) / Math.max(1, warmupSteps);
    }
    const t = (step - warmupSteps) / Math.max(1, TOTAL_STEPS - warmupSteps);
    return LR_MIN + 0.5 * (lrMax - LR_MIN) * (1 + Math.cos(Math.PI * t));
  }

  const W = 540;
  const H = 200;
  const PAD = 30;
  const plotW = W - PAD * 2;
  const plotH = H - PAD * 2;

  // Sample 200 points for a smooth curve. y is scaled relative to lrMax so
  // the curve uses the full plot height regardless of slider.
  const samples = React.useMemo(() => {
    const n = 200;
    const pts: { x: number; y: number; step: number; lr: number }[] = [];
    for (let i = 0; i <= n; i++) {
      const step = (i / n) * TOTAL_STEPS;
      const lr = lrAt(step);
      const x = PAD + (step / TOTAL_STEPS) * plotW;
      const y = PAD + plotH - (lr / lrMax) * plotH;
      pts.push({ x, y, step, lr });
    }
    return pts;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lrMax, warmupSteps]);

  const pathD = samples.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');

  const cursorX = PAD + (cursor / TOTAL_STEPS) * plotW;
  const cursorLr = lrAt(cursor);
  const cursorY = PAD + plotH - (cursorLr / lrMax) * plotH;

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">
        Learning-rate schedule — linear warmup + cosine decay
      </div>

      <p className="text-[12px] text-foreground/85">
        The standard recipe: linear warmup for ~10% of steps from <code>0</code> to <code>lr_max</code>, then cosine
        decay down to ~<code>1e-5</code>. The warmup keeps early gradients from blowing up while the optimizer's moving
        averages haven't filled in yet; the decay lets late training settle into a low-loss plateau.
      </p>

      <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label="Learning-rate schedule">
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
        {/* y ticks at 0, max */}
        <text x={PAD - 4} y={PAD + plotH + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.55}>
          0
        </text>
        <text x={PAD - 4} y={PAD + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.55}>
          {lrMax.toExponential(0)}
        </text>
        <text
          x={10}
          y={PAD + plotH / 2 + 3}
          fontSize={9}
          fill="currentColor"
          fillOpacity={0.55}
          transform={`rotate(-90, 10, ${PAD + plotH / 2 + 3})`}
        >
          learning rate
        </text>

        {/* warmup region shading */}
        <rect
          x={PAD}
          y={PAD}
          width={(warmupSteps / TOTAL_STEPS) * plotW}
          height={plotH}
          fill="oklch(0.7 0.15 60)"
          fillOpacity={0.08}
        />
        <text
          x={PAD + (warmupSteps / TOTAL_STEPS) * plotW * 0.5}
          y={PAD + 14}
          fontSize={10}
          textAnchor="middle"
          fill="oklch(0.7 0.15 60)"
          fillOpacity={0.9}
        >
          warmup
        </text>
        <text
          x={PAD + (warmupSteps / TOTAL_STEPS) * plotW + ((TOTAL_STEPS - warmupSteps) / TOTAL_STEPS) * plotW * 0.5}
          y={PAD + 14}
          fontSize={10}
          textAnchor="middle"
          fill="oklch(0.65 0.13 250)"
          fillOpacity={0.9}
        >
          cosine decay
        </text>

        {/* curve */}
        <path d={pathD} fill="none" stroke="oklch(0.65 0.13 250)" strokeWidth={2} />

        {/* cursor */}
        <line
          x1={cursorX}
          y1={PAD}
          x2={cursorX}
          y2={H - PAD}
          stroke="currentColor"
          strokeOpacity={0.4}
          strokeDasharray="3 3"
        />
        <circle cx={cursorX} cy={cursorY} r={5} fill="oklch(0.7 0.15 60)" />
      </svg>

      <div className="space-y-2">
        <label className="block text-[11px]">
          <div className="mb-0.5 flex justify-between font-mono text-muted-foreground">
            <span>peak LR (lr_max)</span>
            <span>{lrMax.toExponential(1)}</span>
          </div>
          <input
            type="range"
            min={Math.log10(1e-5)}
            max={Math.log10(3e-3)}
            step={0.05}
            value={Math.log10(lrMax)}
            onChange={(e) => setLrMax(Math.pow(10, Number(e.target.value)))}
            className="w-full"
          />
        </label>
        <label className="block text-[11px]">
          <div className="mb-0.5 flex justify-between font-mono text-muted-foreground">
            <span>warmup steps ({((warmupSteps / TOTAL_STEPS) * 100).toFixed(0)}% of total)</span>
            <span>{warmupSteps.toLocaleString()}</span>
          </div>
          <input
            type="range"
            min={0}
            max={TOTAL_STEPS / 2}
            step={50}
            value={warmupSteps}
            onChange={(e) => setWarmupSteps(Number(e.target.value))}
            className="w-full"
          />
        </label>
        <label className="block text-[11px]">
          <div className="mb-0.5 flex justify-between font-mono text-muted-foreground">
            <span>step cursor</span>
            <span>
              step {cursor.toLocaleString()} · lr = {cursorLr.toExponential(2)}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={TOTAL_STEPS}
            step={50}
            value={cursor}
            onChange={(e) => setCursor(Number(e.target.value))}
            className="w-full"
          />
        </label>
      </div>

      <p className="text-[11px] text-muted-foreground">
        Slide warmup to 0 and watch the curve start at <code>lr_max</code> — that's what happens without warmup. Slide
        the peak LR much higher (e.g. <code>1e-3</code>) and you can see why early training would diverge without
        gradient clipping: <em>any</em> large step on a fresh model takes the weights somewhere they can't recover from.
      </p>
    </div>
  );
}
