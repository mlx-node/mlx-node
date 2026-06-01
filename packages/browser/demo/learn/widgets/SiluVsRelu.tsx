import * as React from 'react';

import { MathDisplay } from '../scaffolding/MathDisplay';

/**
 * Chapter 8 supplement — why the SwiGLU MLP uses SiLU instead of ReLU.
 *
 * SiLU(z) = z·σ(z) is a smooth "soft gate": it lets a little negative signal
 * through (dipping to a minimum of ≈ -0.278 near z ≈ -1.278) and has no hard
 * kink, so gradients keep flowing for small negative inputs. ReLU hard-zeros
 * every negative, killing the gradient there. Both curves are sampled from
 * their exact formulas — no model, no worker.
 *
 * Qwen3.5-0.8B's per-layer MLP is a dense SwiGLU FFN whose activation is SiLU.
 */

const Z_MIN = -6;
const Z_MAX = 6;
const SAMPLES = 161;
const Y_MIN = -0.4;
const Y_MAX = 6;

const WIDTH = 520;
const HEIGHT = 220;
const PAD_LEFT = 36;
const PAD_RIGHT = 10;
const PAD_TOP = 12;
const PAD_BOTTOM = 28;
const INNER_W = WIDTH - PAD_LEFT - PAD_RIGHT;
const INNER_H = HEIGHT - PAD_TOP - PAD_BOTTOM;

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}
function silu(z: number): number {
  return z * sigmoid(z);
}
function relu(z: number): number {
  return Math.max(0, z);
}

function xFor(z: number): number {
  return PAD_LEFT + ((z - Z_MIN) / (Z_MAX - Z_MIN)) * INNER_W;
}
function yFor(v: number): number {
  return PAD_TOP + INNER_H - ((v - Y_MIN) / (Y_MAX - Y_MIN)) * INNER_H;
}

const SAMPLE_ZS = Array.from({ length: SAMPLES }, (_, i) => Z_MIN + (i / (SAMPLES - 1)) * (Z_MAX - Z_MIN));

function pathFor(fn: (z: number) => number): string {
  return 'M ' + SAMPLE_ZS.map((z) => `${xFor(z).toFixed(2)} ${yFor(fn(z)).toFixed(2)}`).join(' L ');
}

function formatSigned(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return '—';
  return value.toFixed(digits);
}

export function SiluVsRelu() {
  const [z, setZ] = React.useState(-1);

  const siluPath = React.useMemo(() => pathFor(silu), []);
  const reluPath = React.useMemo(() => pathFor(relu), []);

  const siluZ = silu(z);
  const reluZ = relu(z);
  const gap = siluZ - reluZ;
  const isNegative = z < 0;

  const markerX = xFor(z);
  const siluY = yFor(siluZ);
  const reluY = yFor(reluZ);
  const zeroY = yFor(0);
  const axisX = xFor(0);

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">SiLU vs ReLU — the soft gate</div>
        <div className="text-[11px] text-muted-foreground">z ∈ [−6, 6]</div>
      </div>

      <MathDisplay latex={String.raw`\mathrm{SiLU}(z)=z\,\sigma(z)=\dfrac{z}{1+e^{-z}}`} />

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="SiLU and ReLU activation curves over z from negative six to six. SiLU is a smooth curve that dips slightly negative near z = -1.278 before rising; ReLU is hard-zero for negative z and linear for positive z."
        className="block h-auto w-full"
      >
        {/* y = 0 axis */}
        <line x1={PAD_LEFT} x2={WIDTH - PAD_RIGHT} y1={zeroY} y2={zeroY} stroke="currentColor" strokeOpacity={0.18} />
        {/* x = 0 axis */}
        <line x1={axisX} x2={axisX} y1={PAD_TOP} y2={PAD_TOP + INNER_H} stroke="currentColor" strokeOpacity={0.12} />

        {/* dashed vertical marker at current z */}
        <line
          x1={markerX}
          x2={markerX}
          y1={PAD_TOP}
          y2={PAD_TOP + INNER_H}
          stroke="currentColor"
          strokeOpacity={0.25}
          strokeDasharray="3 3"
        />
        <text x={markerX + 3} y={PAD_TOP + 9} fontSize={9} fill="currentColor" fillOpacity={0.55}>
          z = {z.toFixed(1)}
        </text>

        {/* curves */}
        <path d={reluPath} fill="none" className="stroke-muted-foreground" strokeWidth={1.5} strokeDasharray="5 3" />
        <path d={siluPath} fill="none" className="stroke-primary" strokeWidth={1.8} />

        {/* dots at current z */}
        <circle cx={markerX} cy={reluY} r={3} className="fill-muted-foreground" />
        <circle cx={markerX} cy={siluY} r={3.5} className="fill-primary" />

        {/* axis labels */}
        <text x={PAD_LEFT} y={HEIGHT - 8} fontSize={9} fill="currentColor" fillOpacity={0.55}>
          z = {Z_MIN}
        </text>
        <text x={axisX} y={HEIGHT - 8} fontSize={9} textAnchor="middle" fill="currentColor" fillOpacity={0.55}>
          0
        </text>
        <text x={WIDTH - PAD_RIGHT} y={HEIGHT - 8} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.55}>
          z = {Z_MAX}
        </text>
        <text x={PAD_LEFT - 4} y={yFor(Y_MAX) + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.55}>
          {Y_MAX}
        </text>
        <text x={PAD_LEFT - 4} y={zeroY + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.55}>
          0
        </text>
      </svg>

      {/* legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <LegendDot className="stroke-primary">SiLU</LegendDot>
        <LegendDot className="stroke-muted-foreground" dashed>
          ReLU
        </LegendDot>
      </div>

      {/* z slider */}
      <div className="space-y-1">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <label htmlFor="silu-vs-relu-z" className="uppercase tracking-wider">
            Input z
          </label>
          <span className="font-mono text-foreground/80">z = {z.toFixed(1)}</span>
        </div>
        <input
          id="silu-vs-relu-z"
          type="range"
          min={Z_MIN}
          max={Z_MAX}
          step={0.1}
          value={z}
          onChange={(e) => setZ(Number(e.target.value))}
          className="w-full accent-primary"
        />
      </div>

      {/* readout */}
      <div className="grid grid-cols-3 gap-2 text-[12px]">
        <Readout label="SiLU(z)" value={formatSigned(siluZ)} accent />
        <Readout label="ReLU(z)" value={formatSigned(reluZ)} />
        <Readout label="gap (SiLU − ReLU)" value={formatSigned(gap)} />
      </div>

      <div
        className={[
          'rounded-md border p-2 text-[12px]',
          isNegative
            ? 'border-primary/50 bg-primary/5 text-foreground/90'
            : 'border-border bg-muted/20 text-foreground/85',
        ].join(' ')}
      >
        {isNegative ? (
          <>
            At <span className="font-mono">z = {z.toFixed(1)}</span> (negative), ReLU hard-zeros to{' '}
            <span className="font-mono">0.000</span>, but SiLU leaks{' '}
            <span className="font-mono text-primary">{formatSigned(siluZ)}</span> — the <strong>soft gate</strong> still
            passes a little signal.
          </>
        ) : (
          <>
            At <span className="font-mono">z = {z.toFixed(1)}</span> (non-negative), both curves are close — but SiLU
            curves smoothly through the origin while ReLU has a hard kink at <span className="font-mono">z = 0</span>.
          </>
        )}
      </div>

      <p className="text-[12px] text-foreground/85">
        SiLU is smooth and dips slightly negative near <span className="font-mono">z = 0</span> (minimum ≈{' '}
        <span className="font-mono">−0.278</span> at <span className="font-mono">z ≈ −1.278</span>) — a soft gate — so
        gradients keep flowing for small negative inputs, unlike ReLU&apos;s hard zero. Qwen3.5-0.8B uses SiLU inside
        its SwiGLU MLP.
      </p>

      <p className="text-[10px] text-muted-foreground">
        Illustrative — SiLU and ReLU are plotted from their exact formulas; Qwen3.5-0.8B&apos;s MLP uses SiLU. Not live
        output from the model.
      </p>
    </div>
  );
}

function LegendDot({
  className,
  dashed = false,
  children,
}: {
  className: string;
  dashed?: boolean;
  children: React.ReactNode;
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <svg width={18} height={6} aria-hidden="true">
        <line
          x1={0}
          x2={18}
          y1={3}
          y2={3}
          className={className}
          strokeWidth={2}
          strokeDasharray={dashed ? '4 3' : undefined}
        />
      </svg>
      <span>{children}</span>
    </span>
  );
}

function Readout({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded-md border border-border bg-background p-2">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className={['font-mono text-sm', accent ? 'text-primary' : 'text-foreground/85'].join(' ')}>{value}</div>
    </div>
  );
}
