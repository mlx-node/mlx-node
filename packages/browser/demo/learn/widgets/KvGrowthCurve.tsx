import * as React from "react";

/**
 * Small SVG line chart showing KV-cache memory as a function of context
 * length, log-scaled on X. Three curves — MHA hypothetical, GQA every
 * layer, and Qwen3.5's actual hybrid (6 full + 18 linear constant state).
 *
 * All numbers are computed in this file from the same model config that
 * chapter 10's main chart uses, kept local so the widget is self-contained.
 */

const NUM_LAYERS = 24;
const NUM_HEADS = 8;
const NUM_KV_HEADS = 2;
const HEAD_DIM = 256;
const BYTES_PER_FLOAT = 2;
const FULL_LAYER_INTERVAL = 4;
const NUM_FULL_LAYERS = NUM_LAYERS / FULL_LAYER_INTERVAL; // 6
const NUM_LINEAR_LAYERS = NUM_LAYERS - NUM_FULL_LAYERS; // 18
const LINEAR_NUM_HEADS = 16;
const LINEAR_HEAD_DIM = 128;
const LINEAR_STATE_BYTES =
  LINEAR_NUM_HEADS * LINEAR_HEAD_DIM * LINEAR_HEAD_DIM * BYTES_PER_FLOAT;

const X_TICKS = [1024, 4096, 16384, 65536, 262144];
const X_MIN = X_TICKS[0]!;
const X_MAX = X_TICKS[X_TICKS.length - 1]!;

function mhaBytes(seqLen: number): number {
  return NUM_LAYERS * 2 * NUM_HEADS * HEAD_DIM * seqLen * BYTES_PER_FLOAT;
}
function gqaBytes(seqLen: number): number {
  return NUM_LAYERS * 2 * NUM_KV_HEADS * HEAD_DIM * seqLen * BYTES_PER_FLOAT;
}
function hybridBytes(seqLen: number): number {
  return (
    NUM_FULL_LAYERS * 2 * NUM_KV_HEADS * HEAD_DIM * seqLen * BYTES_PER_FLOAT +
    NUM_LINEAR_LAYERS * LINEAR_STATE_BYTES
  );
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return `${n}`;
}

const SAMPLES = 64;
function logSpace(): number[] {
  const lo = Math.log10(X_MIN);
  const hi = Math.log10(X_MAX);
  return Array.from({ length: SAMPLES }, (_, i) => {
    const t = i / (SAMPLES - 1);
    return Math.round(10 ** (lo + t * (hi - lo)));
  });
}

const WIDTH = 560;
const HEIGHT = 240;
const PAD_LEFT = 60;
const PAD_RIGHT = 14;
const PAD_TOP = 16;
const PAD_BOTTOM = 32;
const INNER_W = WIDTH - PAD_LEFT - PAD_RIGHT;
const INNER_H = HEIGHT - PAD_TOP - PAD_BOTTOM;

export function KvGrowthCurve() {
  const samples = React.useMemo(logSpace, []);
  const yMaxBytes = mhaBytes(X_MAX);
  const yMinBytes = Math.max(1024, hybridBytes(X_MIN));
  const logYMin = Math.log10(yMinBytes);
  const logYMax = Math.log10(yMaxBytes);
  const logXMin = Math.log10(X_MIN);
  const logXMax = Math.log10(X_MAX);

  function xFor(seqLen: number): number {
    const t = (Math.log10(Math.max(X_MIN, seqLen)) - logXMin) / (logXMax - logXMin);
    return PAD_LEFT + Math.max(0, Math.min(1, t)) * INNER_W;
  }
  function yFor(bytes: number): number {
    const clamped = Math.max(1024, bytes);
    const t = (Math.log10(clamped) - logYMin) / (logYMax - logYMin);
    return PAD_TOP + INNER_H - Math.max(0, Math.min(1, t)) * INNER_H;
  }

  function pathFor(fn: (n: number) => number): string {
    return (
      "M " +
      samples
        .map((n) => `${xFor(n).toFixed(2)} ${yFor(fn(n)).toFixed(2)}`)
        .join(" L ")
    );
  }

  const mhaPath = pathFor(mhaBytes);
  const gqaPath = pathFor(gqaBytes);
  const hybridPath = pathFor(hybridBytes);

  const yTicks: number[] = [];
  for (let exp = Math.ceil(logYMin); exp <= Math.floor(logYMax); exp++) {
    yTicks.push(10 ** exp);
  }

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          KV cache growth · log-log
        </div>
        <div className="text-[11px] text-muted-foreground">
          Whole-model bytes vs context length
        </div>
      </div>

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Three KV-cache memory curves vs context length, log-log scale: MHA hypothetical, GQA every layer, and Qwen3.5 hybrid."
        className="block h-auto w-full"
      >
        {yTicks.map((b, idx) => {
          const y = yFor(b);
          return (
            <g key={`y-${idx}`}>
              <line
                x1={PAD_LEFT}
                x2={WIDTH - PAD_RIGHT}
                y1={y}
                y2={y}
                stroke="currentColor"
                strokeOpacity={0.07}
              />
              <text
                x={PAD_LEFT - 6}
                y={y + 3}
                fontSize={9}
                textAnchor="end"
                fill="currentColor"
                fillOpacity={0.55}
              >
                {formatBytes(b)}
              </text>
            </g>
          );
        })}
        {X_TICKS.map((t, idx) => {
          const x = xFor(t);
          return (
            <g key={`x-${idx}`}>
              <line
                x1={x}
                x2={x}
                y1={PAD_TOP}
                y2={PAD_TOP + INNER_H}
                stroke="currentColor"
                strokeOpacity={0.07}
              />
              <text
                x={x}
                y={HEIGHT - 14}
                fontSize={9}
                textAnchor="middle"
                fill="currentColor"
                fillOpacity={0.55}
              >
                {formatTokens(t)}
              </text>
            </g>
          );
        })}

        <text
          x={(PAD_LEFT + WIDTH - PAD_RIGHT) / 2}
          y={HEIGHT - 2}
          fontSize={10}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.55}
        >
          context length (tokens, log scale)
        </text>

        <path
          d={mhaPath}
          fill="none"
          stroke="#94a3b8"
          strokeWidth={1.5}
          strokeDasharray="4 3"
        />
        <path d={gqaPath} fill="none" stroke="#f59e0b" strokeWidth={1.8} />
        <path d={hybridPath} fill="none" stroke="#22c55e" strokeWidth={2} />
      </svg>

      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <LegendDot color="#94a3b8" dashed>
          MHA hypothetical
        </LegendDot>
        <LegendDot color="#f59e0b">GQA every layer</LegendDot>
        <LegendDot color="#22c55e">Hybrid (Qwen3.5 actual)</LegendDot>
      </div>

      <div className="rounded-md border border-border/60 bg-muted/20 p-2 font-mono text-[11px] leading-relaxed text-foreground/85">
        <code className="bg-transparent p-0">
          bytes_per_full_layer = 2 · kv_heads · d · seq · bf16
          {"\n"}
          {"             "}= 2 · {NUM_KV_HEADS} · {HEAD_DIM} · seq · {BYTES_PER_FLOAT}
          {"\n"}
          bytes_per_linear_layer = constant ({formatBytes(LINEAR_STATE_BYTES)})
          {"\n"}
          total = {NUM_FULL_LAYERS} · bytes_per_full_layer + {NUM_LINEAR_LAYERS} ·
          bytes_per_linear_layer
        </code>
      </div>

      <div className="space-y-1 text-[11px] text-muted-foreground">
        <div>
          <span className="font-mono text-foreground/80">2</span> — keys and
          values are stored separately.
        </div>
        <div>
          <span className="font-mono text-foreground/80">kv_heads</span> — the
          {" "}
          number of K/V heads per layer (after GQA grouping).
        </div>
        <div>
          <span className="font-mono text-foreground/80">d</span> — the per-head
          dimension <code>head_dim</code>.
        </div>
        <div>
          <span className="font-mono text-foreground/80">seq</span> — context
          length in tokens (the X axis above).
        </div>
        <div>
          <span className="font-mono text-foreground/80">bf16</span> — 2 bytes
          per float in the cache.
        </div>
      </div>
    </div>
  );
}

function LegendDot({
  color,
  dashed = false,
  children,
}: {
  color: string;
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
          stroke={color}
          strokeWidth={2}
          strokeDasharray={dashed ? "4 3" : undefined}
        />
      </svg>
      <span>{children}</span>
    </span>
  );
}
