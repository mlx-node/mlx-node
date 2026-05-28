import * as React from 'react';

import { Button } from '../../components/ui/button';

/**
 * Chapter 4 supplement — visualize the multi-head concat step.
 *
 *   Z_0  Z_1  ...  Z_{H-1}   →   concat → wide Z   ×   W_O   →   Z_attention
 *
 * Each head produces a [seq, d_head] output. The H outputs are stacked
 * side-by-side along the feature axis to form a [seq, d_head*H] matrix,
 * then projected by W_O back to [seq, d_model]. This widget shows that
 * sequence as colored columns sliding into a single block.
 *
 * Inspired by the encoder-s4 concat panel in An-Jhon/Hand-Drawn-Transformer.
 *
 * Pure JS, no model data — the point is the *shape* of the operation, not
 * the values. We use H=8 (Qwen3.5-0.8B's actual head count) and a short
 * fixed seq_len so the diagram fits horizontally.
 */

const H = 8; // num heads (matches Qwen3.5-0.8B)
const D_HEAD = 64;
const SEQ_LEN = 5;
const D_MODEL = H * D_HEAD; // 512

// One color per head — rotate around an oklch hue wheel so neighbours are
// always distinguishable but the palette stays low-saturation enough not to
// overwhelm the page.
function headColor(i: number, alpha = 0.7): string {
  const hue = (i * 360) / H;
  return `oklch(0.7 0.13 ${hue} / ${alpha})`;
}

export function MultiheadConcat() {
  // Replay key — bumping it remounts the animated svg so the slide-in
  // restarts. Lets the user re-play the animation on demand.
  const [playKey, setPlayKey] = React.useState(0);

  const W = 560;
  const totalH = 280;

  // Top row: H individual head outputs spread across the width.
  const HEAD_W = 38;
  const HEAD_H = 70;
  const TOP_Y = 28;
  const headGap = (W - 40 - HEAD_W * H) / Math.max(1, H - 1);
  function headX(i: number): number {
    return 20 + i * (HEAD_W + headGap);
  }

  // Bottom row: concat (wide) → W_O → Z_attention.
  const CONCAT_X = 20;
  const CONCAT_Y = 168;
  const CONCAT_W = HEAD_W * H + 6; // narrower than top so heads visibly converge
  const CONCAT_H = 70;

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Concat heads → project with W<sub>O</sub>
        </div>
        <Button size="sm" variant="outline" onClick={() => setPlayKey((k) => k + 1)} className="text-[11px]">
          Replay animation
        </Button>
      </div>

      <p className="text-[12px] text-foreground/85">
        Each of the <span className="font-mono">{H}</span> heads produces a slice of shape{' '}
        <span className="font-mono">[seq_len, d_head] = [{SEQ_LEN}, {D_HEAD}]</span>. We stack them side-by-side along
        the feature axis to recover a <span className="font-mono">[{SEQ_LEN}, {D_MODEL}]</span> matrix, then project by{' '}
        the learned output matrix <span className="font-mono">W<sub>O</sub></span> back to{' '}
        <span className="font-mono">d_model</span>. Heads don't talk to each other inside attention — they only mix
        afterward, here.
      </p>

      <svg viewBox={`0 0 ${W} ${totalH}`} className="block h-auto w-full" role="img" aria-label="Multi-head concat diagram">
        <defs>
          <marker id="mh-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 Z" fill="currentColor" opacity={0.55} />
          </marker>
        </defs>

        {/* H individual head outputs at the top */}
        {Array.from({ length: H }, (_, i) => {
          const x = headX(i);
          return (
            <g key={`top-${playKey}-${i}`} style={{ animation: `mhFade 600ms ease-out ${i * 70}ms both` }}>
              <rect
                x={x}
                y={TOP_Y}
                width={HEAD_W}
                height={HEAD_H}
                rx={4}
                fill={headColor(i, 0.7)}
                stroke="currentColor"
                strokeOpacity={0.3}
              />
              <text
                x={x + HEAD_W / 2}
                y={TOP_Y - 6}
                fontSize={10}
                textAnchor="middle"
                fill="currentColor"
                fillOpacity={0.85}
                fontFamily="ui-monospace, monospace"
              >
                Z<tspan baselineShift="sub" fontSize={7}>{i}</tspan>
              </text>
              <text
                x={x + HEAD_W / 2}
                y={TOP_Y + HEAD_H + 12}
                fontSize={8}
                textAnchor="middle"
                fill="currentColor"
                fillOpacity={0.55}
              >
                head {i}
              </text>
            </g>
          );
        })}

        {/* labels on first head only — shape annotation */}
        <text x={20} y={TOP_Y + HEAD_H / 2 + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.55}>
          {''}
        </text>
        <text x={W - 8} y={TOP_Y + HEAD_H / 2 + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.5}>
          [seq={SEQ_LEN}, d_head={D_HEAD}] each
        </text>

        {/* slide arrows from each head down into the concat slot */}
        {Array.from({ length: H }, (_, i) => {
          const srcX = headX(i) + HEAD_W / 2;
          const dstSlotW = CONCAT_W / H;
          const dstX = CONCAT_X + dstSlotW * (i + 0.5);
          return (
            <line
              key={`drop-${playKey}-${i}`}
              x1={srcX}
              y1={TOP_Y + HEAD_H + 18}
              x2={dstX}
              y2={CONCAT_Y - 4}
              stroke="currentColor"
              strokeOpacity={0.25}
              strokeDasharray="2 3"
              style={{ animation: `mhFade 400ms ease-out ${i * 70 + 200}ms both` }}
            />
          );
        })}

        {/* Concatenated wide matrix, drawn as H vertical bands matching head colors */}
        {Array.from({ length: H }, (_, i) => {
          const slotW = CONCAT_W / H;
          const x = CONCAT_X + slotW * i;
          return (
            <rect
              key={`band-${playKey}-${i}`}
              x={x}
              y={CONCAT_Y}
              width={slotW}
              height={CONCAT_H}
              fill={headColor(i, 0.6)}
              stroke="currentColor"
              strokeOpacity={0.2}
              style={{ animation: `mhSlideIn 600ms ease-out ${i * 70 + 280}ms both` }}
            />
          );
        })}
        {/* concat outer border */}
        <rect
          x={CONCAT_X}
          y={CONCAT_Y}
          width={CONCAT_W}
          height={CONCAT_H}
          fill="none"
          stroke="currentColor"
          strokeOpacity={0.45}
        />
        <text
          x={CONCAT_X + CONCAT_W / 2}
          y={CONCAT_Y + CONCAT_H + 14}
          fontSize={10}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.8}
        >
          concat · [seq={SEQ_LEN}, {H} × {D_HEAD} = {D_MODEL}]
        </text>

        {/* × W_O step */}
        <text
          x={CONCAT_X + CONCAT_W + 14}
          y={CONCAT_Y + CONCAT_H / 2 + 4}
          fontSize={20}
          fill="oklch(0.7 0.12 60)"
          style={{ animation: `mhFade 400ms ease-out 1000ms both` }}
        >
          ×
        </text>
        <rect
          x={CONCAT_X + CONCAT_W + 32}
          y={CONCAT_Y + 6}
          width={56}
          height={CONCAT_H - 12}
          rx={4}
          fill="currentColor"
          fillOpacity={0.08}
          stroke="currentColor"
          strokeOpacity={0.35}
          style={{ animation: `mhFade 400ms ease-out 1050ms both` }}
        />
        <text
          x={CONCAT_X + CONCAT_W + 60}
          y={CONCAT_Y + CONCAT_H / 2 + 4}
          fontSize={12}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.85}
          fontFamily="ui-monospace, monospace"
          style={{ animation: `mhFade 400ms ease-out 1050ms both` }}
        >
          W<tspan baselineShift="sub" fontSize={8}>O</tspan>
        </text>
        <text
          x={CONCAT_X + CONCAT_W + 60}
          y={CONCAT_Y + CONCAT_H + 14}
          fontSize={9}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.5}
          style={{ animation: `mhFade 400ms ease-out 1050ms both` }}
        >
          [{D_MODEL}, {D_MODEL}]
        </text>

        {/* arrow to final Z_attention */}
        <line
          x1={CONCAT_X + CONCAT_W + 92}
          y1={CONCAT_Y + CONCAT_H / 2}
          x2={CONCAT_X + CONCAT_W + 124}
          y2={CONCAT_Y + CONCAT_H / 2}
          stroke="currentColor"
          strokeOpacity={0.55}
          markerEnd="url(#mh-arrow)"
          style={{ animation: `mhFade 400ms ease-out 1200ms both` }}
        />
        <rect
          x={CONCAT_X + CONCAT_W + 130}
          y={CONCAT_Y + 6}
          width={88}
          height={CONCAT_H - 12}
          rx={5}
          className="fill-primary/15"
          stroke="oklch(0.7 0.18 25)"
          strokeOpacity={0.6}
          style={{ animation: `mhFade 400ms ease-out 1250ms both` }}
        />
        <text
          x={CONCAT_X + CONCAT_W + 174}
          y={CONCAT_Y + CONCAT_H / 2 - 2}
          fontSize={11}
          textAnchor="middle"
          className="fill-primary"
          fontFamily="ui-monospace, monospace"
          style={{ animation: `mhFade 400ms ease-out 1250ms both` }}
        >
          Z<tspan baselineShift="sub" fontSize={7}>attn</tspan>
        </text>
        <text
          x={CONCAT_X + CONCAT_W + 174}
          y={CONCAT_Y + CONCAT_H / 2 + 12}
          fontSize={9}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.55}
          style={{ animation: `mhFade 400ms ease-out 1250ms both` }}
        >
          [seq={SEQ_LEN}, {D_MODEL}]
        </text>
      </svg>

      <div className="text-[11px] text-muted-foreground">
        Color = head identity. In the concat block each color owns a fixed band of <span className="font-mono">{D_HEAD}</span>{' '}
        feature dimensions; <span className="font-mono">W<sub>O</sub></span> is the only place head outputs ever mix.
        This is also why <em>cross-head</em> reasoning is shallow — anything heads have to share has to be encoded
        through the residual stream across multiple layers.
      </div>
    </div>
  );
}
