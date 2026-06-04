import * as React from 'react';

/**
 * Chapter 7 supplement — the gated MLP drawn as a neuron-graph instead of
 * matrix multiplications, so the "wide intermediate scratch space" claim
 * has a visual referent.
 *
 * We use a symbolic 6 → 12 → 6 layout (not Qwen's 1024 → 3584 → 1024, which
 * would be an unreadable carpet of lines). The dual gate/up paths into the
 * elementwise product, then down_proj back to hidden, are all drawn — the
 * topology is what we're teaching, not the dimensions.
 *
 * Inspired by the encoder-s5 FFN-as-neurons panel in
 * An-Jhon/Hand-Drawn-Transformer.
 */

const HIDDEN = 6;
const INTER = 12;

function evenY(n: number, top: number, bottom: number, i: number): number {
  if (n === 1) return (top + bottom) / 2;
  return top + (i / (n - 1)) * (bottom - top);
}

export function FfnNeurons() {
  const W = 560;
  const H = 280;

  const HIDDEN_R = 6;
  const INTER_R = 5;

  // Three column x positions: input hidden (left), intermediate gate+up
  // (middle, two stacks), intermediate elementwise product (just left of
  // right column conceptually — we render it as the same intermediate
  // column but in a third stack), output hidden (right).
  const xIn = 60;
  const xGate = 200;
  const xUp = 200;
  const xProd = 360;
  const xOut = 500;

  const TOP = 30;
  const BOT = 230;
  const GATE_TOP = 18;
  const GATE_BOT = 130;
  const UP_TOP = 150;
  const UP_BOT = 262;

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">Same MLP, drawn as neurons</div>
      <p className="text-[12px] text-foreground/85">
        The matmul view above tells you the dimensions. The neuron view tells you the <em>topology</em>: two parallel
        wide projections (the gate and the value), an element-wise product, and a narrow projection back. Symbolic
        widths shown — Qwen3.5-0.8B uses <span className="font-mono">1024 → 3584 → 1024</span>.
      </p>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="block h-auto w-full"
        role="img"
        aria-label="Gated MLP drawn as a node graph"
      >
        {/* schematic note — make it explicit this is topology, not real activations */}
        <text
          x={W - 6}
          y={H - 2}
          fontSize={9}
          textAnchor="end"
          fill="currentColor"
          fillOpacity={0.45}
          fontStyle="italic"
        >
          schematic — 6 / 12 nodes shown, model uses 1024 / 3584
        </text>
        {/* column headers */}
        <text x={xIn} y={14} fontSize={10} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
          hidden in
        </text>
        <text x={(xGate + xUp) / 2} y={14} fontSize={10} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
          intermediate (gate & up)
        </text>
        <text x={xProd} y={14} fontSize={10} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
          silu(gate) ⊙ up
        </text>
        <text x={xOut} y={14} fontSize={10} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
          hidden out
        </text>

        {/* edges: input → gate */}
        {Array.from({ length: HIDDEN }, (_, i) => {
          const yi = evenY(HIDDEN, TOP, BOT, i);
          return Array.from({ length: INTER }, (_, j) => {
            const yj = evenY(INTER, GATE_TOP, GATE_BOT, j);
            return (
              <line
                key={`gw-${i}-${j}`}
                x1={xIn}
                y1={yi}
                x2={xGate}
                y2={yj}
                stroke="currentColor"
                strokeOpacity={0.08}
                strokeWidth={0.6}
              />
            );
          });
        })}
        {/* edges: input → up */}
        {Array.from({ length: HIDDEN }, (_, i) => {
          const yi = evenY(HIDDEN, TOP, BOT, i);
          return Array.from({ length: INTER }, (_, j) => {
            const yj = evenY(INTER, UP_TOP, UP_BOT, j);
            return (
              <line
                key={`uw-${i}-${j}`}
                x1={xIn}
                y1={yi}
                x2={xUp}
                y2={yj}
                stroke="currentColor"
                strokeOpacity={0.08}
                strokeWidth={0.6}
              />
            );
          });
        })}
        {/* edges: gate→prod and up→prod combining into the same product nodes */}
        {Array.from({ length: INTER }, (_, j) => {
          const yGate = evenY(INTER, GATE_TOP, GATE_BOT, j);
          const yUp = evenY(INTER, UP_TOP, UP_BOT, j);
          const yProd = evenY(INTER, TOP - 12, BOT + 14, j);
          return (
            <g key={`prod-edges-${j}`}>
              <line
                x1={xGate + 4}
                y1={yGate}
                x2={xProd - 4}
                y2={yProd}
                stroke="oklch(0.7 0.13 60)"
                strokeOpacity={0.35}
                strokeWidth={0.7}
              />
              <line
                x1={xUp + 4}
                y1={yUp}
                x2={xProd - 4}
                y2={yProd}
                stroke="oklch(0.7 0.13 220)"
                strokeOpacity={0.35}
                strokeWidth={0.7}
              />
            </g>
          );
        })}
        {/* edges: prod → output */}
        {Array.from({ length: INTER }, (_, j) => {
          const yj = evenY(INTER, TOP - 12, BOT + 14, j);
          return Array.from({ length: HIDDEN }, (_, k) => {
            const yk = evenY(HIDDEN, TOP, BOT, k);
            return (
              <line
                key={`dw-${j}-${k}`}
                x1={xProd}
                y1={yj}
                x2={xOut}
                y2={yk}
                stroke="currentColor"
                strokeOpacity={0.08}
                strokeWidth={0.6}
              />
            );
          });
        })}

        {/* nodes: input hidden */}
        {Array.from({ length: HIDDEN }, (_, i) => (
          <circle
            key={`in-${i}`}
            cx={xIn}
            cy={evenY(HIDDEN, TOP, BOT, i)}
            r={HIDDEN_R}
            fill="currentColor"
            fillOpacity={0.18}
            stroke="currentColor"
            strokeOpacity={0.55}
          />
        ))}
        {/* nodes: intermediate gate (warm) */}
        {Array.from({ length: INTER }, (_, j) => (
          <circle
            key={`gate-${j}`}
            cx={xGate}
            cy={evenY(INTER, GATE_TOP, GATE_BOT, j)}
            r={INTER_R}
            fill="oklch(0.7 0.13 60 / 0.35)"
            stroke="oklch(0.7 0.13 60)"
            strokeOpacity={0.6}
          />
        ))}
        {/* nodes: intermediate up (cool) */}
        {Array.from({ length: INTER }, (_, j) => (
          <circle
            key={`up-${j}`}
            cx={xUp}
            cy={evenY(INTER, UP_TOP, UP_BOT, j)}
            r={INTER_R}
            fill="oklch(0.7 0.13 220 / 0.35)"
            stroke="oklch(0.7 0.13 220)"
            strokeOpacity={0.6}
          />
        ))}
        {/* nodes: product (purple = mix) */}
        {Array.from({ length: INTER }, (_, j) => (
          <circle
            key={`prod-${j}`}
            cx={xProd}
            cy={evenY(INTER, TOP - 12, BOT + 14, j)}
            r={INTER_R}
            className="fill-primary/30"
            stroke="oklch(0.7 0.18 25)"
            strokeOpacity={0.6}
          />
        ))}
        {/* nodes: output hidden */}
        {Array.from({ length: HIDDEN }, (_, k) => (
          <circle
            key={`out-${k}`}
            cx={xOut}
            cy={evenY(HIDDEN, TOP, BOT, k)}
            r={HIDDEN_R}
            fill="currentColor"
            fillOpacity={0.18}
            stroke="currentColor"
            strokeOpacity={0.55}
          />
        ))}

        {/* operator labels */}
        <text x={(xIn + xGate) / 2} y={H - 12} fontSize={10} textAnchor="middle" fill="oklch(0.7 0.13 60)">
          gate_proj
        </text>
        <text x={(xIn + xUp) / 2} y={H - 2} fontSize={10} textAnchor="middle" fill="oklch(0.7 0.13 220)">
          up_proj
        </text>
        <text x={(xGate + xProd) / 2} y={H - 12} fontSize={10} textAnchor="middle" className="fill-primary">
          silu ⊙
        </text>
        <text x={(xProd + xOut) / 2} y={H - 12} fontSize={10} textAnchor="middle" fill="currentColor" fillOpacity={0.7}>
          down_proj
        </text>
      </svg>
      <div className="text-[11px] text-muted-foreground">
        Two parallel projections widen <span className="font-mono">hidden → intermediate</span>. Their element-wise
        product is the gated activation — the gate branch decides which features survive, the up branch carries the
        values. <span className="font-mono">down_proj</span> collapses back. Together these three matrices are about a
        third of Qwen3.5-0.8B (≈264M), roughly on par with the embedding table.
      </div>
    </div>
  );
}
