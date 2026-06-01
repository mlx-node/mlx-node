import * as React from 'react';

import { SegmentedToggle } from '../scaffolding/SegmentedToggle';

/**
 * Chapter 7 supplement — why a 3-matrix SwiGLU FFN costs about the same as
 * the classic 2-matrix FFN.
 *
 * Toggle between the two designs: the formula swaps and the parameter bar
 * re-proportions, but its total length stays fixed. Flipping to SwiGLU, the
 * `gate` matrix "grows in" from zero while `up`/`down` shrink to make room —
 * because the intermediate is scaled to ~2/3 of the classic 4x width, three
 * matrices land back at the same parameter budget as two.
 *
 * Params are counted in units of h^2 (hidden_dim^2), with the classic FFN
 * intermediate fixed at the conventional 4x hidden:
 *   plain  = up(h x 4h) + down(4h x h)            = 8 h^2
 *   swiglu = (gate + up + down), each (h x 8/3 h) = 8 h^2   (break-even)
 *
 * That bar is the IDEALIZED break-even. Real checkpoints often run wider:
 * Qwen3.5-0.8B uses a 3584 intermediate (3.5x its 1024 hidden), so its three
 * matrices total ~10.5 h^2 — about 30% more than a plain FFN. The widget shows
 * the break-even; the caption carries the actual Qwen figure.
 *
 * Colors match the gate (warm) / up (cool) palette used in FfnNeurons.
 */

type Mode = 'plain' | 'swiglu';

const GATE_BG = 'oklch(0.7 0.13 60 / 0.24)';
const UP_BG = 'oklch(0.7 0.13 220 / 0.24)';
const DOWN_BG = 'oklch(0.72 0.04 280 / 0.22)';

const EASE = 'cubic-bezier(0.4, 0, 0.2, 1)';

function Segment({ label, units, widthPct, bg }: { label: string; units: string; widthPct: number; bg: string }) {
  return (
    <div
      className="flex h-full min-w-0 flex-col items-center justify-center gap-0.5 overflow-hidden border-r border-border/60 last:border-r-0"
      style={{ width: `${widthPct}%`, backgroundColor: bg, transition: `width 550ms ${EASE}` }}
      aria-hidden={widthPct === 0}
    >
      {widthPct > 9 ? (
        <>
          <span className="truncate font-mono text-[10px] text-foreground/90">{label}</span>
          <span className="font-mono text-[9px] text-muted-foreground">{units} h²</span>
        </>
      ) : null}
    </div>
  );
}

export function SwiGluVsPlain() {
  const [mode, setMode] = React.useState<Mode>('swiglu');
  const swiglu = mode === 'swiglu';

  // Segment widths as a % of the fixed 8-unit (8 h^2) total.
  const gatePct = swiglu ? (8 / 3 / 8) * 100 : 0; // 33.3% | 0
  const upPct = swiglu ? (8 / 3 / 8) * 100 : (4 / 8) * 100; // 33.3% | 50%
  const downPct = upPct; // gate/up/down share equally in swiglu; up==down in plain
  const units = swiglu ? '2⅔' : '4';

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">Three matrices, same budget</div>
        <SegmentedToggle
          value={mode}
          onChange={setMode}
          options={[
            { value: 'plain', label: 'Plain FFN' },
            { value: 'swiglu', label: 'SwiGLU' },
          ]}
        />
      </div>

      {/* Formula — cross-fades between the two designs. The two lines are
          grid-stacked in a single cell (gridArea 1/1) so the row sizes to the
          wider formula; the wrapper scrolls horizontally rather than overflowing
          the card on narrow screens, and whitespace-nowrap keeps the height fixed. */}
      <div className="relative h-6 overflow-x-auto">
        <div className="grid h-full w-max items-center">
          <span
            className="whitespace-nowrap font-mono text-[12px] text-foreground/90"
            style={{ gridArea: '1 / 1', opacity: swiglu ? 0 : 1, transition: 'opacity 450ms ease' }}
            aria-hidden={swiglu}
          >
            down(&nbsp;ReLU(&nbsp;up·x&nbsp;)&nbsp;)
            <span className="ml-2 text-muted-foreground">· 2 matrices</span>
          </span>
          <span
            className="whitespace-nowrap font-mono text-[12px] text-foreground/90"
            style={{ gridArea: '1 / 1', opacity: swiglu ? 1 : 0, transition: 'opacity 450ms ease' }}
            aria-hidden={!swiglu}
          >
            down(&nbsp;SiLU(gate·x)&nbsp;⊙&nbsp;up·x&nbsp;)
            <span className="ml-2 text-muted-foreground">· 3 matrices</span>
          </span>
        </div>
      </div>

      {/* Parameter-budget bar — total length is identical in both modes. */}
      <div>
        <div className="mb-1 flex items-center justify-between text-[11px] text-muted-foreground">
          <span>Parameters (classic 4× intermediate = 8 h²)</span>
          <span className="font-mono">≈ 8 h² either way</span>
        </div>
        <div
          className="flex h-9 w-full overflow-hidden rounded-md border border-border"
          role="img"
          aria-label={
            swiglu
              ? 'SwiGLU at break-even: three matrices (gate, up, down), each two and two-thirds h squared, totalling 8 h squared.'
              : 'Plain FFN: two matrices (up, down), each four h squared, totalling 8 h squared.'
          }
        >
          <Segment label="gate_proj" units={units} widthPct={gatePct} bg={GATE_BG} />
          <Segment label="up_proj" units={units} widthPct={upPct} bg={UP_BG} />
          <Segment label="down_proj" units={units} widthPct={downPct} bg={DOWN_BG} />
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">
        The bar shows the <strong>idealized break-even</strong>: three matrices shrunk to{' '}
        <span className="font-mono">8⁄3 · hidden</span> weigh the same <span className="font-mono">8 h²</span> as a
        plain FFN's two 4×-wide matrices — which is how the <span className="font-mono">gate</span> gets added "for
        free." That's the trade-off Llama and Mistral take. Real checkpoints often run a touch wider:{' '}
        <strong>Qwen3.5-0.8B</strong> uses a <span className="font-mono">3584</span> intermediate (3.5× its 1024
        hidden), so its three matrices total <span className="font-mono">≈ 10.5 h²</span> —{' '}
        <strong>about 30% more</strong> than a plain block. The gate is cheap, not literally free.
      </p>
    </div>
  );
}
