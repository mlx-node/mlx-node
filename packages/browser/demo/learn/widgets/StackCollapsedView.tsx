import * as React from 'react';

/**
 * Chapter 8 supplement — the anatomy of a single transformer block, with
 * the middle 22 layers collapsed under a "same structure" placeholder.
 *
 * The 3D tower next to this chapter shows the *stack* (heights, colors,
 * which layers are full-attention). This 2D strip shows the *anatomy* of
 * one layer — and uses the Hand-Drawn-Transformer "结构同上 / same as
 * above" pattern to make the point that "Layer N's body is identical to
 * Layer 0's body" without redrawing it 24 times.
 *
 * Inspired by the encoder-s6/s7 collapsed-stack panel in
 * An-Jhon/Hand-Drawn-Transformer.
 *
 * Pure JSX — no model data needed. Static for now; we could later highlight
 * the currently-selected layer's anatomy box if the 3D viewer's selectedLayer
 * was wired through.
 */

const NUM_LAYERS = 24; // Qwen3.5-0.8B

type StepKind = 'norm' | 'attn' | 'add' | 'mlp';

function LayerStrip({ label, dim = false }: { label: string; dim?: boolean }) {
  const steps: Array<{ id: string; kind: StepKind; label: string; sub?: string }> = [
    { id: 'rmsnorm-1', kind: 'norm', label: 'RMSNorm' },
    { id: 'attn', kind: 'attn', label: 'Attention', sub: 'multi-head + RoPE' },
    { id: 'add-1', kind: 'add', label: '+', sub: 'residual' },
    { id: 'rmsnorm-2', kind: 'norm', label: 'RMSNorm' },
    { id: 'mlp', kind: 'mlp', label: 'MLP', sub: 'gated · SwiGLU' },
    { id: 'add-2', kind: 'add', label: '+', sub: 'residual' },
  ];

  return (
    <div
      className={[
        'rounded-md border bg-background p-3 transition-opacity',
        dim ? 'border-dashed border-border/60 opacity-70' : 'border-border',
      ].join(' ')}
    >
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <div className="text-[11px] font-mono uppercase tracking-wider text-muted-foreground">{label}</div>
        <div className="text-[10px] text-muted-foreground">x_in → x_out</div>
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        {steps.map((step, i) => (
          <React.Fragment key={step.id}>
            {step.kind === 'add' ? (
              <div className="flex h-9 w-9 flex-col items-center justify-center rounded-full border border-amber-500/60 bg-amber-500/10 text-base font-light text-amber-700 dark:text-amber-300">
                {step.label}
              </div>
            ) : (
              <div
                className={[
                  'flex h-9 flex-col items-start justify-center rounded border px-2.5 text-[11px] leading-none',
                  step.kind === 'norm'
                    ? 'border-primary/40 bg-primary/10 text-primary'
                    : step.kind === 'attn'
                      ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
                      : 'border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300',
                ].join(' ')}
              >
                <span className="font-mono">{step.label}</span>
                {step.sub ? <span className="text-[9px] opacity-70">{step.sub}</span> : null}
              </div>
            )}
            {i < steps.length - 1 ? <div className="text-muted-foreground/50">→</div> : null}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

function CollapsedSlab({ from, to }: { from: number; to: number }) {
  return (
    <div className="relative flex items-center justify-center rounded-md border-2 border-dashed border-border/60 bg-muted/20 px-4 py-6 text-center">
      <div>
        <div className="text-xs font-mono text-muted-foreground">
          Layers {from} … {to}
        </div>
        <div className="mt-0.5 text-[11px] text-muted-foreground/80">
          structure same as above · {to - from + 1} more
        </div>
      </div>
      {/* Decorative tick marks to suggest "many of these" without drawing every one. */}
      <div className="pointer-events-none absolute inset-x-3 top-1 flex gap-1 opacity-30" aria-hidden="true">
        {Array.from({ length: to - from + 1 }, (_, k) => (
          <div key={k} className="h-1 flex-1 rounded-full bg-muted-foreground/60" />
        ))}
      </div>
      <div className="pointer-events-none absolute inset-x-3 bottom-1 flex gap-1 opacity-30" aria-hidden="true">
        {Array.from({ length: to - from + 1 }, (_, k) => (
          <div key={k} className="h-1 flex-1 rounded-full bg-muted-foreground/60" />
        ))}
      </div>
    </div>
  );
}

export function StackCollapsedView() {
  const lastIdx = NUM_LAYERS - 1;
  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Anatomy of one layer · {NUM_LAYERS}× stacked
        </div>
        <div className="text-[11px] text-muted-foreground">{NUM_LAYERS} layers, identical body</div>
      </div>

      <p className="text-[12px] text-foreground/85">
        Each of Qwen3.5's <span className="font-mono">{NUM_LAYERS}</span> layers has the same six-step body. The 3D
        tower to the right gives you the <em>stack</em> view; this strip gives you the <em>anatomy</em> view, with
        layers 1–{lastIdx - 1} collapsed because they all look exactly like layer 0.
      </p>

      <div className="space-y-2">
        <LayerStrip label="Layer 0 (input)" />
        <CollapsedSlab from={1} to={lastIdx - 1} />
        <LayerStrip label={`Layer ${lastIdx} (final, feeds into Final RMSNorm + LM head)`} />
      </div>

      <div className="text-[11px] text-muted-foreground">
        The two <span className="font-mono text-amber-700 dark:text-amber-300">+</span> circles are the two residual
        adds — that's the highway the rest of the course keeps pointing at. Note that linear-attention layers (most of
        Qwen3.5's stack) swap the green attention box for GatedDeltaNet; the rest of the body is the same.
      </div>
    </div>
  );
}
