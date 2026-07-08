import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

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
type Step = { id: string; kind: StepKind; label: string; sub?: string };

// Each layer is two norm → sub-block → "+" trios. Grouping them lets the
// skip arc span exactly one trio, ending on its "+" circle.
const STEP_GROUPS: Array<{ id: string; steps: Step[] }> = [
  {
    id: 'attn-trio',
    steps: [
      { id: 'rmsnorm-1', kind: 'norm', label: 'RMSNorm' },
      { id: 'attn', kind: 'attn', label: 'Attention', sub: 'multi-head + RoPE' },
      { id: 'add-1', kind: 'add', label: '+', sub: 'residual' },
    ],
  },
  {
    id: 'mlp-trio',
    steps: [
      { id: 'rmsnorm-2', kind: 'norm', label: 'RMSNorm' },
      { id: 'mlp', kind: 'mlp', label: 'MLP', sub: 'gated · SwiGLU' },
      { id: 'add-2', kind: 'add', label: '+', sub: 'residual' },
    ],
  },
];

const COPY = {
  en: {
    stepGroups: STEP_GROUPS,
    skipLabel: 'skip',
    header: `Anatomy of one layer · ${NUM_LAYERS}× stacked`,
    identicalBody: `${NUM_LAYERS} layers, identical body`,
    intro: (lastIdx: number) => (
      <>
        Each of Qwen3.5's <span className="font-mono">{NUM_LAYERS}</span> layers has the same six-step body. The 3D
        tower to the right gives you the <em>stack</em> view; this strip gives you the <em>anatomy</em> view, with
        layers 1–{lastIdx - 1} collapsed because they all look exactly like layer 0.
      </>
    ),
    layer0Label: 'Layer 0 (input)',
    layer0Caption: "generic anatomy placeholder — Qwen3.5's actual layer 0 is GatedDeltaNet, not full attention",
    layersRange: (from: number, to: number) => (
      <>
        Layers {from} … {to}
      </>
    ),
    sameAsAbove: (count: number) => `structure same as above · ${count} more`,
    finalLayerLabel: (lastIdx: number) => `Layer ${lastIdx} (final, feeds into Final RMSNorm + LM head)`,
    outro: (
      <>
        The two <span className="font-mono text-amber-700 dark:text-amber-300">+</span> circles are the two residual
        adds — that's the highway the rest of the course keeps pointing at. The dashed arc is the skip path: the input
        jumps over the norm + sub-block untouched and meets the sub-block's output at the{' '}
        <span className="font-mono text-amber-700 dark:text-amber-300">+</span> — two paths merging, not one more step
        in a chain. Note that linear-attention layers (most of
        Qwen3.5's stack) swap the green attention box for GatedDeltaNet; the rest of the body is the same.
      </>
    ),
  },
  zh: {
    stepGroups: [
      {
        id: 'attn-trio',
        steps: [
          { id: 'rmsnorm-1', kind: 'norm', label: 'RMSNorm' },
          { id: 'attn', kind: 'attn', label: 'Attention', sub: '多头 + RoPE' },
          { id: 'add-1', kind: 'add', label: '+', sub: '残差' },
        ],
      },
      {
        id: 'mlp-trio',
        steps: [
          { id: 'rmsnorm-2', kind: 'norm', label: 'RMSNorm' },
          { id: 'mlp', kind: 'mlp', label: 'MLP', sub: '门控 · SwiGLU' },
          { id: 'add-2', kind: 'add', label: '+', sub: '残差' },
        ],
      },
    ] as Array<{ id: string; steps: Step[] }>,
    skipLabel: '跳过',
    header: `单层解剖 · 堆叠 ${NUM_LAYERS} 次`,
    identicalBody: `${NUM_LAYERS} 层，结构完全相同`,
    intro: (lastIdx: number) => (
      <>
        Qwen3.5 的 <span className="font-mono">{NUM_LAYERS}</span> 层中的每一层都有同样的六步结构。右侧的 3D
        塔给你的是<em>堆叠</em>视角；这条横带给你的是<em>解剖</em>视角——第 1–{lastIdx - 1} 层被折叠起来，因为它们和第
        0 层长得一模一样。
      </>
    ),
    layer0Label: '第 0 层（输入）',
    layer0Caption: '通用结构示意——Qwen3.5 实际的第 0 层是 GatedDeltaNet，并非全量注意力',
    layersRange: (from: number, to: number) => (
      <>
        第 {from} … {to} 层
      </>
    ),
    sameAsAbove: (count: number) => `结构同上 · 还有 ${count} 层`,
    finalLayerLabel: (lastIdx: number) => `第 ${lastIdx} 层（最后一层，输出进入最终 RMSNorm + LM head）`,
    outro: (
      <>
        两个 <span className="font-mono text-amber-700 dark:text-amber-300">+</span>{' '}
        圆圈就是两次残差相加——也就是课程其余部分一直在指的那条“高速公路”。虚线弧是跳跃路径：输入原封不动地越过归一化 +
        子块，在 <span className="font-mono text-amber-700 dark:text-amber-300">+</span>{' '}
        处与子块的输出汇合——是两条路径的合流，而不是链条上多出的一步。注意，线性注意力层（Qwen3.5
        堆叠中的大多数）把绿色的注意力盒子换成 GatedDeltaNet；其余结构完全相同。
      </>
    ),
  },
} as const;

function StepBox({ step }: { step: Step }) {
  if (step.kind === 'add') {
    return (
      <div className="flex h-9 w-9 flex-col items-center justify-center rounded-full border border-amber-500/60 bg-amber-500/10 text-base font-light text-amber-700 dark:text-amber-300">
        {step.label}
      </div>
    );
  }
  return (
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
  );
}

/**
 * The residual skip path: a dashed arc from the trio's input, OVER the
 * norm + sub-block, down into the "+" circle. It makes the "+" read as a
 * junction merging two paths (skip + sub-block output), not a sequential
 * step. Absolutely positioned over the trio; `right-[18px]` lands the arc's
 * end on the horizontal center of the trailing 36px-wide "+" circle.
 * `preserveAspectRatio="none"` stretches the 100×16 viewBox to the trio's
 * width; `vectorEffect="non-scaling-stroke"` keeps the stroke crisp.
 */
function SkipArc({ showLabel = false }: { showLabel?: boolean }) {
  const copy = COPY[useLocale()];
  return (
    <>
      <svg
        className="pointer-events-none absolute left-0 right-[18px] top-0 h-4 text-muted-foreground/50"
        viewBox="0 0 100 16"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <path
          d="M 1 15 Q 50 -5 99 15"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.2}
          strokeDasharray="3 2"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      {showLabel ? (
        <span className="pointer-events-none absolute left-1/2 top-0 -translate-x-1/2 -translate-y-1/2 rounded bg-background px-1 text-[9px] leading-none text-muted-foreground">
          {copy.skipLabel}
        </span>
      ) : null}
    </>
  );
}

function LayerStrip({
  label,
  dim = false,
  caption,
}: {
  label: string;
  dim?: boolean;
  caption?: string;
}) {
  const copy = COPY[useLocale()];
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
      {caption ? <div className="mb-2 text-[10px] italic text-muted-foreground/80">{caption}</div> : null}
      <div className="flex flex-wrap items-end gap-1.5">
        {copy.stepGroups.map((group, gi) => (
          <React.Fragment key={group.id}>
            {/* pt-3 reserves headroom for the skip arc above the boxes. */}
            <div className="relative inline-flex items-center gap-1.5 pt-3">
              <SkipArc showLabel={gi === 0} />
              {group.steps.map((step, i) => (
                <React.Fragment key={step.id}>
                  <StepBox step={step} />
                  {i < group.steps.length - 1 ? <div className="text-muted-foreground/50">→</div> : null}
                </React.Fragment>
              ))}
            </div>
            {gi < STEP_GROUPS.length - 1 ? <div className="pb-2 text-muted-foreground/50">→</div> : null}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

function CollapsedSlab({ from, to }: { from: number; to: number }) {
  const copy = COPY[useLocale()];
  return (
    <div className="relative flex items-center justify-center rounded-md border-2 border-dashed border-border/60 bg-muted/20 px-4 py-6 text-center">
      <div>
        <div className="text-xs font-mono text-muted-foreground">{copy.layersRange(from, to)}</div>
        <div className="mt-0.5 text-[11px] text-muted-foreground/80">{copy.sameAsAbove(to - from + 1)}</div>
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
  const copy = COPY[useLocale()];
  const lastIdx = NUM_LAYERS - 1;
  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <div className="text-[11px] text-muted-foreground">{copy.identicalBody}</div>
      </div>

      <p className="text-[12px] text-foreground/85">{copy.intro(lastIdx)}</p>

      <div className="space-y-2">
        <LayerStrip label={copy.layer0Label} caption={copy.layer0Caption} />
        <CollapsedSlab from={1} to={lastIdx - 1} />
        <LayerStrip label={copy.finalLayerLabel(lastIdx)} />
      </div>

      <div className="text-[11px] text-muted-foreground">{copy.outro}</div>
    </div>
  );
}
