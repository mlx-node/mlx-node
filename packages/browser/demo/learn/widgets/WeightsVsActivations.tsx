import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 1 (overview) — weights vs activations, the two kinds of numbers.
 *
 * Self-contained: NO worker, NO model, NO WASM. Two lanes:
 *  - left, the WEIGHTS: a dense frozen block (852,985,920 numbers, learned in
 *    training, read-only at inference, identical for every request);
 *  - right, the ACTIVATIONS: per-token vectors born from YOUR prompt, flowing
 *    downward through the layer slots and discarded after the forward pass.
 *
 * A short looping animation walks an activation pulse down the right lane and
 * then throws it away, while the weights block visibly never changes (frozen
 * badge, static cells). With prefers-reduced-motion the loop is replaced by a
 * static final frame: the whole activation path lit at once.
 */

const PARAMS = 852_985_920;

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

const COPY = {
  en: {
    title: 'Weights vs activations',
    weightsHead: 'Weights',
    weightsSub: 'the brains',
    frozen: '❄ frozen',
    weightsFacts: [
      `${PARAMS.toLocaleString()} numbers`,
      'learned once, during training',
      'read-only at inference — identical for every request',
    ],
    actsHead: 'Activations',
    actsSub: "your prompt's data",
    fresh: '↻ fresh each request',
    slots: ['your prompt', 'embeddings', 'layer 1', '⋯ layers 2–23 ⋯', 'layer 24', 'logits'],
    reads: '× weights',
    discarded: 'discarded after the forward pass',
    legendWeights: 'weights — fixed, shared by everyone',
    legendActs: 'activations — computed fresh for your prompt',
    footnote: "Schematic, to show the split — not the model's live state.",
  },
  zh: {
    title: '权重 vs 激活值',
    weightsHead: '权重',
    weightsSub: '“大脑”',
    frozen: '❄ 冻结',
    weightsFacts: [
      `${PARAMS.toLocaleString()} 个数字`,
      '只在训练时学到一次',
      '推理时只读——对每个请求都一模一样',
    ],
    actsHead: '激活值',
    actsSub: '为你的提示词现算的数据',
    fresh: '↻ 每次请求重新计算',
    slots: ['你的提示词', '嵌入向量', '第 1 层', '⋯ 第 2–23 层 ⋯', '第 24 层', 'logits'],
    reads: '× 权重',
    discarded: '前向传播结束后即被丢弃',
    legendWeights: '权重——固定不变，人人相同',
    legendActs: '激活值——为你的提示词现场算出',
    footnote: '示意图，只为呈现这组对比——并非模型的实时状态。',
  },
} as const;

// Frames: 0..slots-1 = the pulse sits on that slot; last frame = discarded.
const SLOT_COUNT = COPY.en.slots.length;
const TOTAL_FRAMES = SLOT_COUNT + 1;
const FRAME_MS = 1000;

// Deterministic per-cell opacity for the dense weights block (SSR-safe; no
// Math.random so server and client render identically).
const WEIGHT_CELLS = Array.from({ length: 60 }, (_, i) => 0.25 + ((i * 37) % 11) * 0.055);

export function WeightsVsActivations() {
  const copy = COPY[useLocale()];
  const reducedMotion = usePrefersReducedMotion();
  const [frame, setFrame] = React.useState(0);

  React.useEffect(() => {
    if (reducedMotion) return;
    const t = window.setInterval(() => setFrame((f) => (f + 1) % TOTAL_FRAMES), FRAME_MS);
    return () => window.clearInterval(t);
  }, [reducedMotion]);

  const discarding = !reducedMotion && frame === TOTAL_FRAMES - 1;

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>

      <div className="grid gap-3 sm:grid-cols-2">
        {/* ── Left lane: the weights — dense, frozen, identical for everyone. */}
        <div className="space-y-2 rounded-md border border-border/60 bg-muted/30 p-2">
          <div className="flex items-center justify-between gap-2">
            <div className="text-[11px] font-medium text-foreground/90">
              {copy.weightsHead} <span className="font-normal text-muted-foreground">— {copy.weightsSub}</span>
            </div>
            <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">{copy.frozen}</span>
          </div>
          <div className="grid grid-cols-10 gap-0.5" aria-hidden>
            {WEIGHT_CELLS.map((o, i) => (
              <div key={i} className="h-3 rounded-[2px] bg-muted-foreground" style={{ opacity: o }} />
            ))}
          </div>
          <ul className="space-y-0.5 text-[11px] text-muted-foreground">
            {copy.weightsFacts.map((f, i) => (
              <li key={i}>{f}</li>
            ))}
          </ul>
        </div>

        {/* ── Right lane: the activations — born from the prompt, flowing down. */}
        <div className="space-y-2 rounded-md border border-border/60 bg-muted/30 p-2">
          <div className="flex items-center justify-between gap-2">
            <div className="text-[11px] font-medium text-foreground/90">
              {copy.actsHead} <span className="font-normal text-muted-foreground">— {copy.actsSub}</span>
            </div>
            <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary">{copy.fresh}</span>
          </div>
          <div className="space-y-1">
            {copy.slots.map((label, i) => {
              // Reduced motion: the whole path lit at once (static final frame).
              const active = reducedMotion ? true : !discarding && frame === i;
              const visited = reducedMotion ? false : !discarding && frame > i;
              return (
                <div
                  key={i}
                  className={[
                    'flex items-center justify-between rounded border px-1.5 py-0.5 font-mono text-[11px] transition-all duration-300',
                    discarding ? 'opacity-25' : '',
                    active
                      ? 'border-primary/40 bg-primary/15 text-primary'
                      : visited
                        ? 'border-primary/20 bg-primary/5 text-foreground/70'
                        : 'border-border/60 bg-background text-muted-foreground',
                  ].join(' ')}
                >
                  <span>{label}</span>
                  {/* Layer slots read the (unchanging) weights on the left. */}
                  {i > 0 && active ? <span className="text-[10px] text-muted-foreground">← {copy.reads}</span> : null}
                </div>
              );
            })}
          </div>
          <div
            className={[
              'text-[10px] transition-opacity duration-300',
              discarding ? 'text-foreground/80 opacity-100' : 'text-muted-foreground opacity-60',
            ].join(' ')}
          >
            ↓ {copy.discarded}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-[2px] bg-muted-foreground/60" aria-hidden />
          {copy.legendWeights}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-[2px] bg-primary/70" aria-hidden />
          {copy.legendActs}
        </span>
      </div>

      <p className="text-[10px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
