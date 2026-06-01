import * as React from 'react';

/**
 * Chapter 5 supplement — the GQA tradeoff as a sweep over #KV heads.
 *
 * This widget isolates the KV-cache cost of the #KV-head choice, not the
 * head-grouping picture (GroupHeadCompare in the chapter already draws which
 * query head reads which K/V head). Pick #KV heads from the divisors of 8 and
 * watch the cache shrink:
 *   • KV cache per token per full-attention layer = 2 (K&V) × #KV × head_dim
 *     → MQA 512, GQA-2 1024, GQA-4 2048, MHA 4096 floats. Shown as a bar vs MHA
 *       plus the savings factor (8×, 4×, 2×, 1×).
 * The quality side of the tradeoff lives in prose (empirically GQA keeps most of
 * MHA's quality), deliberately NOT drawn as a fake "expressivity" bar — this toy
 * sweep measures cache exactly but cannot measure quality.
 *
 * #KV = 2 is highlighted as Qwen3.5-0.8B's actual choice (4× cache savings for
 * little quality loss). All numbers are formula-derived from the architecture
 * constants — no model, no worker.
 */

// Architecture constants (single source of truth: 14-architecture.tsx).
const NUM_HEADS = 8;
const HEAD_DIM = 256;
const QWEN_NUM_KV_HEADS = 2;

// Divisors of 8: 1 = MQA, 2 = Qwen3.5 GQA, 4 = GQA, 8 = MHA.
const KV_OPTIONS = [1, 2, 4, 8] as const;

type Variant = { kv: number; name: string; tag: string };

const VARIANTS: Record<number, Variant> = {
  1: { kv: 1, name: 'MQA', tag: '1 KV head' },
  2: { kv: 2, name: 'GQA', tag: '2 KV heads' },
  4: { kv: 4, name: 'GQA', tag: '4 KV heads' },
  8: { kv: 8, name: 'MHA', tag: '8 KV heads' },
};

// KV cache floats per token, per full-attention layer: 2 (K and V) × kv × head_dim.
function cacheFloats(kv: number): number {
  return 2 * kv * HEAD_DIM;
}
const MHA_FLOATS = cacheFloats(NUM_HEADS);

function formatFloats(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(n % 1000 === 0 ? 0 : 1)}k`;
  return `${n}`;
}

export function GqaGroupSweep() {
  const [kv, setKv] = React.useState(QWEN_NUM_KV_HEADS);
  const variant = VARIANTS[kv]!;
  const groupSize = NUM_HEADS / kv;
  const floats = cacheFloats(kv);
  const cachePct = (floats / MHA_FLOATS) * 100;
  const savings = MHA_FLOATS / floats;
  const isQwen = kv === QWEN_NUM_KV_HEADS;

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">GQA sweep — KV cache vs quality</div>
        <div className="text-[11px] text-muted-foreground">
          {NUM_HEADS} query heads, head_dim {HEAD_DIM}
        </div>
      </div>

      {/* Selector over #KV heads (divisors of 8). */}
      <div className="space-y-1">
        <span className="block text-xs text-muted-foreground">Number of KV heads</span>
        <div
          className="inline-flex flex-wrap gap-1 rounded-md border border-border p-0.5"
          role="group"
          aria-label="Number of KV heads"
        >
          {KV_OPTIONS.map((option) => {
            const active = option === kv;
            const v = VARIANTS[option]!;
            return (
              <button
                key={option}
                type="button"
                aria-pressed={active}
                onClick={() => setKv(option)}
                className={[
                  'rounded px-2.5 py-1 text-xs font-medium transition-colors',
                  active ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
                ].join(' ')}
              >
                {option} — {v.name}
                {option === QWEN_NUM_KV_HEADS ? ' ★' : ''}
              </button>
            );
          })}
        </div>
        <p className="text-[11px] text-muted-foreground">
          ★ = Qwen3.5-0.8B&apos;s choice. <span className="font-mono">1</span> = MQA (one shared K/V),{' '}
          <span className="font-mono">8</span> = MHA (one K/V per query head).
        </p>
      </div>

      {/* Selected variant headline. */}
      <div
        className={[
          'rounded-md border p-2 text-[12px]',
          isQwen ? 'border-primary/50 bg-primary/5 text-foreground/90' : 'border-border bg-muted/20 text-foreground/85',
        ].join(' ')}
      >
        <span className="font-mono font-semibold">{variant.name}</span> — {variant.tag}, group size{' '}
        <span className="font-mono">{groupSize}</span> (each K/V head serves {groupSize} query head
        {groupSize === 1 ? '' : 's'}).
        {isQwen ? <span className="ml-1 text-primary">This is Qwen3.5-0.8B&apos;s chosen balance.</span> : null}
      </div>

      {/* KV cache bar, relative to MHA. */}
      <div className="space-y-1">
        <div className="flex items-baseline justify-between text-[11px]">
          <span className="text-muted-foreground">KV cache / token / full-attn layer</span>
          <span className="font-mono text-foreground/85">
            {formatFloats(floats)} floats · {savings}× smaller than MHA
          </span>
        </div>
        <div
          className="relative h-6 w-full overflow-hidden rounded-sm bg-muted/60"
          role="img"
          aria-label={`KV cache for ${variant.tag}: ${floats} floats, ${savings} times smaller than MHA`}
        >
          <div
            className={['absolute inset-y-0 left-0 transition-all', isQwen ? 'ring-1 ring-primary/50' : ''].join(' ')}
            style={{ width: `${cachePct.toFixed(1)}%`, backgroundColor: isQwen ? '#22c55e' : '#0ea5e9' }}
          />
        </div>
        <div className="text-[11px] text-muted-foreground">
          <span className="font-mono">
            2 × {kv} × {HEAD_DIM}
          </span>{' '}
          = {floats} floats. MHA baseline is {MHA_FLOATS} floats (full width of the bar).
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">
        The KV cache shrinks linearly with the number of KV heads. Every variant keeps all{' '}
        <strong>{NUM_HEADS} query heads</strong>, so the query side is untouched — but sharing K/V means fewer distinct
        key/value subspaces, a real (if usually modest) cut in capacity, not zero capacity. Empirically GQA still
        recovers most of MHA&apos;s quality at a fraction of the cache (a general result, not something this toy sweep
        measures), which is why Qwen3.5-0.8B uses <strong>{QWEN_NUM_KV_HEADS} KV heads</strong> — a{' '}
        <span className="font-mono">{MHA_FLOATS / cacheFloats(QWEN_NUM_KV_HEADS)}×</span> cache saving over MHA.
      </p>

      <p className="text-[10px] text-muted-foreground">
        Illustrative — the cache floats are exact (2 · #KV · head_dim); not live output from the model.
      </p>
    </div>
  );
}
