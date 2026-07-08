import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { SegmentedToggle } from '../scaffolding/SegmentedToggle';

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

type Variant = { kv: number; name: string };

const VARIANTS: Record<number, Variant> = {
  1: { kv: 1, name: 'MQA' },
  2: { kv: 2, name: 'GQA' },
  4: { kv: 4, name: 'GQA' },
  8: { kv: 8, name: 'MHA' },
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

// Per-locale copy: every user-visible string lives here so /zh localizes while
// the English output stays byte-identical.
const COPY = {
  en: {
    header: 'GQA sweep — KV cache vs quality',
    headsLine: (
      <>
        {NUM_HEADS} query heads, head_dim {HEAD_DIM}
      </>
    ),
    kvCountLabel: 'Number of KV heads',
    kvCountAria: 'Number of KV heads',
    starNote: (
      <>
        ★ = Qwen3.5-0.8B&apos;s choice. <span className="font-mono">1</span> = MQA (one shared K/V),{' '}
        <span className="font-mono">8</span> = MHA (one K/V per query head).
      </>
    ),
    variantTag: (kv: number) => `${kv} KV head${kv === 1 ? '' : 's'}`,
    headline: (tag: string, groupSize: number) => (
      <>
        {' '}
        — {tag}, group size <span className="font-mono">{groupSize}</span> (each K/V head serves {groupSize} query head
        {groupSize === 1 ? '' : 's'}).
      </>
    ),
    qwenChoice: "This is Qwen3.5-0.8B's chosen balance.",
    cacheLabel: 'KV cache / token / full-attn layer',
    cacheValue: (floats: number, savings: number) => (
      <>
        {formatFloats(floats)} floats · {savings}× smaller than MHA
      </>
    ),
    cacheBarAria: (tag: string, floats: number, savings: number) =>
      `KV cache for ${tag}: ${floats} floats, ${savings} times smaller than MHA`,
    formulaTail: (floats: number) => (
      <>
        {' '}
        = {floats} floats. MHA baseline is {MHA_FLOATS} floats (full width of the bar).
      </>
    ),
    body: (
      <>
        The KV cache shrinks linearly with the number of KV heads. Every variant keeps all{' '}
        <strong>{NUM_HEADS} query heads</strong>, so the query side is untouched — but sharing K/V means fewer distinct
        sets of K/V features for the heads to match against, a real (if usually modest) cut in capacity, not zero
        capacity. Empirically GQA still recovers most of MHA&apos;s quality at a fraction of the cache (a general
        result, not something this toy sweep measures), which is why Qwen3.5-0.8B uses{' '}
        <strong>{QWEN_NUM_KV_HEADS} KV heads</strong> — a{' '}
        <span className="font-mono">{MHA_FLOATS / cacheFloats(QWEN_NUM_KV_HEADS)}×</span> cache saving over MHA.
      </>
    ),
    footnote: 'Illustrative — the cache floats are exact (2 · #KV · head_dim); not live output from the model.',
  },
  zh: {
    header: 'GQA 扫描——KV 缓存 vs 质量',
    headsLine: (
      <>
        {NUM_HEADS} 个 query 头，head_dim {HEAD_DIM}
      </>
    ),
    kvCountLabel: 'KV 头数量',
    kvCountAria: 'KV 头数量',
    starNote: (
      <>
        ★ = Qwen3.5-0.8B 的选择。<span className="font-mono">1</span> = MQA（共享一份 K/V），
        <span className="font-mono">8</span> = MHA（每个 query 头一份 K/V）。
      </>
    ),
    variantTag: (kv: number) => `${kv} 个 KV 头`,
    headline: (tag: string, groupSize: number) => (
      <>
        ——{tag}，组大小 <span className="font-mono">{groupSize}</span>（每个 K/V 头服务 {groupSize} 个 query 头）。
      </>
    ),
    qwenChoice: '这是 Qwen3.5-0.8B 选定的平衡点。',
    cacheLabel: 'KV 缓存 / token / 全量注意力层',
    cacheValue: (floats: number, savings: number) => (
      <>
        {formatFloats(floats)} 个浮点数 · 比 MHA 小 {savings}×
      </>
    ),
    cacheBarAria: (tag: string, floats: number, savings: number) =>
      `${tag}的 KV 缓存：${floats} 个浮点数，比 MHA 小 ${savings} 倍`,
    formulaTail: (floats: number) => (
      <>
        {' '}
        = {floats} 个浮点数。MHA 基线是 {MHA_FLOATS} 个浮点数（长条的全宽）。
      </>
    ),
    body: (
      <>
        KV 缓存随 KV 头数线性缩小。每个变体都保留全部 <strong>{NUM_HEADS} 个 query 头</strong>，所以 query
        侧不受影响——但共享 K/V 意味着各头可以匹配的不同 K/V
        特征组变少了，这是一次真实（尽管通常不大）的容量削减，而不是零削减。经验上，GQA 仍能用一小部分缓存拿回 MHA
        的大部分质量（这是一个普遍结论，不是这个玩具扫描能测出来的），这正是 Qwen3.5-0.8B 使用{' '}
        <strong>{QWEN_NUM_KV_HEADS} 个 KV 头</strong>的原因——相比 MHA 节省{' '}
        <span className="font-mono">{MHA_FLOATS / cacheFloats(QWEN_NUM_KV_HEADS)}×</span> 的缓存。
      </>
    ),
    footnote: '示意——缓存浮点数是精确的（2 · #KV · head_dim）；不是模型的实时输出。',
  },
} as const;

export function GqaGroupSweep() {
  const copy = COPY[useLocale()];
  const [kv, setKv] = React.useState(QWEN_NUM_KV_HEADS);
  const variant = VARIANTS[kv]!;
  const variantTag = copy.variantTag(kv);
  const groupSize = NUM_HEADS / kv;
  const floats = cacheFloats(kv);
  const cachePct = (floats / MHA_FLOATS) * 100;
  const savings = MHA_FLOATS / floats;
  const isQwen = kv === QWEN_NUM_KV_HEADS;

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <div className="text-[11px] text-muted-foreground">{copy.headsLine}</div>
      </div>

      {/* Selector over #KV heads (divisors of 8). */}
      <div className="space-y-1">
        <span className="block text-xs text-muted-foreground">{copy.kvCountLabel}</span>
        <SegmentedToggle
          value={kv}
          onChange={setKv}
          ariaLabel={copy.kvCountAria}
          wrap
          options={KV_OPTIONS.map((option) => {
            const v = VARIANTS[option]!;
            return {
              value: option,
              label: (
                <>
                  {option} — {v.name}
                  {option === QWEN_NUM_KV_HEADS ? ' ★' : ''}
                </>
              ),
            };
          })}
        />
        <p className="text-[11px] text-muted-foreground">{copy.starNote}</p>
      </div>

      {/* Selected variant headline. */}
      <div
        className={[
          'rounded-md border p-2 text-[12px]',
          isQwen ? 'border-primary/50 bg-primary/5 text-foreground/90' : 'border-border bg-muted/20 text-foreground/85',
        ].join(' ')}
      >
        <span className="font-mono font-semibold">{variant.name}</span>
        {copy.headline(variantTag, groupSize)}
        {isQwen ? <span className="ml-1 text-primary">{copy.qwenChoice}</span> : null}
      </div>

      {/* KV cache bar, relative to MHA. */}
      <div className="space-y-1">
        <div className="flex items-baseline justify-between text-[11px]">
          <span className="text-muted-foreground">{copy.cacheLabel}</span>
          <span className="font-mono text-foreground/85">{copy.cacheValue(floats, savings)}</span>
        </div>
        <div
          className="relative h-6 w-full overflow-hidden rounded-sm bg-muted/60"
          role="img"
          aria-label={copy.cacheBarAria(variantTag, floats, savings)}
        >
          <div
            className={['absolute inset-y-0 left-0 transition-all', isQwen ? 'ring-1 ring-primary/50' : ''].join(' ')}
            style={{ width: `${cachePct.toFixed(1)}%`, backgroundColor: isQwen ? '#22c55e' : '#0ea5e9' }}
          />
        </div>
        <div className="text-[11px] text-muted-foreground">
          <span className="font-mono">
            2 × {kv} × {HEAD_DIM}
          </span>
          {copy.formulaTail(floats)}
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">{copy.body}</p>

      <p className="text-[10px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
