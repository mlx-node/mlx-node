import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Directions in embedding space — analogy arithmetic + the plurality direction.
 *
 * All numbers are PRECOMPUTED offline from this exact checkpoint's
 * `embed_tokens.weight` ([248,320 × 1,024] bf16): raw input embeddings before
 * any transformer layer, cosines in fp32 over the full vocabulary, with the
 * input words excluded from neighbor lists. The widget itself runs no model —
 * it is a static, locale-aware visualization of that measurement.
 */

// -----------------------------------------------------------------------------
// Baked data (locale-independent). Token strings are byte-exact, including
// leading spaces — `renderToken` makes the leading space visible as '·'.
// -----------------------------------------------------------------------------

type Neighbor = { token: string; cos: number };

type AnalogyPreset = {
  id: 'king' | 'paris';
  /** [a, b, c] for the expression E(a) − E(b) + E(c). */
  expr: [string, string, string];
  /** The "textbook answer" token to highlight. */
  answer: string;
  /** Top-10 neighbors of the arithmetic result vector. */
  result: Neighbor[];
  /** Raw neighbors of E(a) alone, when we measured them (king only). */
  raw: { of: string; list: Neighbor[] } | null;
};

const ANALOGY_PRESETS: AnalogyPreset[] = [
  {
    id: 'king',
    expr: ['king', 'man', 'woman'],
    answer: ' queen',
    result: [
      { token: ' queen', cos: 0.578 },
      { token: 'queen', cos: 0.493 },
      { token: ' kings', cos: 0.487 },
      { token: ' KING', cos: 0.469 },
      { token: ' princess', cos: 0.466 },
      { token: 'Queen', cos: 0.465 },
      { token: 'King', cos: 0.457 },
      { token: '國王', cos: 0.433 },
      { token: ' Queen', cos: 0.424 },
      { token: ' wanita', cos: 0.423 },
    ],
    raw: {
      of: ' king',
      list: [
        { token: ' King', cos: 0.703 },
        { token: 'King', cos: 0.643 },
        { token: ' kings', cos: 0.621 },
        { token: ' queen', cos: 0.592 },
        { token: ' KING', cos: 0.561 },
        { token: '国王', cos: 0.482 },
        { token: ' emperor', cos: 0.47 },
        { token: ' Queen', cos: 0.458 },
        { token: ' kingdom', cos: 0.451 },
        { token: ' royal', cos: 0.449 },
      ],
    },
  },
  {
    id: 'paris',
    expr: ['Paris', 'France', 'Japan'],
    answer: ' Tokyo',
    result: [
      { token: ' Tokyo', cos: 0.512 },
      { token: 'Japan', cos: 0.431 },
      { token: 'Paris', cos: 0.421 },
      { token: ' Japanese', cos: 0.404 },
      { token: ' London', cos: 0.382 },
      { token: '东京', cos: 0.371 },
      { token: '日本', cos: 0.366 },
      { token: 'Japanese', cos: 0.34 },
      { token: ' Osaka', cos: 0.331 },
      { token: ' Kyoto', cos: 0.322 },
    ],
    raw: null,
  },
];

/** Dot products with the plurality direction plur = E(cats) − E(cat). */
const PLURALITY_PAIRS: Array<{ singular: string; plural: string; sCos: number; pCos: number }> = [
  { singular: 'dog', plural: 'dogs', sCos: -0.05, pCos: 0.13 },
  { singular: 'house', plural: 'houses', sCos: -0.0, pCos: 0.07 },
  { singular: 'car', plural: 'cars', sCos: -0.1, pCos: 0.07 },
  { singular: 'book', plural: 'books', sCos: -0.02, pCos: 0.06 },
  { singular: 'tree', plural: 'trees', sCos: 0.0, pCos: 0.09 },
  { singular: 'idea', plural: 'ideas', sCos: 0.02, pCos: 0.04 },
];

const PLURALITY_NUMBERS: Array<{ word: string; cos: number }> = [
  { word: 'one', cos: -0.04 },
  { word: 'two', cos: -0.02 },
  { word: 'three', cos: 0.0 },
  { word: 'four', cos: 0.0 },
  { word: 'five', cos: 0.02 },
];

const GENDER_CONSISTENCY_COS = 0.292;

// -----------------------------------------------------------------------------
// Per-locale copy
// -----------------------------------------------------------------------------

const COPY = {
  en: {
    title: 'Directions in embedding space',
    subtitle: 'Measured on Qwen3.5-0.8B · raw embed_tokens.weight',
    analogyHeading: 'Analogy arithmetic',
    presetLabels: { king: 'king − man + woman', paris: 'Paris − France + Japan' } as Record<'king' | 'paris', string>,
    presetGroupAria: 'Analogy preset',
    resultArrow: '→ nearest tokens to the result vector',
    viewResult: (a: string, b: string, c: string) => `after − ${b} + ${c}`,
    viewRaw: (of: string) => `raw neighbors of '${of}'`,
    viewToggleAria: 'Switch between raw neighbors and post-arithmetic neighbors',
    answerBadge: 'answer',
    barAria: (token: string, cos: string) => `${token}, cosine similarity ${cos}`,
    kingCaption: (
      <>
        Honesty check: <code>·queen</code> is <em>already</em> king's #4 raw neighbor — behind the casing/plural variants{' '}
        <code>·King</code>, <code>King</code> (no leading space), <code>·kings</code>. What <code>− man + woman</code> really does
        is <strong>promote queen to #1</strong>, past every king-variant. Flip the toggle above to compare.
      </>
    ),
    parisCaption: (
      <>
        The 248,320-token vocabulary is multilingual, so neighbors surface across languages: <code>东京</code> (Tokyo)
        and <code>日本</code> (Japan) rank alongside <code>·Osaka</code> and <code>·Kyoto</code>.
      </>
    ),
    pluralityHeading: 'A plurality direction — real but faint',
    pluralityIntro: (
      <>
        Take <code>plur = E(cats) − E(cat)</code> and dot it (cosine) against other words. Plural nouns land
        consistently positive, singulars near zero or negative — but the magnitudes are tiny.
      </>
    ),
    singularCol: 'singular',
    pluralCol: 'plural',
    numbersLabel: (
      <>
        Numbers <code>one → five</code> (a hoped-for "increasing plurality" trend — barely there):
      </>
    ),
    pluralityBarAria: (word: string, cos: string) => `${word}, dot with plurality direction ${cos}`,
    footnote: (
      <>
        Computed offline from this checkpoint's <code>embed_tokens.weight</code> ([248,320 × 1,024]): raw input
        embeddings <em>before</em> any transformer layer, fp32 cosines over the full vocabulary, input words excluded
        from neighbor lists. The arithmetic is approximate, not exact — this matrix was trained for next-token
        prediction with tied input/output weights, not for analogies. One symptom:{' '}
        <code>cosine(woman − man, queen − king) = {GENDER_CONSISTENCY_COS}</code>, so the two "gender directions" are
        only loosely parallel.
      </>
    ),
  },
  zh: {
    title: '嵌入空间中的方向',
    subtitle: '基于 Qwen3.5-0.8B 实测 · 原始 embed_tokens.weight',
    analogyHeading: '类比算术',
    presetLabels: { king: 'king − man + woman', paris: 'Paris − France + Japan' } as Record<'king' | 'paris', string>,
    presetGroupAria: '类比预设',
    resultArrow: '→ 与结果向量最近的 token',
    viewResult: (a: string, b: string, c: string) => `做完 − ${b} + ${c} 之后`,
    viewRaw: (of: string) => `'${of}' 的原始近邻`,
    viewToggleAria: '在原始近邻与算术后近邻之间切换',
    answerBadge: '答案',
    barAria: (token: string, cos: string) => `${token}，余弦相似度 ${cos}`,
    kingCaption: (
      <>
        诚实核查：<code>·queen</code> 本来<em>就是</em> king 的第 4 名原始近邻——排在大小写/复数变体 <code>·King</code>
        、<code>King</code>（无前导空格）、<code>·kings</code> 之后。<code>− man + woman</code> 真正做到的，是
        <strong>把 queen 提升到第 1 名</strong>，越过所有 king 变体。点上面的开关对比一下。
      </>
    ),
    parisCaption: (
      <>
        这个 248,320 个 token 的词表是多语言的，所以近邻会跨语言出现：<code>东京</code>和<code>日本</code>与{' '}
        <code>·Osaka</code>、<code>·Kyoto</code> 并排上榜。
      </>
    ),
    pluralityHeading: '复数方向——真实存在，但很微弱',
    pluralityIntro: (
      <>
        取 <code>plur = E(cats) − E(cat)</code>，再与其他词做点积（余弦）。复数名词的结果一致为正，单数接近 0
        或为负——但数值都非常小。
      </>
    ),
    singularCol: '单数',
    pluralCol: '复数',
    numbersLabel: (
      <>
        数词 <code>one → five</code>（理想中的“复数程度递增”趋势——几乎看不出来）：
      </>
    ),
    pluralityBarAria: (word: string, cos: string) => `${word}，与复数方向的点积 ${cos}`,
    footnote: (
      <>
        所有数字均由该 checkpoint 的 <code>embed_tokens.weight</code>（[248,320 × 1,024]）离线算出：进入任何
        Transformer 层<em>之前</em>的原始输入嵌入，在完整词表上用 fp32 计算余弦，近邻列表排除了输入词本身。这种算术是
        近似的，不是精确的——这个矩阵是为“预测下一个 token”训练的（且输入/输出权重绑定），不是为类比训练的。一个症状：
        <code>cosine(woman − man, queen − king) = {GENDER_CONSISTENCY_COS}</code>
        ，两条“性别方向”只是大致平行而已。
      </>
    ),
  },
} as const;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/** Make a leading space visible, matching CosineSimilarityTool's convention. */
function renderToken(text: string): string {
  if (text.startsWith(' ')) return '·' + text.slice(1);
  return text;
}

function TokenChip({ children, highlight = false }: { children: React.ReactNode; highlight?: boolean }) {
  return (
    <span
      className={[
        'rounded border px-1.5 py-0.5 font-mono text-[12px]',
        highlight
          ? 'border-transparent bg-[var(--primary)] text-[var(--primary-foreground)]'
          : 'border-border/50 bg-background text-foreground/90',
      ].join(' ')}
    >
      {children}
    </span>
  );
}

/** Cosine bars share a fixed scale so the two analogy tabs are comparable. */
const COS_BAR_SCALE = 0.8;

function NeighborBars({
  list,
  answer,
  barAria,
}: {
  list: Neighbor[];
  answer: string;
  barAria: (token: string, cos: string) => string;
}) {
  return (
    <ol className="space-y-1">
      {list.map((n, rank) => {
        const isAnswer = n.token === answer;
        const pct = Math.max(0, Math.min(1, n.cos / COS_BAR_SCALE)) * 100;
        return (
          <li key={n.token} className="grid grid-cols-[1.5rem_7.5rem_minmax(0,1fr)_3.25rem] items-center gap-2 text-xs">
            <span className="font-mono text-muted-foreground">#{rank + 1}</span>
            <span className="truncate">
              <TokenChip highlight={isAnswer}>{renderToken(n.token)}</TokenChip>
            </span>
            <span
              className="relative h-3.5 overflow-hidden rounded-sm bg-muted/60"
              role="img"
              aria-label={barAria(n.token, n.cos.toFixed(3))}
            >
              <span
                className="absolute inset-y-0 left-0 rounded-sm"
                style={{
                  width: `${pct.toFixed(1)}%`,
                  backgroundColor: isAnswer ? 'var(--primary)' : 'var(--muted-foreground)',
                  opacity: isAnswer ? 1 : 0.45,
                }}
              />
            </span>
            <span className="text-right font-mono text-foreground/80">{n.cos.toFixed(3)}</span>
          </li>
        );
      })}
    </ol>
  );
}

/** Diverging bar centered at 0; positive → primary (right), negative → muted (left). */
const PLUR_BAR_SCALE = 0.15;

function DivergingBar({ value, ariaLabel }: { value: number; ariaLabel: string }) {
  const frac = Math.max(-1, Math.min(1, value / PLUR_BAR_SCALE));
  const halfPct = Math.abs(frac) * 50;
  const positive = value > 0;
  return (
    <span className="relative block h-3.5 w-full overflow-hidden rounded-sm bg-muted/60" role="img" aria-label={ariaLabel}>
      {/* zero line */}
      <span aria-hidden="true" className="absolute inset-y-0 left-1/2 w-px bg-foreground/30" />
      <span
        className="absolute inset-y-0"
        style={{
          left: positive ? '50%' : `${(50 - halfPct).toFixed(1)}%`,
          width: `${halfPct.toFixed(1)}%`,
          backgroundColor: positive ? 'var(--primary)' : 'var(--muted-foreground)',
          opacity: positive ? 0.9 : 0.5,
        }}
      />
    </span>
  );
}

function formatSigned(value: number): string {
  // Avoid "-0.00" for the −0.0 entry in the data.
  const v = Object.is(value, -0) || value === 0 ? 0 : value;
  return `${v > 0 ? '+' : ''}${v.toFixed(2)}`;
}

// -----------------------------------------------------------------------------
// Widget
// -----------------------------------------------------------------------------

export function EmbeddingDirections() {
  const copy = COPY[useLocale()];
  const [presetId, setPresetId] = React.useState<'king' | 'paris'>('king');
  const [showRaw, setShowRaw] = React.useState(false);

  const preset = ANALOGY_PRESETS.find((p) => p.id === presetId)!;
  const showingRaw = showRaw && preset.raw != null;
  const list = showingRaw ? preset.raw!.list : preset.result;
  const [a, b, c] = preset.expr;

  return (
    <div className="space-y-4 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="text-[11px] text-muted-foreground">{copy.subtitle}</div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Panel 1 — analogy explorer                                          */}
      {/* ------------------------------------------------------------------ */}
      <div className="space-y-3">
        <div className="text-xs font-medium text-foreground">{copy.analogyHeading}</div>

        <div className="flex flex-wrap gap-2" role="group" aria-label={copy.presetGroupAria}>
          {ANALOGY_PRESETS.map((p) => {
            const active = p.id === presetId;
            return (
              <button
                key={p.id}
                type="button"
                aria-pressed={active}
                onClick={() => {
                  setPresetId(p.id);
                  setShowRaw(false);
                }}
                className={[
                  'rounded-full border px-2.5 py-1 font-mono text-xs transition-colors',
                  'focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50',
                  active
                    ? 'border-foreground/30 bg-foreground/10 text-foreground'
                    : 'border-border bg-muted/40 text-muted-foreground hover:text-foreground',
                ].join(' ')}
              >
                {copy.presetLabels[p.id]}
              </button>
            );
          })}
        </div>

        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <TokenChip>E({a})</TokenChip>
          <span className="text-muted-foreground">−</span>
          <TokenChip>E({b})</TokenChip>
          <span className="text-muted-foreground">+</span>
          <TokenChip>E({c})</TokenChip>
          <span className="ml-1 text-muted-foreground">{copy.resultArrow}</span>
        </div>

        {preset.raw != null ? (
          <button
            type="button"
            aria-pressed={showingRaw}
            aria-label={copy.viewToggleAria}
            onClick={() => setShowRaw((v) => !v)}
            className={[
              'rounded-md border px-2 py-1 text-[11px] transition-colors',
              'focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50',
              showingRaw
                ? 'border-foreground/30 bg-foreground/10 text-foreground'
                : 'border-dashed border-muted-foreground/40 bg-muted/30 text-muted-foreground hover:text-foreground',
            ].join(' ')}
          >
            {showingRaw ? copy.viewResult(a, b, c) : copy.viewRaw(preset.raw.of)}
          </button>
        ) : null}

        <NeighborBars list={list} answer={preset.answer} barAria={copy.barAria} />

        <p className="text-[11px] text-muted-foreground">{preset.id === 'king' ? copy.kingCaption : copy.parisCaption}</p>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Panel 2 — plurality direction                                       */}
      {/* ------------------------------------------------------------------ */}
      <div className="space-y-3 border-t border-border/60 pt-3">
        <div className="text-xs font-medium text-foreground">{copy.pluralityHeading}</div>
        <p className="text-[11px] text-muted-foreground">{copy.pluralityIntro}</p>

        <div className="space-y-1">
          <div className="grid grid-cols-[5.5rem_minmax(0,1fr)_2.75rem_5.5rem_minmax(0,1fr)_2.75rem] items-center gap-x-2 text-[10px] uppercase tracking-wider text-muted-foreground">
            <span>{copy.singularCol}</span>
            <span />
            <span />
            <span>{copy.pluralCol}</span>
            <span />
            <span />
          </div>
          {PLURALITY_PAIRS.map((pair) => (
            <div
              key={pair.singular}
              className="grid grid-cols-[5.5rem_minmax(0,1fr)_2.75rem_5.5rem_minmax(0,1fr)_2.75rem] items-center gap-x-2 text-xs"
            >
              <span className="truncate font-mono text-foreground/90">{pair.singular}</span>
              <DivergingBar value={pair.sCos} ariaLabel={copy.pluralityBarAria(pair.singular, formatSigned(pair.sCos))} />
              <span className="text-right font-mono text-[11px] text-foreground/80">{formatSigned(pair.sCos)}</span>
              <span className="truncate font-mono text-foreground/90">{pair.plural}</span>
              <DivergingBar value={pair.pCos} ariaLabel={copy.pluralityBarAria(pair.plural, formatSigned(pair.pCos))} />
              <span className="text-right font-mono text-[11px] text-foreground/80">{formatSigned(pair.pCos)}</span>
            </div>
          ))}
        </div>

        <p className="text-[11px] text-muted-foreground">{copy.numbersLabel}</p>
        <div className="space-y-1">
          {PLURALITY_NUMBERS.map((n) => (
            <div key={n.word} className="grid grid-cols-[5.5rem_minmax(0,1fr)_2.75rem] items-center gap-x-2 text-xs">
              <span className="font-mono text-foreground/90">{n.word}</span>
              <DivergingBar value={n.cos} ariaLabel={copy.pluralityBarAria(n.word, formatSigned(n.cos))} />
              <span className="text-right font-mono text-[11px] text-foreground/80">{formatSigned(n.cos)}</span>
            </div>
          ))}
        </div>
      </div>

      <p className="border-t border-border/60 pt-3 text-[11px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
