import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Curated cosine-similarity pairs. Values are illustrative ballpark numbers
 * meant to teach the *shape* of cosine similarities in a real LLM embedding
 * matrix — they should be treated as accurate to about +/- 0.1 against any
 * particular Qwen3.5-0.8B build. The point is comparison: synonyms cluster
 * tighter than antonyms, antonyms tighter than cross-language, and so on.
 */

type Pair = {
  a: string;
  b: string;
  similarity: number;
};

// Token pairs and similarity values are locale-independent; the per-pair
// teaching notes live in COPY below, aligned by index.
const PAIRS: Pair[] = [
  { a: 'cat', b: 'cat', similarity: 1.0 },
  { a: 'cat', b: ' cat', similarity: 0.72 },
  { a: 'king', b: 'queen', similarity: 0.65 },
  { a: 'Paris', b: 'France', similarity: 0.58 },
  { a: 'hot', b: 'cold', similarity: 0.48 },
  { a: 'dog', b: 'perro', similarity: 0.22 },
  { a: 'cat', b: 'stapler', similarity: 0.18 },
  { a: 'xyz', b: 'thrombosis', similarity: 0.05 },
];

const COPY = {
  en: {
    title: 'Cosine similarity · curated pairs',
    subtitle: 'Illustrative ballpark values · Qwen3.5-0.8B-style',
    notes: [
      'Identical token — sanity check. Cosine of a vector with itself is exactly 1.',
      'Same word, different leading-space token. Close, but not identical — they really are two ids.',
      'Synonym-like pair from the same semantic role.',
      'Different categories (city, country) but tightly related in training text.',
      'Antonyms are often surprisingly close — they appear in similar grammatical slots.',
      'Cross-language: shared concept, very different surface form. Weak overlap.',
      'Unrelated common nouns. Low but non-zero — both are just nouns to the model.',
      'Rare and unrelated. Near-orthogonal in the embedding space.',
    ],
    barAria: (sim: string) => `cosine similarity ${sim}`,
    footnote: (
      <>
        Values are illustrative; real measurements may vary by about +/- 0.1 between model builds. The teaching point is
        the *ordering*: identical &gt; synonym &gt; related &gt; antonym &gt; cross-language &gt; unrelated.
      </>
    ),
  },
  zh: {
    title: '余弦相似度 · 精选词对',
    subtitle: '示意性大致数值 · Qwen3.5-0.8B 风格',
    notes: [
      '完全相同的 token——健全性检查。向量与自身的余弦恰好是 1。',
      '同一个词，带不带前导空格是两个 token。很接近，但不相同——它们确实是两个 id。',
      '同一语义角色下的类同义词对。',
      '类别不同（城市、国家），但在训练文本中紧密相关。',
      '反义词常常近得出人意料——它们出现在相似的语法槽位里。',
      '跨语言：概念相同，表面形式迥异。重叠很弱。',
      '不相关的普通名词。低但不为零——在模型眼里它们都只是名词。',
      '罕见且不相关。在嵌入空间中接近正交。',
    ],
    barAria: (sim: string) => `余弦相似度 ${sim}`,
    footnote: (
      <>
        数值仅为示意；不同模型构建之间的实测可能相差约 ±0.1。要教的结论是*排序*：完全相同 &gt; 同义词 &gt; 相关 &gt;
        反义词 &gt; 跨语言 &gt; 不相关。
      </>
    ),
  },
} as const;

function similarityHue(sim: number): string {
  // Low similarity → muted gray, high → primary teal/blue tone.
  // hsl traversed from a desaturated 215 hue toward a saturated 215.
  const clamped = Math.max(0, Math.min(1, sim));
  const sat = 10 + clamped * 65;
  const light = 60 - clamped * 25;
  return `hsl(215 ${sat.toFixed(0)}% ${light.toFixed(0)}%)`;
}

function renderToken(text: string): string {
  if (text.startsWith(' ')) return '·' + text.slice(1);
  return text;
}

export function CosineSimilarityTool() {
  const copy = COPY[useLocale()];
  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="text-[11px] text-muted-foreground">{copy.subtitle}</div>
      </div>

      <div className="space-y-1.5">
        {PAIRS.map((p, i) => {
          const pct = Math.max(0, Math.min(1, p.similarity)) * 100;
          const color = similarityHue(p.similarity);
          const note = copy.notes[i]!;
          return (
            <div
              key={`${p.a}-${p.b}-${i}`}
              className="grid grid-cols-1 gap-2 rounded-md border border-border/60 bg-muted/20 p-2 sm:grid-cols-[14rem_minmax(0,1fr)_3.5rem]"
              title={note}
            >
              <div className="flex items-center gap-1.5 font-mono text-[12px]">
                <span className="rounded bg-background px-1.5 py-0.5 text-foreground/90 border border-border/50">
                  {renderToken(p.a)}
                </span>
                <span className="text-muted-foreground">↔</span>
                <span className="rounded bg-background px-1.5 py-0.5 text-foreground/90 border border-border/50">
                  {renderToken(p.b)}
                </span>
              </div>
              <div className="flex items-center">
                <div
                  className="relative h-4 w-full overflow-hidden rounded-sm bg-muted/60"
                  role="img"
                  aria-label={copy.barAria(p.similarity.toFixed(2))}
                >
                  <div
                    className="absolute inset-y-0 left-0 transition-all"
                    style={{
                      width: `${pct.toFixed(1)}%`,
                      backgroundColor: color,
                    }}
                  />
                </div>
              </div>
              <div className="text-right font-mono text-[12px] text-foreground/80">{p.similarity.toFixed(2)}</div>
              <div className="col-span-full text-[11px] text-muted-foreground">{note}</div>
            </div>
          );
        })}
      </div>

      <p className="text-[11px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
