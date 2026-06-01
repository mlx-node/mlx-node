import * as React from 'react';

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
  note: string;
};

const PAIRS: Pair[] = [
  {
    a: 'cat',
    b: 'cat',
    similarity: 1.0,
    note: 'Identical token — sanity check. Cosine of a vector with itself is exactly 1.',
  },
  {
    a: 'cat',
    b: ' cat',
    similarity: 0.72,
    note: 'Same word, different leading-space token. Close, but not identical — they really are two ids.',
  },
  {
    a: 'king',
    b: 'queen',
    similarity: 0.65,
    note: 'Synonym-like pair from the same semantic role.',
  },
  {
    a: 'Paris',
    b: 'France',
    similarity: 0.58,
    note: 'Different categories (city, country) but tightly related in training text.',
  },
  {
    a: 'hot',
    b: 'cold',
    similarity: 0.48,
    note: 'Antonyms are often surprisingly close — they appear in similar grammatical slots.',
  },
  {
    a: 'dog',
    b: 'perro',
    similarity: 0.22,
    note: 'Cross-language: shared concept, very different surface form. Weak overlap.',
  },
  {
    a: 'cat',
    b: 'stapler',
    similarity: 0.18,
    note: 'Unrelated common nouns. Low but non-zero — both are just nouns to the model.',
  },
  {
    a: 'xyz',
    b: 'thrombosis',
    similarity: 0.05,
    note: 'Rare and unrelated. Near-orthogonal in the embedding space.',
  },
];

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
  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">Cosine similarity · curated pairs</div>
        <div className="text-[11px] text-muted-foreground">Illustrative ballpark values · Qwen3.5-0.8B-style</div>
      </div>

      <div className="space-y-1.5">
        {PAIRS.map((p, i) => {
          const pct = Math.max(0, Math.min(1, p.similarity)) * 100;
          const color = similarityHue(p.similarity);
          return (
            <div
              key={`${p.a}-${p.b}-${i}`}
              className="grid grid-cols-1 gap-2 rounded-md border border-border/60 bg-muted/20 p-2 sm:grid-cols-[14rem_minmax(0,1fr)_3.5rem]"
              title={p.note}
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
                  aria-label={`cosine similarity ${p.similarity.toFixed(2)}`}
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
              <div className="col-span-full text-[11px] text-muted-foreground">{p.note}</div>
            </div>
          );
        })}
      </div>

      <p className="text-[11px] text-muted-foreground">
        Values are illustrative; real measurements may vary by about +/- 0.1 between model builds. The teaching point is
        the *ordering*: identical &gt; synonym &gt; related &gt; antonym &gt; cross-language &gt; unrelated.
      </p>
    </div>
  );
}
