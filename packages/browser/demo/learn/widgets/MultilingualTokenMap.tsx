import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Multilingual token map — the SAME sentence ("the cat sat on the mat")
 * written in six scripts, tokenized into very different token counts by a
 * SINGLE tokenizer (Qwen3.5's 248,320-entry vocabulary). High-resource
 * languages get whole-word merges (Chinese is the most compact here at 4
 * tokens); Spanish and Hindi cost the most; rarer scripts and emoji fall back
 * to raw byte fragments.
 *
 * Token strings and counts are REAL — captured offline from Qwen3.5's
 * tokenizer.json (encode without special tokens, then decode each id
 * individually). The widget shows only the per-token chip text and counts,
 * never the integer ids. A leading space is rendered as the middle-dot `·`,
 * matching chapter 1's convention.
 */

// Per-script chip palette — one palette color per language so each row reads
// as a distinct color band. Mirrors chapter 1's CHIP_PALETTE hues.
const SCRIPT_PALETTE = {
  sky: 'bg-sky-100 dark:bg-sky-950/40 text-sky-900 dark:text-sky-100',
  amber: 'bg-amber-100 dark:bg-amber-950/40 text-amber-900 dark:text-amber-100',
  emerald: 'bg-emerald-100 dark:bg-emerald-950/40 text-emerald-900 dark:text-emerald-100',
  rose: 'bg-rose-100 dark:bg-rose-950/40 text-rose-900 dark:text-rose-100',
  violet: 'bg-violet-100 dark:bg-violet-950/40 text-violet-900 dark:text-violet-100',
} as const;

type PaletteKey = keyof typeof SCRIPT_PALETTE;

type ScriptRow = {
  /**
   * LTR row label, e.g. "中文 (Chinese)". The English label doubles as the
   * stable React key; the displayed label comes from COPY.rowLabels.
   */
  label: string;
  /** The parallel phrase in this script. */
  phrase: string;
  /** Real per-token strings from the tokenizer; `·` prefixes a leading-space token. */
  tokens: string[];
  /** One palette color per script. */
  palette: PaletteKey;
  /** Render the chip strip right-to-left (Arabic only). */
  rtl?: boolean;
};

// Six scripts, one parallel sentence ("the cat sat on the mat"). Each token
// string is the verbatim output of decoding one id from Qwen3.5's tokenizer;
// a leading space is shown as `·`. Counts are the exact encode lengths.
const ROWS: ScriptRow[] = [
  {
    label: 'English',
    phrase: 'The cat sat on the mat',
    tokens: ['The', '·cat', '·sat', '·on', '·the', '·mat'],
    palette: 'sky',
  },
  {
    label: '中文 (Chinese)',
    phrase: '猫坐在垫子上',
    tokens: ['猫', '坐在', '垫', '子上'],
    palette: 'emerald',
  },
  {
    label: '日本語 (Japanese)',
    phrase: '猫がマットの上に座った',
    tokens: ['猫', 'が', 'マット', 'の上に', '座', 'った'],
    palette: 'amber',
  },
  {
    label: 'العربية (Arabic, RTL)',
    phrase: 'جلس القط على الحصيرة',
    tokens: ['جلس', '·القط', '·على', '·الحص', 'يرة'],
    palette: 'rose',
    rtl: true,
  },
  {
    label: 'हिन्दी (Hindi)',
    phrase: 'बिल्ली चटाई पर बैठ गई',
    tokens: ['ब', 'िल', '्ली', '·च', 'टा', 'ई', '·पर', '·बैठ', '·गई'],
    palette: 'violet',
  },
  {
    label: 'Español (Spanish)',
    phrase: 'El gato se sentó en la alfombra',
    tokens: ['El', '·gato', '·se', '·sent', 'ó', '·en', '·la', '·alf', 'ombra'],
    palette: 'sky',
  },
];

// Bars are sized against a fixed max so the busiest rows (9 tokens) fill the bar.
const MAX_COUNT = 9;

const COPY = {
  en: {
    title: 'Multilingual token map — one sentence, six scripts',
    subtitle: "Captured from Qwen3.5's tokenizer.json; ids omitted.",
    // Display labels, aligned with ROWS by index.
    rowLabels: [
      'English',
      '中文 (Chinese)',
      '日本語 (Japanese)',
      'العربية (Arabic, RTL)',
      'हिन्दी (Hindi)',
      'Español (Spanish)',
    ],
    tokensFor: (label: string) => `Tokens for ${label}`,
    countAria: (n: number, label: string) => `${n} tokens for ${label}`,
    countLabel: (n: number) => `${n} tokens`,
    body: (
      <>
        The same sentence costs a different number of tokens in each language — but it is not simply &ldquo;English is
        cheapest.&rdquo; Qwen3.5&apos;s 248,320-entry vocabulary has learned whole-word merges for high-resource
        languages, so Chinese here is actually the <strong>most compact</strong> (4 tokens), while Spanish and Hindi
        cost the most (9). Token counts only compare <strong>within this one tokenizer</strong>.
      </>
    ),
    fallbackNote:
      'When text falls outside those learned merges — emoji, or rarely-seen scripts — the tokenizer drops to raw bytes: each emoji like 🦫 is 3 tokens, and Amharic runs about 1.7 tokens per character.',
    footnote:
      'Actual Qwen3.5-0.8B tokenizer output, captured offline — exact token splits (token ids omitted), not a live run in your browser.',
  },
  zh: {
    title: '多语言 token 地图——一句话，六种文字',
    subtitle: '采自 Qwen3.5 的 tokenizer.json；id 已省略。',
    rowLabels: [
      '英语（English）',
      '中文',
      '日语（日本語）',
      '阿拉伯语（العربية，RTL）',
      '印地语（हिन्दी）',
      '西班牙语（Español）',
    ],
    tokensFor: (label: string) => `${label}的 token`,
    countAria: (n: number, label: string) => `${label}共 ${n} 个 token`,
    countLabel: (n: number) => `${n} 个 token`,
    body: (
      <>
        同一句话在每种语言里花费的 token 数都不一样——但这并不是简单的“英语最便宜”。Qwen3.5 的 248,320
        条词表为高资源语言学出了整词合并，所以这里中文反而<strong>最紧凑</strong>（4 个
        token），而西班牙语和印地语最贵（9 个）。token 数只在<strong>同一个分词器内部</strong>才可比。
      </>
    ),
    fallbackNote:
      '当文本落在这些学到的合并之外——emoji，或少见的文字——分词器就退回原始字节：像 🦫 这样的 emoji 每个要 3 个 token，阿姆哈拉语大约每字符 1.7 个 token。',
    footnote:
      '真实的 Qwen3.5-0.8B 分词器输出，离线采集——精确的 token 切分（省略 token id），不是在你浏览器里的实时运行。',
  },
} as const;

export function MultilingualTokenMap() {
  const copy = COPY[useLocale()];
  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="text-[11px] text-muted-foreground">{copy.subtitle}</div>
      </div>

      <dl className="space-y-3">
        {ROWS.map((row, ri) => {
          const count = row.tokens.length;
          const barPct = (count / MAX_COUNT) * 100;
          const palette = SCRIPT_PALETTE[row.palette];
          const rowLabel = copy.rowLabels[ri]!;
          return (
            <div
              key={row.label}
              className="grid grid-cols-1 gap-2 rounded-md border border-border/60 bg-muted/20 p-3 sm:grid-cols-[12rem_minmax(0,1fr)]"
            >
              <div>
                <dt className="text-[11px] uppercase tracking-wider text-muted-foreground">{rowLabel}</dt>
                <dd className="mt-1 break-words font-mono text-[12px] text-foreground/90">{row.phrase}</dd>
              </div>
              <div className="space-y-1.5">
                <div
                  role="list"
                  aria-label={copy.tokensFor(rowLabel)}
                  dir={row.rtl ? 'rtl' : undefined}
                  className="flex flex-wrap gap-1"
                >
                  {row.tokens.map((tok, ti) => (
                    <span
                      key={`${row.label}-${ti}-${tok}`}
                      role="listitem"
                      className={[
                        'inline-flex items-center rounded px-1.5 py-1 text-[12px] font-mono leading-none border border-transparent',
                        palette,
                      ].join(' ')}
                    >
                      {tok}
                    </span>
                  ))}
                </div>
                {/* Visual count comparison: bar width ∝ token count. */}
                <div className="flex items-center gap-2">
                  <div
                    className="h-1.5 flex-1 overflow-hidden rounded bg-muted/40"
                    role="img"
                    aria-label={copy.countAria(count, rowLabel)}
                  >
                    <div className="h-full rounded bg-primary/30" style={{ width: `${barPct}%` }} />
                  </div>
                  <span className="shrink-0 font-mono text-[11px] text-foreground/85">{copy.countLabel(count)}</span>
                </div>
              </div>
            </div>
          );
        })}
      </dl>

      <p className="text-[12px] text-foreground/85">{copy.body}</p>

      <p className="text-[11px] text-muted-foreground">{copy.fallbackNote}</p>

      <p className="text-[10px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
