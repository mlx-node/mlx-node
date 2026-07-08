import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Tricky tokenization cases — a static grid of curated inputs that
 * demonstrate the quirks BPE introduces. Token strings are hard-coded to a
 * plausible Qwen3-style split; exact ids vary per tokenizer release, so the
 * widget only displays the *text* of each chip, not the integer id.
 */

// Chip palette mirrors chapter 1's CHIP_PALETTE for visual consistency.
const CHIP_PALETTE = [
  'bg-sky-100 dark:bg-sky-950/40 text-sky-900 dark:text-sky-100',
  'bg-amber-100 dark:bg-amber-950/40 text-amber-900 dark:text-amber-100',
  'bg-emerald-100 dark:bg-emerald-950/40 text-emerald-900 dark:text-emerald-100',
  'bg-rose-100 dark:bg-rose-950/40 text-rose-900 dark:text-rose-100',
  'bg-violet-100 dark:bg-violet-950/40 text-violet-900 dark:text-violet-100',
];

// Each entry: a tokenizer-quirk demo and its expected token strings. The
// per-case prose (label + teaching takeaway) lives in COPY below, aligned by
// index. The `display` for each token follows chapter 1's convention — a
// leading space is rendered as the middle-dot `·`.
type Case = {
  input: string;
  tokens: string[];
};

const CASES: Case[] = [
  { input: 'cat', tokens: ['cat'] },
  { input: ' cat', tokens: ['·cat'] },
  { input: "can't", tokens: ['can', "'t"] },
  { input: 'can not', tokens: ['can', '·not'] },
  { input: 'https://example.com', tokens: ['https', '://', 'example', '.com'] },
  { input: 'const x = 42;', tokens: ['const', '·x', '·=', '·', '4', '2', ';'] },
  { input: '你好', tokens: ['你好'] },
  { input: 'नमस्ते', tokens: ['न', 'म', 'स्', 'ते'] },
  { input: '1,000,000', tokens: ['1', ',', '0', '0', '0', ',', '0', '0', '0'] },
];

type CaseCopy = { label: string; takeaway: string };

const COPY = {
  en: {
    title: 'Tricky tokenization cases',
    subtitle: 'Token strings are plausible; exact ids vary by tokenizer release.',
    tokensFor: (input: string) => `Tokens for ${input}`,
    cases: [
      {
        label: 'Leading whitespace',
        takeaway: 'Bare "cat" is one token. Compare with " cat" below — they are different vocabulary entries.',
      },
      {
        label: 'Leading whitespace',
        takeaway: 'The leading space is part of the token. " cat" is its own id, not the same as "cat".',
      },
      {
        label: 'Contraction',
        takeaway: 'BPE splits the apostrophe off — "\'t" is a frequent enough suffix to earn its own token.',
      },
      {
        label: 'Expanded form',
        takeaway:
          'Two whole words, two tokens. Two tokens here vs two for "can\'t" — but the model sees very different ids.',
      },
      {
        label: 'URL',
        takeaway: 'URLs fragment by punctuation. The model sees scheme, separator, host, TLD as distinct pieces.',
      },
      {
        label: 'Code',
        takeaway:
          'Code merges common keywords ("const") but splits punctuation. Whitespace is absorbed into the following token.',
      },
      {
        label: 'CJK',
        takeaway:
          'CJK characters usually land as one token each, but common bigrams like "你好" can merge into a single token too — much like English "the" earning its own id.',
      },
      {
        label: 'Devanagari',
        takeaway:
          'Rarer scripts decompose into more, smaller pieces than English — here 6 characters become 4 tokens — but each piece is still a legible sub-syllable, not an opaque byte fragment. The chars/token ratio drops well below 1.',
      },
      {
        label: 'Number with commas',
        takeaway:
          'Numbers are split into individual digits — there is no "000" token. The commas stay separate too, so the model reasons over numbers one digit at a time.',
      },
    ] as readonly CaseCopy[],
  },
  zh: {
    title: '刁钻的分词案例',
    subtitle: 'token 字符串为合理示意；具体 id 随分词器版本而变。',
    tokensFor: (input: string) => `${input} 的 token`,
    cases: [
      {
        label: '前导空格',
        takeaway: '裸的 "cat" 是一个 token。与下面的 " cat" 对比——它们是两个不同的词表条目。',
      },
      {
        label: '前导空格',
        takeaway: '前导空格是 token 的一部分。" cat" 有自己的 id，和 "cat" 不是同一个。',
      },
      {
        label: '缩写',
        takeaway: 'BPE 把撇号拆了出来——"\'t" 这个后缀足够常见，挣到了自己的 token。',
      },
      {
        label: '展开形式',
        takeaway: '两个完整的词，两个 token。这里两个，"can\'t" 也是两个——但模型看到的 id 截然不同。',
      },
      {
        label: 'URL',
        takeaway: 'URL 按标点碎裂。在模型眼里，协议、分隔符、主机名、顶级域名是各自独立的片段。',
      },
      {
        label: '代码',
        takeaway: '代码会合并常见关键字（"const"），但把标点拆开。空白被吸进后面的 token。',
      },
      {
        label: 'CJK',
        takeaway: 'CJK 字符通常一字一个 token，但像 "你好" 这样常见的双字组合也可能合并成一个 token——很像英文 "the" 独占一个 id。',
      },
      {
        label: '天城文',
        takeaway:
          '少见文字会分解成比英文单词更多、更细的片段——这里 6 个字符变成 4 个 token——但每一片仍是可读的音节片段，而不是不透明的字节碎片。chars/token 比值远低于 1。',
      },
      {
        label: '带逗号的数字',
        takeaway:
          '数字被拆成一个个数位——并没有 "000" 这个 token。逗号也单独成块，所以模型是一位一位地对数字做推理。',
      },
    ] as readonly CaseCopy[],
  },
} as const;

function renderFragment(raw: string): string {
  if (raw === '') return '∅';
  let text = raw.replace(/\n/g, '↵').replace(/\t/g, '→');
  if (text.length > 18) text = text.slice(0, 17) + '…';
  return text;
}

export function TrickyCases() {
  const copy = COPY[useLocale()];
  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="text-[11px] text-muted-foreground">{copy.subtitle}</div>
      </div>

      <dl className="space-y-3">
        {CASES.map((c, ci) => (
          <div
            key={`${c.input}-${ci}`}
            className="grid grid-cols-1 gap-2 rounded-md border border-border/60 bg-muted/20 p-3 sm:grid-cols-[12rem_minmax(0,1fr)]"
          >
            <div>
              <dt className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.cases[ci]!.label}</dt>
              <dd className="mt-1 break-all font-mono text-[12px] text-foreground/90">{JSON.stringify(c.input)}</dd>
            </div>
            <div className="space-y-1.5">
              <div role="list" aria-label={copy.tokensFor(c.input)} className="flex flex-wrap gap-1">
                {c.tokens.map((tok, ti) => {
                  const palette = CHIP_PALETTE[ti % CHIP_PALETTE.length]!;
                  return (
                    <span
                      key={`${ci}-${ti}-${tok}`}
                      role="listitem"
                      className={[
                        'inline-flex items-center rounded px-1.5 py-1 text-[12px] font-mono leading-none border border-transparent',
                        palette,
                      ].join(' ')}
                    >
                      {renderFragment(tok)}
                    </span>
                  );
                })}
              </div>
              <p className="text-[11px] text-muted-foreground">{copy.cases[ci]!.takeaway}</p>
            </div>
          </div>
        ))}
      </dl>
    </div>
  );
}
