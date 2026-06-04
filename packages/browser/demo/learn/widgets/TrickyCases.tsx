import * as React from 'react';

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

// Each entry: a tokenizer-quirk demo, its expected token strings, and the
// teaching takeaway. The `display` for each token follows chapter 1's
// convention — a leading space is rendered as the middle-dot `·`.
type Case = {
  label: string;
  input: string;
  tokens: string[];
  takeaway: string;
};

const CASES: Case[] = [
  {
    label: 'Leading whitespace',
    input: 'cat',
    tokens: ['cat'],
    takeaway: 'Bare "cat" is one token. Compare with " cat" below — they are different vocabulary entries.',
  },
  {
    label: 'Leading whitespace',
    input: ' cat',
    tokens: ['·cat'],
    takeaway: 'The leading space is part of the token. " cat" is its own id, not the same as "cat".',
  },
  {
    label: 'Contraction',
    input: "can't",
    tokens: ['can', "'t"],
    takeaway: 'BPE splits the apostrophe off — "\'t" is a frequent enough suffix to earn its own token.',
  },
  {
    label: 'Expanded form',
    input: 'can not',
    tokens: ['can', '·not'],
    takeaway:
      'Two whole words, two tokens. Two tokens here vs two for "can\'t" — but the model sees very different ids.',
  },
  {
    label: 'URL',
    input: 'https://example.com',
    tokens: ['https', '://', 'example', '.com'],
    takeaway: 'URLs fragment by punctuation. The model sees scheme, separator, host, TLD as distinct pieces.',
  },
  {
    label: 'Code',
    input: 'const x = 42;',
    tokens: ['const', '·x', '·=', '·42', ';'],
    takeaway:
      'Code merges common keywords ("const") but splits punctuation. Whitespace is absorbed into the following token.',
  },
  {
    label: 'CJK',
    input: '你好',
    tokens: ['你', '好'],
    takeaway: 'CJK characters usually land as one token each — large vocab coverage, no leading-space marker needed.',
  },
  {
    label: 'Devanagari',
    input: 'नमस्ते',
    tokens: ['⟨b⟩', '⟨b⟩', '⟨b⟩', '⟨b⟩'],
    takeaway:
      'Rarer scripts decompose into byte-level fragments that do not line up with visible characters — here 6 characters become 4 opaque byte tokens (shown as ⟨b⟩). The chars/token ratio drops well below 1.',
  },
  {
    label: 'Number with commas',
    input: '1,000,000',
    tokens: ['1', ',', '0', '0', '0', ',', '0', '0', '0'],
    takeaway:
      'Numbers are split into individual digits — there is no "000" token. The commas stay separate too, so the model reasons over numbers one digit at a time.',
  },
];

function renderFragment(raw: string): string {
  if (raw === '') return '∅';
  let text = raw.replace(/\n/g, '↵').replace(/\t/g, '→');
  if (text.length > 18) text = text.slice(0, 17) + '…';
  return text;
}

export function TrickyCases() {
  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">Tricky tokenization cases</div>
        <div className="text-[11px] text-muted-foreground">
          Token strings are plausible; exact ids vary by tokenizer release.
        </div>
      </div>

      <dl className="space-y-3">
        {CASES.map((c, ci) => (
          <div
            key={`${c.label}-${ci}`}
            className="grid grid-cols-1 gap-2 rounded-md border border-border/60 bg-muted/20 p-3 sm:grid-cols-[12rem_minmax(0,1fr)]"
          >
            <div>
              <dt className="text-[11px] uppercase tracking-wider text-muted-foreground">{c.label}</dt>
              <dd className="mt-1 break-all font-mono text-[12px] text-foreground/90">{JSON.stringify(c.input)}</dd>
            </div>
            <div className="space-y-1.5">
              <div role="list" aria-label={`Tokens for ${c.input}`} className="flex flex-wrap gap-1">
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
              <p className="text-[11px] text-muted-foreground">{c.takeaway}</p>
            </div>
          </div>
        ))}
      </dl>
    </div>
  );
}
