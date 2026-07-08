import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * SamplingToolCallingSection widget (chapter 11 "Sampling" sub-chapter,
 * "Tool calling & constrained decoding") — grammar-masked decoding, one step
 * at a time, over THIS course's own tool-call tag format (the same
 * `<tool_call><function=NAME><parameter=NAME>VALUE</parameter></function></tool_call>`
 * syntax chapter 15 already introduces, verbatim the same worked example:
 * get_weather / city / Paris).
 *
 * Self-contained: NO worker, NO model, NO WASM — pure React state over fixed,
 * scripted illustrative logits (three decode steps), so it server-renders for
 * crawlers and needs no randomness.
 *
 * IMPORTANT — what this widget illustrates vs. what this repo does: the
 * grammar shown here is a HYPOTHETICAL finite-state grammar someone could
 * write for this exact tag format (the same idea as Outlines' regex/CFG → FSM
 * compilation, or llama.cpp's GBNF) — a defensible, illustrative design where
 * a bare `<` is disallowed inside a string value (it would be ambiguous with
 * a real tag) and the "arguments" schema for `get_weather` is known to have
 * exactly one field, `city`. It is NOT a claim about this repo's actual
 * parser, which (crates/mlx-core/src/tools/mod.rs, `parse_function_tool_call`)
 * does plain post-hoc STRING SCANNING of already-generated text and enforces
 * no such FSM — see the section's own honesty callout for that gap.
 *
 * The mechanism illustrated IS how this chapter's own sampling machinery
 * works: `apply_top_k` / `apply_top_p` / `apply_min_p`
 * (crates/mlx-core/src/sampling.rs) already build a boolean keep-mask over
 * logits and set every excluded entry to -inf before renormalizing — grammar
 * masking is the exact same operation with a different (validity-based, not
 * probability-based) mask.
 *
 * Three decode steps over the SAME running completion, each with 5 candidate
 * next tokens (illustrative logits, sum to 1 per step):
 *   1. right after `<parameter=city>` — value hasn't started; a loose
 *      constraint (any non-`<` text is a valid string, including nonsense).
 *   2. after `...city>Paris` — closing or extending are both valid; adding a
 *      second parameter or leaving `<parameter>` unclosed are not.
 *   3. after `...Paris</parameter>` — get_weather takes exactly one
 *      parameter, so exactly ONE token is grammar-valid: `</function>`. This
 *      is the tight case: the mask alone decides the next token, before
 *      temperature/top-p/argmax even get a say.
 */

type ReasonKey =
  | 'freeText'
  | 'weirdFreeText'
  | 'rawAngleBracket'
  | 'wrongSyntax'
  | 'emptyRequired'
  | 'closesValue'
  | 'extraParam'
  | 'unclosedParam'
  | 'onlyValid'
  | 'freeTextNotAllowed'
  | 'strayClose';

type Candidate = { token: string; prob: number; valid: boolean; reason: ReasonKey };

type StepId = 1 | 2 | 3;

// Locale-independent: the running prefix, candidate tokens, and their
// probabilities/validity never change between languages — only the
// human-readable reason strings (COPY.notes) and labels translate.
const STEPS: Record<StepId, { soFar: string; candidates: Candidate[] }> = {
  1: {
    soFar: '<tool_call><function=get_weather><parameter=city>',
    candidates: [
      { token: 'Paris', prob: 0.52, valid: true, reason: 'freeText' },
      { token: ' I', prob: 0.09, valid: true, reason: 'weirdFreeText' },
      { token: '<function=', prob: 0.19, valid: false, reason: 'rawAngleBracket' },
      { token: '{', prob: 0.12, valid: false, reason: 'wrongSyntax' },
      { token: '</parameter>', prob: 0.08, valid: false, reason: 'emptyRequired' },
    ],
  },
  2: {
    soFar: '<tool_call><function=get_weather><parameter=city>Paris',
    candidates: [
      { token: '</parameter>', prob: 0.46, valid: true, reason: 'closesValue' },
      { token: ', France', prob: 0.11, valid: true, reason: 'freeText' },
      { token: '<parameter=', prob: 0.17, valid: false, reason: 'extraParam' },
      { token: '</function>', prob: 0.15, valid: false, reason: 'unclosedParam' },
      { token: '{', prob: 0.11, valid: false, reason: 'wrongSyntax' },
    ],
  },
  3: {
    soFar: '<tool_call><function=get_weather><parameter=city>Paris</parameter>',
    candidates: [
      { token: '</function>', prob: 0.55, valid: true, reason: 'onlyValid' },
      { token: '<parameter=', prob: 0.14, valid: false, reason: 'extraParam' },
      { token: ' The', prob: 0.13, valid: false, reason: 'freeTextNotAllowed' },
      { token: '</parameter>', prob: 0.1, valid: false, reason: 'strayClose' },
      { token: '{', prob: 0.08, valid: false, reason: 'wrongSyntax' },
    ],
  },
};

const FINAL_COMPLETION = '<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>';

const COPY = {
  en: {
    header: 'Masking the grammar, one token at a time',
    stepLabel: (n: StepId) => `Step ${n}`,
    stepCaption: {
      1: 'start the value',
      2: 'extend or close it',
      3: 'finish the call',
    } as Record<StepId, string>,
    soFarLabel: 'Completion so far',
    rawHeader: 'Raw probabilities (unconstrained)',
    maskedHeader: 'Grammar-masked → renormalized',
    massKept: (mass: string, kept: number, total: number) => `Mass kept: ${mass} — ${kept} of ${total} candidates survive`,
    sampledTag: 'picked',
    validTag: 'valid',
    invalidTag: 'invalid',
    finalLabel: 'Full completion, step by step:',
    notes: {
      freeText: 'plain text — any content without a raw `<` is a valid string value',
      weirdFreeText: 'also valid — the grammar checks syntax, not meaning',
      rawAngleBracket: 'a raw `<` here would start what looks like a new tag',
      wrongSyntax: 'this tag format never uses JSON braces',
      emptyRequired: 'closes the value before any content — city can’t be empty',
      closesValue: 'closes the value now that it has content',
      extraParam: 'get_weather’s schema defines only `city` — no second parameter',
      unclosedParam: 'would leave this `<parameter>` tag unclosed',
      onlyValid: 'the only continuation left — `city` was the one required parameter',
      freeTextNotAllowed: 'free text isn’t allowed here — outside any value slot',
      strayClose: 'no open `<parameter>` left to close',
    } as Record<ReasonKey, string>,
    footnote:
      'Illustrative probabilities for a hypothetical grammar over this exact tag format — the same idea as Outlines’ regex/CFG-to-FSM compilation or llama.cpp’s GBNF grammars — not a live run of the model, and not what this repo’s own generation loop does (see below).',
    mechanismNote:
      'This is the same machinery as the rest of this chapter: build a keep/reject mask over the vocabulary, set every rejected logit to −∞, renormalize, then run argmax or top-p over what survives. The only thing that changes is what decides the mask — a probability cutoff for top-p, grammar validity here.',
  },
  zh: {
    header: '给语法上 mask，一次一个 token',
    stepLabel: (n: StepId) => `第 ${n} 步`,
    stepCaption: {
      1: '开始填值',
      2: '续写或收尾',
      3: '结束这次调用',
    } as Record<StepId, string>,
    soFarLabel: '目前为止的补全',
    rawHeader: '原始概率（无约束）',
    maskedHeader: '语法 mask → 重新归一化',
    massKept: (mass: string, kept: number, total: number) => `保留的概率质量：${mass} —— ${total} 个候选中有 ${kept} 个存活`,
    sampledTag: '被选中',
    validTag: '合法',
    invalidTag: '不合法',
    finalLabel: '完整的补全，一步步拼出来：',
    notes: {
      freeText: '纯文本——只要不含裸露的 `<`，任何内容都是合法的字符串值',
      weirdFreeText: '同样合法——语法只检查句法，不检查语义',
      rawAngleBracket: '这里出现裸露的 `<` 就会像是在开始一个新标签',
      wrongSyntax: '这套 tag 格式从不使用 JSON 花括号',
      emptyRequired: '还没写任何内容就收尾——city 不能是空的',
      closesValue: '值已经有内容了，可以收尾',
      extraParam: 'get_weather 的 schema 只定义了 `city` 这一个参数——没有第二个',
      unclosedParam: '会让这个 `<parameter>` 标签没法闭合',
      onlyValid: '唯一剩下的延续——`city` 是那个必需的参数',
      freeTextNotAllowed: '这个位置不允许自由文本——已经不在任何取值槽内',
      strayClose: '没有打开的 `<parameter>` 可以闭合',
    } as Record<ReasonKey, string>,
    footnote:
      '这是针对这套 tag 格式假想出的一个语法所用的示意概率——和 Outlines 把 regex/CFG 编译成 FSM、或 llama.cpp 的 GBNF 语法是同一个思路——不是模型的实时运行，也不是本仓库自己的生成循环实际在做的事（见下文）。',
    mechanismNote:
      '这和本章其余部分是同一套机制：在词表上建一个保留/剔除的 mask，把每个被剔除的 logit 设为 −∞，重新归一化，再对存活下来的部分跑 argmax 或 top-p。唯一变化的是谁来决定这个 mask——top-p 用的是概率截断，这里用的是语法合法性。',
  },
} as const;

function TokenRow({
  candidate,
  pct,
  displayProb,
  reasonText,
  emphasize,
  sampledTag,
}: {
  candidate: Candidate;
  pct: number;
  displayProb: string;
  reasonText: string;
  emphasize: boolean;
  sampledTag?: string;
}) {
  return (
    <div className={['space-y-0.5 rounded px-1.5 py-1', candidate.valid ? '' : 'opacity-60'].join(' ')}>
      <div className="grid grid-cols-[1fr_minmax(0,2fr)_3rem] items-center gap-2 text-[12px]">
        <span className="flex items-center gap-1 truncate font-mono">
          <span className={candidate.valid ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}>
            {candidate.valid ? '✓' : '✕'}
          </span>
          <span className={emphasize ? 'font-semibold text-primary' : 'text-foreground/85'}>{candidate.token}</span>
          {emphasize && sampledTag ? (
            <span className="ml-1 rounded bg-primary/15 px-1 py-0.5 text-[10px] font-semibold text-primary">
              {sampledTag}
            </span>
          ) : null}
        </span>
        <div className="relative h-3.5 w-full overflow-hidden rounded bg-muted">
          <div
            className={['h-full origin-left', candidate.valid ? 'bg-primary' : 'bg-foreground/30'].join(' ')}
            style={{ width: `${Math.max(3, Math.min(100, pct))}%` }}
          />
        </div>
        <span className="text-right font-mono text-muted-foreground">{displayProb}</span>
      </div>
      <div className="pl-5 text-[10.5px] leading-snug text-muted-foreground">{reasonText}</div>
    </div>
  );
}

export function ConstrainedDecodeMask() {
  const copy = COPY[useLocale()];
  const [step, setStep] = React.useState<StepId>(1);
  const { soFar, candidates } = STEPS[step];

  const maxRawProb = Math.max(...candidates.map((c) => c.prob));
  const survivors = candidates.filter((c) => c.valid);
  const keptMass = survivors.reduce((a, c) => a + c.prob, 0);
  const maxSurvivorProb = Math.max(...survivors.map((c) => c.prob / keptMass));
  const argmaxSurvivor = survivors.reduce((best, c) => (c.prob > best.prob ? c : best), survivors[0]!);

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <div className="flex gap-1 rounded-full border border-border bg-muted/40 p-0.5 text-[11px]">
          {([1, 2, 3] as const).map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => setStep(n)}
              className={[
                'rounded-full px-2.5 py-1 font-medium transition-colors',
                step === n ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground',
              ].join(' ')}
              aria-pressed={step === n}
            >
              {copy.stepLabel(n)} · {copy.stepCaption[n]}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.soFarLabel}</div>
        <div className="overflow-x-auto rounded border border-border/60 bg-muted/20 p-2 font-mono text-[11px] text-foreground/90">
          {soFar}
          <span className="ml-0.5 bg-primary/70 px-[3px] text-background">▍</span>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-1">
          <div className="text-[11px] text-muted-foreground">{copy.rawHeader}</div>
          <div className="space-y-0.5 rounded-md border border-border bg-background p-2">
            {candidates.map((c) => (
              <TokenRow
                key={c.token}
                candidate={c}
                pct={(c.prob / maxRawProb) * 100}
                displayProb={c.prob.toFixed(2)}
                reasonText={copy.notes[c.reason]}
                emphasize={false}
              />
            ))}
          </div>
        </div>

        <div className="space-y-1">
          <div className="text-[11px] text-muted-foreground">{copy.maskedHeader}</div>
          <div className="space-y-0.5 rounded-md border border-primary/30 bg-background p-2">
            {survivors.map((c) => (
              <TokenRow
                key={c.token}
                candidate={c}
                pct={((c.prob / keptMass) / maxSurvivorProb) * 100}
                displayProb={(c.prob / keptMass).toFixed(2)}
                reasonText={copy.notes[c.reason]}
                emphasize={c.token === argmaxSurvivor.token}
                sampledTag={copy.sampledTag}
              />
            ))}
          </div>
          <div className="text-[10px] text-muted-foreground">
            {copy.massKept(keptMass.toFixed(2), survivors.length, candidates.length)}
          </div>
        </div>
      </div>

      <p className="text-[11px] text-foreground/80">{copy.mechanismNote}</p>

      <div className="rounded border border-border/60 bg-muted/10 p-2 text-[11px]">
        <div className="text-muted-foreground">{copy.finalLabel}</div>
        <div className="mt-0.5 overflow-x-auto font-mono text-foreground/85">{FINAL_COMPLETION}</div>
      </div>

      <p className="text-[10px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
