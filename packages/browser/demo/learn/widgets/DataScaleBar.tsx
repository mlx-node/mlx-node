import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { SegmentedToggle } from '../scaffolding/SegmentedToggle';

/**
 * Chapter 15 supplement — the staggering scale gulf between pretraining data
 * and post-training (SFT) data.
 *
 * Pretraining sees trillions of *tokens* of generic text; SFT only needs a few
 * thousand to a few million curated *examples* to reshape behavior. The two
 * quantities are measured in different units — tokens vs conversations — so the
 * comparison is an order-of-magnitude scale story, NOT a literal apples-to-apples
 * bar. The toggle makes that explicit:
 *   • LINEAR: width ∝ raw count. Pretraining fills the track; the SFT bar is a
 *     sub-pixel sliver (≈0.00003% of the pretraining width) — so it's floored to
 *     a hairline min-width and disclosed, because the whole point is that SFT
 *     *vanishes* at linear scale.
 *   • LOG: width ∝ log10(count). Both bars become visible and the
 *     multiple-orders-of-magnitude gap reads clearly.
 *
 * Numbers are illustrative orders of magnitude (real corpora vary widely). The
 * bar idiom mirrors SwiGluVsPlain.
 */

type Scale = 'linear' | 'log';

// Illustrative, order-of-magnitude representative values (real corpora vary).
const PRETRAIN_TOKENS = 3_000_000_000_000; // ≈ 3T tokens
const SFT_EXAMPLES = 1_000_000; // ≈ 1M examples

// On the linear scale the SFT bar is sub-pixel, so floor it to a hairline so it
// stays visible — and disclose that we did so.
const LINEAR_FLOOR_PCT = 0.5;

const PRETRAIN_BG = 'oklch(0.7 0.13 220 / 0.32)';
const SFT_BG = 'oklch(0.7 0.13 60 / 0.4)';

const EASE = 'cubic-bezier(0.4, 0, 0.2, 1)';

// Per-locale copy. The amounts ("≈ 3T tokens" / "≈ 1M examples") are dataset
// sizes and stay byte-identical in both locales.
const COPY = {
  en: {
    header: 'Pretraining vs SFT — data scale',
    linear: 'Linear',
    log: 'Log',
    pretrainLabel: 'Pretraining (next-token prediction)',
    pretrainAria: 'Pretraining: approximately 3 trillion tokens, filling the full width of the track.',
    sftLabel: 'Instruction tuning (SFT)',
    sftAriaLog:
      'Instruction tuning: approximately 1 million examples, about 48 percent of the track on a log scale — six orders of magnitude smaller than pretraining.',
    sftAriaLinear:
      'Instruction tuning: approximately 1 million examples, a sub-pixel sliver next to pretraining, floored to a hairline so it stays visible.',
    sftLinearNote:
      'SFT bar floored to remain visible on the linear scale — its true width here is ≈ 0.00003% of the pretraining bar.',
    summary: (oom: number) => (
      <>
        <span className="font-mono">≈ 3T tokens</span> vs <span className="font-mono">≈ 1M examples</span> — about{' '}
        <span className="font-mono">10^{oom}×</span> apart in raw count. (Illustrative of scale only: those are
        different units — <em>tokens</em> of generic text vs curated <em>examples</em> — so this is not a strict
        apples-to-apples ratio.)
      </>
    ),
    body: (
      <>
        Pretraining digests <strong>trillions of tokens</strong> to build the model's knowledge; SFT needs only{' '}
        <strong>thousands to millions of curated examples</strong> to reshape behavior into a helpful assistant. On the{' '}
        <strong>linear</strong> scale the SFT data all but vanishes — that's the point. Switch to <strong>log</strong>{' '}
        to see the gap as the multiple-orders-of-magnitude story it really is. SFT is{' '}
        <em>tiny next to pretraining</em>, which is why post-training is comparatively cheap and fast.
      </>
    ),
    footnote:
      'Illustrative orders of magnitude — real corpora vary widely. Tokens (pretraining) and examples (SFT) are different units, so the ratio reads as scale, not a literal apples-to-apples comparison.',
  },
  zh: {
    header: '预训练 vs SFT——数据规模',
    linear: '线性',
    log: '对数',
    pretrainLabel: '预训练（下一个 token 预测）',
    pretrainAria: '预训练：约 3 万亿 token，填满整条轨道。',
    sftLabel: '指令微调（SFT）',
    sftAriaLog: '指令微调：约 100 万条示例，对数刻度下约占轨道的 48%——比预训练小六个数量级。',
    sftAriaLinear: '指令微调：约 100 万条示例，在预训练旁边只是一道亚像素的细缝，已被垫高成一条发丝线以保持可见。',
    sftLinearNote: '为了在线性刻度下保持可见，SFT 条已被垫高——它在这里的真实宽度约为预训练条的 0.00003%。',
    summary: (oom: number) => (
      <>
        <span className="font-mono">≈ 3T tokens</span> vs <span className="font-mono">≈ 1M examples</span>——原始数量相差约{' '}
        <span className="font-mono">10^{oom}×</span>。（仅用于示意规模：两者单位不同——一个是普通文本的{' '}
        <em>token</em>，另一个是精心整理的<em>示例</em>——所以这不是严格的同类比较。）
      </>
    ),
    body: (
      <>
        预训练消化<strong>数万亿 token</strong> 来构建模型的知识；SFT 只需要<strong>几千到几百万条精心整理的示例</strong>
        就能把行为重塑成一个有用的助手。在<strong>线性</strong>刻度下，SFT 的数据几乎消失不见——这正是重点。切换到
        <strong>对数</strong>刻度，才能看清这道相差多个数量级的鸿沟。SFT <em>在预训练面前微不足道</em>
        ，这正是后训练相对便宜、快速的原因。
      </>
    ),
    footnote:
      '示意性的数量级——真实语料差异很大。token（预训练）和示例（SFT）是不同的单位，所以这个比值只用来读规模，不是字面意义上的同类比较。',
  },
} as const;

function Bar({
  label,
  amount,
  widthPct,
  bg,
  ariaLabel,
  note,
}: {
  label: string;
  amount: string;
  widthPct: number;
  bg: string;
  ariaLabel: string;
  note?: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between text-[11px]">
        <span className="text-foreground/85">{label}</span>
        <span className="font-mono text-muted-foreground">{amount}</span>
      </div>
      <div
        className="flex h-9 w-full overflow-hidden rounded-md border border-border bg-muted/40"
        role="img"
        aria-label={ariaLabel}
      >
        <div
          className="flex h-full min-w-0 items-center overflow-hidden rounded-[3px] px-2"
          style={{ width: `${widthPct}%`, backgroundColor: bg, transition: `width 550ms ${EASE}` }}
        >
          {widthPct > 12 ? <span className="truncate font-mono text-[10px] text-foreground/90">{amount}</span> : null}
        </div>
      </div>
      {note ? <p className="text-[10px] text-muted-foreground">{note}</p> : null}
    </div>
  );
}

export function DataScaleBar() {
  const copy = COPY[useLocale()];
  const [scale, setScale] = React.useState<Scale>('linear');
  const log = scale === 'log';

  // LINEAR: width ∝ raw count, pretraining pinned to the full track. SFT is
  // sub-pixel (≈0.00003%), so floor it to a hairline and disclose the floor.
  const sftRawPct = (SFT_EXAMPLES / PRETRAIN_TOKENS) * 100; // ≈ 0.0000333
  const sftLinearPct = Math.max(sftRawPct, LINEAR_FLOOR_PCT);

  // LOG: width ∝ log10(count), pretraining pinned to the full track.
  const pretrainLog = Math.log10(PRETRAIN_TOKENS); // ≈ 12.48
  const sftLog = Math.log10(SFT_EXAMPLES); // 6
  const sftLogPct = (sftLog / pretrainLog) * 100; // ≈ 48.1

  const pretrainPct = 100;
  const sftPct = log ? sftLogPct : sftLinearPct;

  // Order-of-magnitude ratio (tokens vs examples → illustrative of scale only).
  const ordersOfMagnitude = Math.round(Math.log10(PRETRAIN_TOKENS / SFT_EXAMPLES)); // 6

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <SegmentedToggle
          value={scale}
          onChange={setScale}
          options={[
            { value: 'linear', label: copy.linear },
            { value: 'log', label: copy.log },
          ]}
        />
      </div>

      <Bar
        label={copy.pretrainLabel}
        amount="≈ 3T tokens"
        widthPct={pretrainPct}
        bg={PRETRAIN_BG}
        ariaLabel={copy.pretrainAria}
      />

      <Bar
        label={copy.sftLabel}
        amount="≈ 1M examples"
        widthPct={sftPct}
        bg={SFT_BG}
        ariaLabel={log ? copy.sftAriaLog : copy.sftAriaLinear}
        note={log ? undefined : copy.sftLinearNote}
      />

      <div className="rounded-md border border-border bg-muted/20 p-2 text-[12px] text-foreground/85">
        {copy.summary(ordersOfMagnitude)}
      </div>

      <p className="text-[12px] text-foreground/85">{copy.body}</p>

      <p className="text-[10px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
