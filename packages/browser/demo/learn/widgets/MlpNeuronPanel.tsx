import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * MlpNeuronPanel — chapter 7, "what does an MLP neuron actually DO?"
 *
 * The chapter teaches the SwiGLU plumbing (gate × up → down) but never what
 * the 3584 intermediate features compute. This panel shows three illustrative
 * features as detectors: each row is one neuron, a strip of example tokens
 * with the neuron's (hand-built) activation painted behind each token, and an
 * arrow into "writes its pattern into the residual stream".
 *
 * Mirrors PatternDetectorPanel from chapter 4: hand-built CSS, NO model call,
 * NO worker, deterministic SSR-safe render, and an explicit "illustrative —
 * real features are messier and discovered, not designed" disclaimer.
 */

type FeatureTokens = Array<{ text: string; act: number }>;

// Example token strips; act = hand-built activation in 0..1. The tokens stay
// English in both locales (they are illustrative model inputs); the per-row
// label prose lives in COPY below, index-aligned.
const FEATURE_TOKENS: FeatureTokens[] = [
  [
    { text: 'I', act: 0 },
    { text: 'flew', act: 0.05 },
    { text: 'from', act: 0 },
    { text: 'Paris', act: 0.95 },
    { text: 'to', act: 0 },
    { text: 'Tokyo', act: 0.9 },
    { text: 'last', act: 0 },
    { text: 'week', act: 0.05 },
  ],
  [
    { text: 'The', act: 0 },
    { text: 'food', act: 0.05 },
    { text: 'was', act: 0 },
    { text: 'not', act: 0.9 },
    { text: 'bad', act: 0.2 },
    { text: 'at', act: 0.05 },
    { text: 'all', act: 0.1 },
  ],
  [
    { text: 'Try', act: 0 },
    { text: 'this', act: 0.05 },
    { text: 'loop:', act: 0.1 },
    { text: 'for', act: 0.85 },
    { text: 'i', act: 0.9 },
    { text: 'in', act: 0.85 },
    { text: 'range(10):', act: 0.95 },
  ],
];

// Per-locale copy: every user-visible string lives here so /zh localizes while
// the English output stays byte-identical.
const COPY = {
  en: {
    header: 'What one MLP neuron fires on · three illustrative features',
    intro: (
      <>
        Each of the 3584 intermediate features acts like a tiny detector: its gate opens on tokens that match its
        pattern and stays shut on everything else. When it fires, its slice of <code>down_proj</code> adds that
        feature&apos;s signature to the token&apos;s residual stream. Darker = stronger firing.
      </>
    ),
    /** Plain-words labels for what each neuron fires on, index-aligned with FEATURE_TOKENS. */
    featureLabels: ['fires on city names', 'fires on negation', 'fires inside code'],
    activationTitle: (act: string) => `activation ${act}`,
    rowAria: (label: string, peak: string) => `Illustrative neuron that ${label}; strongest on the token ${peak}`,
    writesNote: 'writes its pattern into the residual stream',
    footnote: (
      <>
        <strong>Illustrative — real features are messier and discovered, not designed.</strong> Nobody assigns a neuron
        to &quot;city names&quot;; training finds whatever detectors lower the loss, many fire on several unrelated
        things at once, and a clean one-neuron-one-concept story is the exception, not the rule.
      </>
    ),
  },
  zh: {
    header: '一个 MLP 神经元在什么上激活 · 三个示意特征',
    intro: (
      <>
        3584 个中间特征中的每一个都像一个小小的检测器：它的门只对匹配自己模式的 token
        打开，对其余一切保持关闭。它一旦激活，它在 <code>down_proj</code> 中的那一片就把该特征的签名加进这个 token
        的残差流。颜色越深 = 激活越强。
      </>
    ),
    featureLabels: ['在城市名上激活', '在否定词上激活', '在代码内部激活'],
    activationTitle: (act: string) => `激活值 ${act}`,
    rowAria: (label: string, peak: string) => `示意神经元：${label}；在 token ${peak} 上最强`,
    writesNote: '把自己的模式写进残差流',
    footnote: (
      <>
        <strong>示意——真实特征更杂乱，是被发现的，不是被设计的。</strong>
        没有人把某个神经元指派给&ldquo;城市名&rdquo;；训练找到的是任何能降低损失的检测器，很多神经元会同时在几件不相干的事情上激活，干净的&ldquo;一个神经元一个概念&rdquo;是例外，不是常态。
      </>
    ),
  },
} as const;

/** One token chip with its activation painted behind the text. */
function TokenChip({ text, act, title }: { text: string; act: number; title: string }) {
  const clamped = Math.max(0, Math.min(1, act));
  return (
    <span
      className="relative rounded-[3px] border border-border/40 px-1.5 py-0.5 font-mono text-[11px] text-foreground/90"
      title={title}
    >
      <span
        aria-hidden="true"
        className="absolute inset-0 rounded-[3px]"
        style={{ backgroundColor: 'var(--primary)', opacity: clamped * 0.45 }}
      />
      <span className="relative">{text}</span>
    </span>
  );
}

function FeatureRow({
  label,
  tokens,
  copy,
}: {
  label: string;
  tokens: FeatureTokens;
  copy: (typeof COPY)[keyof typeof COPY];
}) {
  // The peak token, named in the row's aria-label so the "where it lights up"
  // story survives without color vision.
  const peak = tokens.reduce((a, b) => (b.act > a.act ? b : a));
  return (
    <div
      className="flex flex-wrap items-center gap-x-3 gap-y-2 rounded-md border border-border/60 bg-muted/20 p-3"
      role="group"
      aria-label={copy.rowAria(label, peak.text)}
    >
      <div className="w-40 shrink-0 text-[12px] font-semibold text-foreground">&quot;{label}&quot;</div>
      <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1">
        {tokens.map((t, i) => (
          <TokenChip
            key={i}
            text={t.text}
            act={t.act}
            title={copy.activationTitle(Math.max(0, Math.min(1, t.act)).toFixed(2))}
          />
        ))}
      </div>
      <div className="flex shrink-0 items-center gap-1.5 text-[11px] text-muted-foreground">
        <span aria-hidden="true" style={{ color: 'var(--primary)' }}>
          ⟶
        </span>
        <span>{copy.writesNote}</span>
      </div>
    </div>
  );
}

export function MlpNeuronPanel() {
  const copy = COPY[useLocale()];
  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
      </div>
      <p className="text-[12px] text-foreground/85">{copy.intro}</p>
      <div className="space-y-2">
        {FEATURE_TOKENS.map((tokens, i) => (
          <FeatureRow key={i} label={copy.featureLabels[i]!} tokens={tokens} copy={copy} />
        ))}
      </div>
      <p className="text-[11px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
