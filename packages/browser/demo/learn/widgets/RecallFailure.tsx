import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { TopKBars } from '../inspector/TopKBars';

/**
 * Chapter 12 (KV cache & hybrid attention) — the full-vs-linear recall
 * trade-off as a "needle in a haystack." Self-contained: NO worker, NO model,
 * NO WASM. Two synthetic next-token distributions over the same candidate set
 * (the buried "needle" token plus a few distractors), shown side by side:
 *
 *   - Full attention recalls the EXACT needle: a sharp spike on the right
 *     token, near-zero mass everywhere else.
 *   - GatedDeltaNet (linear) recalls APPROXIMATELY: its fixed-size recurrent
 *     state has mixed the whole history together, so the mass is smeared
 *     across the needle and several distractors — the needle is at best
 *     slightly highest, never dominant.
 *
 * This is exactly why Qwen3.5 keeps 6 full-attention layers for the
 * "find that one specific token" jobs. The distributions are hand-picked to
 * show the shape, not live output from the model.
 */

// The unique token buried in the long context. The query asks to recall it.
const NEEDLE = 'X7-Q';

// Candidate next tokens: the needle plus plausible distractors that the
// linear layer's blurred state confuses it with.
const CANDIDATES: ReadonlyArray<string> = [NEEDLE, 'X7-G', 'K3-Q', 'X1-Q', 'B7-Q'];
const NEEDLE_ID = 0;

type Panel = {
  key: 'full' | 'linear';
  // Probability per candidate, aligned to CANDIDATES.
  probs: number[];
};

const PANELS: Panel[] = [
  {
    key: 'full',
    // Sharp spike on the exact needle — correct exact recall.
    probs: [0.91, 0.03, 0.025, 0.02, 0.015],
  },
  {
    key: 'linear',
    // Blurred / flat — the fixed-size state mixed the history together.
    probs: [0.3, 0.2, 0.18, 0.17, 0.15],
  },
];

// Per-locale copy. COPY.en is the original English, moved verbatim.
const COPY = {
  en: {
    title: 'Needle in a haystack — exact recall',
    recallToken: 'Recall token ',
    setCodeTo: '… set the access code to ',
    beforeContinuing: 'before continuing. ',
    filler: '[ 4,000 tokens of filler … ]',
    accessCodeWas: ' The access code was',
    panels: {
      full: {
        label: 'Full attention',
        note: 'Softmax attention scores the needle against every cached key directly, so it can put almost all the mass on the one exact token it was asked to recall.',
      },
      linear: {
        label: 'GatedDeltaNet (linear)',
        note: 'A fixed-size recurrent state compresses the whole history into one constant-size tensor; here that mixing leaves the needle only barely ahead of look-alike distractors, rather than pinned down.',
      },
    },
    sharpRecall: 'sharp recall',
    blurredRecall: 'blurred recall',
    summary: (
      <>
        Full-attention layers keep direct, per-token access to every cached key, so they are well suited to pulling out
        one exact token on demand. A fixed-state recurrent layer (GatedDeltaNet) compresses the whole history into a
        constant-size tensor, so under recall pressure it is <em>less reliable</em> at pinning down an arbitrary token
        — as in this illustrative case, where the needle barely leads its look-alikes. Neither is absolute (attention
        can still mis-attend, and a recurrent state isn&apos;t always blurred), but that reliability gap is why Qwen3.5
        keeps <strong>6 full-attention layers</strong> for the &quot;find that one specific token&quot; jobs its {18}
        -layer linear majority handles less precisely.
      </>
    ),
    footnote: 'Illustrative distributions — hand-picked to show the exact-vs-blurred shape, not live output from the model.',
  },
  zh: {
    title: '大海捞针——精确回忆',
    recallToken: '回忆 token ',
    setCodeTo: '……把访问码设为 ',
    beforeContinuing: '后再继续。',
    filler: '[ 4,000 个 token 的填充文本…… ]',
    accessCodeWas: ' 访问码是',
    panels: {
      full: {
        label: '全量注意力',
        note: 'Softmax 注意力直接拿“针”与每一个缓存的 key 打分，所以它能把几乎全部概率质量放在被要求回忆的那一个确切 token 上。',
      },
      linear: {
        label: 'GatedDeltaNet（线性）',
        note: '固定大小的循环状态把全部历史压缩进一个常数大小的张量；这种混合让“针”只比形近的干扰项略高一点，而不是被牢牢锁定。',
      },
    },
    sharpRecall: '精确回忆',
    blurredRecall: '模糊回忆',
    summary: (
      <>
        全量注意力层保留对每一个缓存 key 的逐 token 直接访问，所以很擅长按需取出某一个确切的
        token。固定状态的循环层（GatedDeltaNet）把全部历史压缩进一个常数大小的张量，在回忆压力下锁定任意一个 token 就
        <em>不那么可靠</em>——正如这个示意场景：那根“针”只略微领先于它的形近干扰项。两者都不是绝对的（注意力也可能看错地方，循环状态也并非总是模糊），但正是这种可靠性差距，让
        Qwen3.5 保留了 <strong>6 个全量注意力层</strong>来处理“找到那个特定 token”的任务——它那 {18} 层的线性主体做这类事不够精确。
      </>
    ),
    footnote: '示意性的分布——为展示“精确 vs 模糊”的形态而手工挑选，并非模型的实时输出。',
  },
} as const;

export function RecallFailure() {
  const copy = COPY[useLocale()];
  const ids = CANDIDATES.map((_, i) => i);
  const texts = CANDIDATES.map((t) => t);

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="text-[11px] text-muted-foreground">
          {copy.recallToken}
          <span className="font-mono text-foreground/80">{NEEDLE}</span>
        </div>
      </div>

      {/* The scripted scenario: a unique needle buried among filler. */}
      <div className="rounded-md border border-border/60 bg-muted/30 p-2 font-mono text-[12px] leading-relaxed text-foreground/90">
        <span className="text-muted-foreground">{copy.setCodeTo}</span>
        <span className="rounded bg-primary/15 px-1 text-primary">{NEEDLE}</span>
        <span className="text-muted-foreground">
          {' '}
          {copy.beforeContinuing}
          <span className="opacity-60">{copy.filler}</span>
          {copy.accessCodeWas}{' '}
        </span>
        <span className="text-foreground/80">▮</span>
      </div>

      {/* Two side-by-side distributions over the same candidate set. */}
      <div className="grid gap-3 sm:grid-cols-2">
        {PANELS.map((panel, i) => {
          const isFull = panel.key === 'full';
          const panelCopy = copy.panels[panel.key];
          return (
            <div key={panel.key} className="space-y-1">
              <div className="flex items-center justify-between">
                <span
                  className={[
                    'rounded px-2 py-0.5 text-[11px] font-medium',
                    isFull
                      ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300'
                      : 'bg-violet-500/15 text-violet-700 dark:text-violet-300',
                  ].join(' ')}
                >
                  {panelCopy.label}
                </span>
                <span className="text-[11px] text-muted-foreground">
                  {isFull ? copy.sharpRecall : copy.blurredRecall}
                </span>
              </div>
              <TopKBars ids={ids} probs={panel.probs} texts={texts} sampledTokenId={NEEDLE_ID} runKey={i + 1} />
              <p className="text-[11px] leading-relaxed text-muted-foreground">{panelCopy.note}</p>
            </div>
          );
        })}
      </div>

      <p className="text-[12px] text-foreground/85">{copy.summary}</p>

      <p className="text-[10px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
