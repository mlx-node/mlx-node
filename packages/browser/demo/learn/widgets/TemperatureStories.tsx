import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 11 (sampling) supplement — the same seed text continued by the REAL
 * Qwen3.5-0.8B checkpoint at three temperatures. Unlike the scripted
 * `SamplingFailureModes` panels, these continuations are genuine recorded
 * model output (see the footnote in COPY for the exact recipe), baked in
 * verbatim so the teaching contrast is stable visit-to-visit. Self-contained:
 * no worker, no model run, no animation.
 *
 * Model outputs are NEVER translated — in the zh locale only the chrome
 * (labels, captions, footnote) switches language; the English continuations
 * stay byte-identical.
 */

// The seed prompt fed to the model — identical for all three runs.
const SEED = 'Once upon a time, there was a';

// Recorded continuations from one local run of the exact checkpoint served by
// this playground (packages/browser/demo/public/model). Byte-identical to the
// model output, including the "\n" newlines and the "\t" at T = 1.5.
// Recipe: Qwen35Model.generate() raw completion (no chat template),
// maxNewTokens = 70, pure temperature sampling (top-p = 1.0, top-k off).
const CONTINUATIONS = {
  greedy:
    ' very special kind of animal called a "pocketed" animal. These animals are very special because they have a special way of living. They live in a very special place called a "pocketed" habitat.\n\nImagine a pocketed animal as a little explorer who lives in a very special place. This place is called a "pocket',
  warm:
    " curious child named Ethan. One sunny morning, Ethan decided to explore the world around him and he found a strange object.\nEthan picked up a wooden block and put it in a circular hole. He used a different material for the block, which made it feel like a solid, but he didn't know why.\nThen, he put the",
  hot: ' boy named Benobils who lived loving today though I Homer gonna doodproof way since Dad Giovanni will offer him up to USA pilgrimage I let Homework set come don accept Pescidio submit these"` Do español Paramal Decindo Unless Lupens Journals few holders Bel En долга Cuban Borders programa greatly\tlario routine gym training regulwoothy Numero dai training So much',
} as const;

type CardKey = keyof typeof CONTINUATIONS;

type CardCopy = {
  badge: string;
  title: string;
  takeaway: string;
};

const COPY = {
  en: {
    header: 'Same seed, three temperatures · real model output',
    subheader: 'One recorded run each · 70 new tokens · this exact Qwen3.5-0.8B checkpoint',
    seedLabel: 'Seed: ',
    cards: {
      greedy: {
        badge: 'T = 0',
        title: 'greedy',
        takeaway:
          'Deterministic — rerun it and you get this exact text again. Safe but circling: "very special" four times, and "pocketed" is already a loop.',
      },
      warm: {
        badge: 'T = 0.7',
        title: 'sampled',
        takeaway:
          'Mild randomness is enough to break out of the safest rut — a named character, a small plot. Still coherent; a rerun gives a different continuation (this is one sample, not a guarantee).',
      },
      hot: {
        badge: 'T = 1.5',
        title: 'too hot',
        takeaway:
          'Each improbable pick becomes context for the next step, so the damage compounds — within a couple of lines it has drifted through three languages into word salad.',
      },
    } satisfies Record<CardKey, CardCopy>,
    footnote:
      'Real pre-recorded output, not scripted: each continuation comes from running this exact Qwen3.5-0.8B checkpoint locally through the native mlx-node Qwen35Model.generate() raw-completion API (no chat template) with maxNewTokens = 70 and pure temperature sampling — no top-p or top-K truncation (top-p = 1.0, top-K off). The T = 0 run is greedy and fully reproducible; the T = 0.7 and T = 1.5 runs are single samples and would differ on a rerun.',
  },
  zh: {
    header: '同一个种子，三种 temperature · 真实模型输出',
    subheader: '各取一次录制的运行 · 每段 70 个新 token · 正是本站使用的 Qwen3.5-0.8B 检查点',
    seedLabel: '种子文本：',
    cards: {
      greedy: {
        badge: 'T = 0',
        title: '贪心',
        takeaway:
          '确定性的——重跑一次会得到一模一样的文本。安全但在原地打转：“very special” 出现了四次，“pocketed” 已经在循环。',
      },
      warm: {
        badge: 'T = 0.7',
        title: '采样',
        takeaway:
          '一点温和的随机性就足以跳出最保险的车辙——有了名字、有了情节。仍然连贯；重跑会得到另一段不同的续写（这只是一次采样，不保证每次都好）。',
      },
      hot: {
        badge: 'T = 1.5',
        title: '过热',
        takeaway:
          '每一个低概率的选择都会成为下一步的上下文，损害不断累积——短短几行就漂过三种语言，变成词语沙拉。',
      },
    } satisfies Record<CardKey, CardCopy>,
    footnote:
      '真实的预录输出，并非人工编写：每段续写都来自在本地运行与本站完全相同的 Qwen3.5-0.8B 检查点，通过原生 mlx-node 的 Qwen35Model.generate() 裸补全 API（不套 chat 模板），maxNewTokens = 70，纯 temperature 采样——没有 top-p 或 top-K 截断（top-p = 1.0，top-K 关闭）。T = 0 的运行是贪心的、完全可复现；T = 0.7 和 T = 1.5 各是一次采样，重跑会得到不同的文本。',
  },
} as const;

// Card order + per-card accent. Greedy reads as "neutral/safe", the sweet
// spot as primary, and the too-hot run as destructive — matching the
// SamplingFailureModes color language.
const CARDS: { key: CardKey; badgeClass: string }[] = [
  { key: 'greedy', badgeClass: 'border-border text-muted-foreground' },
  { key: 'warm', badgeClass: 'border-primary/50 bg-primary/10 text-primary' },
  { key: 'hot', badgeClass: 'border-destructive/50 bg-destructive/10 text-destructive' },
];

export function TemperatureStories() {
  const copy = COPY[useLocale()];

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <div className="text-[10px] text-muted-foreground">{copy.subheader}</div>
      </div>

      <div className="space-y-3">
        {CARDS.map(({ key, badgeClass }) => {
          const card = copy.cards[key];
          return (
            <div key={key} className="rounded-md border border-border/60 bg-muted/20 p-3">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <span
                  className={['rounded border px-1.5 py-0.5 font-mono text-[11px] font-semibold', badgeClass].join(
                    ' ',
                  )}
                >
                  {card.badge}
                </span>
                <span className="text-[11px] uppercase tracking-wider text-muted-foreground">{card.title}</span>
              </div>
              {/* whitespace-pre-wrap keeps the model's own newlines (and the
                  tab at T = 1.5) byte-faithful on screen. */}
              <div className="whitespace-pre-wrap break-words font-mono text-[12px] leading-relaxed">
                <span className="text-muted-foreground">
                  {copy.seedLabel}
                  {SEED}
                </span>
                <span className="text-foreground/90">{CONTINUATIONS[key]}</span>
              </div>
              <p className="mt-2 text-[11px] text-muted-foreground">{card.takeaway}</p>
            </div>
          );
        })}
      </div>

      <p className="text-[10px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
