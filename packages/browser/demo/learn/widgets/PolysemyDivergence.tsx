import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 4 (self-attention) supplement — the polysemy endpoint demo. Three
 * prompts that all END in the byte-identical token " bank" (id 5883), so the
 * embedding lookup hands the layer stack the exact same starting vector at
 * the final position — yet the model's real next-token predictions diverge
 * completely, because attention (and the linear layers) mixed in different
 * context on the way up.
 *
 * The numbers are REAL recorded output of the exact Qwen3.5-0.8B checkpoint
 * this playground serves (see the footnote in COPY for the capture recipe),
 * baked in verbatim so the teaching contrast is stable. Self-contained: no
 * worker, no model run, no animation.
 *
 * Model inputs/outputs are NEVER translated — in the zh locale only the
 * chrome switches language; prompts and predicted tokens stay byte-identical.
 */

// Shared final token across all three prompts (verified single token).
const BANK_TOKEN_ID = 5883;

type Prediction = {
  /** Token text with the leading space shown as the · convention. */
  display: string;
  prob: number;
  /** Marks the rows that reveal the contextual sense. */
  tell?: boolean;
};

type ContextRun = {
  key: 'finance' | 'aviation' | 'river';
  /** The prompt fed to the model, byte-identical, final token " bank". */
  prompt: string;
  top5: Prediction[];
};

// Top-5 next-token candidates per context. Probabilities are the softmax
// over the top-12 captured logits, exactly as the chapter-10 LM-head
// inspector displays them (3 decimals).
const RUNS: ContextRun[] = [
  {
    key: 'finance',
    prompt: 'My rent is paid straight from the bank',
    top5: [
      { display: '·account', prob: 0.351, tell: true },
      { display: '.', prob: 0.275 },
      { display: ',', prob: 0.241 },
      { display: '·and', prob: 0.036 },
      { display: '·in', prob: 0.022 },
    ],
  },
  {
    key: 'aviation',
    prompt: 'The aircraft rolled into a 30-degree bank',
    top5: [
      { display: '·angle', prob: 0.277, tell: true },
      { display: ',', prob: 0.188 },
      { display: '·at', prob: 0.16 },
      { display: '.', prob: 0.126 },
      { display: '·and', prob: 0.111 },
    ],
  },
  {
    key: 'river',
    prompt: 'They had a picnic on the river bank',
    top5: [
      { display: '.', prob: 0.468, tell: true },
      { display: ',', prob: 0.224 },
      { display: '·and', prob: 0.051 },
      { display: '·in', prob: 0.049 },
      { display: '·where', prob: 0.04 },
    ],
  },
];

type CardCopy = { label: string; reading: string };

const COPY = {
  en: {
    header: 'One token, three contexts · real model output',
    subheader: 'Top-5 next-token predictions · this exact Qwen3.5-0.8B checkpoint',
    chipBadge: `token id ${BANK_TOKEN_ID}`,
    chipNote:
      'All three prompts end in the byte-identical token ·bank — the embedding lookup returns the same row of embed_tokens.weight every time, so the layer stack starts from the same vector at that position.',
    cards: {
      finance: {
        label: 'money',
        reading: 'The model lines up ·account at #1 — it read the “rent is paid from” context, not the word alone.',
      },
      aviation: {
        label: 'flying',
        reading: '·angle jumps to #1 (“bank angle” is the aviation phrase) — a continuation the other contexts never surface.',
      },
      river: {
        label: 'riverside',
        reading: 'Here the model mostly thinks the sentence is complete (47% for “.”), with location words like ·where further down.',
      },
    } satisfies Record<ContextRun['key'], CardCopy>,
    barAria: 'Top-5 next-token predictions',
    footnote:
      'Real recorded output, not scripted: each list was captured by running these exact prompts (raw completion, no chat template) through the LM-head inspector of this playground — the same Qwen3.5-0.8B checkpoint the rest of the course runs — which captures the top-12 next-token logits at the final position; the probabilities shown are the softmax over those captured logits, as the inspector displays them. The “same starting vector” claim is by construction: the embedding lookup is a table indexed by token id, and all three prompts end in token id 5883.',
  },
  zh: {
    header: '同一个 token，三种上下文 · 真实模型输出',
    subheader: '下一个 token 的 top-5 预测 · 正是本站使用的 Qwen3.5-0.8B 检查点',
    chipBadge: `token id ${BANK_TOKEN_ID}`,
    chipNote:
      '三条 prompt 都以字节完全相同的 token ·bank 结尾——embedding 查表每次返回 embed_tokens.weight 的同一行，所以层堆栈在这个位置拿到的起点向量一模一样。',
    cards: {
      finance: {
        label: '金钱',
        reading: '模型把 ·account 排到第 1 名——它读懂了“房租从哪里扣”的上下文，而不是只看这个词本身。',
      },
      aviation: {
        label: '飞行',
        reading: '·angle 跃居第 1（“bank angle” 是航空术语“坡度角”）——这个续写在另外两种上下文里完全不会浮现。',
      },
      river: {
        label: '河畔',
        reading: '在这里模型基本认为句子已经说完（“.” 占 47%），位置词如 ·where 排在更后面。',
      },
    } satisfies Record<ContextRun['key'], CardCopy>,
    barAria: '下一个 token 的 top-5 预测',
    footnote:
      '真实的录制输出，并非人工编写：每张列表都来自把这些 prompt 原样（裸补全，不套 chat 模板）送进本站第 10 章的 LM-head 检查面板——与课程其余部分完全相同的 Qwen3.5-0.8B 检查点——它会捕获最后一个位置上 top-12 的下一 token logits；图中概率是对这些已捕获 logits 做 softmax 的结果，与检查面板的显示方式一致。“同一个起点向量”这一点是由构造保证的：embedding 查表只按 token id 取行，而三条 prompt 的最后一个 token id 都是 5883。',
  },
} as const;

/** Highlight the shared final token inside a prompt. */
function PromptLine({ prompt }: { prompt: string }) {
  const head = prompt.slice(0, prompt.length - ' bank'.length);
  return (
    <div className="font-mono text-[12px] leading-relaxed break-words">
      <span className="text-muted-foreground">{head}</span>
      <span className="rounded bg-primary/15 px-0.5 font-semibold text-primary">·bank</span>
    </div>
  );
}

export function PolysemyDivergence() {
  const copy = COPY[useLocale()];

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <div className="text-[10px] text-muted-foreground">{copy.subheader}</div>
      </div>

      <div className="flex flex-wrap items-center gap-2 rounded-md border border-border/60 bg-muted/20 p-2">
        <span className="rounded border border-primary/50 bg-primary/10 px-1.5 py-0.5 font-mono text-[12px] font-semibold text-primary">
          ·bank
        </span>
        <span className="rounded border border-border px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
          {copy.chipBadge}
        </span>
        <p className="m-0 min-w-[12rem] flex-1 text-[11px] leading-snug text-muted-foreground">{copy.chipNote}</p>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        {RUNS.map((run) => {
          const card = copy.cards[run.key];
          const maxProb = run.top5[0].prob;
          return (
            <div key={run.key} className="space-y-2 rounded-md border border-border/60 bg-muted/20 p-3">
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{card.label}</div>
              <PromptLine prompt={run.prompt} />
              <ol aria-label={copy.barAria} className="m-0 list-none space-y-1 p-0">
                {run.top5.map((t) => (
                  <li key={t.display} className="flex items-center gap-2">
                    <span
                      className={[
                        'w-[4.5rem] shrink-0 truncate font-mono text-[11px]',
                        t.tell ? 'font-semibold text-primary' : 'text-foreground/85',
                      ].join(' ')}
                    >
                      {t.display}
                    </span>
                    <span className="h-2 flex-1 overflow-hidden rounded-sm bg-border/40">
                      <span
                        className={['block h-full', t.tell ? 'bg-primary' : 'bg-primary/40'].join(' ')}
                        style={{ width: `${(t.prob / maxProb) * 100}%` }}
                      />
                    </span>
                    <span className="w-10 shrink-0 text-right font-mono text-[10px] text-muted-foreground">
                      {t.prob.toFixed(3)}
                    </span>
                  </li>
                ))}
              </ol>
              <p className="m-0 text-[11px] leading-snug text-muted-foreground">{card.reading}</p>
            </div>
          );
        })}
      </div>

      <p className="text-[10px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
