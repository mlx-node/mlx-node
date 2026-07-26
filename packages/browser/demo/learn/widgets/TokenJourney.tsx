import { cn } from '@/lib/utils';
import { ChapterLink } from '../scaffolding/ChapterLink';
import * as React from 'react';

import { Button } from '../../components/ui/button';
import { useLocale } from '../../lib/i18n-react';
import { TopKBars, cleanupTokenText } from '../inspector/TopKBars';
import { DiagramFrame, PanBox, RX, SW, useStepPlayer } from '../motion';

/**
 * Course capstone — "Follow one token's journey".
 *
 * A scrubber that walks ONE concrete example ("The cat sat on the" → " floor")
 * through the whole transformer forward pass, one stage at a time, so a
 * beginner finally sees the end-to-end flow in a single place. Every earlier
 * chapter zooms into one box; this widget puts them back on one rail and keeps
 * a "right now the data is …" spine under every stage.
 *
 * Self-contained by design: no worker, no model, no async, no network. Every
 * number is a precomputed deterministic constant — visuals vary by stage/index,
 * never by `Math.random` or `Date.now` — so it renders instantly even while the
 * real model is still downloading. The numbers and heatmaps are schematic
 * stand-ins for the real 1024-dim vectors and 248,320 logits, not live output.
 *
 * ── ANIMATION MODEL ────────────────────────────────────────────────────────
 *
 * `useStepPlayer` + `DiagramFrame` from `learn/motion`, replacing this file's
 * former hand-rolled `setInterval`, its private `usePrefersReducedMotion`, the
 * card shell, the header row, its own Play / Pause / Replay button and its own
 * `aria-live` status region. The player is 0-indexed and everything else here
 * is 1-indexed, so there is exactly ONE conversion — `stage = frame + 1` — and
 * nothing downstream of it changes.
 *
 * Three things stay widget-owned on purpose, because `DiagramFrame` renders
 * play/step and nothing else:
 *   - the 6-chip stage rail, wired through `player.goTo` (the frame has no
 *     scrubber, and the rail is also what says which stage is on screen);
 *   - the Prev / Next pair (the frame's Step only goes forward);
 *   - the visible "Stage n / 6" counter — the frame has no counter slot.
 *
 * The caption is the status sentence the old `sr-only role="status"` region
 * announced, moved verbatim into the frame's caption slot: the frame owns the
 * a11y contract now, so that sentence has to live where the frame can both show
 * it and announce it. It is the one string that changes on every beat and is
 * built from the same per-locale `STAGES` table the rail and the spine read, so
 * `captions` is that table mapped one-to-one — every arm, by construction.
 *
 * That sentence is now VISIBLE as well as spoken, and it restates the panel
 * header and the spine badge. Every other per-beat string in this widget is
 * already placed by its own design — the stage prose sits above its visual, the
 * data shape sits in the spine — so there was no spare sentence for the slot,
 * and inventing one, demoting the prose into it, or deleting the spine to free
 * it are all CONTENT calls, not conversion calls. The conversion took the option
 * that removes nothing: the announcement is preserved exactly, and if the
 * repetition is unwanted the fix is one line here plus one deletion, either the
 * spine badge or the panel's stage/title row.
 *
 * Reduced motion parks on stage 1 (`restFrame: 0`), not on `useStepPlayer`'s
 * default last frame: this is a guided walk whose reduced-motion hint tells the
 * reader to step forward with Prev / Next, and stage 6 is the answer.
 *
 * COLOUR CARVE-OUT. This widget runs FOUR independent colour channels — the
 * diverging teal/amber heatmap (sign of a value), the red query row (which token
 * is asking), the 24-step blue depth ramp, and the green winning logit. None of
 * them maps onto the skin's two hues, so only the STRUCTURAL tokens are adopted
 * (`RX` on the rects, `SW` on the two hand-tuned stroke widths). Every hue in
 * this file is left exactly as it was, and `RED` / `EMERALD` are deliberately
 * not imported.
 */

// ---------------------------------------------------------------------------
// The running example — exact constants shared with the rest of the course.
// ---------------------------------------------------------------------------

const PROMPT = 'The cat sat on the';
const TOKEN_TEXTS: readonly string[] = ['The', ' cat', ' sat', ' on', ' the'];
const TOKEN_IDS: readonly number[] = [760, 7993, 7338, 383, 279];

const NUM_LAYERS = 24;

// Stage-6 distribution (precomputed; probs sum to 1.00). " floor" is the greedy
// continuation this course actually measured for the prompt (see ChapterIndex and
// the LM-head chapter) — deliberately NOT the cliché " mat", so the journey shows
// what the real model does, not what a human would guess.
const DIST_TEXTS: readonly string[] = [' floor', ' mat', ' ground', ' table', ' sofa', ' grass'];
const DIST_PROBS: readonly number[] = [0.41, 0.19, 0.13, 0.12, 0.09, 0.06];
const DIST_IDS: readonly number[] = [6220, 5344, 4751, 1898, 30051, 15879];
const SAMPLED_TOKEN_ID = 6220; // the " floor" row, highlighted by TopKBars

// Stage 3 — illustrative causal-attention weights from the last token (" the")
// back toward [The, cat, sat, on, the-self]. "cat"/"sat" are made prominent.
const ATTENTION_WEIGHTS: readonly number[] = [0.3, 0.34, 0.16, 0.12, 0.08];

const AUTOPLAY_MS = 2200;
const TOTAL_STAGES = 6;

// ---------------------------------------------------------------------------
// Deterministic illustrative embedding heatmap — 5 rows × 12 cells in [-1, 1].
// Varies by (row, col), never by randomness.
// ---------------------------------------------------------------------------

const HEATMAP_ROWS = TOKEN_TEXTS.length; // 5
const HEATMAP_COLS = 12;

function heatmapValue(row: number, col: number): number {
  return Math.sin(row * 1.7 + col * 0.6) * 0.8;
}

/** Map an illustrative value in [-1, 1] to a diverging fill (teal +, amber −). */
function heatmapFill(value: number): string {
  const mag = Math.min(1, Math.abs(value));
  const alpha = (0.12 + mag * 0.68).toFixed(3);
  return value >= 0
    ? `oklch(0.72 0.12 185 / ${alpha})` // teal for positive
    : `oklch(0.78 0.14 60 / ${alpha})`; // amber for negative
}

// ---------------------------------------------------------------------------
// Stage metadata. Copy with **bold** spans is rendered as JSX per stage (so the
// emphasis is real <strong>, not parsed markup); the rail/spine read this table.
// ---------------------------------------------------------------------------

type StageMeta = {
  /** 1-indexed position on the rail. */
  n: number;
  title: string;
  /** The through-line "data right now" string shown in the spine badge. */
  dataNow: string;
  /** Label + chapterId for the "Learn more" cross-link. */
  chapterLabel: string;
  chapterId: string;
};

// Stage metadata per locale (titles/data-shape strings/chapter labels are
// prose; n and chapterId are shared structure).
const STAGES: Record<'en' | 'zh', readonly StageMeta[]> = {
  en: [
    {
      n: 1,
      title: 'Text → tokens',
      dataNow: '[760, 7993, 7338, 383, 279]  ·  5 token ids',
      chapterLabel: 'Tokenization',
      chapterId: 'tokenization',
    },
    {
      n: 2,
      title: 'Tokens → vectors',
      dataNow: '5 tokens × 1024 numbers',
      chapterLabel: 'Embeddings',
      chapterId: 'embeddings',
    },
    {
      n: 3,
      title: 'Tokens share context',
      dataNow: '5 × 1024 numbers  ·  now context-mixed',
      chapterLabel: 'Self-attention',
      chapterId: 'attention',
    },
    {
      n: 4,
      title: 'Refine, ×24 layers',
      dataNow: '5 × 1024 numbers  ·  after 24 layers',
      chapterLabel: 'Full transformer block',
      chapterId: 'full-block',
    },
    {
      n: 5,
      title: 'Top vector → 248,320 scores',
      dataNow: '248,320 scores (logits)',
      chapterLabel: 'LM head',
      chapterId: 'lm-head',
    },
    {
      n: 6,
      title: 'Scores → the next token',
      dataNow: '" floor"  ·  appended, then loop',
      chapterLabel: 'Sampling',
      chapterId: 'sampling',
    },
  ],
  zh: [
    {
      n: 1,
      title: '文本 → token',
      dataNow: '[760, 7993, 7338, 383, 279]  ·  5 个 token id',
      chapterLabel: '分词',
      chapterId: 'tokenization',
    },
    {
      n: 2,
      title: 'token → 向量',
      dataNow: '5 个 token × 1024 个数',
      chapterLabel: '词嵌入',
      chapterId: 'embeddings',
    },
    {
      n: 3,
      title: 'token 之间共享上下文',
      dataNow: '5 × 1024 个数  ·  已混入上下文',
      chapterLabel: '自注意力',
      chapterId: 'attention',
    },
    {
      n: 4,
      title: '精炼，×24 层',
      dataNow: '5 × 1024 个数  ·  经过 24 层',
      chapterLabel: '完整 Transformer 块',
      chapterId: 'full-block',
    },
    {
      n: 5,
      title: '顶端向量 → 248,320 个分数',
      dataNow: '248,320 个分数（logits）',
      chapterLabel: 'LM head',
      chapterId: 'lm-head',
    },
    {
      n: 6,
      title: '分数 → 下一个 token',
      dataNow: '" floor"  ·  追加，然后循环',
      chapterLabel: '采样',
      chapterId: 'sampling',
    },
  ],
};

// ---------------------------------------------------------------------------
// Per-locale copy. Model tokens ("The cat sat on the", " floor", " mat") and
// math identifiers (query, MLP, logits, softmax) stay English in both locales.
// ---------------------------------------------------------------------------

const COPY = {
  en: {
    headerTitle: "Follow one token's journey",
    leadingSpaceNote: 'the leading space is part of the token.',
    heatmapAria:
      "Illustrative embedding heatmap: 5 token rows by 12 columns of coloured cells, teal for positive values and amber for negative, standing in for each token's 1024-number vector.",
    heatmapCaption: '5 tokens × 1024 numbers (12 shown)',
    arcsAria:
      "Causal attention arcs: the last token 'the' (the query) reaches back to each earlier token, with thicker arcs toward 'cat' and 'sat' showing more attention; no arc points forward.",
    arcsCaption:
      'Arc thickness = how much the last token pulls from each earlier one (it can only look backward).',
    layersAria:
      'Twenty-four stacked layer bars from input at the top to output at the bottom; each lower bar is slightly more saturated, showing the running vector growing richer with depth.',
    layersIn: 'in',
    layersOut: 'out',
    layersBadge: 'attention + MLP layers',
    layersNote: 'Each layer reads the running total and adds a small refinement.',
    logitsAria:
      "The last token's small vector on the left, an arrow, then a wide thin strip of 248,320 logits drawn as many short bars of varying height — one raw score per vocabulary token.",
    lastVector: 'last vector',
    logitsLabel: '248,320 logits',
    logitsCaption: 'One raw score per vocabulary token — only a handful of the 248,320 bars are drawn here.',
    stageBody: [
      <>
        The model can&apos;t read raw text. The tokenizer first chops your text into known chunks called{' '}
        <strong>tokens</strong> — here, 5 of them — and looks up each one&apos;s <strong>id</strong> (a plain
        integer). From here on, the model only ever sees these numbers.
      </>,
      <>
        Each id is looked up in a big table and becomes a <strong>vector</strong> — a list of 1024 numbers placing
        that token in &quot;meaning space&quot;. The same id always maps to the same row; <em>where</em> a token
        sits in the sentence is folded in later, inside <strong>attention</strong> — not here.
      </>,
      <>
        <strong>Attention</strong> lets each token look back at the earlier ones and pull in what it needs. To guess
        what follows &quot;the&quot;, the last token gathers meaning from &quot;cat&quot; and &quot;sat&quot;.
        Crucially, a token can only look <strong>backward</strong>, never forward.
      </>,
      <>
        Attention plus a small per-token network (the <strong>MLP</strong>) make up one <strong>layer</strong>.
        Qwen3.5 stacks <strong>24</strong> of them. Each layer reads the running total, adds a small refinement, and
        passes it up — so the vectors grow richer with depth.
      </>,
      <>
        Only the <strong>last</strong> token&apos;s vector decides the next word. The <strong>LM head</strong>{' '}
        scores that vector against the whole vocabulary, producing one raw score — a <strong>logit</strong> — for
        every one of the 248,320 tokens the model knows.
      </>,
      <>
        <strong>Softmax</strong> turns those raw scores into probabilities that sum to 1, and the model picks one.
        This small model&apos;s top pick here is <strong>&quot; floor&quot;</strong> — a real model is often less
        predictable than the obvious &quot; mat&quot;. That token is appended to the text and the whole journey runs
        again — that&apos;s how a sentence is written, one token at a time.
      </>,
    ] as readonly React.ReactNode[],
    stage6Append: <>→ append &apos; floor&apos;, then run the whole journey again (the generation loop).</>,
    railAria: 'Forward-pass stages',
    goToStage: (n: number, title: string) => `Go to stage ${n}: ${title}`,
    rightNow: 'Right now the data is:',
    srStatus: (stage: number, total: number, title: string, dataNow: string) =>
      `Stage ${stage} of ${total}: ${title}. Data now: ${dataNow}.`,
    stageCount: (stage: number, total: number) => `Stage ${stage} / ${total}`,
    learnMore: (label: string) => `Learn more: ${label} →`,
    // Play / Pause / Step live in the shared `StepControls` now, which carries
    // its own en/zh pair so every animated widget in the course says the same
    // word. Prev / Next stay here: the frame owns no backward step.
    prev: '‹ Prev',
    next: 'Next ›',
    hintReduced: 'Use Prev / Next to step through the forward pass.',
    hintPlay: 'Scrub the rail, or press Play to watch the forward pass run.',
    footer:
      'Illustrative — a schematic of the real forward pass; the numbers and heatmaps are stand-ins for the real 1024-dim vectors and 248,320 logits, not live model output.',
  },
  zh: {
    headerTitle: '跟随一个 token 的旅程',
    leadingSpaceNote: '前导空格是 token 的一部分。',
    heatmapAria:
      '示意嵌入热力图：5 行 token × 12 列彩色格子，青色为正值、琥珀色为负值，代表每个 token 那 1024 个数的向量。',
    heatmapCaption: '5 个 token × 1024 个数（显示 12 个）',
    arcsAria:
      '因果注意力弧线：最后一个 token “the”（query）回连到每个更早的 token，指向 “cat” 和 “sat” 的弧线更粗，表示注意力更多；没有任何弧线指向前方。',
    arcsCaption: '弧线粗细 = 最后一个 token 从每个更早 token 拉取的程度（它只能向后看）。',
    layersAria:
      '24 根堆叠的层条，从顶部的输入到底部的输出；越靠下的条颜色越饱和，表示随深度增加，流动中的向量越来越丰富。',
    layersIn: '入',
    layersOut: '出',
    layersBadge: '注意力 + MLP 层',
    layersNote: '每一层读取累计结果，加上一点小修正。',
    logitsAria:
      '左侧是最后一个 token 的小向量，经过一个箭头，右侧是一条又宽又薄的 248,320 个 logits 条带，画成许多高低不一的短条——词表中每个 token 一个原始分数。',
    lastVector: '最后的向量',
    logitsLabel: '248,320 个 logits',
    logitsCaption: '词表中每个 token 一个原始分数——这里只画出 248,320 根条中的一小撮。',
    stageBody: [
      <>
        模型读不了原始文本。分词器先把你的文本切成认识的片段，叫作 <strong>token</strong>——这里是 5
        个——再查出每个 token 的 <strong>id</strong>（一个普通整数）。从这里开始，模型看到的只有这些数字。
      </>,
      <>
        每个 id 在一张大表里查出一行，变成一个<strong>向量</strong>——一列 1024 个数，把这个 token
        安放在“含义空间”里。相同的 id 永远映射到同一行；token 在句子里的<em>位置</em>要到后面、在
        <strong>注意力</strong>里才被折算进来——不在这里。
      </>,
      <>
        <strong>注意力</strong>让每个 token 回头看更早的 token，把需要的信息拉进来。为了猜 &quot;the&quot;
        后面是什么，最后一个 token 从 &quot;cat&quot; 和 &quot;sat&quot; 那里收集含义。关键是：token 只能向
        <strong>后</strong>看，永远不能向前看。
      </>,
      <>
        注意力加上一个小的逐 token 网络（<strong>MLP</strong>）组成一<strong>层</strong>。Qwen3.5 叠了{' '}
        <strong>24</strong> 层。每一层读取累计结果，加上一点小修正，再传上去——向量随深度越来越丰富。
      </>,
      <>
        只有<strong>最后</strong>一个 token 的向量决定下一个词。<strong>LM head</strong>{' '}
        拿这个向量对整个词表打分，为模型认识的全部 248,320 个 token 各产生一个原始分数——也就是一个{' '}
        <strong>logit</strong>。
      </>,
      <>
        <strong>Softmax</strong> 把这些原始分数变成总和为 1 的概率，模型从中挑出一个。这个小模型在这里的第一候选是{' '}
        <strong>&quot; floor&quot;</strong>——真实模型常常不像“显而易见”的 &quot; mat&quot; 那样可预测。这个 token
        被追加到文本末尾，整趟旅程再跑一遍——一句话就是这样一次一个 token 写出来的。
      </>,
    ] as readonly React.ReactNode[],
    stage6Append: <>→ 追加 &apos; floor&apos;，然后把整趟旅程再跑一遍（生成循环）。</>,
    railAria: '前向传播阶段',
    goToStage: (n: number, title: string) => `跳到第 ${n} 步：${title}`,
    rightNow: '此刻的数据是：',
    srStatus: (stage: number, total: number, title: string, dataNow: string) =>
      `第 ${stage} / ${total} 步：${title}。当前数据：${dataNow}。`,
    stageCount: (stage: number, total: number) => `第 ${stage} / ${total} 步`,
    learnMore: (label: string) => `深入了解：${label} →`,
    prev: '‹ 上一步',
    next: '下一步 ›',
    hintReduced: '用“上一步 / 下一步”逐步走完这次前向传播。',
    hintPlay: '在阶段条上点选，或按“播放”观看前向传播完整跑一遍。',
    footer:
      '仅为示意——真实前向传播的简化示意图；图中的数字和热力图只是真实 1024 维向量与 248,320 个 logits 的替身，并非模型实时输出。',
  },
} as const;

// ---------------------------------------------------------------------------
// Stage 1 — token chips: token text above its id, leading space made visible.
// ---------------------------------------------------------------------------

function TokenChips() {
  const copy = COPY[useLocale()];
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-stretch gap-2">
        {TOKEN_TEXTS.map((raw, i) => {
          const hasLeadingSpace = raw.startsWith(' ');
          const visible = raw.replace(/^ /, '');
          return (
            <div
              key={`${TOKEN_IDS[i]}-${i}`}
              className="flex min-w-[3.5rem] flex-col items-center rounded-md border border-border bg-muted/30 px-2 py-1.5"
            >
              <span className="font-mono text-[13px] text-foreground/90">
                {hasLeadingSpace ? (
                  <span
                    aria-hidden="true"
                    className="mr-0.5 inline-block w-2 rounded-sm border-b border-dashed border-foreground/40 align-middle"
                  />
                ) : null}
                {visible}
              </span>
              <span className="mt-0.5 font-mono text-[10px] tabular-nums text-muted-foreground">{TOKEN_IDS[i]}</span>
            </div>
          );
        })}
      </div>
      <p className="text-[11px] text-muted-foreground">{copy.leadingSpaceNote}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stage 2 — illustrative 5×12 embedding heatmap.
// ---------------------------------------------------------------------------

function EmbeddingHeatmap() {
  const copy = COPY[useLocale()];
  const cellW = 26;
  const cellH = 22;
  const labelW = 40;
  const gap = 3;
  const width = labelW + HEATMAP_COLS * (cellW + gap);
  const height = HEATMAP_ROWS * (cellH + gap);
  return (
    <div className="space-y-1.5">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={copy.heatmapAria}
        className="block h-auto w-full max-w-[460px]"
      >
        {TOKEN_TEXTS.map((raw, r) => {
          const y = r * (cellH + gap);
          return (
            <g key={`row-${r}`}>
              <text
                x={labelW - 6}
                y={y + cellH / 2 + 3.5}
                fontSize={10}
                textAnchor="end"
                fill="currentColor"
                fillOpacity={0.7}
                style={{ fontFamily: 'var(--font-mono, monospace)' }}
              >
                {cleanupTokenText(raw)}
              </text>
              {Array.from({ length: HEATMAP_COLS }).map((_, c) => {
                const v = heatmapValue(r, c);
                return (
                  <rect
                    key={`cell-${r}-${c}`}
                    x={labelW + c * (cellW + gap)}
                    y={y}
                    width={cellW}
                    height={cellH}
                    rx={RX}
                    fill={heatmapFill(v)}
                    stroke="currentColor"
                    strokeOpacity={0.08}
                  />
                );
              })}
            </g>
          );
        })}
      </svg>
      <p className="font-mono text-[10px] text-muted-foreground">{copy.heatmapCaption}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stage 3 — causal attention arcs from the last token back to earlier ones.
// Mirrors the quadratic-path arc idiom from the KV-cache DecodeAnimation.
// ---------------------------------------------------------------------------

function AttentionArcs() {
  const locale = useLocale();
  const copy = COPY[locale];
  const slotW = 78;
  const padX = 12;
  const rowY = 78;
  const slotH = 30;
  const queryY = 14;
  const width = padX * 2 + TOKEN_TEXTS.length * slotW;
  const height = 130;
  const queryIdx = TOKEN_TEXTS.length - 1;

  function slotCx(i: number) {
    return padX + i * slotW + slotW / 2;
  }

  const qCx = slotCx(queryIdx);
  const qCy = queryY + 22;
  const maxW = Math.max(...ATTENTION_WEIGHTS);

  function arcTo(targetIdx: number): string {
    const tx = slotCx(targetIdx);
    const ty = rowY;
    const dx = tx - qCx;
    const midX = qCx + dx / 2;
    const midY = (qCy + ty) / 2 - Math.max(14, Math.abs(dx) * 0.22);
    return `M ${qCx} ${qCy} Q ${midX} ${midY} ${tx} ${ty}`;
  }

  return (
    <div className="space-y-1.5">
      {/* Legibility floor: viewBox is 414 wide (padX*2 + 5*slotW) and the
          smallest face in here is the 9px weight row, so the canvas may not
          scale below ceil(8 * 414 / 9) = 368px or that row drops under the
          course's 8 CSS px floor (it rendered at ~6.4px in a 320px column).
          PanBox pans what no longer fits; the caption stays outside it. */}
      <PanBox locale={locale}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={copy.arcsAria}
          className="block h-auto w-full min-w-[368px] max-w-[460px]"
        >
          {/* Query bubble above the last token. */}
          <rect
            x={qCx - slotW / 2 + 6}
            y={queryY}
            width={slotW - 12}
            height={slotH - 6}
            rx={RX}
            fill="oklch(0.65 0.2 25)"
          />
          <text
            x={qCx}
            y={queryY + 16}
            fontSize={11}
            textAnchor="middle"
            fill="white"
            style={{ fontFamily: 'var(--font-mono, monospace)' }}
          >
            query
          </text>

          {/* Backward arcs, thickness/opacity ∝ attention weight. */}
          {ATTENTION_WEIGHTS.map((w, i) => {
            const t = w / maxW;
            return (
              <path
                key={`arc-${i}`}
                d={arcTo(i)}
                fill="none"
                stroke="oklch(0.65 0.2 25)"
                strokeOpacity={0.3 + t * 0.6}
                strokeWidth={1 + t * 3.5}
              />
            );
          })}

          {/* Token row. */}
          {TOKEN_TEXTS.map((raw, i) => {
            const x = padX + i * slotW;
            const isQuery = i === queryIdx;
            return (
              <g key={`tok-${i}`}>
                <rect
                  x={x + 4}
                  y={rowY}
                  width={slotW - 8}
                  height={slotH}
                  rx={RX}
                  fill={isQuery ? 'oklch(0.65 0.2 25)' : 'currentColor'}
                  fillOpacity={isQuery ? 0.18 : 0.06}
                  stroke={isQuery ? 'oklch(0.65 0.2 25)' : 'currentColor'}
                  strokeOpacity={isQuery ? 0.8 : 0.18}
                  strokeWidth={isQuery ? SW.OUTER : SW.INNER}
                />
                <text
                  x={slotCx(i)}
                  y={rowY + slotH / 2 + 4}
                  fontSize={11}
                  textAnchor="middle"
                  fill="currentColor"
                  fillOpacity={0.85}
                  style={{ fontFamily: 'var(--font-mono, monospace)' }}
                >
                  {cleanupTokenText(raw)}
                </text>
                <text
                  x={slotCx(i)}
                  y={rowY + slotH + 14}
                  fontSize={9}
                  textAnchor="middle"
                  fill="currentColor"
                  fillOpacity={0.5}
                  style={{ fontFamily: 'var(--font-mono, monospace)' }}
                >
                  {ATTENTION_WEIGHTS[i]?.toFixed(2)}
                </text>
              </g>
            );
          })}
        </svg>
      </PanBox>
      <p className="text-[11px] text-muted-foreground">{copy.arcsCaption}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stage 4 — 24 stacked layer bars, the last token's vector intensifying down.
// ---------------------------------------------------------------------------

function LayerStack() {
  const locale = useLocale();
  const copy = COPY[locale];
  const width = 320;
  const barH = 5;
  const gap = 2.5;
  const padTop = 4;
  const padX = 44;
  const height = padTop * 2 + NUM_LAYERS * (barH + gap);
  return (
    // The svg used to be a bare flex child next to the "× 24" badge column, and
    // a flex child with `w-full` has no width of its own to defend: measured at
    // 375px the row squeezed this canvas to ~183px, scale 0.57, which put the
    // 7.5px layer numbers at 4.3px and the 9px in/out labels at 5.1px. Two
    // things fix it, and both are needed. (1) The `min-w` floor below stops the
    // shrink — and because PanBox sets `overflow-x`, its own automatic minimum
    // size as a flex item is zero, so the ROW still fits the column and pans
    // inside instead of pushing the page sideways. (2) Below `sm` the row
    // stacks, so the diagram gets the whole column rather than sharing ~320px
    // with a 160px badge and panning through the ~200px that would be left.
    <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
      {/* Legibility floor: viewBox 320 wide, smallest face 7.5px (the every-4th
          layer number), so ceil(8 * 320 / 7.5) = 342px. The old `max-w-[340px]`
          cap moves to 342 to match — 2px, and a cap BELOW the floor would just
          be dead CSS, since min-width wins over max-width. */}
      <PanBox locale={locale}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={copy.layersAria}
          className="block h-auto w-full min-w-[342px] max-w-[342px]"
        >
          <text x={padX - 6} y={padTop + 8} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.55}>
            {copy.layersIn}
          </text>
          <text
            x={padX - 6}
            y={height - padTop - 2}
            fontSize={9}
            textAnchor="end"
            fill="currentColor"
            fillOpacity={0.55}
          >
            {copy.layersOut}
          </text>
          {Array.from({ length: NUM_LAYERS }).map((_, i) => {
            const y = padTop + i * (barH + gap);
            // Intensify with depth: deeper layers carry a richer running vector.
            const t = i / (NUM_LAYERS - 1);
            const alpha = (0.22 + t * 0.66).toFixed(3);
            return (
              <g key={`layer-${i}`}>
                <rect
                  x={padX}
                  y={y}
                  width={width - padX - 8}
                  height={barH}
                  rx={RX}
                  fill={`oklch(0.62 0.16 265 / ${alpha})`}
                />
                {(i + 1) % 4 === 0 ? (
                  <text
                    x={width - 4}
                    y={y + barH}
                    fontSize={7.5}
                    textAnchor="end"
                    fill="currentColor"
                    fillOpacity={0.4}
                    style={{ fontFamily: 'var(--font-mono, monospace)' }}
                  >
                    {i + 1}
                  </text>
                ) : null}
              </g>
            );
          })}
        </svg>
      </PanBox>
      <div className="flex flex-col justify-center self-stretch">
        <div className="rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1 text-center">
          <div className="font-mono text-lg font-semibold text-primary">× 24</div>
          <div className="text-[10px] text-muted-foreground">{copy.layersBadge}</div>
        </div>
        <p className="mt-2 max-w-[10rem] text-[11px] text-muted-foreground">{copy.layersNote}</p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stage 5 — last token's vector → arrow → a wide thin logits strip.
// ---------------------------------------------------------------------------

// Deterministic illustrative logit bar heights across the long axis.
const LOGIT_BAR_COUNT = 40;
function logitBarHeight(i: number): number {
  // Vary by index, never randomly. A couple of taller spikes among low scores.
  return 0.18 + 0.5 * Math.abs(Math.sin(i * 0.9 + 0.4)) * (0.4 + 0.6 * Math.abs(Math.cos(i * 0.27)));
}

function LogitsStrip() {
  const locale = useLocale();
  const copy = COPY[locale];
  const width = 480;
  const height = 96;
  const vecCells = 6;
  const vecCellW = 14;
  const vecCellH = 14;
  const vecX = 6;
  const vecY = height / 2 - vecCellH / 2;
  const arrowX0 = vecX + vecCells * (vecCellW + 2) + 6;
  const arrowX1 = arrowX0 + 30;
  const stripX = arrowX1 + 8;
  const stripW = width - stripX - 8;
  const stripTop = 16;
  const stripH = 56;
  const barGap = stripW / LOGIT_BAR_COUNT;
  return (
    <div className="space-y-1.5">
      {/* Legibility floor: viewBox 480 wide and the only two labels here are
          9px, so ceil(8 * 480 / 9) = 427px. This svg carried no width floor and
          no cap at all, so a 320px column scaled it to 0.67 and both labels
          landed at ~5.5 CSS px. */}
      <PanBox locale={locale}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={copy.logitsAria}
          className="block h-auto w-full min-w-[427px]"
        >
          {/* Last token's vector — a small row of cells. */}
          {Array.from({ length: vecCells }).map((_, i) => (
            <rect
              key={`vec-${i}`}
              x={vecX + i * (vecCellW + 2)}
              y={vecY}
              width={vecCellW}
              height={vecCellH}
              rx={RX}
              fill={heatmapFill(heatmapValue(HEATMAP_ROWS - 1, i))}
              stroke="currentColor"
              strokeOpacity={0.1}
            />
          ))}
          <text x={vecX} y={vecY - 6} fontSize={9} fill="currentColor" fillOpacity={0.55}>
            {copy.lastVector}
          </text>

          {/* Arrow. */}
          <line
            x1={arrowX0}
            y1={height / 2}
            x2={arrowX1 - 4}
            y2={height / 2}
            stroke="currentColor"
            strokeOpacity={0.5}
            strokeWidth={SW.INNER}
          />
          <path
            d={`M ${arrowX1 - 4} ${height / 2 - 4} L ${arrowX1 + 3} ${height / 2} L ${arrowX1 - 4} ${height / 2 + 4} Z`}
            fill="currentColor"
            fillOpacity={0.5}
          />

          {/* Wide logits strip baseline + bars. */}
          <line
            x1={stripX}
            y1={stripTop + stripH}
            x2={stripX + stripW}
            y2={stripTop + stripH}
            stroke="currentColor"
            strokeOpacity={0.2}
          />
          {Array.from({ length: LOGIT_BAR_COUNT }).map((_, i) => {
            const h = logitBarHeight(i) * stripH;
            const isWinner = i === 4; // one prominent spike standing in for the winning token
            const bh = isWinner ? Math.max(h, stripH * 0.92) : h;
            return (
              <rect
                key={`logit-${i}`}
                x={stripX + i * barGap + 0.6}
                y={stripTop + stripH - bh}
                width={Math.max(1.5, barGap - 1.2)}
                height={bh}
                rx={RX}
                fill={isWinner ? 'oklch(0.68 0.17 150)' : 'currentColor'}
                fillOpacity={isWinner ? 1 : 0.4}
              />
            );
          })}
          <text x={stripX} y={stripTop - 5} fontSize={9} fill="currentColor" fillOpacity={0.55}>
            {copy.logitsLabel}
          </text>
        </svg>
      </PanBox>
      <p className="text-[11px] text-muted-foreground">{copy.logitsCaption}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Per-stage explanation copy (with real <strong> emphasis) and visual.
// ---------------------------------------------------------------------------

function StageBody({ stage }: { stage: number }) {
  const copy = COPY[useLocale()];
  const body = <p className="text-[12px] leading-relaxed text-foreground/85">{copy.stageBody[stage - 1]}</p>;
  switch (stage) {
    case 1:
      return (
        <>
          {body}
          <TokenChips />
        </>
      );
    case 2:
      return (
        <>
          {body}
          <EmbeddingHeatmap />
        </>
      );
    case 3:
      return (
        <>
          {body}
          <AttentionArcs />
        </>
      );
    case 4:
      return (
        <>
          {body}
          <LayerStack />
        </>
      );
    case 5:
      return (
        <>
          {body}
          <LogitsStrip />
        </>
      );
    case 6:
    default:
      return (
        <>
          <p className="text-[12px] leading-relaxed text-foreground/85">{copy.stageBody[5]}</p>
          <TopKBars
            ids={[...DIST_IDS]}
            probs={[...DIST_PROBS]}
            texts={[...DIST_TEXTS]}
            sampledTokenId={SAMPLED_TOKEN_ID}
            runKey={stage}
          />
          <p className="text-[11px] text-muted-foreground">{copy.stage6Append}</p>
        </>
      );
  }
}

// ---------------------------------------------------------------------------
// Main widget.
// ---------------------------------------------------------------------------

export function TokenJourney() {
  const locale = useLocale();
  const copy = COPY[locale];
  const stages = STAGES[locale];
  // `loop: false` — the old autoplay parked on the last stage instead of
  // wrapping, and `useStepPlayer.toggle` turns a Play press on a parked sweep
  // back into the "Replay" this widget used to spell out on its own button.
  //
  // `initialPlaying: false` restores what this widget did BEFORE the skin
  // conversion (`useState(false)`), which `useStepPlayer`'s `useState(true)`
  // default had silently flipped. This is the capstone: the payoff is stage 6,
  // the answer. At 2200ms x 6 stages the sweep is over in ~11s and it is not
  // viewport-gated, so a reader scrolling down to it arrives at a sweep that
  // already ran and parked on the ending — the setup never seen, and `loop:
  // false` means it never comes back around. It is also what makes
  // `copy.hintPlay` ("press Play …") true again: with autoplay on, the hint
  // said Play while the button said Pause.
  const player = useStepPlayer(TOTAL_STAGES, {
    frameMs: AUTOPLAY_MS,
    loop: false,
    restFrame: 0,
    initialPlaying: false,
  });
  // The only 0-indexed ↔ 1-indexed conversion in the file.
  const stage = player.frame + 1;

  const atStart = stage <= 1;
  const atEnd = stage >= TOTAL_STAGES;
  const current = stages[stage - 1]!;

  // The per-beat sentence. The old `sr-only role="status"` region held exactly
  // this string; `DiagramFrame` owns the a11y contract now, so it moves into the
  // caption slot — visible, and announced when the reader pauses or steps.
  const caption = copy.srStatus(stage, TOTAL_STAGES, current.title, current.dataNow);

  // Every caption this sweep can reach, so DiagramFrame reserves the tallest and
  // the chapter body cannot hop when the stage changes. The selector above has
  // exactly ONE arm — `srStatus` called with the current stage — and `stage`
  // walks [1, TOTAL_STAGES], which is precisely the index set of `stages`. So
  // mapping that table is the whole reachable set, per locale, by construction.
  const captions = stages.map((s) => copy.srStatus(s.n, TOTAL_STAGES, s.title, s.dataNow));

  return (
    <DiagramFrame
      title={copy.headerTitle}
      player={player}
      locale={locale}
      caption={caption}
      captions={captions}
      note={copy.footer}
    >
      {/* The body keeps its own tighter rhythm; DiagramFrame's `space-y-4` is
          for the frame's own parts. */}
      <div className="space-y-3">
        {/* The running example. It used to share the header row with the title;
            DiagramFrame owns that row (title + StepControls), so it moves in
            here. */}
        <div className="font-mono text-[11px] text-muted-foreground">
          &quot;{PROMPT}&quot; <span className="text-foreground/70">→ ?</span>
        </div>

        {/* Stage rail: 6 numbered, clickable chips. Not a frame control — the
            frame renders play/step only — so the scrubber stays widget-owned
            and drives the player through `goTo`, which pauses first. */}
        <div className="flex flex-wrap gap-1.5" role="group" aria-label={copy.railAria}>
          {stages.map((s) => {
            const isCurrent = s.n === stage;
            return (
              <button
                key={s.n}
                type="button"
                onClick={() => player.goTo(s.n - 1)}
                aria-current={isCurrent ? 'step' : undefined}
                aria-label={copy.goToStage(s.n, s.title)}
                className={cn(
                  'flex items-center gap-1.5 rounded-md border px-2 py-1 max-sm:px-2.5 max-sm:py-1.5 max-sm:min-h-[40px] text-left text-[11px] transition-colors',
                  player.reducedMotion && 'transition-none',
                  isCurrent
                    ? 'border-primary/50 bg-primary/15 text-primary'
                    : 'border-border/60 bg-muted/20 text-muted-foreground hover:bg-foreground/5',
                )}
              >
                <span
                  className={cn(
                    'inline-flex h-4 w-4 max-sm:h-5 max-sm:w-5 items-center justify-center rounded-full font-mono text-[10px] tabular-nums',
                    isCurrent ? 'bg-primary text-primary-foreground' : 'bg-foreground/10 text-foreground/70',
                  )}
                >
                  {s.n}
                </span>
                <span className="hidden font-medium sm:inline">{s.title}</span>
              </button>
            );
          })}
        </div>

        {/* Through-line spine: always-visible "right now the data is" badge. */}
        <div className="flex flex-wrap items-center gap-2 rounded-md border border-border/70 bg-muted/30 px-3 py-2">
          <span className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.rightNow}</span>
          <span className="font-mono text-[12px] font-medium text-foreground/90">{current.dataNow}</span>
        </div>

        {/* Stage panel. The visually-hidden status region that used to live here
            is gone: DiagramFrame owns the a11y contract, and the sentence it
            announced is now the caption. The visible "Stage n / 6" counter
            stays, because the frame has no counter slot. The verbose SVG
            aria-labels sit outside any live region, so they are not re-read on
            every scrub. */}
        <div className="space-y-3 rounded-md border border-border/60 bg-muted/10 p-3">
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-[11px] text-muted-foreground">{copy.stageCount(stage, TOTAL_STAGES)}</span>
            <span className="text-sm font-semibold text-foreground">{current.title}</span>
          </div>
          <StageBody stage={stage} />
          <div className="text-[12px]">
            <ChapterLink
              chapterId={current.chapterId}
              className="text-primary underline-offset-4 hover:underline"
            >
              {copy.learnMore(current.chapterLabel)}
            </ChapterLink>
          </div>
        </div>

        {/* Prev / Next. Play, Pause and Step are the frame's now; these two stay
            because `StepControls` has no backward step and this is a walk the
            reader is meant to be able to back up through.

            `aria-disabled`, never the real `disabled` attribute. A keyboard
            reader walking this sweep is holding focus on ONE of these two
            buttons, and the button they are pressing is exactly the one that
            goes inert at the boundary — a real `disabled` drops focus to
            `<body>`, so reaching stage 1 or stage 6 costs them their place in
            the page entirely and they have to Tab back in from the top. Kept
            focusable and announced as unavailable instead: the handler no-ops,
            and the styling is the same `pointer-events-none opacity-50` the
            Button's own `disabled:` variants paint, so nothing looks different. */}
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              if (!atStart) player.goTo(player.frame - 1);
            }}
            aria-disabled={atStart}
            className={cn(atStart && 'pointer-events-none opacity-50')}
          >
            {copy.prev}
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              if (!atEnd) player.goTo(player.frame + 1);
            }}
            aria-disabled={atEnd}
            className={cn(atEnd && 'pointer-events-none opacity-50')}
          >
            {copy.next}
          </Button>
          <span className="ml-auto text-[11px] text-muted-foreground">
            {player.reducedMotion ? copy.hintReduced : copy.hintPlay}
          </span>
        </div>
      </div>
    </DiagramFrame>
  );
}
