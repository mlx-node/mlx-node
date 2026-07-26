import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { TopKBars, cleanupTokenText } from '../inspector/TopKBars';
import { DiagramFrame, useStepPlayer } from '../motion';
import { SegmentedToggle } from '../scaffolding/SegmentedToggle';

/**
 * Chapter 12 (KV cache & hybrid attention) — the generation loop with KV
 * reuse, plus a cache-vs-no-cache cost contrast. Self-contained: NO worker,
 * NO model, NO WASM. A scripted prompt is prefilled in parallel, then a few
 * tokens are decoded one at a time; per step we show the next-token
 * distribution (reused TopKBars) and the per-step attention work.
 *
 * The cache toggle is the lesson:
 *   - ON  → each decode step (re)processes only the ONE new token (the prefix's
 *           K/V is read from cache, not recomputed). Positions processed per step
 *           stay flat at 1. (The new token still attends over the whole cached
 *           prefix, so the attention read itself still grows with context.)
 *   - OFF → each decode step re-processes the entire prefix plus everything
 *           generated so far. Positions processed GROW, so total work is quadratic.
 *
 * This is deliberately distinct from the in-chapter `DecodeAnimation` in the
 * Try-It panel (which draws bespoke SVG attention arcs and has no cache
 * toggle): here it's a token strip + per-step TopKBars + a per-step work bar.
 *
 * Animation model: `useStepPlayer` + `DiagramFrame` from `learn/motion`,
 * replacing this file's former hand-rolled `setInterval`, its private
 * `prefersReducedMotion()` read inside a `useState` initializer, and its own
 * play/pause button, header, caption paragraph and footnote. There is no `<svg>`
 * here at all, so this is a KIT-ONLY adoption — none of the skin's drawing
 * tokens (RX / SW / DASH / PanelFrame …) have anything to apply to.
 *
 * The KV-cache toggle is NOT a frame control, so it stays with the widget,
 * inside `children` directly above the body — and because it swaps one whole
 * family of captions for another, BOTH families are listed in `captions` so the
 * slot is sized for the tallest either mode can produce.
 */

// Scripted sequence. Prompt is prefilled once; the rest is decoded one token
// at a time. Matches the chapter's DECODE_GENERATED_TOKENS = [' floor', '.', ' It']
// (10-kv-cache.tsx) so the two views agree on the measured continuation.
const PROMPT: ReadonlyArray<string> = ['The', ' cat', ' sat', ' on', ' the'];
const GENERATED: ReadonlyArray<string> = [' floor', '.', ' It'];

// Per-decode-step next-token candidates; index 0 is the sampled (greedy) token.
type Cand = { text: string; prob: number };
const DECODE_STEPS: ReadonlyArray<{ cands: Cand[] }> = [
  {
    cands: [
      { text: ' floor', prob: 0.42 },
      { text: ' mat', prob: 0.19 },
      { text: ' rug', prob: 0.12 },
      { text: ' couch', prob: 0.08 },
      { text: ' bed', prob: 0.05 },
    ],
  },
  {
    cands: [
      { text: '.', prob: 0.54 },
      { text: ',', prob: 0.18 },
      { text: ' and', prob: 0.1 },
      { text: ' beside', prob: 0.05 },
      { text: ' near', prob: 0.04 },
    ],
  },
  {
    cands: [
      { text: ' It', prob: 0.31 },
      { text: ' The', prob: 0.17 },
      { text: ' A', prob: 0.09 },
      { text: ' She', prob: 0.07 },
      { text: ' Then', prob: 0.05 },
    ],
  },
];

// Per-locale copy. COPY.en is the original English, moved verbatim.
//
// Play / Pause are NOT here any more: those verbs now belong to the shared
// `StepControls`, which carries its own en/zh pair so every animated widget in
// the course says the same word.
const COPY = {
  en: {
    title: 'The generation loop — KV reuse vs recompute',
    kvCacheLabel: 'KV cache',
    toggleAria: 'KV cache toggle',
    toggleOn: 'on',
    toggleOff: 'off',
    hintOn: 'prefix K/V reused — no re-projecting the prefix each step',
    hintOff: 'prefix re-processed every step — quadratic total',
    prefillStrip: 'prefill',
    decodeStrip: 'decode',
    prefillStep: 'Prefill (one parallel pass)',
    decodeStep: (step: number, total: number) => `Decode step ${step} / ${total}`,
    positionsProcessed: (n: number) => `positions (re)processed: ${n}`,
    workBarAria: (work: number, max: number) => `Token positions (re)processed this step: ${work} of up to ${max}`,
    prefillExplain: (promptLen: number) =>
      `Prefill processes all ${promptLen} prompt tokens together and writes their K/V into the cache.`,
    cacheOnExplain: (priorLen: number) =>
      `Cache ON: only the 1 new token's K/V is computed; the ${priorLen}-token prefix is read from cache, not recomputed — positions (re)processed stay flat at 1. (The new token still attends over all ${priorLen} cached keys, so that read still grows with context.)`,
    cacheOffExplain: (priorLen: number, work: number) =>
      `Cache OFF: the model re-processes all ${priorLen} prior tokens plus the new one (${work} total). The bar grows every step — that is the quadratic blow-up.`,
    totalsTitle: 'Total positions processed so far',
    totalOnRow: (terms: string) => `cache ON: ${terms}`,
    totalOffRow: (terms: string) => `cache OFF: ${terms}`,
    barsPrefill: 'Next-token scores (first decode step, not sampled yet)',
    barsDecode: 'Next token → keep the top bar',
    summary: (
      <>
        The KV cache is why decode stays affordable. With it (green), each step computes Q/K/V for just the single new
        token instead of re-projecting the whole prefix — so the positions it (re)processes stay flat at 1, and across{' '}
        <code>n</code> tokens that is linear rather than quadratic. The new token still attends over every cached key,
        so that attention read does grow with context — the cache removes the prefix re-projection, not the growing
        attention. Without it (red), every step re-runs the entire prefix from scratch, so the positions processed
        climb each step and the total work is quadratic in the sequence length. The running totals above make the gap
        concrete: after the three decode steps here, cache ON has processed 5 + 1 + 1 + 1 = 8 positions, cache OFF 5 +
        6 + 7 + 8 = 26 — and the gap widens with every extra token.
      </>
    ),
    footnote: "Scripted to show the loop's shape and the cache-vs-recompute cost — not live output from the model.",
  },
  zh: {
    title: '生成循环——KV 复用 vs 重新计算',
    kvCacheLabel: 'KV 缓存',
    toggleAria: 'KV 缓存开关',
    toggleOn: '开',
    toggleOff: '关',
    hintOn: '复用前缀 K/V——每一步都不再重新投影前缀',
    hintOff: '每一步都重新处理前缀——总量呈平方级',
    prefillStrip: 'prefill',
    decodeStrip: '解码',
    prefillStep: 'Prefill（一次并行传播）',
    decodeStep: (step: number, total: number) => `解码第 ${step} / ${total} 步`,
    positionsProcessed: (n: number) => `（重新）处理的位置数：${n}`,
    workBarAria: (work: number, max: number) => `本步（重新）处理的 token 位置数：${work}，上限 ${max}`,
    prefillExplain: (promptLen: number) =>
      `Prefill 把全部 ${promptLen} 个提示词 token 一起处理，并把它们的 K/V 写入缓存。`,
    cacheOnExplain: (priorLen: number) =>
      `缓存开：只为这 1 个新 token 计算 K/V；${priorLen} 个 token 的前缀直接从缓存读出、不再重新计算——（重新）处理的位置数恒为 1。（新 token 仍要对全部 ${priorLen} 个缓存的 key 做注意力，所以那次读取仍随上下文增长。）`,
    cacheOffExplain: (priorLen: number, work: number) =>
      `缓存关：模型重新处理全部 ${priorLen} 个旧 token 外加这个新 token（共 ${work} 个）。这条柱每一步都在变长——这就是平方级爆炸。`,
    totalsTitle: '迄今处理的总位置数',
    totalOnRow: (terms: string) => `缓存开：${terms}`,
    totalOffRow: (terms: string) => `缓存关：${terms}`,
    barsPrefill: '下一个 token 的分数（第一步解码，尚未采样）',
    barsDecode: '下一个 token → 保留最高的那条',
    summary: (
      <>
        KV 缓存是解码依然负担得起的原因。开启它（绿色），每一步只为这一个新 token 计算
        Q/K/V，而不是重新投影整个前缀——所以它（重新）处理的位置数恒为 1，跨 <code>n</code> 个 token
        的总量是线性而非平方。新 token 仍要对每一个缓存的 key
        做注意力，所以那次注意力读取确实随上下文增长——缓存去掉的是前缀的重新投影，而不是不断增长的注意力。关闭它（红色），每一步都从头重跑整个前缀，处理的位置数步步攀升，总工作量随序列长度呈平方级。上面的累计总数把差距摆得很具体：这里三步解码之后，缓存开共处理了
        5 + 1 + 1 + 1 = 8 个位置，缓存关则是 5 + 6 + 7 + 8 = 26——而且每多一个 token，差距就再拉大一截。
      </>
    ),
    footnote: '脚本化演示，只为展示循环的形态与“缓存 vs 重新计算”的代价——并非模型的实时输出。',
  },
} as const;

const PROMPT_LEN = PROMPT.length;
// Frame 0 = prefill, frames 1..GENERATED.length = decode steps.
const TOTAL_FRAMES = 1 + GENERATED.length;
const FRAME_MS = 1300;

export function GenerationLoopTrace() {
  const locale = useLocale();
  const copy = COPY[locale];
  const player = useStepPlayer(TOTAL_FRAMES, { frameMs: FRAME_MS });
  const { frame } = player;
  const [cacheOn, setCacheOn] = React.useState(true);

  const isPrefill = frame === 0;
  const decodeIdx = frame - 1; // valid only when !isPrefill

  // Tokens generated so far (newest flashes). During prefill none exist yet.
  const generatedCount = isPrefill ? 0 : decodeIdx + 1;
  const generated = GENERATED.slice(0, generatedCount);
  const flashIndex = isPrefill ? -1 : decodeIdx;

  // Sequence length already committed BEFORE this decode step's new token:
  // prompt + tokens generated in prior steps.
  const priorLen = PROMPT_LEN + (isPrefill ? 0 : decodeIdx);

  // Per-step attention work (number of token positions processed this step):
  //   cache ON  → just the 1 new token (prefix K/V reused).
  //   cache OFF → re-process the whole prefix + the new token.
  const workThisStep = isPrefill ? PROMPT_LEN : cacheOn ? 1 : priorLen + 1;
  // Worst-case work for the bar's full width: no-cache on the LAST decode step.
  const maxWork = PROMPT_LEN + GENERATED.length;
  const workPct = (workThisStep / maxWork) * 100;

  // Running totals — the cumulative divergence is the whole "quadratic" claim.
  // Per-step terms after the 5-token prefill:
  //   ON  → 5, +1, +1, +1   → 5, 6, 7, 8
  //   OFF → 5, +6, +7, +8   → 5, 11, 18, 26
  // Both expressions are shown so the learner can read the sum, not just trust it.
  const onTerms = [PROMPT_LEN, ...Array.from({ length: generatedCount }, () => 1)];
  const offTerms = [PROMPT_LEN, ...Array.from({ length: generatedCount }, (_, k) => PROMPT_LEN + k + 1)];
  const totalOn = onTerms.reduce((s, v) => s + v, 0);
  const totalOff = offTerms.reduce((s, v) => s + v, 0);

  // Current decode step's distribution (prefill shows the first step's bars,
  // dimmed via sampledTokenId = -1 so nothing is "kept" yet).
  const stepForBars = isPrefill ? 0 : decodeIdx;
  const cands = DECODE_STEPS[stepForBars].cands;
  const ids = cands.map((_, i) => i);
  const probs = cands.map((c) => c.prob);
  const texts = cands.map((c) => c.text);
  const sampledTokenId = isPrefill ? -1 : 0;

  // The per-beat sentence. Exactly three arms, and the cache toggle picks
  // between the last two.
  const caption = isPrefill
    ? copy.prefillExplain(PROMPT_LEN)
    : cacheOn
      ? copy.cacheOnExplain(priorLen)
      : copy.cacheOffExplain(priorLen, workThisStep);

  // Every caption this sweep can reach, so DiagramFrame reserves the tallest and
  // the chapter body cannot hop when the beat changes. All three arms are
  // builders, so each is mapped over the full set of arguments it can be called
  // with — read straight off the derivations above:
  //   prefill  → prefillExplain(PROMPT_LEN), one fixed call (frame 0 only).
  //   decode   → decodeIdx = frame - 1 walks [0, GENERATED.length), so
  //              priorLen = PROMPT_LEN + k and, in the OFF arm,
  //              workThisStep = priorLen + 1. Both cache modes are listed
  //              because the toggle swaps one whole family for the other and the
  //              OFF sentences are the longer ones.
  const decodeIdxs = Array.from({ length: GENERATED.length }, (_, k) => k);
  const captions = [
    copy.prefillExplain(PROMPT_LEN),
    ...decodeIdxs.map((k) => copy.cacheOnExplain(PROMPT_LEN + k)),
    ...decodeIdxs.map((k) => copy.cacheOffExplain(PROMPT_LEN + k, PROMPT_LEN + k + 1)),
  ];

  return (
    <DiagramFrame
      title={copy.title}
      player={player}
      locale={locale}
      caption={caption}
      captions={captions}
      note={
        // `summary` is body prose and would normally sit OUTSIDE the frame (see
        // GenerationLoop). Here it is authored BEFORE the footnote, and moving
        // only one of them out would flip the reading order — so both stay, and
        // `summary` carries its own tier explicitly rather than inheriting the
        // 11px footnote size.
        <>
          <p className="text-[12px] text-foreground/85">{copy.summary}</p>
          <p>{copy.footnote}</p>
        </>
      }
    >
      {/* The body keeps its own tighter rhythm; DiagramFrame's `space-y-4` is
          for the frame's own parts. */}
      <div className="space-y-3">
        {/* Cache toggle — an extra control the frame does not own, so it stays
            with the widget, directly above the diagram body. */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.kvCacheLabel}</span>
          <SegmentedToggle
            value={cacheOn}
            onChange={setCacheOn}
            ariaLabel={copy.toggleAria}
            options={[
              { value: true, label: copy.toggleOn },
              { value: false, label: copy.toggleOff },
            ]}
          />
          <span className="text-[11px] text-muted-foreground">{cacheOn ? copy.hintOn : copy.hintOff}</span>
        </div>

        {/* Token strip: wide/dim prompt (prefilled in parallel) + decode chips. */}
        <div className="flex flex-wrap items-center gap-1 rounded-md border border-border/60 bg-muted/30 p-2">
          <span className="mr-1 text-[10px] uppercase tracking-wider text-muted-foreground">{copy.prefillStrip}</span>
          {PROMPT.map((t, i) => (
            <span
              key={`p-${i}`}
              className={[
                'rounded px-2 py-0.5 font-mono text-[11px] tracking-wide transition-colors',
                isPrefill ? 'bg-primary/15 text-primary ring-1 ring-primary/40' : 'bg-background text-muted-foreground',
              ].join(' ')}
            >
              {cleanupTokenText(t)}
            </span>
          ))}
          <span className="mx-1 text-muted-foreground" aria-hidden>
            →
          </span>
          <span className="mr-1 text-[10px] uppercase tracking-wider text-muted-foreground">{copy.decodeStrip}</span>
          {generated.map((t, i) => (
            <span
              key={`g-${i}`}
              className={[
                'rounded px-1.5 py-0.5 font-mono text-[11px] transition-colors',
                i === flashIndex
                  ? 'bg-primary/20 text-primary ring-1 ring-primary/40'
                  : 'bg-primary/10 text-foreground/90',
              ].join(' ')}
            >
              {cleanupTokenText(t)}
            </span>
          ))}
          <span className="ml-0.5 font-mono text-[11px] text-muted-foreground" aria-hidden>
            ▮
          </span>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          {/* Per-step work meter — the heart of the cache contrast. */}
          <div className="space-y-2">
            <div className="flex items-baseline justify-between text-[11px]">
              <span className="text-muted-foreground">
                {isPrefill ? copy.prefillStep : copy.decodeStep(decodeIdx + 1, GENERATED.length)}
              </span>
              <span className="font-mono text-foreground/85">{copy.positionsProcessed(workThisStep)}</span>
            </div>
            <div
              className="relative h-6 w-full overflow-hidden rounded-sm bg-muted/60"
              role="img"
              aria-label={copy.workBarAria(workThisStep, maxWork)}
            >
              <div
                className="absolute inset-y-0 left-0 transition-all"
                style={{
                  width: `${Math.max(2, Math.min(100, workPct)).toFixed(1)}%`,
                  backgroundColor: isPrefill ? '#94a3b8' : cacheOn ? '#22c55e' : '#ef4444',
                }}
              />
            </div>

            {/* The per-step explanation that used to sit here is now the frame's
                caption — one accessible copy, and a slot sized for the tallest
                sentence either cache mode can reach.

                Running totals — the cumulative gap is the point. Both rows stay
                visible; the active cache mode is accented. */}
            <div className="rounded-md border border-border/60 bg-muted/30 p-2">
              <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">{copy.totalsTitle}</div>
              <div
                className={[
                  'flex items-baseline justify-between gap-2 font-mono text-[12px]',
                  cacheOn ? 'font-semibold text-emerald-600 dark:text-emerald-400' : 'text-muted-foreground',
                ].join(' ')}
              >
                <span>{copy.totalOnRow(onTerms.join(' + '))}</span>
                <span className="text-sm">= {totalOn}</span>
              </div>
              <div
                className={[
                  'flex items-baseline justify-between gap-2 font-mono text-[12px]',
                  !cacheOn ? 'font-semibold text-red-600 dark:text-red-400' : 'text-muted-foreground',
                ].join(' ')}
              >
                <span>{copy.totalOffRow(offTerms.join(' + '))}</span>
                <span className="text-sm">= {totalOff}</span>
              </div>
            </div>
          </div>

          {/* Current decode step's next-token distribution. */}
          <div className="space-y-1">
            <div className="text-[11px] text-muted-foreground">{isPrefill ? copy.barsPrefill : copy.barsDecode}</div>
            <TopKBars ids={ids} probs={probs} texts={texts} sampledTokenId={sampledTokenId} runKey={frame + 1} />
          </div>
        </div>
      </div>
    </DiagramFrame>
  );
}
