import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * SpecDecodeVariantsDiagram — "pick your drafter": the family of speculative
 * decoding methods, all sharing the SAME draft → verify → accept skeleton the
 * MTP sub-chapter already taught. They differ ONLY in WHERE the cheap draft
 * comes from.
 *
 * Self-contained: NO worker, NO model, NO WASM — pure SVG + React state, so it
 * server-renders for crawlers (createRoot replaces it on boot) and is robust
 * while the model downloads. There is NO autoplay and no time-dependent state,
 * so it needs no prefers-reduced-motion handling — the variant <button>s are the
 * only interaction, and the first render is deterministic (variant 0 = MTP).
 *
 * The shared skeleton (drawn once, identical for every variant):
 *   committed prefix → a "drafter" box (the ONLY thing that changes per variant)
 *   → D dashed draft cells → ONE full-model verify pass → emerald ✓ accepts.
 * Selecting a variant swaps the drafter box's label + a readout strip
 *   (draft source · typical draft length · best-for) + a plain-language blurb.
 *
 * Every per-variant fact is grounded in the task brief / primary sources and
 * quoted, not paraphrased from memory:
 *   - MTP head ...... Qwen3.5's own built-in multi-token-prediction head — the
 *                     "you are here" variant the course already covered.
 *   - Draft-target .. a SECOND smaller model (≥10× fewer params, same family /
 *                     tokenizer) drafts; original target verifies. Most overhead
 *                     (two models' weights/activations/KV) but no training.
 *   - Medusa ........ fine-tune the target to add a handful of extra decoder
 *                     HEADS (the paper trains 5, sometimes 3-4 suffice at
 *                     inference), each emitting one draft token; no separate
 *                     model. Limited draft count + acceptance; inspired EAGLE.
 *   - EAGLE ......... a purpose-built tiny draft model (often <1B) trained from
 *                     scratch, fed the target's HIDDEN STATES (early+mid+late
 *                     layer); its draft head is a SEPARATE small autoregressive
 *                     module that runs its own few extra small forward passes
 *                     (cheap — one tiny layer) to build up to ~8 draft tokens
 *                     (~2× Medusa) at high acceptance. The go-to for general use.
 *   - n-gram / lookahead — NO draft model. Prefill builds an n-gram dictionary
 *                     (token → observed suffix); decode looks up the suffix as
 *                     the draft (can exceed 10 tokens). High acceptance ONLY when
 *                     output ~ input (CODE, predictable syntax); flops on prose.
 *                     Lookahead Decoding builds the n-grams on the fly.
 *
 * Shared rules stated in copy: speculative decoding helps most at LOW batch
 * sizes (spare compute to verify) and is dynamically DISABLED at high batch
 * (compute saturated); higher temperature → harder-to-predict tokens → lower
 * acceptance; and ALL variants are LOSSLESS — at T=0 verify keeps a draft only
 * when it matches the real model's own top pick, and with sampling it runs the
 * same accept-with-probability-ratio / resample-on-reject test either way, so
 * they change SPEED, never the answer.
 *
 * HONESTY: the in-browser demo runs NONE of these (plain autoregressive, one
 * token per pass); the project's native engine runs the MTP variant (~1.06–1.46×
 * there). The selector is the menu a production server picks from.
 */

type VariantId = 'mtp' | 'draftTarget' | 'medusa' | 'eagle' | 'ngram';

const VARIANT_IDS: VariantId[] = ['mtp', 'draftTarget', 'medusa', 'eagle', 'ngram'];

const EMERALD = '#10b981'; // emerald-500 — accepted

// ── Per-locale copy. COPY.en strings are the original English, verbatim. ─────
// zh keeps the GLOSSARY TERMS IN ENGLISH (speculative decoding, draft, verify,
// accept, MTP, draft-target, Medusa, EAGLE, n-gram, lookahead, hidden states,
// decoder head, batch, token, Qwen3.5) and uses full-width punctuation in prose.
const COPY = {
  en: {
    title: 'Pick your drafter — the speculative decoding family',
    selectorLabel: 'drafter',
    // The shared skeleton's static SVG labels.
    committedLabel: 'committed so far',
    drafterLabel: 'cheap drafter',
    draftLabel: 'D draft tokens',
    verifyLabel: 'full-model verify · one pass',
    draftTag: 'draft (dashed = unverified)',
    acceptTag: 'kept only if verify accepts it (exact match at T=0)',
    skeletonNote: 'Same skeleton every time — only the drafter box changes.',
    // Readout strip labels.
    sourceLabel: 'draft source',
    lengthLabel: 'typical draft length',
    bestForLabel: 'best for',
    youAreHere: 'you are here · Qwen3.5',
    // Per-variant short button label (must fit a small button in BOTH locales).
    buttonLabels: {
      mtp: 'MTP head',
      draftTarget: 'draft-target',
      medusa: 'Medusa',
      eagle: 'EAGLE',
      ngram: 'n-gram / lookahead',
    } as Record<VariantId, string>,
    // The label drawn inside the drafter box in the SVG (kept SHORT — EN is the
    // wider locale, so size the box for these).
    boxLabels: {
      mtp: 'MTP head',
      draftTarget: 'small model',
      medusa: 'extra heads',
      eagle: 'tiny model',
      ngram: 'n-gram table',
    } as Record<VariantId, string>,
    // Readout values.
    source: {
      mtp: "the model's own built-in MTP head",
      draftTarget: 'a second, smaller model (same family)',
      medusa: 'a handful of extra decoder heads on the target',
      eagle: 'a tiny model fed the target hidden states',
      ngram: 'an n-gram dictionary — no draft model',
    } as Record<VariantId, string>,
    length: {
      mtp: '1 token (Qwen3.5 default depth)',
      draftTarget: 'a few tokens',
      medusa: '2–4 tokens (limited)',
      eagle: 'up to ~8 tokens (~2× Medusa)',
      ngram: 'can exceed 10 tokens',
    } as Record<VariantId, string>,
    bestFor: {
      mtp: 'what this course already covered',
      draftTarget: 'quick setup, no training',
      medusa: 'no separate model, modest gain',
      eagle: 'general chat — the go-to choice',
      ngram: 'code completion / predictable syntax',
    } as Record<VariantId, string>,
    // Plain-language blurb per variant.
    blurbs: {
      mtp: (
        <>
          The model drafts for <strong>itself</strong> through its built-in{' '}
          <span className="font-mono">MTP</span> head — no second model, no training. This is exactly the{' '}
          draft → verify → accept loop the course already walked through, and the variant{' '}
          <strong>this project&apos;s native engine</strong> actually runs.
        </>
      ),
      draftTarget: (
        <>
          A <strong>second, smaller</strong> model — at least <strong>10× fewer</strong> params, same family and{' '}
          tokenizer — writes the draft; the original <strong>target</strong> model verifies. Easiest to set up{' '}
          (no training), but the <strong>most overhead</strong> of any method: you store and coordinate two{' '}
          models&apos; weights, activations, and KV.
        </>
      ),
      medusa: (
        <>
          Fine-tune the target model to grow <strong>a handful of extra decoder heads</strong> (the paper trains 5,
          sometimes 3–4 suffice at inference), each emitting one draft token — so there is{' '}
          <strong>no separate model</strong>. The draft count and acceptance are limited, but Medusa is what{' '}
          <strong>inspired EAGLE</strong>.
        </>
      ),
      eagle: (
        <>
          A purpose-built <strong>tiny draft model</strong> (often under 1B params), trained from scratch, that is
          fed the target&apos;s internal <strong>hidden-state features</strong> instead of just the token ids (the later
          versions, <span className="font-mono">EAGLE-3</span>, fuse features from several layers). Its draft head is a{' '}
          <strong>separate small autoregressive module</strong> that runs its own few extra small forward passes
          (cheap, since it&apos;s a single tiny layer) to build up to <strong>~8 draft tokens</strong> (about 2× Medusa) at
          high acceptance, before the target&apos;s one verify pass — the trade EAGLE makes for higher acceptance than
          Medusa&apos;s single-pass heads, and the <strong>go-to</strong> for general use.
        </>
      ),
      ngram: (
        <>
          <strong>No draft model at all.</strong> During prefill, build an <span className="font-mono">n-gram</span>{' '}
          dictionary mapping a token → an observed suffix; during decode, look up the current token&apos;s suffix{' '}
          and use it as the draft (it can exceed 10 tokens). Acceptance is high <strong>only</strong> when the{' '}
          output closely matches the input — <strong>code</strong> completion, predictable syntax — and it flops{' '}
          on prose. <span className="font-mono">Lookahead</span> Decoding builds the n-grams on the fly during{' '}
          inference (more general, costs extra compute).
        </>
      ),
    } as Record<VariantId, React.ReactNode>,
    // Shared rules + honesty, always shown.
    rulesTitle: 'true for every drafter',
    rules: [
      <>
        <strong>All lossless.</strong> At T=0 verify keeps a draft only when it matches the real model&apos;s own top
        pick; with sampling it runs the same probability-ratio accept/resample test instead, so{' '}
        every variant changes <strong>speed, never the answer</strong>.
      </>,
      <>
        Helps most at <strong>low batch sizes</strong> (spare compute to verify); at high batch it is dynamically{' '}
        <strong>disabled</strong> — the GPU is already saturated.
      </>,
      <>
        Higher <strong>temperature</strong> → harder-to-predict tokens → <strong>lower acceptance</strong>.
      </>,
    ],
    honesty: (
      <>
        The <strong>in-browser demo runs none of these</strong> — it does plain autoregressive decode, one token
        per pass. The project&apos;s native engine runs the <strong>MTP</strong> variant (~1.06–1.46× there). Read
        this selector as the menu a <strong>production server</strong> picks from — <span className="font-mono">n-gram</span>{' '}
        for code, <strong>EAGLE</strong> for general chat — all sharing the <strong>same lossless verify</strong>{' '}
        the course already proved.
      </>
    ),
    ariaLabel:
      'Diagram of the speculative decoding family: a shared draft → verify → accept skeleton where the committed prefix feeds a cheap drafter, which proposes D dashed draft tokens, the full model verifies them in one pass, and matching drafts are accepted in emerald. Selecting a variant changes only the drafter box: the MTP head, a draft-target model, Medusa heads, EAGLE, or an n-gram table.',
  },
  zh: {
    title: '挑选你的 drafter——speculative decoding 家族',
    selectorLabel: 'drafter',
    committedLabel: '已提交',
    drafterLabel: '廉价 drafter',
    draftLabel: 'D 个 draft token',
    verifyLabel: '全模型 verify · 一遍',
    draftTag: 'draft（虚线＝未验证）',
    acceptTag: '只有 verify 接受才保留（T=0 时保证严格匹配）',
    skeletonNote: '骨架每次都一样——只有 drafter 这个盒子在变。',
    sourceLabel: 'draft 来源',
    lengthLabel: '典型 draft 长度',
    bestForLabel: '最适合',
    youAreHere: 'you are here · Qwen3.5',
    buttonLabels: {
      mtp: 'MTP head',
      draftTarget: 'draft-target',
      medusa: 'Medusa',
      eagle: 'EAGLE',
      ngram: 'n-gram / lookahead',
    } as Record<VariantId, string>,
    boxLabels: {
      mtp: 'MTP head',
      draftTarget: 'small model',
      medusa: 'extra heads',
      eagle: 'tiny model',
      ngram: 'n-gram table',
    } as Record<VariantId, string>,
    source: {
      mtp: '模型自带的 MTP head',
      draftTarget: '第二个更小的模型（同一家族）',
      medusa: 'target 上若干额外的 decoder head',
      eagle: '读取 target hidden states 的微型模型',
      ngram: '一个 n-gram 字典——没有 draft model',
    } as Record<VariantId, string>,
    length: {
      mtp: '1 个 token（Qwen3.5 默认 depth）',
      draftTarget: '几个 token',
      medusa: '2–4 个 token（有限）',
      eagle: '最多约 8 个 token（约 2× Medusa）',
      ngram: '可超过 10 个 token',
    } as Record<VariantId, string>,
    bestFor: {
      mtp: '本课程已经讲过的部分',
      draftTarget: '快速搭建、无需训练',
      medusa: '无需独立模型、收益适中',
      eagle: '通用对话——首选',
      ngram: '代码补全 / 可预测的语法',
    } as Record<VariantId, string>,
    blurbs: {
      mtp: (
        <>
          模型通过自带的 <span className="font-mono">MTP</span> head 为<strong>自己</strong>起草——没有第二个模型，也无需训练。这正是本课程已经走过的{' '}
          draft → verify → accept 循环，也是<strong>本项目原生引擎</strong>真正运行的那一种。
        </>
      ),
      draftTarget: (
        <>
          一个<strong>第二、更小</strong>的模型——参数至少<strong>少 10×</strong>，同一家族、同一 tokenizer——负责起草；由原始的{' '}
          <strong>target</strong> 模型来 verify。它最容易搭建（无需训练），但<strong>开销也最大</strong>：你要存储并协调两个模型的权重、激活与 KV。
        </>
      ),
      medusa: (
        <>
          对 target 模型做微调，长出<strong>若干额外的 decoder head</strong>（论文默认训练 5 个，推理时有时 3–4 个就够），每个输出一个 draft token——因此<strong>没有独立模型</strong>。draft 数量与接受率都有限，但 Medusa 正是{' '}
          <strong>启发了 EAGLE</strong> 的那一步。
        </>
      ),
      eagle: (
        <>
          一个专门打造的<strong>微型 draft model</strong>（通常不到 1B 参数），从零训练，被喂入 target 的内部{' '}
          <strong>hidden-state 特征</strong>（而不只是 token id；其较新版本 <span className="font-mono">EAGLE-3</span>{' '}
          会融合多层特征）。它的 draft head 是一个<strong>独立的小型自回归模块</strong>，会自己再跑几次额外的小前向（很便宜，因为只有单独一层），从而在 target 的一次 verify 前向之前起草最多<strong>约 8 个 token</strong>（约为 Medusa 的 2 倍）且接受率很高——这正是 EAGLE 用更多算力换取比 Medusa 单次前向 head 更高接受率的取舍，也是通用场景的<strong>首选</strong>。
        </>
      ),
      ngram: (
        <>
          <strong>完全不用 draft model。</strong>prefill 时构建一个 <span className="font-mono">n-gram</span> 字典，把一个 token 映射到一段观察到的后缀；decode 时查出当前 token 的后缀，直接当作 draft（可超过 10 个 token）。只有当输出与输入高度接近时接受率才高——<strong>代码</strong>补全、可预测的语法——遇到散文就失灵。<span className="font-mono">Lookahead</span> Decoding 则在推理过程中即时构建 n-gram（更通用，但要花额外算力）。
        </>
      ),
    } as Record<VariantId, React.ReactNode>,
    rulesTitle: '对每一种 drafter 都成立',
    rules: [
      <>
        <strong>全部无损。</strong>在 T=0 时，verify 只有在 draft 与真实模型自己的顶选一致时才保留；带采样时，它跑的是同一套概率比接受/重采样判据，所以每一种变体只改变<strong>速度，绝不改变答案</strong>。
      </>,
      <>
        在 <strong>低 batch</strong> 时收益最大（有富余算力去 verify）；在高 batch 时会被动态<strong>关闭</strong>——GPU 已经被占满了。
      </>,
      <>
        <strong>temperature</strong> 越高 → token 越难预测 → <strong>接受率越低</strong>。
      </>,
    ],
    honesty: (
      <>
        <strong>浏览器内的演示这些都不跑</strong>——它只做普通的自回归解码，每遍一个 token。本项目的原生引擎运行的是 <strong>MTP</strong> 变体（在那里约 1.06–1.46×）。请把这个选择器看作一台<strong>生产服务器</strong>会从中挑选的菜单——代码用 <span className="font-mono">n-gram</span>，通用对话用 <strong>EAGLE</strong>——它们共享着课程已经证明过的<strong>同一套无损 verify</strong>。
      </>
    ),
    ariaLabel:
      'speculative decoding 家族示意图：一个共享的 draft → verify → accept 骨架，已提交的前缀喂给一个廉价 drafter，它提出 D 个虚线 draft token，全模型在一遍内 verify，匹配的 draft 以翠绿色被接受。选择不同变体只会改变 drafter 这个盒子：MTP head、draft-target 模型、Medusa heads、EAGLE，或 n-gram 表。',
  },
} as const;

// ── SVG geometry ─────────────────────────────────────────────────────────────
// One shared skeleton. Sized so EN labels (the wider locale) fit every box.
const VB_W = 760;
const VB_H = 188;
const ROW_Y = 78;
const CELL_H = 46;

// committed prefix: two compact cells (last shown), then the drafter, then the
// D dashed draft cells under one verify bracket.
const PREFIX_X = 16;
const PREFIX_W = 64;
const PREFIX_GAP = 6;

const DRAFTER_X = 168;
const DRAFTER_W = 132;

const DRAFT_X = 332;
const DRAFT_W = 88;
const DRAFT_GAP = 8;
const D = 3; // illustrative draft cells in the skeleton

function prefixCellX(i: number): number {
  return PREFIX_X + i * (PREFIX_W + PREFIX_GAP);
}
function draftCellX(j: number): number {
  return DRAFT_X + j * (DRAFT_W + DRAFT_GAP);
}

// Illustrative skeleton tokens (stand in for raw model tokens, not UI copy).
const PREFIX_TOKENS = ['…', 'is'];
const DRAFT_TOKENS = ['Paris', ',', 'a'];

export function SpecDecodeVariantsDiagram() {
  const copy = COPY[useLocale()];
  const [variantIdx, setVariantIdx] = React.useState(0);
  const variant = VARIANT_IDS[variantIdx];
  const isMtp = variant === 'mtp';

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header + variant selector */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
          <span className="hidden sm:inline">{copy.selectorLabel}:</span>
          <div className="flex flex-wrap overflow-hidden rounded border border-border">
            {VARIANT_IDS.map((id, i) => {
              const active = i === variantIdx;
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => setVariantIdx(i)}
                  aria-pressed={active}
                  className={[
                    'px-2 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                    i > 0 ? 'border-l border-border' : '',
                  ].join(' ')}
                  style={{
                    background: active ? 'var(--primary)' : 'transparent',
                    color: active ? 'var(--background)' : 'var(--muted-foreground)',
                  }}
                >
                  {copy.buttonLabels[id]}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* "you are here" badge for the MTP variant */}
      <div className="min-h-[1.25rem]">
        {isMtp ? (
          <span className="inline-flex items-center rounded bg-primary/15 px-1.5 py-0.5 text-[11px] font-medium text-primary">
            {copy.youAreHere}
          </span>
        ) : null}
      </div>

      {/* The shared skeleton */}
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.ariaLabel}>
        {/* Section labels (wide labels above the boxes, never between them) */}
        <text x={PREFIX_X} y={ROW_Y - 18} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
          {copy.committedLabel}
        </text>
        <text
          x={DRAFTER_X + DRAFTER_W / 2}
          y={ROW_Y - 18}
          textAnchor="middle"
          style={{ fill: 'var(--primary)' }}
          className="text-[11px] font-medium"
        >
          {copy.drafterLabel}
        </text>
        <text
          x={draftCellX(0)}
          y={ROW_Y - 18}
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[11px]"
        >
          {copy.draftLabel}
        </text>

        {/* committed prefix cells */}
        {PREFIX_TOKENS.map((tok, i) => (
          <g key={`p${i}`}>
            <rect
              x={prefixCellX(i)}
              y={ROW_Y}
              width={PREFIX_W}
              height={CELL_H}
              rx={6}
              style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
              strokeWidth={1.5}
            />
            <text
              x={prefixCellX(i) + PREFIX_W / 2}
              y={ROW_Y + CELL_H / 2 + 5}
              textAnchor="middle"
              style={{ fill: 'var(--muted-foreground)' }}
              className="font-mono text-[14px]"
            >
              {tok}
            </text>
          </g>
        ))}

        {/* connector: prefix → drafter */}
        <line
          x1={prefixCellX(1) + PREFIX_W}
          y1={ROW_Y + CELL_H / 2}
          x2={DRAFTER_X}
          y2={ROW_Y + CELL_H / 2}
          style={{ stroke: 'var(--border)' }}
          strokeWidth={1.5}
        />

        {/* the drafter box — the ONLY thing that changes per variant */}
        <g>
          <rect
            x={DRAFTER_X}
            y={ROW_Y - 2}
            width={DRAFTER_W}
            height={CELL_H + 4}
            rx={8}
            style={{ fill: 'var(--primary)', fillOpacity: 0.12, stroke: 'var(--primary)' }}
            strokeWidth={2}
          />
          <text
            x={DRAFTER_X + DRAFTER_W / 2}
            y={ROW_Y + CELL_H / 2 + 5}
            textAnchor="middle"
            style={{ fill: 'var(--foreground)' }}
            className="font-mono text-[14px] font-semibold"
          >
            {copy.boxLabels[variant]}
          </text>
        </g>

        {/* connector: drafter → draft cells */}
        <line
          x1={DRAFTER_X + DRAFTER_W}
          y1={ROW_Y + CELL_H / 2}
          x2={DRAFT_X}
          y2={ROW_Y + CELL_H / 2}
          style={{ stroke: 'var(--border)' }}
          strokeWidth={1.5}
        />

        {/* verify bracket over all D draft cells (one pass) */}
        <rect
          x={draftCellX(0) - 5}
          y={ROW_Y - 6}
          width={D * DRAFT_W + (D - 1) * DRAFT_GAP + 10}
          height={CELL_H + 12}
          rx={9}
          fill="var(--primary)"
          fillOpacity={0.06}
          stroke="var(--primary)"
          strokeWidth={1.5}
          strokeDasharray="2 3"
        />

        {/* D dashed draft cells → accepted in emerald (matches the real pick) */}
        {DRAFT_TOKENS.map((tok, j) => (
          <g key={`d${j}`}>
            <rect
              x={draftCellX(j)}
              y={ROW_Y}
              width={DRAFT_W}
              height={CELL_H}
              rx={6}
              fill={EMERALD}
              fillOpacity={0.14}
              stroke={EMERALD}
              strokeWidth={2}
              strokeDasharray="5 4"
            />
            <text
              x={draftCellX(j) + DRAFT_W / 2}
              y={ROW_Y + CELL_H / 2 + 5}
              textAnchor="middle"
              fill={EMERALD}
              className="font-mono text-[14px] font-semibold"
            >
              {tok}
            </text>
            <text
              x={draftCellX(j) + DRAFT_W - 8}
              y={ROW_Y + 16}
              textAnchor="end"
              fill={EMERALD}
              className="text-[13px] font-bold"
            >
              ✓
            </text>
          </g>
        ))}

        {/* verify label sits ABOVE the bracket (wide EN label, never between cells) */}
        <text
          x={draftCellX(0)}
          y={ROW_Y + CELL_H + 22}
          style={{ fill: 'var(--primary)' }}
          className="text-[11px] font-medium"
        >
          ↑ {copy.verifyLabel}
        </text>
        {/* accept caption under the cells */}
        <text
          x={draftCellX(0)}
          y={ROW_Y + CELL_H + 38}
          fill={EMERALD}
          className="text-[11px]"
        >
          {copy.acceptTag}
        </text>
      </svg>

      {/* "skeleton is fixed" note */}
      <p className="text-[10px] italic text-muted-foreground/70">{copy.skeletonNote}</p>

      {/* Readout strip — draft source · typical length · best-for */}
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
        <div className="rounded border border-border bg-card/40 p-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.sourceLabel}</div>
          <div className="mt-0.5 text-[12px] text-foreground/90">{copy.source[variant]}</div>
        </div>
        <div className="rounded border border-border bg-card/40 p-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.lengthLabel}</div>
          <div className="mt-0.5 text-[12px] text-foreground/90">{copy.length[variant]}</div>
        </div>
        <div className="rounded border border-border bg-card/40 p-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.bestForLabel}</div>
          <div className="mt-0.5 text-[12px] text-foreground/90">{copy.bestFor[variant]}</div>
        </div>
      </div>

      {/* Plain-language blurb for the selected variant */}
      <p className="min-h-[4.5rem] text-[13px] text-foreground/90">{copy.blurbs[variant]}</p>

      {/* Shared rules (true for every drafter) */}
      <div className="rounded border border-border bg-muted/40 p-2.5">
        <div className="mb-1.5 text-[10px] uppercase tracking-wider text-muted-foreground">{copy.rulesTitle}</div>
        <ul className="space-y-1">
          {copy.rules.map((r, i) => (
            <li key={i} className="flex gap-2 text-[12px] leading-snug text-foreground/90">
              <span aria-hidden className="mt-[1px] select-none text-primary">
                ·
              </span>
              <span>{r}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Honesty framing */}
      <p className="border-t border-border/60 pt-3 text-[12px] text-muted-foreground">{copy.honesty}</p>
    </div>
  );
}
