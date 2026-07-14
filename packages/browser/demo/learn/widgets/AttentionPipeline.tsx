import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { MathDisplay } from '../scaffolding/MathDisplay';
import { SegmentedToggle } from '../scaffolding/SegmentedToggle';

/**
 * Chapter 5 supplement — the WHOLE attention data flow, end to end, on one
 * diagram. Every other widget in this chapter zooms into one piece (grouping,
 * cache, head patterns); this one connects them so a learner can see how
 * hidden `h` becomes the attention output.
 *
 * A toggle swaps the sequence mixer:
 *   • Full attention (GQA): h → Q/K/V projections → RoPE on Q,K → QKᵀ/√d →
 *     causal mask → softmax → ·V → concat heads → W_O → output. Quadratic in
 *     sequence length; the KV cache grows with the sequence.
 *   • Linear (GatedDeltaNet): a high-level alternate path with a FIXED-SIZE
 *     recurrent state (no QKᵀ, no growing KV cache), updated per token.
 *
 * A "step" control highlights one stage at a time; hovering a stage also
 * emphasises it. All labels/constants are single-sourced from the architecture
 * chapter — no model, no worker.
 *
 * Visual idiom (rounded bordered boxes + arrows, color-coded by kind) follows
 * StackCollapsedView / the forward-pass flow used elsewhere in the course.
 */

// Architecture constants (single source of truth: 14-architecture.tsx).
const HIDDEN_DIM = 1024;
const HEAD_DIM = 256;
const NUM_HEADS = 8;
const NUM_KV_HEADS = 2;
const NUM_LAYERS = 24;
const NUM_FULL = 6; // every 4th layer
const NUM_LINEAR = NUM_LAYERS - NUM_FULL; // 18
const GROUP_SIZE = NUM_HEADS / NUM_KV_HEADS; // 4

type Mixer = 'full' | 'linear';

type Kind = 'io' | 'proj' | 'norm' | 'rope' | 'score' | 'mask' | 'softmax' | 'mix' | 'gate' | 'out' | 'state';

type Stage = {
  id: string;
  kind: Kind;
  label: string;
  sub?: string;
  detail: string;
};

// Per-kind colours, reusing the course palette (primary / emerald / sky / amber
// / violet / teal). Kept as Tailwind class triplets so the boxes match siblings.
const KIND_STYLE: Record<Kind, string> = {
  io: 'border-border bg-muted/30 text-foreground/85',
  proj: 'border-primary/40 bg-primary/10 text-primary',
  norm: 'border-indigo-500/40 bg-indigo-500/10 text-indigo-700 dark:text-indigo-300',
  rope: 'border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300',
  score: 'border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300',
  mask: 'border-muted-foreground/40 bg-muted/40 text-foreground/80',
  softmax: 'border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300',
  mix: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
  gate: 'border-fuchsia-500/40 bg-fuchsia-500/10 text-fuchsia-700 dark:text-fuchsia-300',
  out: 'border-violet-500/40 bg-violet-500/10 text-violet-700 dark:text-violet-300',
  state: 'border-teal-500/40 bg-teal-500/10 text-teal-700 dark:text-teal-300',
};

// Per-locale copy: all user-visible strings (stage labels/subs/details, header,
// toggle labels, aria labels, framing notes, footnote) live here so the /zh
// pages localize while the English output stays byte-identical.
const COPY = {
  en: {
    header: 'Attention, end to end',
    mixerAria: 'Sequence mixer',
    fullOption: 'Full attention (GQA)',
    linearOption: 'Linear (GatedDeltaNet)',
    fullFlowAria: 'Full grouped-query attention data flow, hidden input to output',
    linearFlowAria: 'Linear GatedDeltaNet data flow, hidden input to output',
    fullStages: [
      {
        id: 'hidden',
        kind: 'io',
        label: 'hidden h',
        sub: `${HIDDEN_DIM}-dim`,
        detail: `The normalized residual vector for the current token enters the attention block — ${HIDDEN_DIM} dims wide.`,
      },
      {
        id: 'qkv',
        kind: 'proj',
        label: 'project Q, K, V',
        sub: `W_Q · ${NUM_HEADS} (+gate) / W_K,W_V · ${NUM_KV_HEADS}`,
        detail: `Learned projections. Q gets ${NUM_HEADS} heads (each head_dim ${HEAD_DIM}); K and V get only ${NUM_KV_HEADS} heads, shared across the ${GROUP_SIZE} query heads in each group — that's GQA. Qwen3.5's q_proj is 2× wide: it emits the queries AND a per-head gate that is used at the very end.`,
      },
      {
        id: 'qknorm',
        kind: 'norm',
        label: 'RMSNorm Q, K',
        sub: 'q_norm / k_norm',
        detail:
          'Qwen3.5 applies a per-head RMSNorm to the queries and keys (q_norm / k_norm) before RoPE, stabilizing the scale of the dot products. V is left un-normalized.',
      },
      {
        id: 'rope',
        kind: 'rope',
        label: 'RoPE on Q, K',
        sub: 'rotate by position',
        detail:
          'Rotary position encoding rotates the Q and K vectors by an angle that depends on token position, so the dot product below becomes position-aware. V is left untouched.',
      },
      {
        id: 'scores',
        kind: 'score',
        label: 'scores = QKᵀ / √d',
        sub: `√${HEAD_DIM} = 16`,
        detail: `Every query dot-products against every key, then divides by √d_head = √${HEAD_DIM} = 16 to keep the softmax in an informative range. This [seq × seq] matrix is quadratic in sequence length.`,
      },
      {
        id: 'mask',
        kind: 'mask',
        label: 'causal mask',
        sub: '+ -∞ above diagonal',
        detail:
          'The upper triangle is set to -∞ so token i can only attend to keys 0..i — no peeking at the future. After softmax those cells become exactly 0.',
      },
      {
        id: 'softmax',
        kind: 'softmax',
        label: 'softmax (per row)',
        sub: 'rows sum to 1',
        detail:
          'Each row of the masked score matrix becomes a probability distribution over earlier positions — the attention pattern for that query token.',
      },
      {
        id: 'mix',
        kind: 'mix',
        label: 'weighted sum · V',
        sub: 'attention · V',
        detail:
          'Multiply the attention pattern by the value vectors: each output position is a weighted blend of the values it attended to.',
      },
      {
        id: 'concat',
        kind: 'mix',
        label: 'concat heads',
        sub: `${NUM_HEADS} × ${HEAD_DIM} → ${NUM_HEADS * HEAD_DIM}`,
        detail: `The ${NUM_HEADS} heads each produced a head_dim-${HEAD_DIM} vector; concatenate them back into one ${NUM_HEADS * HEAD_DIM}-wide vector.`,
      },
      {
        id: 'gate',
        kind: 'gate',
        label: '× sigmoid(gate)',
        sub: 'output gating',
        detail:
          'The per-head gate emitted back at q_proj passes through a sigmoid and multiplies the attention output element-wise — an output gate that lets the model damp or pass each head before the final projection.',
      },
      {
        id: 'oproj',
        kind: 'out',
        label: 'output proj W_O',
        sub: `→ ${HIDDEN_DIM}-dim`,
        detail: `A final linear projection mixes the gated, concatenated heads and returns to the ${HIDDEN_DIM}-dim residual stream. Equivalently: each head's ${HEAD_DIM}-wide slice of W_O up-projects that head's output to ${HIDDEN_DIM} dims, and the ${NUM_HEADS} results are summed.`,
      },
      {
        id: 'out',
        kind: 'io',
        label: 'output',
        sub: `${HIDDEN_DIM}-dim`,
        detail: 'The attention output, added back onto the residual stream for the next sub-block.',
      },
    ] as Stage[],
    linearStages: [
      {
        id: 'hidden',
        kind: 'io',
        label: 'hidden h',
        sub: `${HIDDEN_DIM}-dim`,
        detail: `The same ${HIDDEN_DIM}-dim normalized residual enters — but this layer mixes the sequence recurrently instead of with attention.`,
      },
      {
        id: 'proj',
        kind: 'proj',
        label: 'project q, k, v, gates',
        sub: 'q/k/v · β, decay',
        detail:
          'Linear projections split the input into query/key/value-like streams plus the β (write strength) and decay (forget) gates that drive the recurrence.',
      },
      {
        id: 'state',
        kind: 'state',
        label: 'fixed-size recurrent state',
        sub: 'updated per token · no QKᵀ',
        detail:
          'The heart of Gated DeltaNet: a fixed-size state is updated one token at a time — β controls how much new information is written, decay controls forgetting. There is no QKᵀ matrix and no growing KV cache; memory is constant at token 10 and token 100,000.',
      },
      {
        id: 'out',
        kind: 'out',
        label: 'gated norm + out proj',
        sub: `→ ${HIDDEN_DIM}-dim`,
        detail: `A gated RMSNorm and an output projection return the block result to the ${HIDDEN_DIM}-dim residual stream.`,
      },
    ] as Stage[],
    fullNote: (
      <>
        The <span className="font-mono">QKᵀ</span> softmax is <strong>quadratic in sequence length</strong>, and the KV
        cache grows with every token ({NUM_HEADS} query / {NUM_KV_HEADS} KV heads, head_dim {HEAD_DIM}). Only {NUM_FULL}{' '}
        of Qwen3.5-0.8B&apos;s {NUM_LAYERS} layers use this path.
      </>
    ),
    linearNote: (
      <>
        No <span className="font-mono">QKᵀ</span> and no growing KV cache — history is folded into a{' '}
        <strong>fixed-size recurrent state</strong>, so memory is constant as the sequence grows. The other {NUM_LINEAR}{' '}
        of Qwen3.5-0.8B&apos;s {NUM_LAYERS} layers use this path; the KV-cache chapter covers the recurrence in detail.
      </>
    ),
    footnote: (
      <>
        Illustrative schematic — the full-attention path mirrors Qwen3.5&apos;s actual gated GQA (QK-norm, output gate);
        the linear path is a high-level sketch. No tensors flow here; not live output from the model.
      </>
    ),
  },
  zh: {
    header: '注意力，端到端',
    mixerAria: '序列混合器',
    fullOption: '全量注意力（GQA）',
    linearOption: '线性（GatedDeltaNet）',
    fullFlowAria: '全量分组查询注意力（GQA）的数据流，从 hidden 输入到输出',
    linearFlowAria: '线性 GatedDeltaNet 的数据流，从 hidden 输入到输出',
    fullStages: [
      {
        id: 'hidden',
        kind: 'io',
        label: 'hidden h',
        sub: `${HIDDEN_DIM} 维`,
        detail: `当前 token 归一化后的残差向量进入注意力模块——宽 ${HIDDEN_DIM} 维。`,
      },
      {
        id: 'qkv',
        kind: 'proj',
        label: '投影 Q、K、V',
        sub: `W_Q · ${NUM_HEADS} (+gate) / W_K,W_V · ${NUM_KV_HEADS}`,
        detail: `可学习的投影。Q 有 ${NUM_HEADS} 个头（每个头 head_dim ${HEAD_DIM}）；K 和 V 只有 ${NUM_KV_HEADS} 个头，被每组 ${GROUP_SIZE} 个 query 头共享——这就是 GQA。Qwen3.5 的 q_proj 有两倍宽：它在输出 query 的同时，还输出一个在最后一步才用到的逐头门控。`,
      },
      {
        id: 'qknorm',
        kind: 'norm',
        label: 'RMSNorm Q、K',
        sub: 'q_norm / k_norm',
        detail:
          'Qwen3.5 在 RoPE 之前对 query 和 key 施加逐头的 RMSNorm（q_norm / k_norm），稳定点积的尺度。V 不做归一化。',
      },
      {
        id: 'rope',
        kind: 'rope',
        label: '对 Q、K 施加 RoPE',
        sub: '按位置旋转',
        detail: '旋转位置编码（RoPE）按一个依赖 token 位置的角度旋转 Q 和 K 向量，让下面的点积带上位置信息。V 保持不动。',
      },
      {
        id: 'scores',
        kind: 'score',
        label: 'scores = QKᵀ / √d',
        sub: `√${HEAD_DIM} = 16`,
        detail: `每个 query 与每个 key 做点积，再除以 √d_head = √${HEAD_DIM} = 16，让 softmax 落在信息充分的区间。这个 [seq × seq] 矩阵与序列长度成平方关系。`,
      },
      {
        id: 'mask',
        kind: 'mask',
        label: '因果掩码',
        sub: '对角线上方置 -∞',
        detail: '上三角被置为 -∞，使 token i 只能关注 key 0..i——不许偷看未来。softmax 之后这些格子恰好变成 0。',
      },
      {
        id: 'softmax',
        kind: 'softmax',
        label: 'softmax（逐行）',
        sub: '每行和为 1',
        detail: '掩码后分数矩阵的每一行变成对更早位置的概率分布——也就是该 query token 的注意力模式。',
      },
      {
        id: 'mix',
        kind: 'mix',
        label: '加权求和 · V',
        sub: '注意力 · V',
        detail: '把注意力模式乘上 value 向量：每个输出位置都是它所关注的 value 的加权混合。',
      },
      {
        id: 'concat',
        kind: 'mix',
        label: '拼接各头',
        sub: `${NUM_HEADS} × ${HEAD_DIM} → ${NUM_HEADS * HEAD_DIM}`,
        detail: `${NUM_HEADS} 个头各产出一个 head_dim 为 ${HEAD_DIM} 的向量；把它们拼接回一个 ${NUM_HEADS * HEAD_DIM} 宽的向量。`,
      },
      {
        id: 'gate',
        kind: 'gate',
        label: '× sigmoid(gate)',
        sub: '输出门控',
        detail:
          '早在 q_proj 就输出的逐头门控经过 sigmoid 后逐元素乘到注意力输出上——这道输出门让模型在最终投影之前压低或放行每个头。',
      },
      {
        id: 'oproj',
        kind: 'out',
        label: '输出投影 W_O',
        sub: `→ ${HIDDEN_DIM} 维`,
        detail: `最后一次线性投影混合门控后拼接的各头，回到 ${HIDDEN_DIM} 维的残差流。等价地说：每个头在 W_O 里那条 ${HEAD_DIM} 宽的切片把该头的输出升回 ${HIDDEN_DIM} 维，再把 ${NUM_HEADS} 个结果相加。`,
      },
      {
        id: 'out',
        kind: 'io',
        label: '输出',
        sub: `${HIDDEN_DIM} 维`,
        detail: '注意力的输出，加回残差流，交给下一个子模块。',
      },
    ] as Stage[],
    linearStages: [
      {
        id: 'hidden',
        kind: 'io',
        label: 'hidden h',
        sub: `${HIDDEN_DIM} 维`,
        detail: `同一个 ${HIDDEN_DIM} 维的归一化残差进入——但这一层用循环而不是注意力来混合序列。`,
      },
      {
        id: 'proj',
        kind: 'proj',
        label: '投影 q、k、v 与门控',
        sub: 'q/k/v · β, decay',
        detail: '线性投影把输入拆成类 query/key/value 的几路流，外加驱动循环的 β（写入强度）和 decay（遗忘）门控。',
      },
      {
        id: 'state',
        kind: 'state',
        label: '固定大小的循环状态',
        sub: '逐 token 更新 · 无 QKᵀ',
        detail:
          'Gated DeltaNet 的核心：一个固定大小的状态逐 token 更新——β 控制写入多少新信息，decay 控制遗忘。没有 QKᵀ 矩阵，也没有增长的 KV 缓存；第 10 个 token 和第 100,000 个 token 占用的内存相同。',
      },
      {
        id: 'out',
        kind: 'out',
        label: '门控归一化 + 输出投影',
        sub: `→ ${HIDDEN_DIM} 维`,
        detail: `一次门控 RMSNorm 加一次输出投影，把模块结果送回 ${HIDDEN_DIM} 维的残差流。`,
      },
    ] as Stage[],
    fullNote: (
      <>
        <span className="font-mono">QKᵀ</span> softmax 与序列长度成<strong>平方关系</strong>，KV 缓存随每个 token 增长（
        {NUM_HEADS} 个 query 头 / {NUM_KV_HEADS} 个 KV 头，head_dim {HEAD_DIM}）。Qwen3.5-0.8B 的 {NUM_LAYERS} 层中只有{' '}
        {NUM_FULL} 层走这条通路。
      </>
    ),
    linearNote: (
      <>
        没有 <span className="font-mono">QKᵀ</span>，也没有增长的 KV 缓存——历史被折叠进一个
        <strong>固定大小的循环状态</strong>，内存不随序列增长。Qwen3.5-0.8B 的 {NUM_LAYERS} 层中其余 {NUM_LINEAR}{' '}
        层走这条通路；KV 缓存一章会详细讲这个循环。
      </>
    ),
    footnote: (
      <>
        示意图——全量注意力通路对应 Qwen3.5 真实的门控
        GQA（QK-norm、输出门）；线性通路只是高层速写。这里没有张量流动；不是模型的实时输出。
      </>
    ),
  },
} as const;

function StageBox({ stage, active, onActivate }: { stage: Stage; active: boolean; onActivate: () => void }) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onMouseEnter={onActivate}
      onFocus={onActivate}
      onClick={onActivate}
      className={[
        'flex h-12 min-w-[7rem] flex-col items-start justify-center rounded border px-2.5 text-left text-[11px] leading-none transition-all',
        KIND_STYLE[stage.kind],
        active ? 'ring-2 ring-primary/60' : 'opacity-85 hover:opacity-100',
      ].join(' ')}
    >
      <span className="font-mono">{stage.label}</span>
      {stage.sub ? <span className="mt-0.5 text-[10px] opacity-80">{stage.sub}</span> : null}
    </button>
  );
}

export function AttentionPipeline() {
  const copy = COPY[useLocale()];
  const [mixer, setMixer] = React.useState<Mixer>('full');
  const stages = mixer === 'full' ? copy.fullStages : copy.linearStages;
  const [activeId, setActiveId] = React.useState<string>(stages[0]!.id);

  // When the mixer flips, the previously-active id may not exist in the new
  // stage list — fall back to the first stage of the new flow.
  const activeStage = stages.find((s) => s.id === activeId) ?? stages[0]!;

  const selectMixer = (m: Mixer) => {
    setMixer(m);
    const next = m === 'full' ? copy.fullStages : copy.linearStages;
    setActiveId(next[0]!.id);
  };

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <SegmentedToggle
          value={mixer}
          onChange={selectMixer}
          ariaLabel={copy.mixerAria}
          options={[
            { value: 'full', label: copy.fullOption },
            { value: 'linear', label: copy.linearOption },
          ]}
        />
      </div>

      <MathDisplay
        latex={String.raw`\text{Attention}(Q,K,V)=\text{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right)V`}
      />

      {/* The flow: labeled boxes joined by arrows. Wraps on narrow screens. */}
      <div
        className="flex flex-wrap items-center gap-1.5"
        role="group"
        aria-label={mixer === 'full' ? copy.fullFlowAria : copy.linearFlowAria}
      >
        {stages.map((stage, i) => (
          <React.Fragment key={stage.id}>
            <StageBox stage={stage} active={stage.id === activeStage.id} onActivate={() => setActiveId(stage.id)} />
            {i < stages.length - 1 ? <span className="text-muted-foreground/50">→</span> : null}
          </React.Fragment>
        ))}
      </div>

      {/* Detail for the active stage. */}
      <div className="rounded-md border border-border/60 bg-muted/20 p-3" aria-live="polite">
        <div className="text-[12px] font-semibold text-foreground">{activeStage.label}</div>
        <p className="mt-1 text-[12px] leading-relaxed text-foreground/85">{activeStage.detail}</p>
      </div>

      {/* Mixer-specific framing note. */}
      <p className="text-[12px] text-foreground/85">{mixer === 'full' ? copy.fullNote : copy.linearNote}</p>

      <p className="text-[10px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
