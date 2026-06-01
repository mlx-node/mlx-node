import * as React from 'react';

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

const FULL_STAGES: Stage[] = [
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
    detail: `A final linear projection mixes the gated, concatenated heads and returns to the ${HIDDEN_DIM}-dim residual stream.`,
  },
  {
    id: 'out',
    kind: 'io',
    label: 'output',
    sub: `${HIDDEN_DIM}-dim`,
    detail: 'The attention output, added back onto the residual stream for the next sub-block.',
  },
];

const LINEAR_STAGES: Stage[] = [
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
];

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
      {stage.sub ? <span className="mt-0.5 text-[9px] opacity-70">{stage.sub}</span> : null}
    </button>
  );
}

export function AttentionPipeline() {
  const [mixer, setMixer] = React.useState<Mixer>('full');
  const stages = mixer === 'full' ? FULL_STAGES : LINEAR_STAGES;
  const [activeId, setActiveId] = React.useState<string>(stages[0]!.id);

  // When the mixer flips, the previously-active id may not exist in the new
  // stage list — fall back to the first stage of the new flow.
  const activeStage = stages.find((s) => s.id === activeId) ?? stages[0]!;

  const selectMixer = (m: Mixer) => {
    setMixer(m);
    const next = m === 'full' ? FULL_STAGES : LINEAR_STAGES;
    setActiveId(next[0]!.id);
  };

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">Attention, end to end</div>
        <SegmentedToggle
          value={mixer}
          onChange={selectMixer}
          ariaLabel="Sequence mixer"
          options={[
            { value: 'full', label: 'Full attention (GQA)' },
            { value: 'linear', label: 'Linear (GatedDeltaNet)' },
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
        aria-label={
          mixer === 'full'
            ? 'Full grouped-query attention data flow, hidden input to output'
            : 'Linear GatedDeltaNet data flow, hidden input to output'
        }
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
      {mixer === 'full' ? (
        <p className="text-[12px] text-foreground/85">
          The <span className="font-mono">QKᵀ</span> softmax is <strong>quadratic in sequence length</strong>, and the
          KV cache grows with every token ({NUM_HEADS} query / {NUM_KV_HEADS} KV heads, head_dim {HEAD_DIM}). Only{' '}
          {NUM_FULL} of Qwen3.5-0.8B&apos;s {NUM_LAYERS} layers use this path.
        </p>
      ) : (
        <p className="text-[12px] text-foreground/85">
          No <span className="font-mono">QKᵀ</span> and no growing KV cache — history is folded into a{' '}
          <strong>fixed-size recurrent state</strong>, so memory is constant as the sequence grows. The other{' '}
          {NUM_LINEAR} of Qwen3.5-0.8B&apos;s {NUM_LAYERS} layers use this path; the KV-cache chapter covers the
          recurrence in detail.
        </p>
      )}

      <p className="text-[10px] text-muted-foreground">
        Illustrative schematic — the full-attention path mirrors Qwen3.5&apos;s actual gated GQA (QK-norm, output gate);
        the linear path is a high-level sketch. No tensors flow here; not live output from the model.
      </p>
    </div>
  );
}
