import { Link } from '@tanstack/react-router';
import * as React from 'react';

import { Prose } from '../Prose';
import { ChapterFrame } from '../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../scaffolding/learning-data';
import { ContextWindow } from '../widgets/ContextWindow';
import { HallucinationDemo } from '../widgets/HallucinationDemo';

/**
 * Chapter 16 — The whole model, end to end (and what it isn't).
 *
 * Two arcs in one capstone. First, a static, text-only, interactive architecture
 * POSTER of Qwen3.5-0.8B. No worker, no WASM, no inference — everything rendered
 * here is inline SVG + React state. Then a closing reflection on the honest
 * limits that fall out of next-token prediction — hallucination, no lookup
 * database, a finite context window, stochastic output — merged in from the
 * former standalone "What the model isn't" chapter (its HallucinationDemo and
 * ContextWindow widgets render inline in that reflection section).
 *
 * The poster is a faithful dark-theme reproduction of the project's
 * landscape architecture artifact (artifacts/qwen35-0.8b-architecture.svg):
 * a central decoder spine, an expanded Dense SwiGLU FFN panel on the left, the
 * Gated DeltaNet (linear) and Gated GQA (full) sequence-mixer panels on the
 * right, and an optional vision path. The poster lives inline in the article
 * body (this chapter has no "Try it now" column); hovering / clicking / tabbing
 * any block updates the detail panel below it. Selecting a block never
 * navigates — a small secondary "Related" link inside each detail entry is the
 * only navigation, and it is never the primary action.
 *
 * Every numeric fact below matches the Qwen3.5-0.8B checkpoint the browser
 * loads (qwen3.5-0.8b-mlx-bf16): vocab_size 248,320, SwiGLU intermediate
 * 3,584, 24 layers (18 linear + 6 full), hidden 1024, RoPE base 1e7. Sourced
 * from the model config (and the architecture poster). Note 248,320 — not the
 * 151,936 of older Qwen text models: the tokenizer's own image_token_id
 * (248,056) already exceeds 151,936, so the larger vocab is provably the real
 * one.
 */

// --------------------------------------------------------------------------
// Verified model facts (single source of truth: the sibling chapters).
// --------------------------------------------------------------------------
const HIDDEN_DIM = 1024; // Qwen3.5-0.8B config: hidden_size
const VOCAB = 248_320; // Qwen3.5-0.8B config: vocab_size (image_token_id 248056 > 151936)
const INTERMEDIATE = 3584; // Qwen3.5-0.8B config: intermediate_size (the 0.6B model is 3072)
const NUM_LAYERS = 24;
const NUM_FULL = 6; // every 4th layer
const NUM_LINEAR = NUM_LAYERS - NUM_FULL; // 18
const MAX_POSITION = 262_144; // 05-rope, 10-kv-cache
const NUM_HEADS = 8; // 04-multihead-gqa
const NUM_KV_HEADS = 2; // 04-multihead-gqa
const HEAD_DIM = 256; // 04-multihead-gqa, 05-rope
const ROPE_DIMS = 64; // 05-rope (head_dim * 0.25)

const VOCAB_LABEL = VOCAB.toLocaleString();

export const learning: ChapterLearningData = {
  chapterId: 'architecture',
  objective:
    "Trace a token's path through all of Qwen3.5-0.8B in one diagram and explain how the hybrid linear/full-attention stack fits together — then name the core limits that fall straight out of next-token prediction: hallucination, no lookup database, a finite context, and stochastic output.",
  problem:
    "Every prior chapter zooms into a single component — it's easy to lose the forest for the trees. This chapter is the map that puts every piece back in its place, then steps back to the honest limits of the whole thing.",
  minutes: 12,
  glossary: [
    {
      term: 'residual stream',
      definition:
        'The un-normalized hidden vector that runs the full depth of the model. Every block reads it, computes a small contribution, and adds the result back.',
    },
    {
      term: 'pre-norm block',
      definition:
        'A sub-block laid out as x + sublayer(rmsnorm(x)): normalize the input, transform it, then add back onto the un-normalized residual highway. Keeps deep stacks trainable.',
    },
    {
      term: 'hybrid attention',
      definition:
        'Mixing two kinds of sequence mixer in one stack. Qwen3.5 alternates linear-attention layers with full softmax-attention layers on a fixed schedule.',
    },
    {
      term: 'Gated DeltaNet (linear attention)',
      definition:
        'A recurrent O(n) sequence mixer that compresses history into a fixed-size state instead of a growing KV cache. Used in 18 of the 24 layers.',
    },
    {
      term: 'grouped-query attention',
      definition:
        'Full softmax attention where several query heads share one K/V head. Qwen3.5 runs 8 query heads over 2 KV heads, shrinking the cache 4×.',
    },
    {
      term: 'weight tying',
      definition:
        'Reusing the embedding matrix as the LM head, so the same table maps token ids to vectors at the bottom and scores vectors into logits at the top.',
    },
    {
      term: 'hallucination',
      definition:
        "A fluent, confident output that isn't true — the result of sampling the most plausible token when no true one is available.",
    },
    {
      term: 'grounding',
      definition:
        'Tying a model output to a verifiable source (e.g. retrieved documents); a base model has none by default.',
    },
    {
      term: 'context window',
      definition: `The maximum number of tokens the model can attend to at once (${MAX_POSITION.toLocaleString()} for Qwen3.5-0.8B).`,
    },
    {
      term: 'parametric memory',
      definition: "Knowledge stored implicitly in the model's weights during training, not in any queryable database.",
    },
    {
      term: 'knowledge cutoff',
      definition: 'The point in time after which the model knows nothing, because its training data ends there.',
    },
    {
      term: 'stochastic',
      definition: 'Random by default — sampling means the same prompt can produce different outputs run to run.',
    },
  ],
  takeaways: [
    'Qwen3.5-0.8B is 24 pre-norm residual blocks; the only thing that changes layer to layer is the sequence mixer — linear (Gated DeltaNet) on 18 layers, full attention on 6.',
    'Full attention is global but its KV cache grows with sequence length; linear attention is cheap and recurrent with a fixed-size state. The 3:1 schedule keeps most of the quality at a fraction of the memory.',
    'The embedding matrix and the LM head are the same weights (tying) — one table both maps tokens in and scores them out. Every layer also carries a dense SwiGLU FFN — there is no mixture-of-experts anywhere.',
    "Hallucination isn't a malfunction: the model samples the most plausible token, and plausible only equals true when the fact was well-represented in training.",
    'The model has no database and no memory between calls — its knowledge lives in fixed weights (with a cutoff), and its working memory is only the current context window.',
    'Sampling makes generation stochastic by default; identical prompts can diverge unless you decode greedily.',
  ],
  exercise: {
    prompt:
      "Two things to try on this page. (1) In the poster, flip the GDN ↔ GQA toggle and read the two right-hand panels — name two things the full-attention layer (GQA) has that the linear layer (Gated DeltaNet) doesn't. (2) In the hallucination widget below, switch to 'No real answer' and compare the top bar's height to the 'Well-known fact' case — does the flatter distribution stop the model from answering?",
    answer:
      "(1) Full attention (GQA) has a KV cache that grows with sequence length and uses partial RoPE + per-head Q/K norm with an all-positions softmax; the linear layer (Gated DeltaNet) instead keeps a fixed-size recurrent state plus a small conv cache and never computes an all-pairs softmax. (2) On the fact the top bar is ~0.93 (very sure); on the made-up paper it's ~0.16, with probability spread across many tokens — the model is less certain but still commits to the single most plausible token and produces a confident, fabricated answer. Low confidence doesn't become 'I don't know' unless the model was trained to say so.",
  },
  quiz: [
    {
      id: 'how-many-full',
      prompt: 'How many of the 24 layers use full attention?',
      options: [
        { id: 'a', label: 'All 24' },
        { id: 'b', label: '6 — every 4th layer' },
        { id: 'c', label: 'None' },
      ],
      correctId: 'b',
      explanation: 'The schedule is 3 linear + 1 full, repeated 6 times — so 6 of the 24 layers are full attention.',
    },
    {
      id: 'weight-tying',
      prompt: 'What does weight tying mean here?',
      options: [
        { id: 'a', label: 'The embedding matrix and the LM head are the same weights' },
        { id: 'b', label: 'Every layer shares the same weights' },
        { id: 'c', label: 'Q and K projections share weights' },
      ],
      correctId: 'a',
      explanation:
        'One table is used twice: token-id → vector at the input, and vector → logits (its transpose) at the output.',
    },
    {
      id: 'other-18',
      prompt: "What's the sequence mixer in the other 18 layers?",
      options: [
        { id: 'a', label: 'Full softmax attention' },
        { id: 'b', label: 'Gated DeltaNet linear attention' },
        { id: 'c', label: 'Just a convolution' },
      ],
      correctId: 'b',
      explanation:
        'The 18 non-full layers are Gated DeltaNet — recurrent linear attention with a fixed-size state, not plain convolution.',
    },
    {
      id: 'q1-why-hallucinate',
      prompt: 'Why does an LLM hallucinate?',
      options: [
        { id: 'a', label: 'It is deliberately lying to the user.' },
        {
          id: 'b',
          label:
            'It samples the most plausible next token, and nothing in the forward pass checks that token against a source of truth.',
        },
        { id: 'c', label: 'Its training data was entirely false.' },
      ],
      correctId: 'b',
      explanation:
        'The model emits a likely-looking token; when no true answer is available it still produces a fluent, confident one. Plausibility is not truth.',
    },
    {
      id: 'q2-knowledge',
      prompt: "Where does an LLM's factual knowledge live at inference time?",
      options: [
        { id: 'a', label: 'In a database it queries for each token.' },
        { id: 'b', label: 'In its fixed weights (parametric memory), set during training — with a knowledge cutoff.' },
        { id: 'c', label: "In the user's previous conversations." },
      ],
      correctId: 'b',
      explanation:
        'A base model looks nothing up; its knowledge is baked into the weights and frozen after training, which is why it has a cutoff.',
    },
    {
      id: 'q3-stochastic',
      prompt: 'Why can the same prompt give two different answers on two runs?',
      options: [
        { id: 'a', label: 'The model rewrites its own weights between runs.' },
        {
          id: 'b',
          label:
            "Generation samples from a probability distribution, so it's stochastic unless you decode greedily (temperature 0).",
        },
        { id: 'c', label: 'The context window changed size.' },
      ],
      correctId: 'b',
      explanation:
        'Sampling introduces randomness by design; greedy decoding (always take the top token) is the deterministic special case.',
    },
  ],
};

// ==========================================================================
// Chapter body — prose + the inline interactive poster.
// ==========================================================================

export function ArchitectureChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <div className="space-y-8">
        <Prose className="lg:max-w-none">
          <h1>The whole model on one page</h1>
          <div className="grid grid-cols-1 gap-x-14 lg:grid-cols-2">
            <p>
              <strong>One sentence:</strong> Qwen3.5-0.8B is a stack of {NUM_LAYERS} near-identical pre-norm residual
              blocks bookended by an embedding table at the bottom and a tied LM head at the top — and almost everything
              you read in the earlier chapters lives inside one of those blocks. The poster below is the map: it lays
              the whole text decoder out as a central spine, with the dense SwiGLU feed-forward network expanded on the
              left and the two sequence mixers — linear (Gated DeltaNet) and full (Gated GQA) — expanded on the right.
              Hover, click, or tab any block to read what it does in the panel underneath.
            </p>
            <p>
              One honest caveat on “whole model”: the spine traces the <strong>text decoder</strong> — the path the
              in-browser demo actually runs. The full Qwen3.5-0.8B checkpoint is multimodal, so the poster also sketches
              the optional vision encoder (greyed at the lower left); it is never exercised in this text-only demo.
            </p>
          </div>
        </Prose>

        <ArchitecturePoster />

        <Prose className="lg:max-w-none">
          <div className="grid grid-cols-1 gap-x-14 lg:grid-cols-2">
            <div>
              <h2>Same block, 24 times</h2>
              <p>
                A token enters as an integer id, gets looked up in the embedding matrix, and becomes a {HIDDEN_DIM}-dim
                vector. That vector is the start of the <strong>residual stream</strong> — the un-normalized highway
                that runs the full depth of the model. Each of the {NUM_LAYERS} layers does the same two things to it: a
                sequence-mixing sub-block (attention of some kind) and a per-token feed-forward sub-block, each wrapped
                in the now-standard pre-norm + residual pattern.
              </p>
              <p>
                Crucially, the feed-forward half never changes: every layer carries a dense <code>SwiGLU</code> MLP (
                {HIDDEN_DIM} → {INTERMEDIATE} → {HIDDEN_DIM}), with no mixture-of-experts routing anywhere in this
                checkpoint. The norm is RMSNorm, applied to the sub-block input only, so the residual highway itself
                stays un-normalized and gradients reach all the way down. After the last layer, a final RMSNorm and the
                LM head turn the hidden state into one logit per vocabulary token.
              </p>
            </div>
            <div>
              <h2>Two kinds of sequence mixer</h2>
              <p>
                The one thing that <em>does</em> vary layer to layer is the sequence mixer. Qwen3.5 is a{' '}
                <strong>hybrid</strong>: it runs a fixed <code>[linear, linear, linear, full]</code> schedule and
                repeats it six times, so {NUM_LINEAR} of the {NUM_LAYERS} layers use linear attention (Gated DeltaNet)
                and the remaining {NUM_FULL} use full grouped-query attention. The full-attention layers are global —
                every position can look back at every other — but their KV cache grows with the sequence. The linear
                layers are recurrent and cheap: they fold history into a fixed-size state, so they cost the same memory
                at token 10 and token 100,000.
              </p>
              <p>
                The 3:1 ratio is the bargain. Full attention is the only mixer that can reliably reach back and recall
                one specific earlier token, so the model keeps a sprinkling of it; the linear layers carry the bulk of
                the language flow at a fraction of the cache. The result has most of the quality of an all-attention
                model with a memory footprint that barely moves as context grows.
              </p>
            </div>
          </div>
        </Prose>

        {/* Optional deep-dive: what the GDN sub-boxes in the poster actually
            mean, in words. Collapsed by default and skippable — beginners can
            ignore it; the curious get the linear-attention mechanics. Centered
            at article width (like the limitations arc below) so it doesn't
            strand in the left third of the wide body. Native <details> so it
            stays keyboard-focusable; styled like the project's Glossary
            disclosure. */}
        <div className="mx-auto w-full max-w-3xl">
          <details className="group not-prose rounded-md border border-border bg-muted/30 px-4 py-3 text-sm">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-2 text-foreground/90 [&::-webkit-details-marker]:hidden">
              <span className="font-medium">
                Advanced: inside a Gated DeltaNet layer{' '}
                <span className="text-muted-foreground">· optional, for the curious</span>
              </span>
              <span
                aria-hidden="true"
                className="text-xs text-muted-foreground transition-transform group-open:rotate-90"
              >
                ▸
              </span>
            </summary>
            <div className="mt-3 space-y-2 text-[13px] leading-relaxed text-foreground/80">
              <p>
                This is what the teal Gated DeltaNet boxes in the poster above mean, unpacked. You can skip it — the 3:1
                bargain above is the whole story for using the model. But here is what actually happens inside one of the{' '}
                {NUM_LINEAR} linear layers:
              </p>
              <ul className="list-disc space-y-1.5 pl-5">
                <li>
                  <strong>Projections.</strong> Linear maps split the input into query/key, a value plus a{' '}
                  <em>z-gate</em>, and a β (beta) / g (decay) pair — the knobs the recurrence below turns.
                </li>
                <li>
                  <strong>A short conv.</strong> A causal <em>depthwise</em> convolution (kernel 4, so each channel only
                  looks at the last 4 tokens) followed by a SiLU activation adds a little local mixing before the
                  recurrence.
                </li>
                <li>
                  <strong>The gated delta-rule recurrence</strong> is the heart of it. It carries one fixed-size state
                  and walks the sequence token by token: β controls how much each new token <em>overwrites</em> that
                  state, and g (decay) controls how fast old information is <em>forgotten</em>. That makes it O(n) in
                  the sequence length — linear, not quadratic — with a state whose size never grows.
                </li>
                <li>
                  <strong>Memory.</strong> Instead of a KV cache that grows with every token, the layer carries only a
                  tiny conv cache (the last few tokens) plus that fixed-size recurrent state — so its memory is{' '}
                  <em>constant regardless of context length</em>.
                </li>
                <li>
                  <strong>Finish.</strong> A z-gated RMSNorm and an out-projection return the result to the{' '}
                  {HIDDEN_DIM}-dim residual stream.
                </li>
              </ul>
              <p>
                That constant-memory recurrence is exactly what lets {NUM_LINEAR} of the {NUM_LAYERS} layers run without
                a growing cache, leaving only the {NUM_FULL} full-attention layers to pay for one.
              </p>
            </div>
          </details>
        </div>

        {/* The merged limitations arc is a reading section, not part of the
            full-width poster — center it at article width inside the wide
            (~1400px) body so it doesn't strand in the left third. */}
        <div className="mx-auto w-full max-w-3xl">
          <Prose className="lg:max-w-none">
            <h2>What the model isn't</h2>
            <p>
              You've now seen the whole machine, end to end — every block, and where it sits. It's worth being equally
              precise about what it is <em>not</em> doing, because nearly every "the LLM is broken" surprise comes from
              expecting a capability the architecture never had. None of the limits below are bolted-on flaws — they
              fall straight out of next-token prediction.
            </p>

            <h2>It samples the plausible, not the true</h2>
            <p>
              The forward pass produces a probability for every next token, and we sample one. Nowhere in that pipeline
              is a fact checked against a source. When the true continuation was common in the training data, plausible
              and true coincide and the model looks like it "knows." When there's no true answer — or the model simply
              never saw it — it <em>still</em> emits a fluent, confident token. That's a <strong>hallucination</strong>:
            </p>
            <HallucinationDemo />
            <p>
              Notice it isn't lying (there's no intent) and it isn't malfunctioning — it's doing exactly what it was
              built to do: produce a likely-looking next token. The flatter the distribution, the less sure it is, but
              it commits to an answer regardless unless training taught it to hedge.
            </p>

            <h2>There is no lookup database</h2>
            <p>
              A base model's "knowledge" is <strong>parametric</strong>: baked into its weights during training, not
              stored in a table it queries at inference time. So it can't reliably cite a source it didn't effectively
              memorize, and it has a <strong>knowledge cutoff</strong> — anything that happened after training simply
              isn't in the weights. (Retrieval and tools can bolt a real database on top, but that's an addition; the
              model you've studied has none.)
            </p>

            <h2>Its memory is the context window — and it's finite</h2>
            <ContextWindow />
            <p>
              The model's only working memory is the tokens currently inside its{' '}
              <Link to="/chapters/$chapterId" params={{ chapterId: 'kv-cache' }} search={(prev) => prev}>
                context window
              </Link>
              . Past that hard limit the oldest tokens have to be dropped to make room, and once dropped they're gone.
              And between two separate conversations it remembers nothing at all — every call starts blank, apart from
              whatever text you re-supply in the prompt.
            </p>

            <h2>Same input, different output</h2>
            <p>
              Because we <em>sample</em>, generation is <strong>stochastic</strong> by default: with temperature above
              zero, the same prompt can produce different answers on different runs. That's a feature for creative
              writing and a footgun for anything you need to reproduce — decode greedily (temperature 0) when you want
              determinism, as the{' '}
              <Link to="/chapters/$chapterId" params={{ chapterId: 'sampling' }} search={(prev) => prev}>
                Sampling chapter
              </Link>{' '}
              showed.
            </p>

            <h2>None of this is a bug</h2>
            <p>
              These aren't defects waiting for a patch — they're the honest shape of a system that predicts one
              plausible token at a time. Knowing them is exactly what separates using an LLM well from being blindsided
              by it. That's the real payoff of having traced the whole forward pass: the model stops being magic, and
              starts being a tool whose strengths and edges you can actually reason about.
            </p>
          </Prose>
        </div>
      </div>
    </ChapterFrame>
  );
}

// ==========================================================================
// Detail data — one entry per selectable block. The `sequence` entry is
// special: it swaps with the mixer toggle (see sequenceDetailFor).
// ==========================================================================

type RelatedLink = { chapterId: string; label: string };
type Detail = { title: string; facts?: string; body: string; related?: RelatedLink[] };

const DETAILS: Record<string, Detail> = {
  // --- central spine ---
  output: {
    title: 'Output logits',
    facts: `${VOCAB_LABEL} scores · one per vocab token`,
    body: 'The very top of the stack: one real number per vocabulary token, before softmax. Argmax (greedy) or sampling over these picks the next token, which is appended and fed back in.',
    related: [{ chapterId: 'sampling', label: 'Sampling' }],
  },
  'lm-head': {
    title: 'Tied LM head',
    facts: `→ ${VOCAB_LABEL} logits · weights tied to embeddings`,
    body: `Projects the final ${HIDDEN_DIM}-dim hidden state to one logit per vocabulary token. The projection weights ARE the embedding matrix (weight tying): the same table that mapped ids to vectors at the bottom scores vectors into logits at the top.`,
    related: [{ chapterId: 'lm-head', label: 'LM head & weight tying' }],
  },
  'final-norm': {
    title: 'Final RMSNorm',
    body: 'One last normalization after all 24 layers, applied before the output projection so the LM head reads a well-scaled hidden state.',
    related: [{ chapterId: 'rmsnorm', label: 'RMSNorm' }],
  },
  decoder: {
    title: `Decoder stack ×${NUM_LAYERS}`,
    facts: `${NUM_LAYERS} layers · ${NUM_LINEAR} linear + ${NUM_FULL} full · hidden ${HIDDEN_DIM}`,
    body: `${NUM_LAYERS} pre-norm residual blocks. The sequence-mixing slot follows a fixed [GDN, GDN, GDN, full-attention] schedule repeated ${NUM_FULL} times — so full attention lands on layers 4, 8, 12, 16, 20, 24 and every other layer is linear. The feed-forward half is a dense SwiGLU MLP in every layer.`,
    related: [{ chapterId: 'kv-cache', label: 'KV cache & hybrid attention' }],
  },
  'pre-norm': {
    title: 'RMSNorm (pre-attention)',
    facts: `per-token L2 → ≈ √${HIDDEN_DIM} ≈ 32`,
    body: 'Normalize the residual stream before the mixer: divide by the root-mean-square, then scale by a learned per-feature gain. This keeps magnitudes stable as the residual accumulates across depth.',
    related: [{ chapterId: 'rmsnorm', label: 'RMSNorm' }],
  },
  'residual-1': {
    title: 'Add the mixer output back',
    body: 'The mixer output is added back into the residual stream. The un-normalized residual is the identity path that lets gradients reach all 24 layers — it is why the stack trains at all.',
    related: [{ chapterId: 'full-block', label: 'Full transformer block' }],
  },
  'post-norm': {
    title: 'RMSNorm (pre-MLP)',
    body: 'A second pre-norm, this time in front of the feed-forward sub-block. Same operation as the pre-attention norm, applied to the (now updated) residual stream.',
    related: [{ chapterId: 'rmsnorm', label: 'RMSNorm' }],
  },
  mlp: {
    title: 'Dense SwiGLU FFN',
    facts: `${HIDDEN_DIM} → ${INTERMEDIATE} → ${HIDDEN_DIM} · SwiGLU · no MoE`,
    body: `A per-token feed-forward network in every layer (dense — no mixture-of-experts). Gate and up projections widen ${HIDDEN_DIM} → ${INTERMEDIATE}, the two are combined as SiLU(gate) · up, then the down projection collapses ${INTERMEDIATE} → ${HIDDEN_DIM} back onto the residual stream. The panel on the left expands exactly this block.`,
    related: [{ chapterId: 'mlp', label: 'MLP block' }],
  },
  'residual-2': {
    title: 'Add the MLP output back',
    body: 'The MLP output is added back into the residual stream. That sum is the layer output — the input the next layer reads. Repeat 24 times.',
    related: [{ chapterId: 'full-block', label: 'Full transformer block' }],
  },
  embeddings: {
    title: 'Token + visual embeddings',
    facts: `embedding table ${VOCAB_LABEL} × ${HIDDEN_DIM}`,
    body: `Each token id is looked up in the embedding matrix and becomes a ${HIDDEN_DIM}-dim vector — the start of the residual stream. In the multimodal checkpoint, projected vision tokens are spliced into this same sequence; the browser demo only uses text.`,
    related: [{ chapterId: 'embeddings', label: 'Embeddings' }],
  },
  input: {
    title: 'Input sequence',
    facts: `context up to ${MAX_POSITION.toLocaleString()} tokens`,
    body: 'Your text becomes a list of integer token ids; the multimodal model can also interleave optional image and video tokens. Each id is just a row index into the embedding matrix — pure lookup keys, nothing is "understood" yet.',
    related: [{ chapterId: 'tokenization', label: 'Tokenization' }],
  },

  // --- left: Dense SwiGLU FFN expanded ---
  'ffn-input': {
    title: 'FFN input (hidden 1024)',
    body: `The normalized residual vector enters the per-token feed-forward network. Same ${HIDDEN_DIM}-dim hidden width everywhere in the stack.`,
    related: [{ chapterId: 'mlp', label: 'MLP block' }],
  },
  'ffn-gate': {
    title: 'Gate projection',
    facts: `Linear ${HIDDEN_DIM} → ${INTERMEDIATE}`,
    body: 'One of two parallel up-projections. Its output is squashed by SiLU to act as a multiplicative gate over the up projection.',
    related: [{ chapterId: 'mlp', label: 'MLP block' }],
  },
  'ffn-up': {
    title: 'Up projection',
    facts: `Linear ${HIDDEN_DIM} → ${INTERMEDIATE}`,
    body: 'The second parallel projection, left linear. It carries the values that the gate modulates.',
    related: [{ chapterId: 'mlp', label: 'MLP block' }],
  },
  'ffn-act': {
    title: 'SwiGLU activation',
    facts: 'SiLU(gate) · up',
    body: 'Elementwise: SiLU(gate) multiplies up. This gating is what makes SwiGLU more expressive than a plain MLP for the same parameter budget.',
    related: [{ chapterId: 'mlp', label: 'MLP block' }],
  },
  'ffn-down': {
    title: 'Down projection',
    facts: `Linear ${INTERMEDIATE} → ${HIDDEN_DIM}`,
    body: `Collapses the wide ${INTERMEDIATE}-dim intermediate back to ${HIDDEN_DIM} so the result can be added back onto the residual stream.`,
    related: [{ chapterId: 'mlp', label: 'MLP block' }],
  },

  // --- right top: Gated DeltaNet (linear attention) expanded ---
  'gdn-input': {
    title: 'GDN hidden input',
    body: 'The normalized residual enters the linear-attention block — used in 18 of the 24 layers.',
    related: [{ chapterId: 'kv-cache', label: 'KV cache & hybrid attention' }],
  },
  'gdn-proj': {
    title: 'Input projections + state',
    facts: 'q,k · v,z · b,a  +  conv & recurrent state',
    body: 'Linear projections split the input into query/key, value/z-gate, and beta/decay (b,a) streams, alongside the carried conv state and recurrent state from previous tokens.',
    related: [{ chapterId: 'kv-cache', label: 'KV cache & hybrid attention' }],
  },
  'gdn-conv': {
    title: 'Depthwise Conv1D',
    facts: 'kernel 4 · SiLU · q/k/v channels',
    body: 'A short causal depthwise convolution (kernel 4) over the q/k/v channels with a SiLU activation — gives the linear-attention layer a little local mixing before the recurrence.',
  },
  'gdn-recurrence': {
    title: 'Gated delta recurrence',
    facts: 'β, g update a fixed-size state',
    body: 'The heart of Gated DeltaNet: a delta-rule recurrence where β controls how much new information overwrites the state and g (decay) controls forgetting. O(n) over the sequence, with a fixed-size state.',
    related: [{ chapterId: 'kv-cache', label: 'KV cache & hybrid attention' }],
  },
  'gdn-cache': {
    title: 'Conv & recurrent caches',
    body: 'Instead of a KV cache that grows with the sequence, GDN carries a tiny conv cache (the last few tokens) and a fixed-size recurrent state — constant memory regardless of context length.',
    related: [{ chapterId: 'kv-cache', label: 'KV cache & hybrid attention' }],
  },
  'gdn-out': {
    title: 'RMSNormGated + out-proj',
    body: 'A z-gated RMSNorm followed by the output projection returns the block result to the 1024-dim residual stream.',
    related: [{ chapterId: 'rmsnorm', label: 'RMSNorm' }],
  },

  // --- right bottom: Gated GQA (full attention) expanded ---
  'gqa-input': {
    title: 'GQA hidden input',
    body: 'The normalized residual enters the full-attention block — used in 6 of the 24 layers (4, 8, 12, 16, 20, 24).',
    related: [{ chapterId: 'kv-cache', label: 'KV cache & hybrid attention' }],
  },
  'gqa-qproj': {
    title: 'Q projection (+ gate)',
    facts: `${NUM_HEADS} query heads · head-dim ${HEAD_DIM}`,
    body: `Projects to ${NUM_HEADS} query heads plus a sigmoid output-gate signal. Gating the attention output is the "gated" in gated GQA.`,
    related: [{ chapterId: 'multi-head-gqa', label: 'Multi-head & GQA' }],
  },
  'gqa-kvproj': {
    title: 'K / V projection',
    facts: `${NUM_KV_HEADS} KV heads (shared by ${NUM_HEADS} queries)`,
    body: `Only ${NUM_KV_HEADS} key/value heads, shared across the ${NUM_HEADS} query heads — that's grouped-query attention, shrinking the KV cache ${NUM_HEADS / NUM_KV_HEADS}×.`,
    related: [{ chapterId: 'multi-head-gqa', label: 'Multi-head & GQA' }],
  },
  'gqa-qnorm': {
    title: 'Q norm (per head)',
    facts: `${NUM_HEADS} heads`,
    body: 'Per-head RMSNorm on the queries before RoPE — stabilizes the attention logits.',
    related: [{ chapterId: 'rmsnorm', label: 'RMSNorm' }],
  },
  'gqa-knorm': {
    title: 'K norm (per head)',
    facts: `${NUM_KV_HEADS} heads`,
    body: 'Per-head RMSNorm on the keys, the K-side mirror of the query norm.',
    related: [{ chapterId: 'rmsnorm', label: 'RMSNorm' }],
  },
  'gqa-rope': {
    title: 'Partial RoPE',
    facts: `${ROPE_DIMS} of ${HEAD_DIM} dims · θ = 1e7`,
    body: `Rotary position encoding applied to only the first ${ROPE_DIMS} of each head's ${HEAD_DIM} dims; the rest stay position-free. The large base θ = 1e7 stretches the rotation across very long context.`,
    related: [{ chapterId: 'rope', label: 'Positional encoding (RoPE)' }],
  },
  'gqa-sdpa': {
    title: 'SDPA + output gate',
    facts: `scaled dot-product · o_proj → ${HIDDEN_DIM}`,
    body: `Scaled dot-product attention over all positions, multiplied by the sigmoid gate, then o_proj returns to the ${HIDDEN_DIM}-dim residual stream. Uses a KV cache that grows with sequence length.`,
    related: [{ chapterId: 'kv-cache', label: 'KV cache & hybrid attention' }],
  },

  // --- vision + text inputs ---
  'vision-input': {
    title: 'Image / video input',
    facts: 'patch 16 · tubelet 2',
    body: 'Optional. Images and video are cut into 16-px patches (temporal tubelet 2 for video) before the vision encoder. Not used in this text-only browser demo.',
  },
  'vision-encoder': {
    title: 'Vision encoder ×12',
    facts: 'hidden 768 · 12 heads',
    body: 'A 12-layer ViT-style encoder (hidden 768) turns image/video patches into visual tokens.',
  },
  'vision-project': {
    title: 'Vision → hidden 1024',
    body: `A projection maps the encoder's output into the ${HIDDEN_DIM}-dim text embedding space, so visual tokens sit in the same residual stream as text.`,
  },
  'text-input': {
    title: 'Text tokens',
    facts: `${VOCAB_LABEL} vocab`,
    body: 'Text becomes integer token ids and is embedded into the residual stream. The only input path the browser demo runs.',
    related: [{ chapterId: 'tokenization', label: 'Tokenization' }],
  },
};

/** Live detail for the central `sequence` block, depending on the mixer toggle. */
function sequenceDetailFor(mixer: Mixer): Detail {
  if (mixer === 'gdn') {
    return {
      title: 'Sequence block — Gated DeltaNet (linear)',
      facts: 'linear attention · conv kernel 4 · fixed recurrent state',
      body: `Recurrent O(n) linear attention, used in ${NUM_LINEAR} of the ${NUM_LAYERS} layers. Projections split the input into q/k/v/z and a separate b/a (beta, decay) pair; a depthwise Conv1D (kernel 4, SiLU) then a gated delta recurrence folds them into a fixed-size state, with b and a controlling keep-vs-overwrite. A z-gated RMSNorm and an out-projection finish the block — see the expanded panel on the right.`,
      related: [{ chapterId: 'kv-cache', label: 'KV cache & hybrid attention' }],
    };
  }
  return {
    title: 'Sequence block — Gated GQA (full)',
    facts: `${NUM_HEADS} Q / ${NUM_KV_HEADS} KV heads · head-dim ${HEAD_DIM} · RoPE ${ROPE_DIMS}/${HEAD_DIM}`,
    body: `Grouped-query full attention, used in ${NUM_FULL} of the ${NUM_LAYERS} layers (4, 8, 12, 16, 20, 24). ${NUM_HEADS} query heads share ${NUM_KV_HEADS} KV heads; per-head Q/K RMSNorm; partial RoPE rotates ${ROPE_DIMS} of ${HEAD_DIM} head dims (θ = 1e7); then scaled-dot-product attention, a sigmoid output gate, and an o-projection — see the expanded panel on the right.`,
    related: [
      { chapterId: 'multi-head-gqa', label: 'Multi-head & GQA' },
      { chapterId: 'rope', label: 'Positional encoding (RoPE)' },
    ],
  };
}

// ==========================================================================
// Interactive poster.
// ==========================================================================

type Mixer = 'gdn' | 'gqa';

export function ArchitecturePoster() {
  const [selectedId, setSelectedId] = React.useState<string>('decoder');
  const [mixer, setMixer] = React.useState<Mixer>('gdn');

  const detail = selectedId === 'sequence' ? sequenceDetailFor(mixer) : (DETAILS[selectedId] ?? DETAILS.decoder!);

  const selectMixer = React.useCallback((m: Mixer) => {
    setMixer(m);
    setSelectedId('sequence');
  }, []);

  return (
    <div className="space-y-4">
      <div className="rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
        A static map of the Qwen3.5-0.8B architecture — no model inference. Hover, click, or tab any block to read what
        it does in the panel below. The toggle highlights which sequence mixer the representative layer is using.
      </div>

      <MixerToggle mixer={mixer} onChange={selectMixer} />

      <div className="overflow-x-auto rounded-md border border-border bg-background p-3">
        <PosterSvg selectedId={selectedId} onSelect={setSelectedId} mixer={mixer} onMixerSelect={selectMixer} />
      </div>

      <DetailPanel detail={detail} />
    </div>
  );
}

// --------------------------------------------------------------------------
// Mixer toggle — real HTML buttons (keyboard-focusable), not SVG controls.
// --------------------------------------------------------------------------
function MixerToggle({ mixer, onChange }: { mixer: Mixer; onChange: (m: Mixer) => void }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs uppercase tracking-wider text-muted-foreground">Representative layer mixer</span>
      <div
        className="inline-flex overflow-hidden rounded-md border border-border"
        role="group"
        aria-label="Sequence mixer"
      >
        <button
          type="button"
          aria-pressed={mixer === 'gdn'}
          onClick={() => onChange('gdn')}
          className={[
            'px-3 py-1 text-xs font-medium transition-colors',
            mixer === 'gdn'
              ? 'bg-teal-400/15 text-teal-200'
              : 'bg-transparent text-muted-foreground hover:text-foreground',
          ].join(' ')}
        >
          Linear (GDN)
        </button>
        <button
          type="button"
          aria-pressed={mixer === 'gqa'}
          onClick={() => onChange('gqa')}
          className={[
            'border-l border-border px-3 py-1 text-xs font-medium transition-colors',
            mixer === 'gqa'
              ? 'bg-violet-400/15 text-violet-200'
              : 'bg-transparent text-muted-foreground hover:text-foreground',
          ].join(' ')}
        >
          Full (GQA)
        </button>
      </div>
      <span className="text-[11px] text-muted-foreground">
        {NUM_LINEAR} layers linear, {NUM_FULL} full — flip to highlight the matching right-hand panel.
      </span>
    </div>
  );
}

// ==========================================================================
// SVG poster. ViewBox ported 1:1 from the landscape artifact (1600×1180) and
// recoloured for the dark theme. Accent colours are inline oklch (independent
// of the Tailwind palette); neutrals use currentColor.
// ==========================================================================

const VB_W = 1600;
const VB_H = 1180;

type Variant = 'norm' | 'seq' | 'state' | 'ffn' | 'rope' | 'io' | 'header';

// Dark-theme accent colours (oklch). Mapping mirrors the artifact's legend:
// normalization=blue, sequence=purple, linear/state=teal, FFN=pink, RoPE=orange.
const ACCENT: Record<'norm' | 'seq' | 'state' | 'ffn' | 'rope', string> = {
  norm: 'oklch(0.72 0.13 235)', // blue
  seq: 'oklch(0.70 0.16 300)', // purple
  state: 'oklch(0.75 0.12 185)', // teal
  ffn: 'oklch(0.72 0.18 350)', // pink
  rope: 'oklch(0.80 0.13 80)', // orange
};

function accentFor(v: Variant): string | null {
  if (v === 'norm' || v === 'seq' || v === 'state' || v === 'ffn' || v === 'rope') return ACCENT[v];
  return null;
}

function PosterSvg({
  selectedId,
  onSelect,
  mixer,
  onMixerSelect,
}: {
  selectedId: string;
  onSelect: (id: string) => void;
  mixer: Mixer;
  onMixerSelect: (m: Mixer) => void;
}) {
  const ib = (props: Omit<IBoxProps, 'selectedId' | 'onSelect'>, key?: React.Key) => (
    <IBox key={key} {...props} selectedId={selectedId} onSelect={onSelect} />
  );

  return (
    <svg
      viewBox={`0 0 ${VB_W} ${VB_H}`}
      className="block h-auto"
      style={{ width: '100%', minWidth: 1280, maxWidth: VB_W }}
      role="group"
      aria-label="Interactive architecture poster of Qwen3.5-0.8B: a central decoder spine, an expanded SwiGLU feed-forward panel on the left, the Gated DeltaNet and Gated GQA sequence-mixer panels on the right, and an optional vision path."
    >
      <Markers />

      {/* ---- background dashed detail panels ---- */}
      <DashedPanel x={80} y={420} w={335} h={430} color={ACCENT.ffn} />
      <DashedPanel x={1160} y={190} w={350} h={496} color={ACCENT.seq} active={mixer === 'gdn'} />
      <DashedPanel x={1160} y={710} w={350} h={430} color={ACCENT.seq} active={mixer === 'gqa'} />
      <DashedPanel x={80} y={910} w={430} h={210} color={ACCENT.state} />

      {/* ---- soft container panels (model facts, decoder, text, legend) ---- */}
      <SoftPanel x={80} y={130} w={330} h={242} />
      <BigPanel x={500} y={320} w={600} h={640} />
      <SoftPanel x={520} y={1010} w={110} h={90} />
      <SoftPanel x={1160} y={110} w={350} h={52} />

      {/* ---- title ---- */}
      <text x={800} y={52} fontSize={30} fontWeight={700} textAnchor="middle" fill="currentColor" fillOpacity={0.95}>
        Qwen3.5-0.8B Hybrid Multimodal Architecture
      </text>
      <text x={800} y={82} fontSize={15} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
        24 dense decoder layers: {NUM_LINEAR} Gated DeltaNet linear-attention + {NUM_FULL} gated GQA full-attention
      </text>

      {/* ---- static text: model facts ---- */}
      <text x={105} y={164} fontSize={16} fontWeight={700} fill="currentColor" fillOpacity={0.9}>
        Model facts
      </text>
      {[
        'Dense 0.8B checkpoint, no MoE router',
        'Hidden size 1024, 24 decoder layers',
        'Vocab 248,320, tied LM head',
        'Context 262,144 tokens',
        'SwiGLU FFN intermediate 3584',
        'Vision path: 12-layer encoder, out 1024',
      ].map((line, i) => (
        <text key={line} x={105} y={196 + i * 26} fontSize={12.5} fill="currentColor" fillOpacity={0.75}>
          {line}
        </text>
      ))}
      <text
        x={105}
        y={354}
        fontSize={11}
        fill="currentColor"
        fillOpacity={0.55}
        style={{ fontFamily: 'var(--font-mono, monospace)' }}
      >
        Text config: Qwen3_5ForConditionalGeneration
      </text>

      {/* ---- static text: legend ---- */}
      <text x={1182} y={132} fontSize={12.5} fill="currentColor" fillOpacity={0.72}>
        Color key: normalization blue, sequence purple,
      </text>
      <text x={1182} y={152} fontSize={12.5} fill="currentColor" fillOpacity={0.72}>
        linear/state teal, FFN pink, RoPE orange.
      </text>

      {/* ---- connector curves (drawn under boxes) ---- */}
      {/* FFN panel → in-stack Dense SwiGLU FFN */}
      <FlowCurve d="M 415 575 C 500 610, 560 595, 660 595" color={ACCENT.ffn} marker="ffn" />
      {/* sequence block → GDN panel / GQA panel (active one emphasised) */}
      <FlowCurve
        d="M 925 781 C 1040 760, 1080 540, 1160 515"
        color={ACCENT.seq}
        marker="seq"
        active={mixer === 'gdn'}
      />
      <FlowCurve
        d="M 925 781 C 1040 840, 1080 885, 1160 885"
        color={ACCENT.seq}
        marker="seq"
        active={mixer === 'gqa'}
      />
      {/* vision project → token embeddings */}
      <FlowCurve d="M 450 1070 C 550 1070, 600 1032, 658 1032" color={ACCENT.state} marker="state" />

      {/* ---- central spine arrows (data flows bottom → top) ---- */}
      <FlowLine x1={800} y1={176} x2={800} y2={142} />
      <FlowLine x1={800} y1={252} x2={800} y2={231} />
      <FlowLine x1={800} y1={326} x2={800} y2={300} />
      <FlowLine x1={800} y1={1005} x2={800} y2={961} />
      <FlowLine x1={800} y1={1108} x2={800} y2={1056} />
      <FlowLine x1={631} y1={1055} x2={658} y2={1035} />

      {/* ---- representative-layer arrows + residual skips ---- */}
      <FlowLine x1={800} y1={910} x2={800} y2={888} />
      <FlowLine x1={800} y1={838} x2={800} y2={810} />
      <FlowLine x1={800} y1={753} x2={800} y2={740} />
      <FlowLine x1={800} y1={696} x2={800} y2={684} />
      <FlowLine x1={800} y1={634} x2={800} y2={624} />
      <FlowLine x1={800} y1={567} x2={800} y2={562} />
      {/* attention residual skip — single L-path, one arrowhead into the ⊕ */}
      <SkipPath d="M 612 864 V 718 H 778" />
      {/* mlp residual skip — single L-path, one arrowhead into the ⊕ */}
      <SkipPath d="M 590 718 V 540 H 778" />

      {/* ---- FFN internal arrows ---- */}
      <ThinLine x1={248} y1={775} x2={174} y2={746} />
      <ThinLine x1={248} y1={775} x2={322} y2={746} />
      <ThinLine x1={174} y1={700} x2={215} y2={674} />
      <ThinLine x1={322} y1={700} x2={282} y2={674} />
      <FlowLine x1={248} y1={628} x2={248} y2={596} />
      <FlowLine x1={248} y1={550} x2={248} y2={520} />

      {/* ---- GDN internal arrows ---- */}
      <ThinLine x1={1335} y1={311} x2={1335} y2={336} />
      <ThinLine x1={1215} y1={383} x2={1265} y2={412} />
      <ThinLine x1={1275} y1={383} x2={1335} y2={412} />
      <ThinLine x1={1335} y1={383} x2={1400} y2={412} />
      <ThinLine x1={1395} y1={383} x2={1395} y2={412} />
      <FlowLine x1={1335} y1={465} x2={1335} y2={490} />
      <ThinLine x1={1270} y1={548} x2={1254} y2={564} />
      <ThinLine x1={1400} y1={548} x2={1416} y2={564} />
      <FlowLine x1={1335} y1={548} x2={1335} y2={628} />

      {/* ---- GQA internal arrows ---- */}
      <ThinLine x1={1300} y1={831} x2={1259} y2={854} />
      <ThinLine x1={1370} y1={831} x2={1426} y2={854} />
      <ThinLine x1={1259} y1={911} x2={1257} y2={940} />
      <ThinLine x1={1426} y1={911} x2={1429} y2={940} />
      <ThinLine x1={1257} y1={985} x2={1300} y2={1006} />
      <ThinLine x1={1429} y1={985} x2={1380} y2={1006} />
      <FlowLine x1={1339} y1={1051} x2={1339} y2={1072} />

      {/* ---- vision internal arrows ---- */}
      <FlowLine x1={234} y1={1000} x2={270} y2={1000} />
      <FlowLine x1={357} y1={1025} x2={357} y2={1050} />

      {/* ---- captions ---- */}
      <text x={248} y={452} fontSize={15} fontWeight={700} textAnchor="middle" fill={ACCENT.ffn}>
        Dense SwiGLU expert
      </text>
      <text x={248} y={473} fontSize={11.5} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
        every layer, not a mixture of experts
      </text>
      <text x={248} y={508} fontSize={13} fontWeight={600} textAnchor="middle" fill="currentColor" fillOpacity={0.8}>
        FFN output
      </text>
      <text x={1335} y={226} fontSize={15} fontWeight={700} textAnchor="middle" fill={ACCENT.seq}>
        Gated DeltaNet block
      </text>
      <text x={1335} y={247} fontSize={11.5} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
        linear attention, recurrent O(n)
      </text>
      <text x={1335} y={746} fontSize={15} fontWeight={700} textAnchor="middle" fill={ACCENT.seq}>
        Gated GQA block
      </text>
      <text x={1335} y={767} fontSize={11.5} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
        full attention, quadratic prefill
      </text>
      <text x={1339} y={1132} fontSize={11} textAnchor="middle" fill="currentColor" fillOpacity={0.55}>
        KV cache: flat by default, optional block-paged text-only path
      </text>
      <text x={295} y={943} fontSize={15} fontWeight={700} textAnchor="middle" fill={ACCENT.state}>
        Optional vision path
      </text>
      <text x={800} y={932} fontSize={12.5} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
        representative decoder layer body
      </text>

      {/* ---- layer schedule (inside decoder panel) ---- */}
      <ScheduleStrip mixer={mixer} onMixerSelect={onMixerSelect} selectedId={selectedId} />

      {/* ---- residual ⊕ nodes ---- */}
      <ResidualNode id="residual-1" cx={800} cy={718} selectedId={selectedId} onSelect={onSelect} />
      <ResidualNode id="residual-2" cx={800} cy={540} selectedId={selectedId} onSelect={onSelect} />

      {/* ================= interactive boxes ================= */}

      {/* central spine */}
      {ib({
        id: 'output',
        x: 700,
        y: 100,
        w: 200,
        h: 40,
        variant: 'io',
        label: 'Output logits',
        sub: `vocab ${VOCAB_LABEL}`,
      })}
      {ib({
        id: 'lm-head',
        x: 690,
        y: 178,
        w: 220,
        h: 52,
        variant: 'ffn',
        label: 'Tied LM head',
        sub: 'embedding weight reused',
      })}
      {ib({ id: 'final-norm', x: 705, y: 255, w: 190, h: 44, variant: 'norm', label: 'Final RMSNorm' })}
      {ib({
        id: 'decoder',
        x: 540,
        y: 336,
        w: 520,
        h: 48,
        variant: 'header',
        label: `Decoder stack ×${NUM_LAYERS}`,
        sub: 'pre-norm residual block · dense FFN every layer',
      })}
      {ib({ id: 'pre-norm', x: 690, y: 840, w: 220, h: 46, variant: 'norm', label: 'RMSNorm', sub: 'pre-attention' })}
      <SequenceBox mixer={mixer} selectedId={selectedId} onSelect={onSelect} />
      {ib({ id: 'post-norm', x: 690, y: 636, w: 220, h: 46, variant: 'norm', label: 'RMSNorm', sub: 'pre-MLP' })}
      {ib({
        id: 'mlp',
        x: 665,
        y: 568,
        w: 270,
        h: 54,
        variant: 'ffn',
        label: 'Dense SwiGLU FFN',
        sub: '1024 → 3584 → 1024, no MoE',
      })}
      {ib({ id: 'embeddings', x: 660, y: 1006, w: 280, h: 48, variant: 'io', label: 'Token + visual embeddings' })}
      {ib({
        id: 'input',
        x: 650,
        y: 1110,
        w: 300,
        h: 44,
        variant: 'io',
        label: 'Input sequence',
        sub: 'text · optional image · optional video',
      })}

      {/* left: FFN expanded */}
      {ib({ id: 'ffn-input', x: 143, y: 775, w: 210, h: 40, variant: 'io', label: 'Input hidden 1024' })}
      {ib({ id: 'ffn-gate', x: 114, y: 700, w: 120, h: 44, variant: 'ffn', label: 'gate Linear', sub: '1024 → 3584' })}
      {ib({ id: 'ffn-up', x: 262, y: 700, w: 120, h: 44, variant: 'ffn', label: 'up Linear', sub: '1024 → 3584' })}
      {ib({
        id: 'ffn-act',
        x: 156,
        y: 628,
        w: 184,
        h: 44,
        variant: 'rope',
        label: 'SiLU(gate) · up',
        sub: 'SwiGLU activation',
      })}
      {ib({ id: 'ffn-down', x: 158, y: 550, w: 180, h: 44, variant: 'ffn', label: 'down Linear', sub: '3584 → 1024' })}

      {/* right top: GDN expanded */}
      {ib({ id: 'gdn-input', x: 1234, y: 275, w: 202, h: 36, variant: 'io', label: 'hidden input' })}
      {[
        { x: 1188, l: 'q,k', s: 'proj' },
        { x: 1248, l: 'v,z', s: 'proj' },
        { x: 1308, l: 'b,a', s: 'proj' },
        { x: 1368, l: 'conv', s: 'state' },
        { x: 1428, l: 'rec', s: 'state' },
      ].map((c, i) =>
        ib(
          {
            id: 'gdn-proj',
            x: c.x,
            y: 338,
            w: 54,
            h: 44,
            variant: 'state',
            label: c.l,
            sub: c.s,
            tiny: true,
          },
          `gdn-proj-${i}`,
        ),
      )}
      {ib({
        id: 'gdn-conv',
        x: 1200,
        y: 414,
        w: 270,
        h: 50,
        variant: 'norm',
        label: 'Depthwise Conv1D, kernel 4',
        sub: 'q/k/v channels, SiLU',
      })}
      {ib({
        id: 'gdn-recurrence',
        x: 1206,
        y: 492,
        w: 258,
        h: 56,
        variant: 'state',
        label: 'Gated delta recurrence',
        sub: 'β and g update recurrent state',
      })}
      {[
        { x: 1190, l: 'conv cache' },
        { x: 1352, l: 'recurrent cache' },
      ].map((c, i) =>
        ib(
          {
            id: 'gdn-cache',
            x: c.x,
            y: 566,
            w: 128,
            h: 36,
            variant: 'seq',
            label: c.l,
            tiny: true,
          },
          `gdn-cache-${i}`,
        ),
      )}
      {ib({ id: 'gdn-out', x: 1244, y: 630, w: 182, h: 38, variant: 'io', label: 'RMSNormGated + out proj' })}

      {/* right bottom: GQA expanded */}
      {ib({ id: 'gqa-input', x: 1234, y: 794, w: 202, h: 36, variant: 'io', label: 'hidden input' })}
      {ib({ id: 'gqa-qproj', x: 1188, y: 856, w: 142, h: 54, variant: 'ffn', label: 'q_proj', sub: 'Q + gate' })}
      {ib({ id: 'gqa-kvproj', x: 1372, y: 856, w: 108, h: 54, variant: 'state', label: 'k/v_proj', sub: '2 KV heads' })}
      {ib({ id: 'gqa-qnorm', x: 1198, y: 942, w: 118, h: 42, variant: 'norm', label: 'Q norm', sub: '8 heads' })}
      {ib({ id: 'gqa-knorm', x: 1370, y: 942, w: 118, h: 42, variant: 'norm', label: 'K norm', sub: '2 heads' })}
      {ib({
        id: 'gqa-rope',
        x: 1220,
        y: 1008,
        w: 238,
        h: 42,
        variant: 'rope',
        label: 'Partial RoPE',
        sub: '64 of 256 dims, θ 1e7',
      })}
      {ib({
        id: 'gqa-sdpa',
        x: 1210,
        y: 1074,
        w: 258,
        h: 42,
        variant: 'seq',
        label: 'SDPA + sigmoid gate',
        sub: 'o_proj → hidden 1024',
      })}

      {/* vision + text inputs */}
      {ib({
        id: 'vision-input',
        x: 118,
        y: 976,
        w: 115,
        h: 48,
        variant: 'state',
        label: 'image/video',
        sub: 'patch 16, tubelet 2',
      })}
      {ib({
        id: 'vision-encoder',
        x: 272,
        y: 976,
        w: 170,
        h: 48,
        variant: 'norm',
        label: 'Vision encoder ×12',
        sub: 'hidden 768, heads 12',
      })}
      {ib({ id: 'vision-project', x: 266, y: 1052, w: 184, h: 38, variant: 'state', label: 'project to hidden 1024' })}
      {ib({ id: 'text-input', x: 524, y: 1018, w: 102, h: 74, variant: 'io', label: 'Text', sub: 'token IDs · 248k' })}
    </svg>
  );
}

// --------------------------------------------------------------------------
// Markers (arrow heads) — one neutral + one per accent used on curves.
// --------------------------------------------------------------------------
function Markers() {
  const head = (id: string, fill: string, opacity = 1) => (
    <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 Z" fill={fill} opacity={opacity} />
    </marker>
  );
  return (
    <defs>
      {head('arch-arrow', 'currentColor', 0.5)}
      {head('arch-arrow-seq', ACCENT.seq)}
      {head('arch-arrow-state', ACCENT.state)}
      {head('arch-arrow-ffn', ACCENT.ffn)}
    </defs>
  );
}

// --------------------------------------------------------------------------
// Decoration: panels & flow lines.
// --------------------------------------------------------------------------
function SoftPanel({ x, y, w, h }: { x: number; y: number; w: number; h: number }) {
  return (
    <rect
      x={x}
      y={y}
      width={w}
      height={h}
      rx={12}
      fill="currentColor"
      fillOpacity={0.02}
      stroke="currentColor"
      strokeOpacity={0.18}
      strokeWidth={1.2}
    />
  );
}
function BigPanel({ x, y, w, h }: { x: number; y: number; w: number; h: number }) {
  return (
    <rect
      x={x}
      y={y}
      width={w}
      height={h}
      rx={12}
      fill="currentColor"
      fillOpacity={0.035}
      stroke="currentColor"
      strokeOpacity={0.25}
      strokeWidth={1.4}
    />
  );
}
function DashedPanel({
  x,
  y,
  w,
  h,
  color,
  active,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
  active?: boolean;
}) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={12}
        fill={color}
        fillOpacity={0.03}
        stroke={color}
        strokeOpacity={0.45}
        strokeWidth={1.6}
        strokeDasharray="8 6"
      />
      {active ? (
        <rect
          x={x - 4}
          y={y - 4}
          width={w + 8}
          height={h + 8}
          rx={14}
          fill="none"
          className="stroke-primary"
          strokeOpacity={0.85}
          strokeWidth={2}
        />
      ) : null}
    </g>
  );
}
function FlowLine({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) {
  return (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke="currentColor"
      strokeOpacity={0.4}
      strokeWidth={2}
      markerEnd="url(#arch-arrow)"
    />
  );
}
function ThinLine({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) {
  return (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke="currentColor"
      strokeOpacity={0.3}
      strokeWidth={1.5}
      markerEnd="url(#arch-arrow)"
    />
  );
}
// Right-angle residual bypass: one continuous path (riser → corner → into the
// ⊕) carrying a single arrowhead at the destination. Drawing it as one path
// avoids the spurious corner arrowhead produced when two separately-markered
// segments meet at the bend.
function SkipPath({ d }: { d: string }) {
  return (
    <path d={d} fill="none" stroke="currentColor" strokeOpacity={0.3} strokeWidth={1.5} markerEnd="url(#arch-arrow)" />
  );
}
function FlowCurve({
  d,
  color,
  marker,
  active,
}: {
  d: string;
  color: string;
  marker: 'seq' | 'state' | 'ffn';
  active?: boolean;
}) {
  // `active` undefined → always-on curve (FFN, vision). Defined → emphasis toggle.
  const on = active === undefined ? true : active;
  return (
    <path
      d={d}
      fill="none"
      stroke={color}
      strokeOpacity={on ? 0.9 : 0.25}
      strokeWidth={1.8}
      strokeDasharray="8 6"
      markerEnd={`url(#arch-arrow-${marker})`}
    />
  );
}

// --------------------------------------------------------------------------
// Interactive box.
// --------------------------------------------------------------------------
type IBoxProps = {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  variant: Variant;
  label: string;
  sub?: string;
  /** Render compact (two tiny stacked lines) for the small projection cells. */
  tiny?: boolean;
  selectedId: string;
  onSelect: (id: string) => void;
};

function IBox({ id, x, y, w, h, variant, label, sub, tiny, selectedId, onSelect }: IBoxProps) {
  const selected = selectedId === id;
  const accent = accentFor(variant);
  const isHeader = variant === 'header';
  const select = () => onSelect(id);

  const cx = x + w / 2;
  const labelSize = isHeader ? 18 : tiny ? 11 : 13;
  const subSize = isHeader ? 12.5 : tiny ? 9.5 : 10.5;
  const labelY = isHeader ? y + 22 : sub ? y + h / 2 - 2 : y + h / 2 + 4.5;
  const subY = isHeader ? y + 40 : y + h / 2 + (tiny ? 11 : 13);

  return (
    <g
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      aria-label={`${label}. Show details.`}
      onMouseEnter={select}
      onFocus={select}
      onClick={select}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          select();
        }
      }}
      style={{ cursor: 'pointer' }}
    >
      {selected ? (
        <rect
          x={x - 3}
          y={y - 3}
          width={w + 6}
          height={h + 6}
          rx={9}
          className="fill-primary/10 stroke-primary"
          strokeWidth={1.8}
        />
      ) : null}
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={isHeader ? 8 : 6}
        fill={accent ?? (isHeader ? 'transparent' : 'currentColor')}
        fillOpacity={accent ? 0.16 : isHeader ? 0 : 0.06}
        stroke={accent ?? (isHeader ? 'transparent' : 'currentColor')}
        strokeOpacity={accent ? 0.7 : isHeader ? 0 : 0.35}
        strokeWidth={1.3}
      />
      <text
        x={cx}
        y={labelY}
        fontSize={labelSize}
        textAnchor="middle"
        fill={accent ?? 'currentColor'}
        fillOpacity={accent ? 1 : 0.9}
        style={{ fontWeight: isHeader ? 700 : 600 }}
      >
        {label}
      </text>
      {sub ? (
        <text x={cx} y={subY} fontSize={subSize} textAnchor="middle" fill="currentColor" fillOpacity={0.58}>
          {sub}
        </text>
      ) : null}
    </g>
  );
}

// The central sequence block — label/colour follow the mixer toggle.
function SequenceBox({
  mixer,
  selectedId,
  onSelect,
}: {
  mixer: Mixer;
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const selected = selectedId === 'sequence';
  const color = mixer === 'gdn' ? ACCENT.state : ACCENT.seq;
  const x = 675;
  const y = 754;
  const w = 250;
  const h = 54;
  const cx = x + w / 2;
  const select = () => onSelect('sequence');
  return (
    <g
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      aria-label={`Sequence block, currently ${mixer === 'gdn' ? 'Gated DeltaNet linear attention' : 'gated GQA full attention'}. Show details.`}
      onMouseEnter={select}
      onFocus={select}
      onClick={select}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          select();
        }
      }}
      style={{ cursor: 'pointer' }}
    >
      {selected ? (
        <rect
          x={x - 3}
          y={y - 3}
          width={w + 6}
          height={h + 6}
          rx={9}
          className="fill-primary/10 stroke-primary"
          strokeWidth={1.8}
        />
      ) : null}
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={6}
        fill={color}
        fillOpacity={0.16}
        stroke={color}
        strokeOpacity={0.7}
        strokeWidth={1.3}
      />
      <text x={cx} y={y + 24} fontSize={14} textAnchor="middle" fill={color} style={{ fontWeight: 600 }}>
        Sequence block
      </text>
      <text x={cx} y={y + 42} fontSize={11} textAnchor="middle" fill="currentColor" fillOpacity={0.62}>
        {mixer === 'gdn' ? 'Gated DeltaNet (linear) →' : 'Gated GQA (full) →'}
      </text>
    </g>
  );
}

// Residual ⊕ node.
function ResidualNode({
  id,
  cx,
  cy,
  selectedId,
  onSelect,
}: {
  id: string;
  cx: number;
  cy: number;
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const selected = selectedId === id;
  const select = () => onSelect(id);
  return (
    <g
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      aria-label="Residual add. Show details."
      onMouseEnter={select}
      onFocus={select}
      onClick={select}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          select();
        }
      }}
      style={{ cursor: 'pointer' }}
    >
      {selected ? <circle cx={cx} cy={cy} r={23} className="fill-primary/10 stroke-primary" strokeWidth={1.8} /> : null}
      <circle
        cx={cx}
        cy={cy}
        r={20}
        fill="currentColor"
        fillOpacity={0.06}
        stroke="currentColor"
        strokeOpacity={0.5}
        strokeWidth={1.6}
      />
      <line x1={cx - 11} y1={cy} x2={cx + 11} y2={cy} stroke="currentColor" strokeOpacity={0.7} strokeWidth={2} />
      <line x1={cx} y1={cy - 11} x2={cx} y2={cy + 11} stroke="currentColor" strokeOpacity={0.7} strokeWidth={2} />
    </g>
  );
}

// --------------------------------------------------------------------------
// Layer schedule — faithful [GDN, GDN, GDN, Gated GQA] ×6 pattern. The three
// GDN cells select the linear mixer; the Gated GQA cell selects full.
// --------------------------------------------------------------------------
function ScheduleStrip({
  mixer,
  onMixerSelect,
  selectedId,
}: {
  mixer: Mixer;
  onMixerSelect: (m: Mixer) => void;
  selectedId: string;
}) {
  const cells: { x: number; w: number; label: string; m: Mixer }[] = [
    { x: 590, w: 78, label: 'GDN', m: 'gdn' },
    { x: 684, w: 78, label: 'GDN', m: 'gdn' },
    { x: 778, w: 78, label: 'GDN', m: 'gdn' },
    { x: 872, w: 92, label: 'Gated GQA', m: 'gqa' },
  ];
  // Schedule cells are click-only (focus must NOT flip the mixer), so they need
  // their own visible focus ring — tabbing to an inactive cell would otherwise
  // give keyboard users no indication of where focus landed.
  const [focusedCell, setFocusedCell] = React.useState<number | null>(null);
  return (
    <g>
      {/* dashed schedule container */}
      <rect
        x={540}
        y={390}
        width={520}
        height={116}
        rx={8}
        fill={ACCENT.seq}
        fillOpacity={0.03}
        stroke={ACCENT.seq}
        strokeOpacity={0.45}
        strokeWidth={1.6}
        strokeDasharray="8 6"
      />
      <text x={800} y={414} fontSize={13.5} fontWeight={700} textAnchor="middle" fill="currentColor" fillOpacity={0.7}>
        Layer type schedule, repeated 6 times
      </text>
      {cells.map((c, i) => {
        const color = c.m === 'gdn' ? ACCENT.state : ACCENT.seq;
        const active = mixer === c.m;
        const sel = selectedId === 'sequence' && active;
        const focused = focusedCell === i;
        const select = () => onMixerSelect(c.m);
        return (
          <g
            key={`sched-${c.x}`}
            role="button"
            tabIndex={0}
            aria-pressed={active}
            aria-label={`${c.label} layer. Highlight the ${c.m === 'gdn' ? 'linear' : 'full'} sequence mixer.`}
            onClick={select}
            onFocus={() => setFocusedCell(i)}
            onBlur={() => setFocusedCell((prev) => (prev === i ? null : prev))}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                select();
              }
            }}
            style={{ cursor: 'pointer' }}
          >
            {sel ? (
              <rect
                x={c.x - 3}
                y={430}
                width={c.w + 6}
                height={50}
                rx={8}
                className="fill-primary/10 stroke-primary"
                strokeWidth={1.6}
              />
            ) : null}
            {focused && !sel ? (
              <rect
                x={c.x - 3}
                y={430}
                width={c.w + 6}
                height={50}
                rx={8}
                fill="none"
                className="stroke-primary"
                strokeWidth={1.6}
                strokeDasharray="4 3"
              />
            ) : null}
            <rect
              x={c.x}
              y={433}
              width={c.w}
              height={44}
              rx={5}
              fill={color}
              fillOpacity={active ? 0.22 : 0.1}
              stroke={color}
              strokeOpacity={active ? 0.9 : 0.45}
              strokeWidth={active ? 1.5 : 1}
            />
            <text
              x={c.x + c.w / 2}
              y={460}
              fontSize={c.label.length > 4 ? 12 : 14}
              textAnchor="middle"
              fill={color}
              style={{ fontWeight: 600 }}
            >
              {c.label}
            </text>
          </g>
        );
      })}
      <text x={1005} y={461} fontSize={14} fontWeight={700} fill="currentColor" fillOpacity={0.7}>
        ×6
      </text>
      <text x={800} y={496} fontSize={11} textAnchor="middle" fill="currentColor" fillOpacity={0.6}>
        Full attention layers: 4, 8, 12, 16, 20, 24. All others use linear attention.
      </text>
    </g>
  );
}

// --------------------------------------------------------------------------
// Detail panel below the poster. Related links are small secondary <Link>s.
// --------------------------------------------------------------------------
function DetailPanel({ detail }: { detail: Detail }) {
  return (
    <div className="rounded-md border border-border bg-card/40 p-4" aria-live="polite">
      <div className="text-sm font-semibold text-foreground">{detail.title}</div>
      {detail.facts ? (
        <div className="mt-1 font-mono text-[11px] uppercase tracking-wider text-muted-foreground">{detail.facts}</div>
      ) : null}
      <p className="mt-2 text-[13px] leading-relaxed text-foreground/85">{detail.body}</p>
      {detail.related && detail.related.length > 0 ? (
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
          {detail.related.map((r) => (
            <Link
              key={r.chapterId}
              to="/chapters/$chapterId"
              params={{ chapterId: r.chapterId }}
              search={(prev) => prev}
              className="text-[11px] text-muted-foreground underline decoration-dotted underline-offset-2 hover:text-foreground"
            >
              Related: {r.label} →
            </Link>
          ))}
        </div>
      ) : null}
    </div>
  );
}
