import * as React from 'react';

import { Button } from '../../components/ui/button';
import { DemoCallout } from '../inspector/DemoCallout';
import { Prose } from '../Prose';
import { ChapterFrame } from '../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../scaffolding/learning-data';
import { GenerationLoopTrace } from '../widgets/GenerationLoopTrace';
import { KvGrowthCurve } from '../widgets/KvGrowthCurve';
import { PrefillVsDecodeChart } from '../widgets/PrefillVsDecodeChart';
import { RecallFailure } from '../widgets/RecallFailure';
import { RunningStateVsCache } from '../widgets/RunningStateVsCache';

/**
 * Chapter 12 — KV cache & hybrid attention.
 *
 * JS-only: every number here is derivable from Qwen3.5-0.8B's config (see
 * packages/browser/demo/public/model/config.json and the layer_types array
 * inside `text_config`). No worker call, no model inference — the widget is
 * a pure SVG/State visualisation with three views:
 *   - a 24-bar layer strip showing the hybrid full/linear interleave
 *   - a memory-vs-context-length chart contrasting MHA, GQA-all-full, and
 *     the actual hybrid layout
 *   - a stepper-driven decode animation that shows the cached K/V grow as
 *     the model generates tokens one at a time
 */

const NUM_LAYERS = 24;
const NUM_HEADS = 8;
const NUM_KV_HEADS = 2;
const HEAD_DIM = 256;
const HIDDEN_DIM = 1024;
const FULL_ATTENTION_INTERVAL = 4;
const BYTES_PER_FLOAT = 2; // bf16
const MAX_POSITION = 262_144;
const DEFAULT_CONTEXT_LEN = 32_768;
const CHART_MAX_CONTEXT = 131_072;
const CHART_MIN_CONTEXT = 256;

// Layer indices that use full softmax attention. Mirrors the layer_types
// array in the model's config.json: every 4th layer (1-indexed) is full.
const FULL_LAYER_INDICES: number[] = Array.from({ length: NUM_LAYERS }, (_, i) => i).filter(
  (i) => (i + 1) % FULL_ATTENTION_INTERVAL === 0,
);
const FULL_LAYER_SET = new Set<number>(FULL_LAYER_INDICES);
const NUM_FULL_LAYERS = FULL_LAYER_INDICES.length;
const NUM_LINEAR_LAYERS = NUM_LAYERS - NUM_FULL_LAYERS;

// GatedDeltaNet per-layer recurrent state. The exact form is a fixed-size
// tensor [linear_num_value_heads, value_head_dim, key_head_dim] = [16, 128, 128]
// that compresses the entire history — independent of sequence length. These
// are the linear-attention dims from Qwen3.5's config (distinct from the
// full-attention 8 heads × 256 head-dim).
const LINEAR_NUM_HEADS = 16;
const LINEAR_HEAD_DIM = 128;
const LINEAR_STATE_SIZE_BYTES = LINEAR_NUM_HEADS * LINEAR_HEAD_DIM * LINEAR_HEAD_DIM * BYTES_PER_FLOAT;

const DECODE_PROMPT_TOKENS: ReadonlyArray<string> = ['The', ' cat', ' sat', ' on', ' the'];
const DECODE_GENERATED_TOKENS: ReadonlyArray<string> = [' mat', '.', ' It'];
const DECODE_TOTAL_STEPS = DECODE_PROMPT_TOKENS.length + DECODE_GENERATED_TOKENS.length;
const DECODE_AUTOPLAY_MS = 1100;

function fullLayerCacheBytes(seqLen: number): number {
  return 2 * NUM_KV_HEADS * HEAD_DIM * seqLen * BYTES_PER_FLOAT;
}
function mhaBaselineBytes(seqLen: number): number {
  return NUM_LAYERS * 2 * NUM_HEADS * HEAD_DIM * seqLen * BYTES_PER_FLOAT;
}
function gqaAllLayersBytes(seqLen: number): number {
  return NUM_LAYERS * fullLayerCacheBytes(seqLen);
}
function hybridBytes(seqLen: number): number {
  return NUM_FULL_LAYERS * fullLayerCacheBytes(seqLen) + NUM_LINEAR_LAYERS * LINEAR_STATE_SIZE_BYTES;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—';
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function formatTokens(n: number): string {
  if (!Number.isFinite(n)) return '—';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return `${n}`;
}

/**
 * Scaffolding metadata for chapter 12 — drives the header, glossary,
 * takeaways, exercise, and quick-check rendered by `<ChapterFrame>`.
 * `chapterId` must match the 'kv-cache' entry in `learn/chapters.ts`.
 */
export const learning: ChapterLearningData = {
  chapterId: 'kv-cache',
  objective:
    "Explain why the KV cache exists, how it grows with context, and what Qwen3.5's hybrid attention buys you in memory.",
  problem:
    'Without caching K and V, each decoded token would redo all of attention from scratch — making generation quadratic in context length.',
  minutes: 10,
  glossary: [
    {
      term: 'KV cache',
      definition:
        'Per-layer stored K and V tensors for every prompt and generated token, reused across decode steps so attention stays roughly linear.',
    },
    {
      term: 'prefill',
      definition:
        'First inference phase: the whole prompt is processed in one matmul-heavy pass that writes K and V for every prompt token.',
    },
    {
      term: 'decode',
      definition:
        "Per-token generation phase: only the new token's Q/K/V is computed, and Q attends over the cached K/V history.",
    },
    {
      term: 'GatedDeltaNet',
      definition:
        "Qwen3.5's linear-attention variant. Stores a fixed-size recurrent state per layer instead of a token-by-token KV cache.",
    },
    {
      term: 'hybrid attention',
      definition:
        'Mixing full softmax-attention layers and linear-attention layers in the same stack. Qwen3.5 uses 6 full + 18 linear.',
    },
    {
      term: 'linear state',
      definition:
        'Per-layer tensor of shape [16, 128, 128] (linear value-heads × value-dim × key-dim) that compresses the entire history; size is independent of sequence length.',
    },
    {
      term: 'recurrent state (RNN-style)',
      definition:
        'A fixed-size memory the model overwrites and rolls forward at every step — like a running total or moving average — instead of keeping every past item. Linear-attention layers use one so their memory stays constant no matter how long the context grows.',
    },
  ],
  takeaways: [
    'KV cache memory scales linearly with context, and at long context it dwarfs the model weights themselves.',
    "Qwen3.5's 6 full + 18 linear layout collapses the cache curve because linear layers contribute a constant, not a per-token chunk.",
    "Full-attention layers handle 'recall this specific token' jobs; linear layers handle the bulk of language flow under a tight memory budget.",
  ],
  exercise: {
    prompt:
      "Drag the context-length slider from 1k to 131k while watching the three KV cache curves. At 131k tokens, roughly how many times smaller is the hybrid 'actual' curve than the MHA hypothetical curve, and which curve overlaps the hybrid line at short contexts?",
    answer:
      "At 131k tokens the hybrid curve sits roughly an order of magnitude below MHA (the 'Savings' tile shows the GQA-to-hybrid ratio at the current context). At short contexts the linear layers' constant state dominates the total, so the hybrid line plateaus while GQA and MHA continue dropping toward it — they only diverge once seq_len grows enough to dominate the per-layer arithmetic.",
  },
  quiz: [
    {
      id: 'q1-why-cache',
      prompt: 'What would happen during decode if the model did not maintain a KV cache?',
      options: [
        { id: 'a', label: 'Decode would still work, just at the same speed.' },
        {
          id: 'b',
          label:
            'Each new token would re-compute K and V for the entire history, so decode cost per step would be as expensive as prefill — quadratic in context length over the whole generation.',
        },
        {
          id: 'c',
          label: 'The model would lose access to RoPE positions.',
        },
      ],
      correctId: 'b',
      explanation:
        "K and V for old tokens don't depend on the new one, so caching them turns decode from quadratic to linear in sequence length.",
    },
    {
      id: 'q2-linear-state-size',
      prompt: "How does a GatedDeltaNet linear-attention layer's memory cost scale with sequence length?",
      options: [
        { id: 'a', label: 'Linearly with seq_len, like a regular KV cache.' },
        {
          id: 'b',
          label:
            "It doesn't — the recurrent state is a fixed-size tensor [16, 128, 128] (linear heads × value-dim × key-dim) regardless of how many tokens have streamed by.",
        },
        {
          id: 'c',
          label: 'Quadratically with seq_len.',
        },
      ],
      correctId: 'b',
      explanation:
        "That's the whole point of a recurrent compression: the state stays the same size as the sequence grows, at the cost of losing exact per-token recall.",
    },
    {
      id: 'q3-why-hybrid',
      prompt: 'Why does Qwen3.5 still keep 6 full-attention layers instead of going fully linear?',
      options: [
        {
          id: 'a',
          label:
            'Linear attention compresses history and loses exact recall; full layers preserve the ability to look back at one specific earlier token when the model needs it.',
        },
        {
          id: 'b',
          label: 'Full layers are required for RoPE to work.',
        },
        {
          id: 'c',
          label: 'Full layers run faster than linear ones at long context.',
        },
      ],
      correctId: 'a',
      explanation:
        "Linear layers handle pattern continuation cheaply; full layers handle 'look back to that one specific token' jobs. The hybrid recovers most of the quality at a fraction of the cache cost.",
    },
  ],
};

export function KvCacheChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>KV cache &amp; hybrid attention</h1>
        <p>
          Every chapter so far has described what happens for <em>one forward pass</em>. But generation is a sequence of
          forward passes — one per output token — and naively each pass would redo all the attention work for every
          token in the history. That would make decode quadratic in the context length, which is obviously hopeless once
          you have a 32k-token document in the prompt. The fix is a piece of engineering, not a new architecture: cache
          the parts that don't change between steps. That cache is what dominates inference memory and shapes most of
          the architectural decisions you see in modern LLMs.
        </p>

        <h2>Prefill versus decode</h2>
        <p>
          Inference has two phases. <strong>Prefill</strong> takes the whole prompt at once and computes K, V and the
          hidden states for every token in one big matmul-heavy pass. <strong>Decode</strong> then generates one token
          at a time: feed the previous output back in, compute K/V/Q for just <em>that</em> single new token, and read
          the K/V of all earlier tokens out of a cache instead of recomputing them. Without the cache, each decode step
          would be as expensive as prefill — with it, decode cost is roughly linear in the cache size.
        </p>

        <h2>Why caching K and V works</h2>
        <p>
          At decode step <code>t</code>, every layer needs the attention output for the new token:{' '}
          <code>softmax(q_t · K^T / √d) · V</code>. The Q is brand new (it comes from the just-computed hidden state).
          The K and V for the new token are also new. But the K and V for tokens <code>0..t-1</code> were already
          computed in earlier steps —{' '}
          <em>
            and they don't depend on token <code>t</code>
          </em>
          . So we save them once and read them back forever.
        </p>
        <p>
          Shape-wise, the cache stores, per layer, two tensors of shape <code>[num_kv_heads, seq_len, head_dim]</code>.
          As <code>seq_len</code> grows, the cache grows linearly. That's the catch: cache memory is proportional to
          context length, and for long contexts it dwarfs the weights.
        </p>

        <GenerationLoopTrace />

        <h2>The arithmetic for Qwen3.5-0.8B</h2>
        <p>
          Per full-attention layer, per token, the cache costs{' '}
          <code>
            2 · num_kv_heads · head_dim · 2 bytes = 2 · {NUM_KV_HEADS} · {HEAD_DIM} · {BYTES_PER_FLOAT} ={' '}
            {fullLayerCacheBytes(1)} bytes
          </code>
          . At a 32k-token context, one full-attention layer holds{' '}
          <code>{formatBytes(fullLayerCacheBytes(32_768))}</code>. If every one of the {NUM_LAYERS} layers were
          full-attention, the whole-model cache would be <code>{formatBytes(gqaAllLayersBytes(32_768))}</code> — for the{' '}
          <em>smallest</em> Qwen3.5 variant. Bigger Qwen3.5 sizes scale from there.
        </p>
        <p>
          This is why KV cache memory, not weights, is the headline number for modern serving. It's also why a
          generation of architectural decisions — multi-query attention, grouped-query attention, sliding windows, and
          linear-attention variants — have all been driven by cutting the cache without giving up too much quality.
        </p>

        <h2>Qwen3.5's answer: hybrid attention</h2>
        <p>
          Qwen3.5 interleaves two different kinds of attention layer. <strong>Six</strong> of its {NUM_LAYERS} layers
          (one in every four, at indices {FULL_LAYER_INDICES.join(', ')}) are conventional full-attention layers — the
          ones from Chapter 4, with the linear-in-
          <code>seq_len</code> KV cache. The other {NUM_LINEAR_LAYERS} layers use a{' '}
          <strong>linear attention variant called GatedDeltaNet</strong>. Instead of caching K and V per token, a
          GatedDeltaNet layer compresses the entire history into a <em>fixed-size recurrent state</em> of shape{' '}
          <code>
            [{LINEAR_NUM_HEADS}, {LINEAR_HEAD_DIM}, {LINEAR_HEAD_DIM}]
          </code>{' '}
          (linear value-heads × value-dim × key-dim — distinct from the {NUM_HEADS} full-attention heads of head-dim{' '}
          {HEAD_DIM}). The state is rolled forward at each step the way an RNN's hidden state is — independent of how
          many tokens have streamed by.
        </p>
        <p>
          With {LINEAR_NUM_HEADS} linear heads at <code>head_dim = {LINEAR_HEAD_DIM}</code>, each linear layer's state
          is a constant <code>{formatBytes(LINEAR_STATE_SIZE_BYTES)}</code> regardless of context. So total cache for
          Qwen3.5-0.8B is <code>6 × full + 18 × constant</code> — the chart on the right plots that against a
          hypothetical Qwen with full attention on every layer.
        </p>
        <p>
          Here is that contrast at the token level: feed the same stream through both kinds of memory and watch one
          append a box per token while the other updates a single slot in place.
        </p>

        <RunningStateVsCache />

        <h2>The trade-off</h2>
        <p>
          Linear attention is approximate. It cannot perfectly recall an arbitrary single token from way back in the
          history the way full softmax attention can — its compressed state mixes everything together. What it does well
          is the bulk of language processing: pattern continuation, syntactic flow, local style. Interleaving with the
          six full-attention layers brings the exact-recall capability back where it's needed. Concretely: linear layers
          run the long- context flow, full layers handle "look back to <em>that</em> one specific token" jobs.
        </p>

        <RecallFailure />

        <h2>Why this is the future</h2>
        <p>
          As context lengths push past 1M tokens, quadratic-memory attention becomes untenable — both as cache and as
          compute. Hybrid attention is one of the techniques the field is exploring. Pure state-space models like Mamba
          take it further still by removing softmax attention entirely. The trend is clear: cache is the bottleneck, and
          architectures will keep getting reshaped around it.
        </p>

        <div className="not-prose my-4 rounded-md border border-amber-500/50 bg-amber-500/5 p-4">
          <div className="text-xs uppercase tracking-wider text-amber-700 dark:text-amber-300">Misconception</div>
          <div className="mt-1 text-sm font-semibold text-foreground">The KV cache is not a Python dict.</div>
          <p className="mt-2 text-[12px] leading-relaxed text-foreground/85">
            People sometimes picture it as a hashmap from token text or position to (k, v) tuples. It isn&apos;t. The
            cache is a pre-allocated tensor of shape{' '}
            <code className="rounded bg-background px-1 py-0.5 font-mono text-[11px]">
              [layers, 2, batch, kv_heads, max_seq, head_dim]
            </code>{' '}
            that each attention layer writes into and reads from, with a position counter telling it how much of the
            tensor is live. The "key" is the position index, not a hash. It&apos;s a buffer, not a hashmap.
          </p>
        </div>

        <h2>Prefill vs decode, in time</h2>
        <p>
          Memory is one axis; wall-clock time is the other. The chart below contrasts prefill (one parallel matmul over
          the whole prompt) with decode (many small sequential matmuls, one per generated token).
        </p>

        <PrefillVsDecodeChart />

        <h2>How the cache grows with context</h2>
        <p>
          And here, plotted in isolation, is the cache-vs-context curve for each layout. The formula is unpacked
          underneath: every factor in <code>2 · L · kv_heads · d · seq · bf16</code> is a knob someone is working on
          right now.
        </p>

        <KvGrowthCurve />

        <p className="mt-6 text-muted-foreground">
          The right-hand widget plots the cache curves for MHA, GQA-only, and Qwen3.5's hybrid layout; drag the context
          slider to read off memory at any length. The layer strip below it shows the exact interleave — click any layer
          to see what its cache looks like at the current context length.
        </p>
      </Prose>
    </ChapterFrame>
  );
}

export type KvCacheDemoProps = {
  // Accepted for API parity with the other chapter Try-It panels even
  // though this chapter never touches the worker.
  workerRef?: React.RefObject<Worker | null>;
  abortRef?: React.RefObject<AbortController | null>;
};

export function KvCacheDemo(_props: KvCacheDemoProps) {
  const [contextLen, setContextLen] = React.useState(DEFAULT_CONTEXT_LEN);
  const [selectedLayer, setSelectedLayer] = React.useState(FULL_LAYER_INDICES[0] ?? 3);
  const [decodeStep, setDecodeStep] = React.useState(DECODE_PROMPT_TOKENS.length - 1);
  const [autoplay, setAutoplay] = React.useState(false);

  React.useEffect(() => {
    if (!autoplay) return undefined;
    const id = window.setInterval(() => {
      setDecodeStep((s) => {
        if (s + 1 >= DECODE_TOTAL_STEPS) {
          setAutoplay(false);
          return s;
        }
        return s + 1;
      });
    }, DECODE_AUTOPLAY_MS);
    return () => window.clearInterval(id);
  }, [autoplay]);

  React.useEffect(() => {
    console.info('[kv-cache-demo] context-length update', {
      contextLen,
      hybridBytes: hybridBytes(contextLen),
      gqaAllLayersBytes: gqaAllLayersBytes(contextLen),
      mhaBaselineBytes: mhaBaselineBytes(contextLen),
    });
  }, [contextLen]);

  return (
    <div className="space-y-4">
      <div className="rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
        All numbers below are derived from Qwen3.5-0.8B's config: num_layers={NUM_LAYERS}, num_heads={NUM_HEADS},
        num_kv_heads=
        {NUM_KV_HEADS}, head_dim={HEAD_DIM}, full_attention_interval=
        {FULL_ATTENTION_INTERVAL}, max_position_embeddings=
        {formatTokens(MAX_POSITION)}. No model inference needed.
      </div>

      <MemoryChart contextLen={contextLen} onContextLen={setContextLen} />

      <ContextSlider contextLen={contextLen} onContextLen={setContextLen} />

      <CacheStats contextLen={contextLen} />

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <LayerStrip selectedLayer={selectedLayer} onSelect={setSelectedLayer} />
        <LayerSidePanel layerIdx={selectedLayer} contextLen={contextLen} />
      </div>

      <DecodeAnimation
        step={decodeStep}
        autoplay={autoplay}
        onStep={(next) => {
          setAutoplay(false);
          setDecodeStep(next);
        }}
        onTogglePlay={() => {
          if (decodeStep >= DECODE_TOTAL_STEPS - 1) {
            setDecodeStep(DECODE_PROMPT_TOKENS.length - 1);
            setAutoplay(true);
          } else {
            setAutoplay((p) => !p);
          }
        }}
      />

      <DemoCallout
        items={[
          "Drag the context slider toward 131k tokens — the MHA-hypothetical curve explodes while Qwen's actual hybrid stays modest.",
          "The MHA total at 32k is ~6 GB — that's what every layer storing full K/V would cost.",
          'Hybrid wins at long context because the linear layers cost a constant 512 KB each regardless of seq_len.',
        ]}
      />
    </div>
  );
}

// -----------------------------------------------------------------------------
// Memory-vs-context-length chart.
// -----------------------------------------------------------------------------

const CHART_SAMPLES = 96;

function logSpacePositions(samples: number): number[] {
  const lo = Math.log10(CHART_MIN_CONTEXT);
  const hi = Math.log10(CHART_MAX_CONTEXT);
  return Array.from({ length: samples }, (_, i) => {
    const t = i / (samples - 1);
    return Math.round(10 ** (lo + t * (hi - lo)));
  });
}

function MemoryChart({ contextLen, onContextLen }: { contextLen: number; onContextLen: (n: number) => void }) {
  const samples = React.useMemo(() => logSpacePositions(CHART_SAMPLES), []);

  const width = 560;
  const height = 240;
  const padLeft = 56;
  const padRight = 14;
  const padTop = 16;
  const padBottom = 38;
  const innerW = width - padLeft - padRight;
  const innerH = height - padTop - padBottom;

  // Y axis: log scale on bytes.
  const yMaxBytes = Math.max(
    mhaBaselineBytes(CHART_MAX_CONTEXT),
    gqaAllLayersBytes(CHART_MAX_CONTEXT),
    hybridBytes(CHART_MAX_CONTEXT),
  );
  const yMinBytes = Math.min(hybridBytes(CHART_MIN_CONTEXT), LINEAR_STATE_SIZE_BYTES * NUM_LINEAR_LAYERS);
  const logYMin = Math.log10(Math.max(1024, yMinBytes));
  const logYMax = Math.log10(yMaxBytes);

  const logXMin = Math.log10(CHART_MIN_CONTEXT);
  const logXMax = Math.log10(CHART_MAX_CONTEXT);

  function xFor(seqLen: number): number {
    const t = (Math.log10(Math.max(CHART_MIN_CONTEXT, seqLen)) - logXMin) / (logXMax - logXMin);
    return padLeft + Math.max(0, Math.min(1, t)) * innerW;
  }
  function yFor(bytes: number): number {
    const clamped = Math.max(1024, bytes);
    const t = (Math.log10(clamped) - logYMin) / (logYMax - logYMin);
    return padTop + innerH - Math.max(0, Math.min(1, t)) * innerH;
  }

  function pathFor(fn: (n: number) => number): string {
    return 'M ' + samples.map((n) => `${xFor(n).toFixed(2)} ${yFor(fn(n)).toFixed(2)}`).join(' L ');
  }

  const mhaPath = pathFor(mhaBaselineBytes);
  const gqaPath = pathFor(gqaAllLayersBytes);
  const hybridPath = pathFor(hybridBytes);

  const xTicks = [1_024, 8_192, 32_768, 131_072];
  const yTicks: number[] = [];
  for (let exp = Math.ceil(logYMin); exp <= Math.floor(logYMax); exp++) {
    yTicks.push(10 ** exp);
  }

  const markerX = xFor(contextLen);
  const mhaAtMarker = mhaBaselineBytes(contextLen);
  const gqaAtMarker = gqaAllLayersBytes(contextLen);
  const hybridAtMarker = hybridBytes(contextLen);

  const svgRef = React.useRef<SVGSVGElement>(null);

  function handlePoint(evt: React.PointerEvent<SVGSVGElement>) {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const pxX = ((evt.clientX - rect.left) / rect.width) * width;
    const tx = (pxX - padLeft) / innerW;
    const tNorm = Math.max(0, Math.min(1, tx));
    const next = Math.round(10 ** (logXMin + tNorm * (logXMax - logXMin)));
    onContextLen(Math.max(CHART_MIN_CONTEXT, Math.min(MAX_POSITION, next)));
  }

  return (
    <div className="rounded-md border border-border bg-background p-3">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          KV cache memory vs context length (log–log)
        </div>
        <div className="text-[11px] text-muted-foreground">Click or drag inside the chart to move the marker.</div>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Three KV-cache memory curves on a log-log chart: hypothetical MHA, every-layer GQA, and Qwen3.5's hybrid layout."
        className="block h-auto w-full"
        onPointerDown={(e) => {
          (e.target as Element).setPointerCapture?.(e.pointerId);
          handlePoint(e);
        }}
        onPointerMove={(e) => {
          if (e.buttons === 0) return;
          handlePoint(e);
        }}
        style={{ cursor: 'ew-resize', touchAction: 'none' }}
      >
        {yTicks.map((b, idx) => {
          const y = yFor(b);
          return (
            <g key={`y-${idx}`}>
              <line x1={padLeft} x2={width - padRight} y1={y} y2={y} stroke="currentColor" strokeOpacity={0.07} />
              <text x={padLeft - 6} y={y + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.55}>
                {formatBytes(b)}
              </text>
            </g>
          );
        })}
        {xTicks.map((t, idx) => {
          const x = xFor(t);
          return (
            <g key={`x-${idx}`}>
              <line x1={x} x2={x} y1={padTop} y2={padTop + innerH} stroke="currentColor" strokeOpacity={0.07} />
              <text x={x} y={height - 20} fontSize={9} textAnchor="middle" fill="currentColor" fillOpacity={0.55}>
                {formatTokens(t)}
              </text>
            </g>
          );
        })}

        <text
          x={(padLeft + width - padRight) / 2}
          y={height - 6}
          fontSize={10}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.55}
        >
          context length (tokens, log scale)
        </text>

        <path d={mhaPath} fill="none" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="4 3" />
        <path d={gqaPath} fill="none" stroke="#f59e0b" strokeWidth={1.8} />
        <path d={hybridPath} fill="none" stroke="#22c55e" strokeWidth={2} />

        <line
          x1={markerX}
          x2={markerX}
          y1={padTop}
          y2={padTop + innerH}
          stroke="currentColor"
          strokeOpacity={0.45}
          strokeDasharray="3 3"
        />
        <circle cx={markerX} cy={yFor(mhaAtMarker)} r={3.5} fill="#94a3b8" />
        <circle cx={markerX} cy={yFor(gqaAtMarker)} r={3.5} fill="#f59e0b" />
        <circle cx={markerX} cy={yFor(hybridAtMarker)} r={3.5} fill="#22c55e" />
        <text
          x={Math.min(width - padRight - 4, markerX + 6)}
          y={padTop + 12}
          fontSize={10}
          fill="currentColor"
          fillOpacity={0.7}
          textAnchor={markerX > padLeft + innerW * 0.75 ? 'end' : 'start'}
        >
          {formatTokens(contextLen)} tok
        </text>
      </svg>
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <LegendDot color="#94a3b8" dashed>
          MHA (hypothetical, num_kv_heads = num_heads)
        </LegendDot>
        <LegendDot color="#f59e0b">GQA every layer (num_kv_heads = {NUM_KV_HEADS})</LegendDot>
        <LegendDot color="#22c55e">Hybrid (Qwen3.5 actual)</LegendDot>
      </div>
    </div>
  );
}

function LegendDot({
  color,
  dashed = false,
  children,
}: {
  color: string;
  dashed?: boolean;
  children: React.ReactNode;
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <svg width={18} height={6} aria-hidden="true">
        <line
          x1={0}
          x2={18}
          y1={3}
          y2={3}
          stroke={color}
          strokeWidth={2}
          strokeDasharray={dashed ? '4 3' : undefined}
        />
      </svg>
      <span>{children}</span>
    </span>
  );
}

// -----------------------------------------------------------------------------
// Context-length slider + cache stats.
// -----------------------------------------------------------------------------

function ContextSlider({ contextLen, onContextLen }: { contextLen: number; onContextLen: (n: number) => void }) {
  // Slider is log-scaled in [CHART_MIN_CONTEXT, MAX_POSITION].
  const logLo = Math.log10(CHART_MIN_CONTEXT);
  const logHi = Math.log10(MAX_POSITION);
  const sliderValue = (Math.log10(Math.max(CHART_MIN_CONTEXT, contextLen)) - logLo) / (logHi - logLo);
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <label htmlFor="kv-cache-demo-context" className="uppercase tracking-wider">
          Context length
        </label>
        <span className="font-mono text-foreground/80">{formatTokens(contextLen)} tokens</span>
      </div>
      <input
        id="kv-cache-demo-context"
        type="range"
        min={0}
        max={1}
        step={0.001}
        value={sliderValue}
        onChange={(e) => {
          const t = Number.parseFloat(e.target.value);
          const next = Math.round(10 ** (logLo + t * (logHi - logLo)));
          onContextLen(Math.max(CHART_MIN_CONTEXT, Math.min(MAX_POSITION, next)));
        }}
        className="h-2 w-full cursor-pointer appearance-none rounded-full bg-muted accent-primary"
      />
    </div>
  );
}

function CacheStats({ contextLen }: { contextLen: number }) {
  const mha = mhaBaselineBytes(contextLen);
  const gqa = gqaAllLayersBytes(contextLen);
  const hybrid = hybridBytes(contextLen);
  const savingsVsGqa = gqa > 0 ? gqa / hybrid : 0;
  return (
    <div className="grid grid-cols-1 gap-2 rounded-md border border-border bg-background p-3 sm:grid-cols-3">
      <StatTile label="MHA total" value={formatBytes(mha)} sub={`24 layers × full · num_kv_heads = ${NUM_HEADS}`} />
      <StatTile
        label="GQA every layer"
        value={formatBytes(gqa)}
        sub={`24 layers × full · num_kv_heads = ${NUM_KV_HEADS}`}
      />
      <StatTile
        label="Hybrid (actual)"
        value={formatBytes(hybrid)}
        sub={`6 full + 18 linear · ${savingsVsGqa.toFixed(1)}× under GQA`}
        accent
      />
    </div>
  );
}

function StatTile({
  label,
  value,
  sub,
  accent = false,
}: {
  label: string;
  value: string;
  sub: string;
  accent?: boolean;
}) {
  return (
    <div
      className={[
        'rounded-md border p-2',
        accent ? 'border-primary/50 bg-primary/5' : 'border-border bg-muted/30',
      ].join(' ')}
    >
      <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="mt-1 font-mono text-sm text-foreground">{value}</div>
      <div className="text-[11px] text-muted-foreground">{sub}</div>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Layer strip.
// -----------------------------------------------------------------------------

function LayerStrip({ selectedLayer, onSelect }: { selectedLayer: number; onSelect: (i: number) => void }) {
  const width = 320;
  const height = 360;
  const padTop = 32;
  const padBottom = 18;
  const padLeftInside = 14;
  const padRightInside = 14;
  const innerH = height - padTop - padBottom;
  const innerW = width - padLeftInside - padRightInside;
  const barH = (innerH - (NUM_LAYERS - 1) * 2) / NUM_LAYERS;

  return (
    <div className="space-y-2 rounded-md border border-border bg-background p-3">
      <div className="flex items-baseline justify-between">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">Layer interleave · 24 layers</div>
        <div className="text-[11px] text-muted-foreground">F = full · L = linear (GatedDeltaNet)</div>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Vertical strip of 24 layer bars, every fourth one (indices 3, 7, 11, 15, 19, 23) coloured amber for full attention, the rest violet for GatedDeltaNet linear attention."
        className="mx-auto block h-auto w-full max-w-[380px]"
      >
        <text x={width / 2} y={18} fontSize={10} textAnchor="middle" fill="currentColor" fillOpacity={0.55}>
          input → output (top to bottom)
        </text>
        {Array.from({ length: NUM_LAYERS }).map((_, i) => {
          const isFull = FULL_LAYER_SET.has(i);
          const isSelected = i === selectedLayer;
          const y = padTop + i * (barH + 2);
          const fill = isFull ? '#f59e0b' : '#7c3aed';
          const labelColor = 'white';
          return (
            <g
              key={`layer-${i}`}
              role="button"
              tabIndex={0}
              aria-label={`Select layer ${i} (${isFull ? 'full attention' : 'linear attention'})`}
              aria-pressed={isSelected}
              onClick={() => onSelect(i)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onSelect(i);
                }
              }}
              style={{ cursor: 'pointer', outline: 'none' }}
            >
              <rect
                x={padLeftInside}
                y={y}
                width={innerW}
                height={barH}
                rx={4}
                fill={fill}
                opacity={isSelected ? 1 : 0.7}
              />
              {isSelected ? (
                <rect
                  x={padLeftInside - 2}
                  y={y - 2}
                  width={innerW + 4}
                  height={barH + 4}
                  rx={5}
                  fill="none"
                  className="stroke-foreground"
                  strokeWidth={1.6}
                />
              ) : null}
              <text
                x={padLeftInside + 8}
                y={y + barH / 2 + 3.5}
                fontSize={Math.min(10, barH - 2)}
                fill={labelColor}
                pointerEvents="none"
                style={{ fontFamily: 'var(--font-mono, monospace)' }}
              >
                {`#${i.toString().padStart(2, '0')}`}
              </text>
              <text
                x={padLeftInside + innerW - 10}
                y={y + barH / 2 + 3.5}
                fontSize={Math.min(10, barH - 2)}
                textAnchor="end"
                fill={labelColor}
                pointerEvents="none"
                style={{
                  fontFamily: 'var(--font-mono, monospace)',
                  fontWeight: 600,
                }}
              >
                {isFull ? 'F' : 'L'}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="text-[11px] text-muted-foreground">
        Full-attention layers at indices {FULL_LAYER_INDICES.join(', ')} — one in every {FULL_ATTENTION_INTERVAL}.
      </div>
    </div>
  );
}

function LayerSidePanel({ layerIdx, contextLen }: { layerIdx: number; contextLen: number }) {
  const isFull = FULL_LAYER_SET.has(layerIdx);
  const groupSize = NUM_HEADS / NUM_KV_HEADS;
  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex items-baseline justify-between">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Layer {layerIdx} · {isFull ? 'Full attention' : 'Linear (GatedDeltaNet)'}
        </div>
        <span
          className={[
            'rounded px-2 py-0.5 font-mono text-[10px]',
            isFull
              ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300'
              : 'bg-violet-500/15 text-violet-700 dark:text-violet-300',
          ].join(' ')}
        >
          {isFull ? 'F' : 'L'}
        </span>
      </div>

      {isFull ? (
        <div className="space-y-2 text-[12px] text-foreground/85">
          <p>Stores K and V for every token. Cache scales linearly with context length.</p>
          <pre className="overflow-x-auto rounded bg-muted p-2 font-mono text-[11px] leading-5">
            <code>{`shape = [2, kv_heads=${NUM_KV_HEADS}, seq_len, d=${HEAD_DIM}]
bytes = 2 · ${NUM_KV_HEADS} · seq_len · ${HEAD_DIM} · ${BYTES_PER_FLOAT}
      = ${fullLayerCacheBytes(1)} · seq_len`}</code>
          </pre>
          <dl className="grid grid-cols-2 gap-x-2 gap-y-1 font-mono text-[11px]">
            <dt className="text-muted-foreground">cache @ {formatTokens(contextLen)}</dt>
            <dd className="text-right text-foreground/90">{formatBytes(fullLayerCacheBytes(contextLen))}</dd>
            <dt className="text-muted-foreground">Q heads / KV head</dt>
            <dd className="text-right text-foreground/90">{groupSize}</dd>
            <dt className="text-muted-foreground">head_dim</dt>
            <dd className="text-right text-foreground/90">{HEAD_DIM}</dd>
            <dt className="text-muted-foreground">cache dtype</dt>
            <dd className="text-right text-foreground/90">bf16</dd>
          </dl>
        </div>
      ) : (
        <div className="space-y-2 text-[12px] text-foreground/85">
          <p>
            Compresses the entire history into a fixed-size recurrent state — independent of <code>seq_len</code>.
          </p>
          <pre className="overflow-x-auto rounded bg-muted p-2 font-mono text-[11px] leading-5">
            <code>{`shape = [heads=${LINEAR_NUM_HEADS}, d=${LINEAR_HEAD_DIM}, d=${LINEAR_HEAD_DIM}]
bytes = ${LINEAR_NUM_HEADS} · ${LINEAR_HEAD_DIM} · ${LINEAR_HEAD_DIM} · ${BYTES_PER_FLOAT}
      = ${LINEAR_STATE_SIZE_BYTES} bytes (constant)`}</code>
          </pre>
          <dl className="grid grid-cols-2 gap-x-2 gap-y-1 font-mono text-[11px]">
            <dt className="text-muted-foreground">state @ {formatTokens(contextLen)}</dt>
            <dd className="text-right text-foreground/90">{formatBytes(LINEAR_STATE_SIZE_BYTES)}</dd>
            <dt className="text-muted-foreground">linear heads</dt>
            <dd className="text-right text-foreground/90">{LINEAR_NUM_HEADS}</dd>
            <dt className="text-muted-foreground">linear head_dim</dt>
            <dd className="text-right text-foreground/90">{LINEAR_HEAD_DIM}</dd>
            <dt className="text-muted-foreground">state dtype</dt>
            <dd className="text-right text-foreground/90">bf16</dd>
          </dl>
        </div>
      )}

      <div className="rounded-md border border-border bg-muted/30 p-2 text-[11px] text-muted-foreground">
        {isFull
          ? `One full layer holds ${formatBytes(fullLayerCacheBytes(contextLen))} at this context. Multiply by ${NUM_FULL_LAYERS} for the model's full-attention contribution.`
          : `One linear layer holds ${formatBytes(LINEAR_STATE_SIZE_BYTES)} regardless of context. Multiply by ${NUM_LINEAR_LAYERS} for the model's linear contribution.`}
      </div>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Decode animation.
// -----------------------------------------------------------------------------

function DecodeAnimation({
  step,
  autoplay,
  onStep,
  onTogglePlay,
}: {
  step: number;
  autoplay: boolean;
  onStep: (next: number) => void;
  onTogglePlay: () => void;
}) {
  const prefillCount = DECODE_PROMPT_TOKENS.length;
  const isPrefill = step < prefillCount;
  const sequence = [
    ...DECODE_PROMPT_TOKENS.map((text, i) => ({
      text,
      kind: 'prompt' as const,
      idx: i,
    })),
    ...DECODE_GENERATED_TOKENS.map((text, i) => ({
      text,
      kind: 'gen' as const,
      idx: prefillCount + i,
    })),
  ];
  // Tokens cached so far: at the end of step t, positions 0..t are cached.
  const cachedThrough = Math.max(0, Math.min(step, sequence.length - 1));
  const activeIdx = cachedThrough; // The token being processed at this step.

  const width = 560;
  const slotW = 56;
  const slotH = 38;
  const padX = 12;
  const rowY = 70;
  const queryY = 18;
  const height = 150;

  function slotX(i: number) {
    return padX + i * slotW;
  }

  // Arc endpoints for the attention reach lines from active Q to cached K/V.
  const activeCx = slotX(activeIdx) + slotW / 2;
  const activeCy = queryY + 18;

  function arcTo(targetIdx: number): string {
    const tx = slotX(targetIdx) + slotW / 2;
    const ty = rowY;
    const dx = tx - activeCx;
    const midX = activeCx + dx / 2;
    const midY = (activeCy + ty) / 2 - Math.max(10, Math.abs(dx) * 0.18);
    return `M ${activeCx} ${activeCy} Q ${midX} ${midY} ${tx} ${ty}`;
  }

  return (
    <div className="space-y-2 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Decode animation · cache grows one token at a time
        </div>
        <div className="text-[11px] text-muted-foreground">
          {isPrefill
            ? `prefill step ${step + 1} / ${prefillCount}`
            : `decode step ${step - prefillCount + 1} / ${DECODE_GENERATED_TOKENS.length}`}
        </div>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Decode animation: a row of token slots where the active token's Q attends to all previously cached K and V positions."
        className="block h-auto w-full"
      >
        <text x={padX} y={queryY - 6} fontSize={10} fill="currentColor" fillOpacity={0.55}>
          ACTIVE Q
        </text>
        <text x={padX} y={rowY + slotH + 16} fontSize={10} fill="currentColor" fillOpacity={0.55}>
          CACHED K, V (per full-attention layer)
        </text>

        {sequence.map((tok, i) => {
          const x = slotX(i);
          const isCached = i <= cachedThrough;
          const isActive = i === activeIdx;
          const isGenerated = tok.kind === 'gen';
          const fill = isCached ? (isGenerated ? '#22c55e' : '#3b82f6') : 'transparent';
          const stroke = isCached ? 'transparent' : 'rgba(148,163,184,0.45)';
          return (
            <g key={`slot-${i}`}>
              {/* Active query bubble above the row. */}
              {isActive ? (
                <g>
                  <rect x={x + 6} y={queryY} width={slotW - 12} height={slotH - 14} rx={5} fill="#fb923c" />
                  <text
                    x={x + slotW / 2}
                    y={queryY + 16}
                    fontSize={11}
                    textAnchor="middle"
                    fill="white"
                    style={{ fontFamily: 'var(--font-mono, monospace)' }}
                  >
                    q
                  </text>
                </g>
              ) : null}

              <rect
                x={x + 2}
                y={rowY}
                width={slotW - 4}
                height={slotH}
                rx={5}
                fill={fill}
                stroke={stroke}
                strokeDasharray={isCached ? undefined : '3 3'}
                strokeWidth={1.2}
              />
              <text
                x={x + slotW / 2}
                y={rowY + slotH / 2 + 4}
                fontSize={11}
                textAnchor="middle"
                fill={isCached ? 'white' : 'currentColor'}
                fillOpacity={isCached ? 1 : 0.5}
                style={{ fontFamily: 'var(--font-mono, monospace)' }}
              >
                {tok.text.trim() === '' ? '·' : tok.text.trim()}
              </text>
              <text
                x={x + slotW / 2}
                y={rowY + slotH + 14}
                fontSize={9}
                textAnchor="middle"
                fill="currentColor"
                fillOpacity={0.5}
              >
                {i}
              </text>
            </g>
          );
        })}

        {/* Attention reach arcs: the active Q attends to every cached K/V (and itself). */}
        {Array.from({ length: cachedThrough + 1 }).map((_, i) => (
          <path key={`arc-${i}`} d={arcTo(i)} fill="none" stroke="#fb923c" strokeOpacity={0.5} strokeWidth={1.2} />
        ))}
      </svg>
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="outline" onClick={() => onStep(Math.max(0, step - 1))} disabled={step === 0}>
          Prev
        </Button>
        <Button size="sm" onClick={onTogglePlay}>
          {autoplay ? 'Pause' : step >= DECODE_TOTAL_STEPS - 1 ? 'Replay' : 'Play'}
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => onStep(Math.min(DECODE_TOTAL_STEPS - 1, step + 1))}
          disabled={step >= DECODE_TOTAL_STEPS - 1}
        >
          Next
        </Button>
        <span className="ml-2 text-[11px] text-muted-foreground">
          {isPrefill
            ? 'Prefill: K/V for every prompt token written in parallel.'
            : 'Decode: K/V for the new token appended; Q attends over the whole cache.'}
        </span>
      </div>
    </div>
  );
}
