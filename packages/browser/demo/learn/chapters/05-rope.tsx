import * as React from 'react';

import { DemoCallout } from '../inspector/DemoCallout';
import { Prose } from '../Prose';
import { ChapterFrame } from '../scaffolding/ChapterFrame';
import type { ChapterLearningData } from '../scaffolding/learning-data';
import { MathDisplay } from '../scaffolding/MathDisplay';

/**
 * Chapter 5 — Positional encoding (RoPE).
 *
 * JS-only: RoPE math is deterministic given the model's config, so this
 * chapter visualises the math directly without calling the worker. The
 * <RopeDemo /> widget renders three views: the frequency spectrum across
 * the rotated dimension pairs, the per-pair rotation as position advances,
 * and the rotated-dot-product curve that demonstrates the relative-position
 * property.
 */

// Qwen3.5-0.8B; see crates/mlx-core/src/models/qwen3_5/config.rs and the
// `text_config.rope_parameters` block in the model's config.json. RoPE only
// rotates the first `head_dim * partial_rotary_factor` = 64 features of each
// 256-dim head, which is 32 pairs.
const HEAD_DIM = 256;
const PARTIAL_ROTARY_FACTOR = 0.25;
const ROPE_DIMS = HEAD_DIM * PARTIAL_ROTARY_FACTOR; // 64 rotated dims
const NUM_PAIRS = ROPE_DIMS / 2; // 32 pairs
const ROPE_THETA = 10_000_000;
const MAX_POSITION = 262_144;

const DEFAULT_SELECTED_PAIR = 4;
const ROTATION_POSITION_MAX = 100;
const DOT_POSITION_MAX = 200;
const DOT_QUERY_POS = 50;
const LOW_FREQ_PAIR = NUM_PAIRS - 4; // index 28 — slow
const HIGH_FREQ_PAIR = 0; // index 0 — fast

/** RoPE pair frequency: `theta_i = base ^ (-2i / rope_dims)`. */
function pairFrequency(i: number, ropeDims: number, base: number): number {
  return base ** ((-2 * i) / ropeDims);
}

/**
 * Rotated dot product of two RoPE'd vectors at positions `qPos` and `kPos`.
 *
 * For q = k = (1, 1, ..., 1) the per-pair contribution reduces to
 *   (q0 q0 + q1 q1) * cos((kPos - qPos) * theta_i) = 2 cos(delta * theta_i)
 * which is what the relative-position property predicts.
 */
function ropeDotProductOnes(qPos: number, kPos: number, ropeDims: number, base: number): number {
  const pairs = ropeDims / 2;
  const delta = kPos - qPos;
  let sum = 0;
  for (let i = 0; i < pairs; i++) {
    sum += 2 * Math.cos(delta * pairFrequency(i, ropeDims, base));
  }
  return sum;
}

function formatFreq(value: number): string {
  if (!Number.isFinite(value)) return '—';
  if (value >= 1) return value.toFixed(3);
  if (value >= 1e-3) return value.toExponential(2);
  return value.toExponential(2);
}

function formatPeriod(value: number): string {
  if (!Number.isFinite(value)) return '—';
  if (value < 100) return value.toFixed(1);
  if (value < 1e4) return value.toFixed(0);
  return value.toExponential(2);
}

/**
 * Scaffolding metadata for chapter 5 — drives the header, glossary,
 * takeaways, exercise, and quick-check rendered by `<ChapterFrame>`.
 * `chapterId` must match `CHAPTERS[4].id` in `learn/chapters.ts`.
 */
export const learning: ChapterLearningData = {
  chapterId: 'rope',
  objective:
    'Explain how RoPE injects token order into attention via per-pair rotations, without learned position embeddings.',
  problem:
    "Self-attention is permutation-invariant — without a positional signal the model cannot tell 'cat sat on mat' from 'mat sat on cat'.",
  minutes: 7,
  glossary: [
    {
      term: 'RoPE',
      definition:
        'Rotary positional embedding. Rotates each (x_2i, x_2i+1) pair of Q and K by an angle m * theta_i before the dot product.',
    },
    {
      term: 'pair frequency (theta_i)',
      definition:
        'Per-dimension rotation rate: theta_i = base^(-2i / rope_dims). Low i rotates fast, high i rotates slowly.',
    },
    {
      term: 'rope_theta (base)',
      definition:
        'The base of the frequency exponent. Qwen3.5 uses 1e7, far higher than the original 1e4 — stretches the spectrum for long context.',
    },
    {
      term: 'partial rotary factor',
      definition:
        "Fraction of each head's dims that RoPE rotates. Qwen3.5 uses 0.25, so 64 of 256 head dims are rotated; the rest pass through.",
    },
    {
      term: 'relative-position property',
      definition:
        'The dot product of RoPE(q,m) and RoPE(k,n) depends only on m-n, so the model only has to learn relative offsets.',
    },
    {
      term: 'unitary rotation',
      definition: 'A rotation preserves vector norms — RoPE changes where vectors point, never how big they are.',
    },
  ],
  takeaways: [
    'RoPE encodes position as rotation, not addition — magnitudes stay constant and only angles carry the position signal.',
    'Low-index pairs are high-frequency (fine local position); high-index pairs are low-frequency (coarse global position).',
    'The dot product after RoPE depends only on m - n, which is why models with RoPE extrapolate to lengths they never saw in training.',
  ],
  exercise: {
    prompt:
      "Drag the 'Token position m' slider from 0 up to 100. Compare the high-frequency rotation panel (pair 0) with the low-frequency one (pair 28). Which one completes a full turn first, and by how much has the other moved when m=100?",
    answer:
      "Pair 0 spins through many full turns by m=100 (its frequency is roughly 1 rad/token in Qwen3.5's spectrum). Pair 28 has barely moved — its frequency is so small the hand is still close to 3 o'clock. That gap is the entire point of using a spectrum of frequencies: short-range vs long-range position in one operation.",
  },
  quiz: [
    {
      id: 'q1-why-rotate',
      prompt: 'What problem does RoPE solve that pure self-attention cannot?',
      options: [
        { id: 'a', label: 'Self-attention output magnitudes are too large.' },
        {
          id: 'b',
          label:
            'Self-attention is permutation-invariant — it has no notion of token order until something injects it.',
        },
        {
          id: 'c',
          label: 'Self-attention cannot represent negative values.',
        },
      ],
      correctId: 'b',
      explanation:
        'Attention computes the same thing if you shuffle inputs (with outputs shuffled to match). Position has to come from somewhere — RoPE injects it into Q and K.',
    },
    {
      id: 'q2-low-vs-high',
      prompt: "Which dimension pairs end up encoding 'coarse global position' under RoPE?",
      options: [
        {
          id: 'a',
          label:
            'High-index pairs — they have the lowest frequencies and rotate slowly enough to remain distinguishable over long contexts.',
        },
        {
          id: 'b',
          label: 'Low-index pairs, because they have the highest frequencies.',
        },
        {
          id: 'c',
          label: "It's random — RoPE assigns roles per training run.",
        },
      ],
      correctId: 'a',
      explanation:
        "theta_i = base^(-2i/d), so large i means tiny theta_i. Those pairs barely rotate even across tens of thousands of tokens, which is what 'coarse global position' looks like.",
    },
    {
      id: 'q3-relative-pos',
      prompt: 'Why is it called the relative-position property?',
      options: [
        {
          id: 'a',
          label:
            "The dot product of RoPE'd Q and K depends only on the offset (m - n), not on the absolute positions m and n.",
        },
        {
          id: 'b',
          label: 'RoPE positions are stored relative to a chosen anchor token.',
        },
        {
          id: 'c',
          label: 'The rotations always relate two adjacent tokens.',
        },
      ],
      correctId: 'a',
      explanation:
        "Once you do the algebra, the position-dependence collapses to m - n. That's why a model trained at 4k tokens can extrapolate to much longer contexts.",
    },
  ],
};

export function RopeChapterBody() {
  return (
    <ChapterFrame learning={learning}>
      <Prose>
        <h1>Positional encoding: how the model knows token order</h1>
        <p>
          Self-attention has a strange property: it is <strong>permutation-invariant</strong>. If you shuffle the input
          tokens, the attention output for each token shuffles right along with them, but the relationships the
          attention layer computes don't change. Without help, a transformer literally cannot tell{' '}
          <em>"the cat sat on the mat"</em> from <em>"mat the on sat cat the"</em>. Something has to inject the order
          back in.
        </p>

        <h2>Two earlier ideas</h2>
        <ul>
          <li>
            <strong>Learned positional embeddings.</strong> Early GPT, BERT, and friends gave every position 0..N-1 its
            own learned vector and
            <em>added</em> it to the token embedding. Simple, but the model can't extrapolate to positions it never saw
            during training, and you have to pick a max length up front.
          </li>
          <li>
            <strong>Sinusoidal embeddings.</strong> The original 2017 transformer used fixed sinusoids of different
            frequencies, also added to the token embedding. Extrapolation is in principle possible, but in practice
            quality degrades past the training window.
          </li>
        </ul>

        <h2>RoPE's idea: rotate, don't add</h2>
        <p>
          <strong>Rotary Positional Embedding</strong> (RoPE) leaves the embeddings alone and instead bakes the position
          directly into the attention math. The trick is to <strong>rotate</strong> the query and key vectors by an
          angle that depends on the token's position,
          <em>before</em> the dot product is computed.
        </p>
        <p>
          Concretely, RoPE pairs up adjacent feature dimensions — <code>(x_0, x_1)</code>, <code>(x_2, x_3)</code>, … —
          and treats each pair as a point in a 2D plane. For pair index <code>i</code> and head dimension <code>d</code>
          , the frequency is:
        </p>
        <MathDisplay latex={String.raw`\theta_i = \text{base}^{-2i / d}`} />
        <p>
          At token position <code>m</code>, the pair is rotated by an angle of{' '}
          <MathDisplay latex={String.raw`m \cdot \theta_i`} inline />:
        </p>
        <MathDisplay
          latex={String.raw`\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}`}
        />
        <p>
          Rotation is <strong>unitary</strong> — it preserves vector norms — so RoPE doesn't change how big Q and K are,
          only where they point. The position information rides on the angle, not the magnitude.
        </p>

        <h2>Wide range of frequencies</h2>
        <p>
          Low-index pairs (small <code>i</code>) get the highest frequencies and rotate fast — one full turn every few
          tokens. High-index pairs get vanishingly small frequencies and barely move even over tens of thousands of
          tokens. The intuition: high-frequency pairs encode
          <strong> fine local position</strong> ("am I 3 tokens from my neighbour?"), low-frequency pairs encode{' '}
          <strong>coarse global position</strong> ("am I in the early part of the document or the late part?").
        </p>
        <p>
          Qwen3.5 uses a head dimension of <code>{HEAD_DIM}</code>, a partial rotary factor of{' '}
          <code>{PARTIAL_ROTARY_FACTOR}</code> (so only the first <code>{ROPE_DIMS}</code> features per head are rotated
          — the rest pass through untouched), and an unusually large RoPE base of{' '}
          <code>{ROPE_THETA.toExponential(0)}</code>. The large base stretches the frequency spectrum so the
          lowest-frequency pairs barely rotate at all over the model's <code>{MAX_POSITION.toLocaleString()}</code>
          -token context window — a key ingredient for long-context extrapolation.
        </p>

        <h2>The relative-position property</h2>
        <p>
          Here is the part that makes RoPE click. If you take the dot product of <code>RoPE(q, m)</code> with{' '}
          <code>RoPE(k, n)</code>, the answer depends <em>only on the difference</em> <code>m - n</code>, not on{' '}
          <code>m</code> and <code>n</code> separately. The model never has to learn what "position 1,427" means — it
          only has to learn how attention should behave at <em>relative</em> offsets, which generalises naturally to
          lengths it has never seen.
        </p>
        <p>
          The third panel on the right makes this concrete. Holding the query position fixed at{' '}
          <code>m = {DOT_QUERY_POS}</code> and varying the key position <code>n</code>, the rotated dot product peaks
          sharply at <code>n - m = 0</code> and decays as you move further apart. That decay is exactly the inductive
          bias that makes attention "want" nearby tokens more than distant ones, with no per-position parameters at all.
        </p>

        <p className="mt-6 text-muted-foreground">
          This chapter visualises RoPE's math directly — no model inference needed. Want to see how this plays out on
          real attention scores? Open <em>Self-attention</em> (Chapter 3) and look at how the score map favours the
          diagonal.
        </p>
      </Prose>
    </ChapterFrame>
  );
}

export type RopeDemoProps = {
  // Accepted for API parity with the other chapter Try-It panels even
  // though this chapter never touches the worker. Letting the parent pass
  // them anyway keeps the wiring in app.tsx uniform.
  workerRef?: React.RefObject<Worker | null>;
  abortRef?: React.RefObject<AbortController | null>;
};

export function RopeDemo(_props: RopeDemoProps) {
  const [selectedPair, setSelectedPair] = React.useState(DEFAULT_SELECTED_PAIR);
  const [rotationPos, setRotationPos] = React.useState(8);
  // "Sweep position m" animation — drives `rotationPos` from 0 → MAX over
  // ~6 s on click. The whole RoPE punchline ("pair 0 spins fast, pair 28
  // barely moves") is invisible until the position changes, and most
  // learners won't drag the slider on their own. The sweep does it for
  // them, and the wedge trail in each RotationPanel records how much
  // angular distance each pair has covered. Bumping `sweepKey` re-fires
  // the effect, restarting the animation from 0.
  const [sweepKey, setSweepKey] = React.useState(0);
  const [isSweeping, setIsSweeping] = React.useState(false);

  React.useEffect(() => {
    if (sweepKey === 0) return;

    const reducedMotion =
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reducedMotion) {
      // Snap to the endpoint — the trails still tell the contrast story
      // statically (full disk vs. invisible sliver).
      setRotationPos(ROTATION_POSITION_MAX);
      setIsSweeping(false);
      return;
    }

    setIsSweeping(true);
    setRotationPos(0);
    const durationMs = 6000;
    let rafId: number | null = null;
    let startTime: number | null = null;
    const step = (t: number) => {
      if (startTime === null) startTime = t;
      const elapsed = t - startTime;
      const progress = Math.min(1, elapsed / durationMs);
      // Ease-out cubic — fast initial ramp so the high-freq clock starts
      // visibly spinning right away, gentle landing near m = MAX.
      const eased = 1 - Math.pow(1 - progress, 3);
      setRotationPos(eased * ROTATION_POSITION_MAX);
      if (progress < 1) {
        rafId = requestAnimationFrame(step);
      } else {
        setIsSweeping(false);
      }
    };
    rafId = requestAnimationFrame(step);

    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      setIsSweeping(false);
    };
  }, [sweepKey]);

  const frequencies = React.useMemo(() => {
    const out = new Float64Array(NUM_PAIRS);
    for (let i = 0; i < NUM_PAIRS; i++) {
      out[i] = pairFrequency(i, ROPE_DIMS, ROPE_THETA);
    }
    return out;
  }, []);

  const dotCurve = React.useMemo(() => {
    const out = new Float64Array(DOT_POSITION_MAX + 1);
    for (let n = 0; n <= DOT_POSITION_MAX; n++) {
      out[n] = ropeDotProductOnes(DOT_QUERY_POS, n, ROPE_DIMS, ROPE_THETA);
    }
    return out;
  }, []);

  const selectedFreq = frequencies[selectedPair] ?? 0;
  const selectedPeriod = selectedFreq > 0 ? (2 * Math.PI) / selectedFreq : Infinity;

  return (
    <div className="space-y-4">
      <div className="rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
        This chapter visualises RoPE's math directly — no model inference needed. The numbers below come from
        Qwen3.5-0.8B's config: head_dim={HEAD_DIM}, rotated dims={ROPE_DIMS} ({NUM_PAIRS} pairs), rope_theta=
        {ROPE_THETA.toExponential(0)}.
      </div>

      <FrequencySpectrum frequencies={frequencies} selectedPair={selectedPair} onSelect={setSelectedPair} />

      <div className="rounded-md bg-muted/40 px-3 py-2 font-mono text-[11px] text-muted-foreground">
        Pair {selectedPair} · freq = {formatFreq(selectedFreq)} rad/token · period ≈ {formatPeriod(selectedPeriod)}{' '}
        tokens
      </div>

      <SliderRow
        id="rope-demo-pair"
        label="Selected pair"
        value={selectedPair}
        min={0}
        max={NUM_PAIRS - 1}
        step={1}
        onChange={(v) => setSelectedPair(Math.round(v))}
        format={(v) => `i = ${v}`}
      />

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <RotationPanel
          title={`High frequency (pair ${HIGH_FREQ_PAIR})`}
          subtitle="fast — fine local position"
          pairIndex={HIGH_FREQ_PAIR}
          position={rotationPos}
        />
        <RotationPanel
          title={`Low frequency (pair ${LOW_FREQ_PAIR})`}
          subtitle="slow — coarse global position"
          pairIndex={LOW_FREQ_PAIR}
          position={rotationPos}
        />
      </div>

      <div className="space-y-2">
        <SliderRow
          id="rope-demo-position"
          label="Token position m"
          value={rotationPos}
          min={0}
          max={ROTATION_POSITION_MAX}
          step={1}
          onChange={(v) => {
            // Manual drag interrupts any in-flight sweep — bump sweepKey
            // would re-fire, so we just write the value and let the rAF
            // cleanup happen on unmount or next sweep.
            setRotationPos(Math.round(v));
          }}
          format={(v) => `m = ${Math.round(v)}`}
        />
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setSweepKey((k) => k + 1)}
            className="inline-flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1 text-xs font-medium text-foreground hover:bg-primary/20 disabled:opacity-50"
          >
            <span aria-hidden="true">▶</span>
            {sweepKey === 0 ? 'Sweep position m' : isSweeping ? 'Sweep running…' : 'Replay sweep'}
          </button>
          <span className="text-[11px] text-muted-foreground">
            Auto-scrubs m from 0 → {ROTATION_POSITION_MAX}. Watch pair 0 spin wildly while pair {LOW_FREQ_PAIR} barely
            twitches — that contrast is what RoPE encodes.
          </span>
        </div>
      </div>

      <DotProductChart values={dotCurve} queryPos={DOT_QUERY_POS} />

      <DemoCallout
        items={[
          'Each pair of dimensions rotates at its own frequency — high-index pairs rotate slowly, low-index pairs rotate fast.',
          'Drag the position slider: pair 0 sweeps through many rotations, pair 28 barely moves.',
          'Relative position emerges from the rotation: token at pos 5 and key at pos 3 see the same relative angle as pos 10 → 8.',
        ]}
      />
    </div>
  );
}

function FrequencySpectrum({
  frequencies,
  selectedPair,
  onSelect,
}: {
  frequencies: Float64Array;
  selectedPair: number;
  onSelect: (i: number) => void;
}) {
  const width = 520;
  const height = 160;
  const padLeft = 36;
  const padRight = 8;
  const padTop = 12;
  const padBottom = 24;
  const innerW = width - padLeft - padRight;
  const innerH = height - padTop - padBottom;

  // Log scale across the full spread of frequencies.
  const logMin = Math.log10(Math.min(...frequencies));
  const logMax = Math.log10(Math.max(...frequencies));
  const logSpan = logMax - logMin || 1;

  const barW = innerW / NUM_PAIRS;

  function yForFreq(f: number): number {
    const norm = (Math.log10(f) - logMin) / logSpan;
    return padTop + innerH - norm * innerH;
  }

  const ticks = [logMin, logMin + logSpan / 2, logMax];

  return (
    <div className="rounded-md border border-border bg-background p-3">
      <div className="mb-2 text-xs uppercase tracking-wider text-muted-foreground">Pair frequencies (log scale)</div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="RoPE pair frequencies on a log scale, one bar per dimension pair"
        className="block h-auto w-full"
      >
        {ticks.map((t, idx) => {
          const y = padTop + innerH - ((t - logMin) / logSpan) * innerH;
          return (
            <g key={idx}>
              <line x1={padLeft} x2={width - padRight} y1={y} y2={y} stroke="currentColor" strokeOpacity={0.08} />
              <text x={padLeft - 4} y={y + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.5}>
                {`1e${t.toFixed(0)}`}
              </text>
            </g>
          );
        })}
        {Array.from(frequencies).map((f, i) => {
          const x = padLeft + i * barW;
          const y = yForFreq(f);
          const isSelected = i === selectedPair;
          return (
            <rect
              key={i}
              x={x + 0.5}
              y={y}
              width={Math.max(1, barW - 1)}
              height={padTop + innerH - y}
              className={isSelected ? 'fill-primary' : 'fill-foreground/40 hover:fill-foreground/70'}
              onClick={() => onSelect(i)}
              style={{ cursor: 'pointer' }}
            >
              <title>{`pair ${i}: ${formatFreq(f)} rad/token`}</title>
            </rect>
          );
        })}
        <text x={padLeft} y={height - 6} fontSize={9} fill="currentColor" fillOpacity={0.5}>
          i = 0
        </text>
        <text x={width - padRight} y={height - 6} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.5}>
          i = {NUM_PAIRS - 1}
        </text>
      </svg>
      <div className="pt-1 text-[11px] text-muted-foreground">
        Each bar is one dimension pair. Click or use the slider below to select a pair and update the rotation panels.
      </div>
    </div>
  );
}

function RotationPanel({
  title,
  subtitle,
  pairIndex,
  position,
}: {
  title: string;
  subtitle: string;
  pairIndex: number;
  position: number;
}) {
  const freq = pairFrequency(pairIndex, ROPE_DIMS, ROPE_THETA);
  const angle = position * freq;
  // Wrap to [0, 2π) for the trail; keep the absolute angle for the readout.
  const wrapped = ((angle % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  const size = 160;
  const cx = size / 2;
  const cy = size / 2;
  const r = size * 0.4;

  const handX = cx + r * Math.cos(wrapped);
  const handY = cy - r * Math.sin(wrapped);

  // Trail = one translucent dot on the rim for every integer position m the
  // hand has visited so far. Overlapping translucent dots create a natural
  // density heatmap: a high-frequency pair sweeps the rim and the dots
  // scatter all around it; a low-frequency pair barely moves and the dots
  // stack on top of each other near 3 o'clock — that clumping IS the lesson.
  const visited = Math.max(0, Math.floor(position));
  const trailDots: React.ReactElement[] = [];
  for (let m = 0; m <= visited; m++) {
    const a = m * freq;
    const wm = ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    const dx = cx + r * Math.cos(wm);
    const dy = cy - r * Math.sin(wm);
    trailDots.push(<circle key={m} cx={dx} cy={dy} r={1.6} className="fill-primary" fillOpacity={0.28} />);
  }

  const turns = angle / (2 * Math.PI);

  return (
    <div className="rounded-md border border-border bg-background p-3">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{title}</div>
      <div className="text-[11px] text-muted-foreground">{subtitle}</div>
      <svg
        viewBox={`0 0 ${size} ${size}`}
        role="img"
        aria-label={`Unit-circle rotation for ${title} at position ${position}`}
        className="mx-auto block h-auto w-full max-w-[200px]"
      >
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="currentColor" strokeOpacity={0.2} />
        <line x1={cx - r - 4} x2={cx + r + 4} y1={cy} y2={cy} stroke="currentColor" strokeOpacity={0.1} />
        <line x1={cx} x2={cx} y1={cy - r - 4} y2={cy + r + 4} stroke="currentColor" strokeOpacity={0.1} />
        {trailDots}
        <line x1={cx} y1={cy} x2={handX} y2={handY} className="stroke-primary" strokeWidth={2} />
        <circle cx={handX} cy={handY} r={3.5} className="fill-primary" />
        <circle cx={cx + r} cy={cy} r={2.2} fill="none" stroke="currentColor" strokeOpacity={0.45} strokeWidth={1} />
      </svg>
      <div className="mt-1 font-mono text-[11px] text-muted-foreground">
        freq {formatFreq(freq)} · angle {wrapped.toFixed(3)} rad ·{' '}
        {turns < 0.01 ? `${(turns * 360).toFixed(2)}°` : `${turns.toFixed(2)} turns`}
      </div>
    </div>
  );
}

function DotProductChart({ values, queryPos }: { values: Float64Array; queryPos: number }) {
  const width = 520;
  const height = 160;
  const padLeft = 36;
  const padRight = 8;
  const padTop = 12;
  const padBottom = 28;
  const innerW = width - padLeft - padRight;
  const innerH = height - padTop - padBottom;

  const finite = Array.from(values).filter(Number.isFinite);
  const minV = Math.min(...finite);
  const maxV = Math.max(...finite);
  // Pad slightly so the peak doesn't sit flush against the top.
  const span = maxV - minV || 1;
  const yMin = minV - span * 0.05;
  const yMax = maxV + span * 0.05;

  function xFor(n: number): number {
    return padLeft + (n / (values.length - 1)) * innerW;
  }
  function yFor(v: number): number {
    return padTop + innerH - ((v - yMin) / (yMax - yMin)) * innerH;
  }

  const path =
    'M ' +
    Array.from(values)
      .map((v, n) => `${xFor(n).toFixed(2)} ${yFor(v).toFixed(2)}`)
      .join(' L ');

  const queryX = xFor(queryPos);
  const zeroY = yFor(0);
  const showZero = 0 >= yMin && 0 <= yMax;

  return (
    <div className="rounded-md border border-border bg-background p-3">
      <div className="mb-2 text-xs uppercase tracking-wider text-muted-foreground">
        Rotated dot product · q at m={queryPos}, k at n=0…{values.length - 1}
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Rotated dot product against the offset between query and key position"
        className="block h-auto w-full"
      >
        {showZero ? (
          <line x1={padLeft} x2={width - padRight} y1={zeroY} y2={zeroY} stroke="currentColor" strokeOpacity={0.1} />
        ) : null}
        <line
          x1={queryX}
          x2={queryX}
          y1={padTop}
          y2={padTop + innerH}
          stroke="currentColor"
          strokeOpacity={0.2}
          strokeDasharray="3 3"
        />
        <text x={queryX + 3} y={padTop + 9} fontSize={9} fill="currentColor" fillOpacity={0.5}>
          n = m ({queryPos})
        </text>
        <path d={path} fill="none" className="stroke-primary" strokeWidth={1.5} />
        <text x={padLeft} y={height - 8} fontSize={9} fill="currentColor" fillOpacity={0.5}>
          n = 0
        </text>
        <text x={width - padRight} y={height - 8} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.5}>
          n = {values.length - 1}
        </text>
        <text x={padLeft - 4} y={yFor(maxV) + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.5}>
          {maxV.toFixed(0)}
        </text>
        <text x={padLeft - 4} y={yFor(yMin) + 3} fontSize={9} textAnchor="end" fill="currentColor" fillOpacity={0.5}>
          {yMin.toFixed(0)}
        </text>
      </svg>
      <div className="pt-1 text-[11px] text-muted-foreground">
        With a constant query and key, the rotated dot product depends only on (n − m). The peak at n = m and smooth
        decay either side is the relative-position property of RoPE.
      </div>
    </div>
  );
}

function SliderRow({
  id,
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <label htmlFor={id} className="uppercase tracking-wider">
          {label}
        </label>
        <span className="font-mono text-foreground/80">{format(value)}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number.parseFloat(e.target.value))}
        className="h-2 w-full cursor-pointer appearance-none rounded-full bg-muted accent-primary"
      />
    </div>
  );
}
