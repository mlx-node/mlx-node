import * as React from 'react';

/**
 * Chapter 9 (LM head) supplement — show that for Qwen3.5 (and most modern
 * decoder-only LLMs) the embedding matrix and the LM head are the *same*
 * tensor, just transposed.
 *
 *   embed_tokens.weight : [V, d]  — bottom of the stack, "id → vector"
 *   lm_head.weight      : [V, d]  — top of the stack, "vector → vocab scores"
 *                                    (we use its transpose: [d, V])
 *
 * Verified for Qwen3.5: in `crates/mlx-core/src/models/qwen3_5/model.rs`,
 * when `tie_word_embeddings=true`, the model never allocates a separate
 * `lm_head` — it uses `embed_tokens.weight.T` directly for the final matmul.
 *
 * The widget animates a token particle traveling through the model: it gets
 * looked up via `embed_tokens.weight` at the top, flows through the 24-layer
 * stack, then gets projected back to a vocab score via the *same matrix*
 * (now `lm_head`) at the bottom. The dashed "tied — same tensor" arc lights
 * up whenever either matrix is "active" to drive home that both are the same
 * floats. Loop, play/pause.
 */

const STEPS = [
  'embedding lookup — read row of embed_tokens.weight',
  'flow through 24 transformer layers (residual stream)',
  'final RMSNorm + LM head — project against the SAME matrix',
  'top-K vocab scores — model predicts " mat"',
] as const;

export function WeightTyingVisual() {
  const [step, setStep] = React.useState(0);
  const [playing, setPlaying] = React.useState(true);

  React.useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => {
      setStep((s) => (s + 1) % STEPS.length);
    }, 1800);
    return () => window.clearInterval(t);
  }, [playing]);

  const W = 540;
  const H = 320;

  const matW = 120;
  const matH = 38;

  const topMatX = 50;
  const topMatY = 30;

  const botMatX = W - 50 - matW;
  const botMatY = H - 30 - matH;

  // Vertical "stack" between them suggests the decoder layers run vertically
  // between embedding (top of input) and LM head (bottom of stack to logits).
  const stackX = W / 2 - 22;
  const stackY = 90;
  const stackW = 44;
  const stackH = 140;

  // Particle position: lerp through 4 anchor points keyed off `step`.
  // step 0 → entering top matrix.   pos ≈ topMatX + matW/2, topMatY + matH/2
  // step 1 → halfway through stack. pos ≈ stackX + stackW/2, stackY + stackH/2
  // step 2 → at bottom matrix.       pos ≈ botMatX + matW/2, botMatY + matH/2
  // step 3 → past bottom matrix (output).
  const ANCHORS: Array<{ x: number; y: number; label: string }> = [
    { x: topMatX + matW / 2, y: topMatY + matH / 2, label: '"the"' },
    { x: stackX + stackW / 2, y: stackY + stackH / 2, label: 'h₀ … h₂₃' },
    { x: botMatX + matW / 2, y: botMatY + matH / 2, label: 'h_last' },
    { x: botMatX + matW + 30, y: botMatY + matH / 2, label: '" mat"' },
  ];
  const particle = ANCHORS[step]!;
  const topActive = step === 0;
  const botActive = step === 2;
  const arcLit = topActive || botActive;

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Weight tying — one matrix, used twice
        </div>
        <div className="inline-flex items-center gap-2">
          <button
            type="button"
            onClick={() => setPlaying((p) => !p)}
            className="rounded border border-border/60 bg-muted/40 px-2 py-0.5 text-[11px] hover:bg-muted/70"
            aria-pressed={playing}
            aria-label={playing ? 'Pause weight-tying animation' : 'Play weight-tying animation'}
          >
            {playing ? 'Pause' : 'Play'}
          </button>
          <span className="font-mono text-[11px] text-muted-foreground">
            step {step + 1}/{STEPS.length}
          </span>
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">
        Qwen3.5-0.8B (and most modern decoder LLMs) sets <span className="font-mono">tie_word_embeddings = true</span>.
        That means the embedding matrix at the input and the LM head at the output are{' '}
        <em>literally the same tensor</em> in memory — the same <span className="font-mono">[248,320 × 1024]</span> grid
        of floats, used once for <span className="font-mono">id → vector</span> and once (transposed) for{' '}
        <span className="font-mono">vector → vocab scores</span>.
      </p>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="block h-auto w-full"
        role="img"
        aria-label="Weight tying animation showing one tensor used at both ends of the model"
      >
        {/* Stack of decoder layers in the middle */}
        <rect
          x={stackX}
          y={stackY}
          width={stackW}
          height={stackH}
          fill="currentColor"
          fillOpacity={0.05}
          stroke="currentColor"
          strokeOpacity={0.3}
        />
        {Array.from({ length: 6 }, (_, i) => (
          <line
            key={`l-${i}`}
            x1={stackX}
            y1={stackY + (stackH / 6) * (i + 1)}
            x2={stackX + stackW}
            y2={stackY + (stackH / 6) * (i + 1)}
            stroke="currentColor"
            strokeOpacity={0.18}
            strokeDasharray="2 3"
          />
        ))}
        <text
          x={stackX + stackW / 2}
          y={stackY + stackH / 2 + 4}
          fontSize={9}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.55}
          fontFamily="monospace"
        >
          24 layers
        </text>

        {/* Top matrix glyph — embedding lookup */}
        <rect
          x={topMatX}
          y={topMatY}
          width={matW}
          height={matH}
          fill="oklch(0.65 0.13 250)"
          fillOpacity={topActive ? 0.45 : 0.18}
          stroke="oklch(0.65 0.13 250)"
          strokeOpacity={topActive ? 1 : 0.6}
          strokeWidth={topActive ? 2 : 1}
          rx={3}
          style={{ transition: 'all 400ms ease-out' }}
        />
        {Array.from({ length: 12 }, (_, i) =>
          Array.from({ length: 4 }, (_, j) => (
            <rect
              key={`tg-${i}-${j}`}
              x={topMatX + 4 + i * 9.5}
              y={topMatY + 4 + j * 8}
              width={7}
              height={6}
              fill="oklch(0.65 0.13 250)"
              fillOpacity={(topActive ? 0.55 : 0.25) + 0.3 * Math.abs(Math.sin(i * 1.3 + j * 0.7))}
              style={{ transition: 'fill-opacity 400ms' }}
            />
          )),
        )}
        <text x={topMatX + matW / 2} y={topMatY - 4} fontSize={10} textAnchor="middle" fill="oklch(0.65 0.13 250)">
          embed_tokens.weight
        </text>
        <text
          x={topMatX + matW / 2}
          y={topMatY + matH + 12}
          fontSize={9}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.55}
        >
          token id → vector (lookup)
        </text>

        {/* Bottom matrix glyph — LM head (transpose of same matrix) */}
        <rect
          x={botMatX}
          y={botMatY}
          width={matW}
          height={matH}
          fill="oklch(0.7 0.15 60)"
          fillOpacity={botActive ? 0.45 : 0.18}
          stroke="oklch(0.7 0.15 60)"
          strokeOpacity={botActive ? 1 : 0.6}
          strokeWidth={botActive ? 2 : 1}
          rx={3}
          style={{ transition: 'all 400ms ease-out' }}
        />
        {Array.from({ length: 12 }, (_, i) =>
          Array.from({ length: 4 }, (_, j) => (
            <rect
              key={`bg-${i}-${j}`}
              x={botMatX + 4 + i * 9.5}
              y={botMatY + 4 + j * 8}
              width={7}
              height={6}
              fill="oklch(0.7 0.15 60)"
              fillOpacity={(botActive ? 0.55 : 0.25) + 0.3 * Math.abs(Math.sin(i * 1.3 + j * 0.7))}
              style={{ transition: 'fill-opacity 400ms' }}
            />
          )),
        )}
        <text x={botMatX + matW / 2} y={botMatY - 4} fontSize={10} textAnchor="middle" fill="oklch(0.7 0.15 60)">
          lm_head.weight (= embed_tokens.weight)
        </text>
        <text
          x={botMatX + matW / 2}
          y={botMatY + matH + 12}
          fontSize={9}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.55}
        >
          vector → vocab scores
        </text>

        {/* Tied-weights arc connecting the two matrices — lights up when either matrix is "active" */}
        <path
          d={`M ${topMatX + matW / 2} ${topMatY + matH / 2} C ${15} ${H / 2}, ${15} ${H / 2}, ${botMatX + matW / 2} ${botMatY + matH / 2}`}
          fill="none"
          stroke="oklch(0.7 0.18 25)"
          strokeWidth={arcLit ? 2 : 1.2}
          strokeDasharray="5 4"
          strokeOpacity={arcLit ? 0.95 : 0.55}
          style={{ transition: 'stroke-opacity 400ms, stroke-width 400ms' }}
        />
        <text
          x={28}
          y={H / 2 - 6}
          fontSize={10}
          fill="oklch(0.7 0.18 25)"
          fillOpacity={arcLit ? 1 : 0.7}
          transform={`rotate(-90, 28, ${H / 2 - 6})`}
        >
          tied — same tensor
        </text>

        {/* The traveling particle — small dot that pulses at the active step */}
        <circle
          cx={particle.x}
          cy={particle.y}
          r={9}
          fill="oklch(0.8 0.15 25)"
          fillOpacity={0.18}
          style={{ transition: 'cx 900ms cubic-bezier(0.4, 0, 0.2, 1), cy 900ms cubic-bezier(0.4, 0, 0.2, 1)' }}
        />
        <circle
          cx={particle.x}
          cy={particle.y}
          r={5}
          fill="oklch(0.7 0.18 25)"
          stroke="oklch(0.85 0.18 25)"
          strokeWidth={1.2}
          style={{ transition: 'cx 900ms cubic-bezier(0.4, 0, 0.2, 1), cy 900ms cubic-bezier(0.4, 0, 0.2, 1)' }}
        />
        <text
          x={particle.x}
          y={particle.y - 16}
          fontSize={10}
          textAnchor="middle"
          fill="oklch(0.85 0.18 25)"
          fontFamily="monospace"
          style={{ transition: 'x 900ms cubic-bezier(0.4, 0, 0.2, 1), y 900ms cubic-bezier(0.4, 0, 0.2, 1)' }}
        >
          {particle.label}
        </text>
      </svg>

      <div className="rounded-md border border-border/60 bg-muted/30 px-3 py-2 text-[12px] text-foreground/95">
        <span className="text-muted-foreground">Step {step + 1}: </span>
        {STEPS[step]}
      </div>

      <div className="rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-[12px]">
        <strong>Parameter savings:</strong> the matrix is{' '}
        <span className="font-mono">248,320 × 1024 ≈ 254.3M floats</span>. Tying skips a second copy at the LM head — a
        ~254M-parameter reduction on a 0.8B-parameter model. That's close to a third of the model, gone, just by reusing
        the dictionary.
      </div>

      <p className="text-[11px] text-muted-foreground">
        Conceptually tying says:{' '}
        <em>
          the same dictionary that maps a token id to its incoming representation also maps an outgoing representation
          back to a vocab score
        </em>
        . Reading and writing share one alphabet. Not every model ties — large GPT-style models sometimes keep them
        separate for a small quality win — but for sub-billion-parameter models, tying is the standard.
      </p>
    </div>
  );
}
