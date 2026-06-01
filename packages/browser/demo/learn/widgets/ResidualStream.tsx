import * as React from 'react';

/**
 * Chapter 8 supplement — make the "residual stream" name concrete.
 *
 * The 3D widget already shows that each layer has the shape (norm → attn →
 * residual add, norm → mlp → residual add), but a reader can still finish the
 * chapter without internalizing that the *same vector* is what's being
 * accumulated into. This widget animates that single accumulation.
 *
 * We render a vertical "stream" of bar-blocks growing upward. A token's
 * hidden state enters at the bottom; each layer's attention and MLP outputs
 * are drawn as side-arrows that *write into* the stream (think `h += attn(h)`).
 * Two layers, four writes, then the LM head reads the top of the stream. The
 * stream's height growing is the residual norm growing — a visual referent
 * for the magnitude climb the 3D tower color-codes.
 *
 * Pure JS animation, no model. Loops, with a play/pause toggle.
 */

const STEP_LABELS = [
  { kind: 'enter', label: 'h₀  (embedding + RoPE)' },
  { kind: 'attn', label: 'h₁ = h₀ + attn(norm(h₀))' },
  { kind: 'mlp', label: 'h₂ = h₁ + mlp(norm(h₁))' },
  { kind: 'attn', label: 'h₃ = h₂ + attn(norm(h₂))' },
  { kind: 'mlp', label: 'h₄ = h₃ + mlp(norm(h₃))' },
  { kind: 'exit', label: 'lm_head(h₄) → logits' },
] as const;

// In overwrite mode the same steps REPLACE instead of add, so the readout
// formulas drop the "hₙ₋₁ +" term (h := Δ). Parallel to STEP_LABELS by index.
const OVERWRITE_STEP_LABELS = [
  'h₀  (embedding + RoPE)',
  'h₁ = attn(norm(h₀))',
  'h₂ = mlp(norm(h₁))',
  'h₃ = attn(norm(h₂))',
  'h₄ = mlp(norm(h₃))',
  'lm_head(h₄) → logits',
] as const;

// Heights chosen to monotonically grow — each layer adds a contribution, so
// L2 norm climbs. Real Qwen3.5 magnitudes climb roughly like this through
// the 24-layer stack; we compress to 4 writes for visualization.
const HEIGHTS = [22, 30, 38, 48, 60];
// Overwrite mode: each sub-block REPLACES the stream with just its own output,
// so the column jumps to roughly one block's magnitude and never accumulates.
// Same length as HEIGHTS so every geometry helper indexes identically.
const OVERWRITE_HEIGHTS = [22, 27, 24, 28, 23];
const COLORS = [
  'oklch(0.65 0.13 250)', // h0 — cool blue, "input"
  'oklch(0.68 0.14 230)',
  'oklch(0.7 0.14 200)',
  'oklch(0.72 0.14 160)',
  'oklch(0.75 0.15 120)', // h4 — warmer, "almost output"
];

const ATTN_COLOR = 'oklch(0.7 0.16 60)'; // warm orange
const MLP_COLOR = 'oklch(0.62 0.18 300)'; // purple

function ToggleButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={[
        'rounded px-2.5 py-1 text-xs font-medium transition-colors',
        active ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
      ].join(' ')}
    >
      {children}
    </button>
  );
}

export function ResidualStream() {
  const [step, setStep] = React.useState(0);
  const [playing, setPlaying] = React.useState(() =>
    typeof window !== 'undefined' ? !window.matchMedia('(prefers-reduced-motion: reduce)').matches : true,
  );
  const [mode, setMode] = React.useState<'add' | 'overwrite'>('add');
  const heights = mode === 'add' ? HEIGHTS : OVERWRITE_HEIGHTS;

  React.useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => {
      setStep((s) => (s + 1) % (STEP_LABELS.length + 1));
    }, 1400);
    return () => window.clearInterval(t);
  }, [playing]);

  const W = 520;
  // 400 / 30 px stream-top headroom — earlier values of H=360, topY=30 put
  // the lm_head box (y=topY-14 to topY+16 ⇒ 16..46) at exactly the same y
  // as the h4 tick at the top of the fully-filled stream (h4 height=60,
  // tick y = baseY-300 = 30), so the box visually swallowed the tick label
  // and the stream column poked through the box's interior. Bumping H to
  // 400 + topY to 20 gives a clean ~34px gap between the lm_head box
  // bottom (y=36) and the h4 tick (y=70), and the arrow from stream top
  // into lm_head now has visible length (~30px).
  const H = 400;
  const streamX = 200;
  const baseY = H - 30;
  const topY = 20;
  const streamW = 28;

  // Cumulative height at each step (after k writes have happened). step=0
  // shows only the input; step=5 shows the final read by lm_head.
  const filledHeight = (s: number): number => {
    if (s <= 0) return heights[0]!;
    if (s >= heights.length) return heights[heights.length - 1]!;
    return heights[s]!;
  };
  const filled = filledHeight(step);

  // Y position of the topmost edge of the stream after `step` writes. The
  // stream grows upward, so smaller Y = taller stream.
  const topOfStreamY = baseY - filled * 5;

  // Side-block geometry: attention writes from the left, MLP from the right.
  const SIDE_OFFSET = 110;

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          The residual stream — one vector, written into 2× per layer
        </div>
        <div className="inline-flex items-center gap-2">
          <div className="inline-flex rounded-md border border-border p-0.5">
            <ToggleButton active={mode === 'add'} onClick={() => setMode('add')}>
              Add (h += Δ)
            </ToggleButton>
            <ToggleButton active={mode === 'overwrite'} onClick={() => setMode('overwrite')}>
              Overwrite (h = new)
            </ToggleButton>
          </div>
          <button
            type="button"
            onClick={() => setPlaying((p) => !p)}
            className="rounded border border-border/60 bg-muted/40 px-2 py-0.5 text-[11px] hover:bg-muted/70"
            aria-pressed={playing}
          >
            {playing ? 'Pause' : 'Play'}
          </button>
          <button
            type="button"
            onClick={() => {
              setPlaying(false);
              setStep((s) => (s - 1 + (STEP_LABELS.length + 1)) % (STEP_LABELS.length + 1));
            }}
            className="rounded border border-border/60 bg-muted/40 px-2 py-0.5 text-[11px] hover:bg-muted/70"
            aria-label="Previous step"
          >
            ◀
          </button>
          <button
            type="button"
            onClick={() => {
              setPlaying(false);
              setStep((s) => (s + 1) % (STEP_LABELS.length + 1));
            }}
            className="rounded border border-border/60 bg-muted/40 px-2 py-0.5 text-[11px] hover:bg-muted/70"
            aria-label="Next step"
          >
            ▶
          </button>
        </div>
      </div>

      {mode === 'add' ? (
        <p className="text-[12px] text-foreground/85">
          The hidden state for one token, drawn as a column that grows as each sub-block writes a correction into it.
          Attention writes from the left, MLP writes from the right. Nothing ever overwrites — every operation is{' '}
          <code>h := h + Δ</code>. Two layers shown; Qwen3.5-0.8B does this 24 times.
        </p>
      ) : (
        <p className="text-[12px] text-foreground/85">
          In this mode each sub-block <strong>replaces</strong> the stream — <code>h := Δ</code> — so the running state
          is thrown away every write. Watch the column fail to climb.
        </p>
      )}

      <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label="Residual stream animation">
        {/* baseline ticks for h0..h4 */}
        {heights.map((h, i) => {
          const y = baseY - h * 5;
          return (
            <g key={`tick-${i}`}>
              <line
                x1={streamX - streamW / 2 - 6}
                y1={y}
                x2={streamX - streamW / 2 - 2}
                y2={y}
                stroke="currentColor"
                strokeOpacity={0.25}
                strokeWidth={0.7}
              />
              <text
                x={streamX - streamW / 2 - 10}
                y={y + 3}
                fontSize={9}
                textAnchor="end"
                fill="currentColor"
                fillOpacity={0.5}
                fontFamily="monospace"
              >
                h{i}
              </text>
            </g>
          );
        })}

        {/* The stream column itself, growing as `filled` grows */}
        <rect
          x={streamX - streamW / 2}
          y={topOfStreamY}
          width={streamW}
          height={baseY - topOfStreamY}
          rx={2}
          fill={COLORS[Math.min(step, COLORS.length - 1)]}
          fillOpacity={0.4}
          stroke={COLORS[Math.min(step, COLORS.length - 1)]}
          strokeOpacity={0.85}
          style={{ transition: 'all 700ms cubic-bezier(0.4, 0, 0.2, 1)' }}
        />

        {/* Floor line — "the stream starts here" */}
        <line
          x1={streamX - streamW / 2 - 18}
          y1={baseY}
          x2={streamX + streamW / 2 + 18}
          y2={baseY}
          stroke="currentColor"
          strokeOpacity={0.35}
          strokeWidth={1}
        />
        <text x={streamX} y={baseY + 18} fontSize={10} textAnchor="middle" fill="currentColor" fillOpacity={0.5}>
          entry — h₀
        </text>

        {/* Sub-block side write arrows. We mark them visible/active based on
            which step we're showing. Attention writes happen at step 1 and 3,
            MLP at 2 and 4. */}
        {STEP_LABELS.map((s, idx) => {
          if (s.kind === 'enter' || s.kind === 'exit') return null;
          // Step idx → which level of the stream this write lands at.
          // Write lands at heights[idx] (idx === 1..4 maps to the 1st..4th write).
          const landingY = baseY - heights[idx]! * 5;
          const isActive = step === idx;
          const wasDone = step > idx;
          const isAttn = s.kind === 'attn';
          const sideX = isAttn ? streamX - SIDE_OFFSET : streamX + SIDE_OFFSET;
          const arrowEndX = isAttn ? streamX - streamW / 2 - 2 : streamX + streamW / 2 + 2;
          const blockColor = isAttn ? ATTN_COLOR : MLP_COLOR;
          const opacity = isActive ? 1 : wasDone ? 0.35 : 0.12;
          return (
            <g key={`write-${idx}`} style={{ transition: 'opacity 600ms', opacity }}>
              {/* sub-block label box */}
              <rect
                x={sideX - 36}
                y={landingY - 12}
                width={72}
                height={24}
                rx={4}
                fill={blockColor}
                fillOpacity={isActive ? 0.25 : 0.12}
                stroke={blockColor}
                strokeOpacity={isActive ? 0.9 : 0.4}
              />
              <text
                x={sideX}
                y={landingY + 1}
                fontSize={10}
                textAnchor="middle"
                fill={blockColor}
                fontFamily="monospace"
              >
                {isAttn ? 'attn(·)' : 'mlp(·)'}
              </text>
              {/* arrow into the stream */}
              <line
                x1={isAttn ? sideX + 36 : sideX - 36}
                y1={landingY}
                x2={arrowEndX}
                y2={landingY}
                stroke={blockColor}
                strokeOpacity={isActive ? 0.95 : 0.45}
                strokeWidth={isActive ? 2 : 1.2}
                markerEnd={`url(#arrow-${isAttn ? 'attn' : 'mlp'})`}
              />
              {/* "+= " / ":= " label */}
              {isActive ? (
                <text
                  x={(sideX + arrowEndX) / 2}
                  y={landingY - 4}
                  fontSize={10}
                  textAnchor="middle"
                  fill={blockColor}
                  fontFamily="monospace"
                >
                  {mode === 'add' ? '+=' : ':='}
                </text>
              ) : null}
            </g>
          );
        })}

        {/* LM head reading at the top, lit when step === final */}
        {step >= STEP_LABELS.length - 1 ? (
          <g style={{ transition: 'opacity 600ms', opacity: step === STEP_LABELS.length - 1 ? 1 : 0.6 }}>
            <line
              x1={streamX}
              y1={topOfStreamY - 4}
              x2={streamX}
              y2={topY + 16}
              stroke="currentColor"
              strokeOpacity={0.7}
              strokeWidth={1.5}
              markerEnd="url(#arrow-out)"
            />
            <rect
              x={streamX - 50}
              y={topY - 14}
              width={100}
              height={30}
              rx={4}
              fill="currentColor"
              fillOpacity={0.08}
              stroke="currentColor"
              strokeOpacity={0.5}
            />
            <text x={streamX} y={topY + 5} fontSize={10} textAnchor="middle" fill="currentColor" fontFamily="monospace">
              lm_head
            </text>
          </g>
        ) : null}

        {/* arrowhead markers */}
        <defs>
          <marker id="arrow-attn" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 Z" fill={ATTN_COLOR} fillOpacity={0.95} />
          </marker>
          <marker id="arrow-mlp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 Z" fill={MLP_COLOR} fillOpacity={0.95} />
          </marker>
          <marker id="arrow-out" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 Z" fill="currentColor" fillOpacity={0.7} />
          </marker>
        </defs>
      </svg>

      <div className="rounded-md border border-border/60 bg-muted/30 px-3 py-2 text-[12px]">
        <span className="text-muted-foreground">Step {step}: </span>
        <span className="font-mono text-foreground/95">
          {step < STEP_LABELS.length
            ? mode === 'add'
              ? STEP_LABELS[step]!.label
              : OVERWRITE_STEP_LABELS[step]!
            : 'next token sampled →'}
        </span>
      </div>

      {mode === 'add' ? (
        <p className="text-[11px] text-muted-foreground">
          Three things to notice. First, the stream never gets <em>narrower</em> — there is no operation in the layer
          that subtracts or replaces. Second, the same column is read and written by every sub-block; it's a single
          running address space the whole network shares. Third, the LM head at the top reads the <em>top</em> of the
          stream — every layer's contribution is visible to the final prediction, not just the last one.
        </p>
      ) : (
        <p className="text-[11px] text-muted-foreground">
          This is a deliberately destructive toy — a pure <code>h := Δ</code> that throws the old state away on every
          write, so the column never accumulates. Real non-residual networks aren&apos;t this extreme (they transform
          the state, <code>h := F(h)</code>), but they still give up what the residual stream buys for free: a clean
          identity path, every layer&apos;s contribution preserved by addition, and stable gradients straight back to
          the input. That is why real transformers <em>add</em> (<code>h += Δ</code>) instead of replacing.
        </p>
      )}

      <p className="text-[10px] text-muted-foreground">
        Illustrative — heights are a synthetic stand-in for the residual-stream magnitude; Qwen3.5-0.8B does this 24
        times. Not live output from the model.
      </p>
    </div>
  );
}
