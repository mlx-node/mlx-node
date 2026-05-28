import * as React from 'react';

/**
 * Chapter 9 (LM head) supplement — animate the final matmul that converts a
 * hidden vector into a vector of logits.
 *
 *   last_hidden ∈ R^d    ×    embed_tokens.weight ∈ R^{V×d}^T  =  logits ∈ R^V
 *
 * For Qwen3.5-0.8B: d=1024, V=151,936. The animation uses symbolic dimensions
 * (a short cell strip for the hidden state, a wide grid for the matrix, a
 * vocab-wide strip for the logits) — the point is the topology, not the
 * actual shapes.
 *
 * A "scan beam" sweeps left-to-right across the output strip, lighting up
 * each output cell in turn. While the beam is over column j, we highlight
 * row j of the weight matrix — that's the row being dotted with the hidden
 * vector to produce logit j. Once-through, then loops.
 */

const HIDDEN_CELLS = 12; // symbolic — stands in for d=1024
const VOCAB_CELLS = 36; // symbolic — stands in for V=152k

export function LmHeadWalkthrough() {
  const [beamCol, setBeamCol] = React.useState(0);
  const [playing, setPlaying] = React.useState(true);

  React.useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => {
      setBeamCol((c) => (c + 1) % VOCAB_CELLS);
    }, 110);
    return () => window.clearInterval(t);
  }, [playing]);

  const W = 640;
  const H = 280;

  // Layout: hidden strip (left), matrix (middle), logits strip (right).
  const hiddenX = 30;
  const hiddenCellW = 14;
  const hiddenY = 70;

  const matX = 130;
  const matRowH = 6;
  const matColW = 11;
  const matH = HIDDEN_CELLS * matRowH;
  const matY = hiddenY;

  const logitsX = matX + VOCAB_CELLS * matColW + 30;
  const logitsCellW = matColW;
  const logitsY = matY + HIDDEN_CELLS * matRowH + 24;

  // A handful of fake logits — taller bars at "the", "mat", "floor"
  // positions to give the output strip some readable shape.
  const fakeLogits = React.useMemo(() => {
    const arr = new Array<number>(VOCAB_CELLS);
    for (let i = 0; i < VOCAB_CELLS; i++) {
      // base noise
      let v = Math.sin(i * 0.7) * 0.15 + Math.cos(i * 0.3) * 0.1;
      // peaks at a few positions
      if (i === 8) v += 0.9;
      if (i === 14) v += 0.65;
      if (i === 22) v += 0.42;
      if (i === 30) v += 0.25;
      arr[i] = v;
    }
    return arr;
  }, []);

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          The final matmul — hidden state → logits
        </div>
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          className="rounded border border-border/60 bg-muted/40 px-2 py-0.5 text-[11px] hover:bg-muted/70"
          aria-pressed={playing}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
      </div>

      <p className="text-[12px] text-foreground/85">
        One matrix-vector product at the very top of the stack:{' '}
        <span className="font-mono">logits = last_hidden @ embed_tokens.weight.T</span>. For Qwen3.5-0.8B that means a
        <span className="font-mono"> [1, 1024]</span> vector multiplied by a{' '}
        <span className="font-mono">[1024, 151936]</span> matrix → a <span className="font-mono">[1, 151936]</span>{' '}
        output, one score per vocab token. The scan beam shows which output column is being produced.
      </p>

      <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label="LM head matmul animation">
        {/* hidden state column */}
        <text
          x={hiddenX + hiddenCellW / 2}
          y={hiddenY - 8}
          fontSize={10}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.6}
        >
          hidden
        </text>
        <text
          x={hiddenX + hiddenCellW / 2}
          y={hiddenY + HIDDEN_CELLS * matRowH + 14}
          fontSize={9}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.45}
        >
          d = 1024
        </text>
        {Array.from({ length: HIDDEN_CELLS }, (_, i) => (
          <rect
            key={`h-${i}`}
            x={hiddenX}
            y={hiddenY + i * matRowH}
            width={hiddenCellW}
            height={matRowH - 1}
            fill="oklch(0.65 0.13 250)"
            fillOpacity={0.25 + 0.55 * Math.abs(Math.sin(i * 1.7))}
          />
        ))}

        {/* the @ symbol */}
        <text
          x={(hiddenX + hiddenCellW + matX) / 2}
          y={hiddenY + (HIDDEN_CELLS * matRowH) / 2 + 4}
          fontSize={16}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.55}
          fontFamily="monospace"
        >
          @
        </text>

        {/* matrix */}
        <text
          x={matX + (VOCAB_CELLS * matColW) / 2}
          y={hiddenY - 8}
          fontSize={10}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.6}
        >
          embed_tokens.weight.T
        </text>
        <text
          x={matX + (VOCAB_CELLS * matColW) / 2}
          y={hiddenY + HIDDEN_CELLS * matRowH + 14}
          fontSize={9}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.45}
        >
          [d=1024, V=151,936]
        </text>
        {/* matrix cells — color modulates by row+col so it doesn't look like a flat block */}
        {Array.from({ length: HIDDEN_CELLS }, (_, r) =>
          Array.from({ length: VOCAB_CELLS }, (_, c) => {
            const isActiveCol = c === beamCol;
            const noise = Math.abs(Math.sin(r * 1.3 + c * 0.4));
            return (
              <rect
                key={`m-${r}-${c}`}
                x={matX + c * matColW}
                y={hiddenY + r * matRowH}
                width={matColW - 0.5}
                height={matRowH - 0.5}
                fill={isActiveCol ? 'oklch(0.7 0.15 60)' : 'oklch(0.6 0.05 250)'}
                fillOpacity={isActiveCol ? 0.55 + 0.4 * noise : 0.06 + 0.18 * noise}
                style={{ transition: 'fill 200ms, fill-opacity 200ms' }}
              />
            );
          }),
        )}
        {/* scan beam — full-height column outline above the matrix to highlight the active column */}
        <rect
          x={matX + beamCol * matColW - 0.5}
          y={hiddenY - 2}
          width={matColW + 1}
          height={HIDDEN_CELLS * matRowH + 4}
          fill="none"
          stroke="oklch(0.75 0.15 60)"
          strokeOpacity={0.85}
          strokeWidth={1.5}
          style={{ transition: 'x 110ms linear' }}
        />

        {/* equals + output logit strip */}
        <text
          x={(matX + VOCAB_CELLS * matColW + logitsX) / 2}
          y={hiddenY + (HIDDEN_CELLS * matRowH) / 2 + 4}
          fontSize={16}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.55}
          fontFamily="monospace"
        >
          =
        </text>

        {/* output bars - one per "vocab token". Heights from fakeLogits, lit
            up incrementally as the beam sweeps through. */}
        <text
          x={logitsX + (VOCAB_CELLS * logitsCellW) / 2}
          y={logitsY - 70}
          fontSize={10}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.6}
        >
          logits
        </text>
        <text
          x={logitsX + (VOCAB_CELLS * logitsCellW) / 2}
          y={logitsY + 16}
          fontSize={9}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.45}
        >
          V = 151,936 entries
        </text>
        {fakeLogits.map((v, i) => {
          const filled = i <= beamCol;
          const barH = 8 + Math.max(0, v) * 48;
          return (
            <rect
              key={`o-${i}`}
              x={logitsX + i * logitsCellW}
              y={logitsY - barH}
              width={logitsCellW - 0.5}
              height={barH}
              fill={filled ? 'oklch(0.7 0.15 60)' : 'oklch(0.5 0.04 250)'}
              fillOpacity={filled ? 0.6 : 0.18}
              style={{ transition: 'fill 200ms, fill-opacity 200ms' }}
            />
          );
        })}

        {/* baseline */}
        <line
          x1={logitsX - 4}
          y1={logitsY}
          x2={logitsX + VOCAB_CELLS * logitsCellW + 4}
          y2={logitsY}
          stroke="currentColor"
          strokeOpacity={0.3}
          strokeWidth={1}
        />
      </svg>

      <p className="text-[11px] text-muted-foreground">
        Every output entry is one inner product:{' '}
        <span className="font-mono">logit_j = sum_i (last_hidden[i] · W[i, j])</span>. Each column of the matrix is the
        "fingerprint" of one vocab token — when that column points in roughly the same direction as the hidden state,
        the logit for that token is high.
      </p>
    </div>
  );
}
