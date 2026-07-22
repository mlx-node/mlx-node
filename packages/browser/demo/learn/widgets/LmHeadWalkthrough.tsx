import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 10 (LM head) supplement — animate the final matmul that converts a
 * hidden vector into a vector of logits.
 *
 *   last_hidden ∈ R^d    ×    embed_tokens.weight ∈ R^{V×d}^T  =  logits ∈ R^V
 *
 * For Qwen3.5-0.8B: d=1024, V=248,320. The animation uses symbolic dimensions
 * (a short cell strip for the hidden state, a wide grid for the matrix, a
 * vocab-wide strip for the logits) — the point is the topology, not the
 * actual shapes.
 *
 * A "scan beam" sweeps left-to-right across the output strip, lighting up
 * each output cell in turn. While the beam is over column j, we highlight
 * column j of the displayed weight.T — that's the column being dotted with
 * the hidden vector to produce logit j. Once-through, then loops.
 */

const HIDDEN_CELLS = 12; // symbolic — stands in for d=1024
const VOCAB_CELLS = 36; // symbolic — stands in for V=248k

// One plausible next-token text per symbolic vocab column (after "The cat
// sat on the"), so the beam can name the token whose fingerprint-column it
// is scoring. The peaks in `fakeLogits` (indices 8, 14, 22, 30) line up with
// ' floor', ' mat', ' rug', ' couch' — the same shortlist the KV-cache
// chapter's scripted decode uses, with ' floor' the course's measured pick.
const SAMPLE_TOKENS: ReadonlyArray<string> = [
  ' the', ' a', ' it', ' his', ' her', ' top', ' old', ' warm',
  ' floor', ' wall', ' step', ' lap', ' box', ' soft', ' mat', ' bed',
  ' chair', ' table', ' grass', ' roof', ' porch', ' shelf', ' rug', ' bench',
  ' tree', ' path', ' seat', ' lawn', ' hill', ' edge', ' couch', ' stone',
  ' sand', ' deck', ' stairs', ' sofa',
];

const COPY = {
  en: {
    header: 'The final matmul — hidden state → logits',
    pause: 'Pause',
    play: 'Play',
    intro: (
      <>
        One matrix-vector product at the very top of the stack:{' '}
        <span className="font-mono">logits = last_hidden @ embed_tokens.weight.T</span>. For Qwen3.5-0.8B that means a
        <span className="font-mono"> [1, 1024]</span> vector multiplied by a{' '}
        <span className="font-mono">[1024, 248320]</span> matrix → a <span className="font-mono">[1, 248320]</span>{' '}
        output, one score per vocab token. The scan beam shows which output column is being produced.
      </>
    ),
    svgAria: 'LM head matmul animation',
    hiddenLabel: 'hidden',
    beamLabel: (col: number, token: string) => `col ${col} = "${token}"`,
    fingerprintLabel: "column j = token j's fingerprint (row j of W_lm)",
    vocabEntriesLabel: 'V = 248,320 entries',
    outro: (
      <>
        Every output entry is one inner product:{' '}
        <span className="font-mono">logit_j = sum_i (last_hidden[i] · W[i, j])</span>. Each column of this{' '}
        <span className="font-mono">[d, V]</span> matrix — equivalently, a row of the untransposed{' '}
        <span className="font-mono">[V, d]</span> weight — is the "fingerprint" of one vocab token; when that fingerprint
        points in roughly the same direction as the hidden state, the logit for that token is high.
      </>
    ),
    footnote:
      'Illustrative/schematic — the 12×36 grid and bar heights are stand-ins for the real [1024 × 248,320] matmul, not actual model logits.',
  },
  zh: {
    header: '最后的矩阵乘法——隐藏状态 → logits',
    pause: '暂停',
    play: '播放',
    intro: (
      <>
        堆叠最顶端的一次矩阵-向量乘法：
        <span className="font-mono">logits = last_hidden @ embed_tokens.weight.T</span>。对 Qwen3.5-0.8B
        来说，就是一个<span className="font-mono"> [1, 1024]</span> 的向量乘以一个{' '}
        <span className="font-mono">[1024, 248320]</span> 的矩阵 → 得到 <span className="font-mono">[1, 248320]</span>{' '}
        的输出，词表中每个 token 一个分数。扫描光束标出当前正在产出的输出列。
      </>
    ),
    svgAria: 'LM head 矩阵乘法动画',
    hiddenLabel: '隐藏状态',
    beamLabel: (col: number, token: string) => `第 ${col} 列 = "${token}"`,
    fingerprintLabel: '第 j 列 = token j 的“指纹”（W_lm 的第 j 行）',
    vocabEntriesLabel: 'V = 248,320 个条目',
    outro: (
      <>
        每个输出条目都是一次内积：
        <span className="font-mono">logit_j = sum_i (last_hidden[i] · W[i, j])</span>。这个{' '}
        <span className="font-mono">[d, V]</span> 矩阵的每一列——等价于未转置的{' '}
        <span className="font-mono">[V, d]</span> 权重的某一行——就是一个词表 token
        的“指纹”；当这枚指纹与隐藏状态指向大致相同的方向时，该 token 的 logit 就高。
      </>
    ),
    footnote: '示意图——12×36 的网格和柱高只是真实 [1024 × 248,320] 矩阵乘法的替身，并非模型的真实 logits。',
  },
} as const;

export function LmHeadWalkthrough() {
  const copy = COPY[useLocale()];
  const [beamCol, setBeamCol] = React.useState(0);
  const [playing, setPlaying] = React.useState(() =>
    typeof window !== 'undefined' ? !window.matchMedia('(prefers-reduced-motion: reduce)').matches : true,
  );

  React.useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => {
      setBeamCol((c) => (c + 1) % VOCAB_CELLS);
    }, 110);
    return () => window.clearInterval(t);
  }, [playing]);

  // Wide enough for hidden strip + matrix + logits strip side by side. The
  // logits strip spans logitsX (556) + 36·11 = 952, so anything narrower
  // clips the output bars (the old 640 cut off two thirds of the strip).
  const W = 970;
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
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          className="rounded border border-border/60 bg-muted/40 px-2 py-0.5 text-[11px] hover:bg-muted/70"
          aria-pressed={playing}
        >
          {playing ? copy.pause : copy.play}
        </button>
      </div>

      <p className="text-[12px] text-foreground/85">{copy.intro}</p>

      {/* The scanning beam lives inside the SVG (an SVG <text>), so mirror its
          label into an sr-only live region. Muted while autoplaying to avoid a
          firehose at the 110ms scan rate; announced once when static/paused. */}
      <div className="sr-only" aria-live={playing ? 'off' : 'polite'} aria-atomic="true">
        {copy.beamLabel(beamCol, SAMPLE_TOKENS[beamCol]!)}
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label={copy.svgAria}>
        {/* hidden state column */}
        <text
          x={hiddenX + hiddenCellW / 2}
          y={hiddenY - 8}
          fontSize={13}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.6}
        >
          {copy.hiddenLabel}
        </text>
        <text
          x={hiddenX + hiddenCellW / 2}
          y={hiddenY + HIDDEN_CELLS * matRowH + 14}
          fontSize={12}
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
          fontSize={20}
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
          fontSize={13}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.6}
        >
          embed_tokens.weight.T
        </text>
        <text
          x={matX + (VOCAB_CELLS * matColW) / 2}
          y={hiddenY + HIDDEN_CELLS * matRowH + 14}
          fontSize={12}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.45}
        >
          [d=1024, V=248,320]
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
        {/* Beam-following label: which token's fingerprint-column is being
            scored right now. x clamped so the text never leaves the matrix. */}
        <text
          x={Math.max(matX + 70, Math.min(matX + VOCAB_CELLS * matColW - 70, matX + beamCol * matColW + matColW / 2))}
          y={matY + matH + 30}
          fontSize={13}
          textAnchor="middle"
          fill="oklch(0.7 0.15 60)"
          fontFamily="monospace"
          style={{ transition: 'x 110ms linear' }}
        >
          {copy.beamLabel(beamCol, SAMPLE_TOKENS[beamCol]!)}
        </text>
        <text
          x={matX + (VOCAB_CELLS * matColW) / 2}
          y={matY + matH + 46}
          fontSize={12}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.55}
        >
          {copy.fingerprintLabel}
        </text>

        {/* equals + output logit strip */}
        <text
          x={(matX + VOCAB_CELLS * matColW + logitsX) / 2}
          y={hiddenY + (HIDDEN_CELLS * matRowH) / 2 + 4}
          fontSize={20}
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
          fontSize={13}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.6}
        >
          logits
        </text>
        <text
          x={logitsX + (VOCAB_CELLS * logitsCellW) / 2}
          y={logitsY + 16}
          fontSize={12}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.45}
        >
          {copy.vocabEntriesLabel}
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

      <p className="text-[11px] text-muted-foreground">{copy.outro}</p>

      <p className="text-[10px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
