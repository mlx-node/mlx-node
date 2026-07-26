import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { DiagramFrame, PanBox, SW, useStepPlayer } from '../motion';

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
 *
 * Animation model: `useStepPlayer` from `learn/motion`, replacing this file's
 * former hand-rolled `setInterval` plus a `playing` flag seeded from
 * `matchMedia` inside a `useState` initializer. Same 36 beats at the same 110ms
 * pace — the beam column IS the frame index. Two behaviour changes ride along
 * and are the kit's contract rather than accidents: the widget now mounts
 * PLAYING for everyone and retargets reduced-motion readers in an effect after
 * mount (which is also what makes flipping the OS preference mid-session work —
 * initializers run once), and those readers now rest on the LAST column, where
 * the whole logits strip is filled, instead of on column 0 where none of it is.
 *
 * ── WHY THE SKIN'S COLOUR TOKENS ARE *NOT* APPLIED HERE ────────────────────
 *
 * `skin` says hue is a box's PERMANENT IDENTITY: "busy right now" is the hatch,
 * "not real yet" is the dash. This diagram breaks that twice — the matrix cells
 * and the logit bars both flip blue 250 → amber 60 — so it is adoptable only if
 * BOTH flips move onto `hatchFill()` + `DIM`. They do not, for two independent
 * reasons, so this file takes the kit and stops.
 *
 *  1. ONLY ONE OF THE TWO FLIPS IS "BUSY NOW". The matrix flip is
 *     `c === beamCol` — one column, this instant — and that is exactly what the
 *     hatch means. The logit-bar flip is `i <= beamCol`: every bar the beam has
 *     ALREADY passed, cumulatively, for the rest of the sweep. That is a
 *     progress readout, not a this-step state, and the hatch cannot say it. The
 *     only channel left for it would be `DIM`, and its nearest pair (FILLED .85
 *     vs EMPTY_SLOT .55) trades a hue flip PLUS a 3.3x alpha step (.6 vs .18)
 *     for a 1.5x alpha step, across 36 bars whose heights already vary — and it
 *     would brighten the not-yet-computed half of the strip from .18 to .55,
 *     asserting the opposite of the lesson (the strip is being PRODUCED, left
 *     to right). Halving an encoding to satisfy a token is not an improvement.
 *
 *  2. THE HATCH CANNOT PHYSICALLY RENDER AT THIS SCALE. `HatchDefs` is
 *     `patternUnits="userSpaceOnUse"` with a 7-unit pitch, sized for CPE's
 *     300x40-unit panels on a canvas whose units are roughly a CSS pixel. This
 *     canvas is 970 units wide inside chapter 10's reading column, which is
 *     PANED (`LmHeadDemo` is the right-hand panel) and therefore ~451px — a
 *     scale of 0.465. The hatch pitch lands at ~3.3 CSS px on matrix cells that
 *     are 11x6 units, i.e. ~5.1x2.8 CSS px. One cell would hold a fragment of a
 *     single stripe. The channel does not exist at this size.
 *
 * So: §1 (the kit) in full, plus the two structural stroke widths that already
 * matched the ladder exactly — `SW.OUTER` on the scan beam, `SW.INNER` on the
 * logits baseline, both zero-pixel renames. Deliberately NOT applied: `RX`
 * (nothing here carries an `rx`, and matrix cells must never gain one), `DASH`
 * (nothing is dashed; adding one would be inventing an encoding), `PanelFrame`
 * (there is no stage box — the three strips are bare cell arrays), `FrameLabel`
 * (no label sits on a stroke — each was checked against the beam rect and the
 * logits baseline), `elbow` (no connectors), and `DIM` (every opacity here is a
 * per-cell magnitude, i.e. the diagram's DATA, not a dimming level).
 */

const HIDDEN_CELLS = 12; // symbolic — stands in for d=1024
const VOCAB_CELLS = 36; // symbolic — stands in for V=248k

/** ms per beam column — the scan pace, unchanged by the kit adoption. */
const FRAME_MS = 110;

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

// Play / Pause are NOT here any more: those verbs belong to the shared
// `StepControls`, which carries its own en/zh pair so every animated widget in
// the course says the same word.
const COPY = {
  en: {
    header: 'The final matmul — hidden state → logits',
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
    beamCaption: (col: number, token: string) => `Hidden state dotted with column ${col} → the logit for "${token}".`,
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
    beamCaption: (col: number, token: string) => `隐藏状态与第 ${col} 列做点积 → "${token}" 的 logit。`,
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
  const locale = useLocale();
  const copy = COPY[locale];
  // The beam column IS the frame index: 36 beats, one per symbolic vocab column.
  const player = useStepPlayer(VOCAB_CELLS, { frameMs: FRAME_MS });
  const beamCol = player.frame;

  // Reduced-motion readers keep Step (StepControls hides only Play), so the CSS
  // transitions would still fire on a press — motion they did not ask for. Drop
  // them outright rather than shortening them.
  const slide = player.reducedMotion ? undefined : `x ${FRAME_MS}ms linear`;
  const fade = player.reducedMotion ? undefined : 'fill 200ms, fill-opacity 200ms';

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

  // The per-beat sentence. It deliberately does NOT repeat the in-svg chip,
  // which already says `col N = "token"` right under the beam: the chip owns
  // the COORDINATE (it is the only thing tying the token name to the beam's x
  // position), so the caption owns the OPERATION — the one thing no per-frame
  // label states. Both are needed: the svg is `role="img"`, so its chip is
  // pruned from the a11y tree and the caption is the only accessible copy.
  // DiagramFrame fills its live region only after the reader pauses, so the
  // 110ms scan still cannot become an announcement firehose.
  const caption = copy.beamCaption(beamCol, SAMPLE_TOKENS[beamCol]!);
  // Every caption the sweep can reach. `beamCaption` is a builder and NOTHING
  // else picks the caption, so mapping it over all 36 columns is the complete
  // set of arms — which is also the only locale-correct way to size the slot:
  // the widest column index and the widest token text do not coincide, and the
  // zh sentence measures differently again.
  const captions = SAMPLE_TOKENS.map((t, i) => copy.beamCaption(i, t));

  return (
    <>
      {/* Body-weight teaching prose that introduces the diagram — it stays
          OUTSIDE the frame rather than being demoted into the 11px `note`
          slot, and it keeps its place immediately above the diagram. */}
      <p className="text-[12px] text-foreground/85">{copy.intro}</p>

      <DiagramFrame
        title={copy.header}
        player={player}
        locale={locale}
        caption={caption}
        captions={captions}
        note={
          // Both were footnote tier before the conversion (11px muted and 10px
          // muted), so both belong in `note`. It renders a div, so these are
          // real paragraphs; the second keeps its smaller size explicitly.
          <>
            <p>{copy.outro}</p>
            <p className="text-[10px]">{copy.footnote}</p>
          </>
        }
      >
        {/* The floor the `PanBox` below needs to mean anything. Nothing here
            stops a `w-full` svg shrinking, so a narrow column scales the whole
            canvas instead of reflowing it, text included — at a 375px viewport
            the column is 320px and every label lands at a third of its size.
            12 is the smallest font in this canvas ("d = 1024", the two shape
            labels, the fingerprint line, "V = 248,320 entries"): all of them
            prose a reader has to read, none of them a superscript, so 12 is
            the size the 8px legibility floor has to be measured against —
            8 * 970 / 12 = 647. Below that the labels go grey smudge; past it
            the box pans sideways instead. */}
        <PanBox locale={locale}>
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="block h-auto w-full min-w-[647px]"
            role="img"
            aria-label={copy.svgAria}
          >
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
                    style={{ transition: fade }}
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
              strokeWidth={SW.OUTER}
              style={{ transition: slide }}
            />
            {/* Beam-following label: which token's fingerprint-column is being
                scored right now. x clamped so the text never leaves the matrix. */}
            <text
              x={Math.max(
                matX + 70,
                Math.min(matX + VOCAB_CELLS * matColW - 70, matX + beamCol * matColW + matColW / 2),
              )}
              y={matY + matH + 30}
              fontSize={13}
              textAnchor="middle"
              fill="oklch(0.7 0.15 60)"
              fontFamily="monospace"
              style={{ transition: slide }}
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
                  style={{ transition: fade }}
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
              strokeWidth={SW.INNER}
            />
          </svg>
        </PanBox>
      </DiagramFrame>
    </>
  );
}
