import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { DIM, DiagramFrame, PanBox, RX, SW, useStepPlayer } from '../motion';

/**
 * EmbeddingLookupDiagram — chapter 3 (embeddings), the core mechanism as a
 * picture: an integer token id indexes one row of the [248,320 × 1,024]
 * embedding table. No multiply, no add — just reading a row.
 *
 * Self-contained: NO worker, NO model, NO WASM — pure SVG + React state, so it
 * server-renders for crawlers (createRoot replaces it on boot) and is robust
 * while the model downloads.
 *
 * Uses the course's running example: token id 9059 = " cat" (the same id the
 * overview chapter's TokenJourney uses). The eight floats shown popping out of
 * the row are illustrative stand-ins for the real 1,024 bf16 values — the
 * teaching point is the SHAPE of the operation (id → row → vector), not the
 * exact numbers.
 *
 * Animation model: `useStepPlayer` from `learn/motion`, replacing this file's
 * former hand-rolled `setInterval` + private `usePrefersReducedMotion`. Same
 * four frames at the same 2100ms pace; reduced-motion readers still rest on the
 * final frame, but they now get there via an effect after mount instead of a
 * `useState` initializer — which is also what makes flipping the OS preference
 * mid-session work (initializers run once, so the old version could not).
 *
 * ── SKIN NOTES (why this widget keeps `var(--primary)`) ────────────────────
 *
 * `skin` names two hues, RED (busy/compute-now) and EMERALD (stored/done), to
 * replace HARDCODED copies of those hexes. This file has never hardcoded a hex:
 * the one accent it draws with is `var(--primary)`, the site theme's own accent
 * (`--book-pink`), and it is the PERMANENT IDENTITY of a single object followed
 * across all four frames — the token, its row, the arrows into and out of that
 * row, and the vector read off it are all the same thing. It is also the hue the
 * chapter's caption copy already points at (`text-primary`, inside COPY). So
 * there is nothing here for the two-hue rule to fix, and swapping the accent
 * would desync the picture from its own sentences.
 *
 * The hatch is deliberately absent for the same kind of reason: hatch means
 * "this box is BUSY this step", and the whole thesis of this diagram — the
 * caption on the last frame says it outright — is that NO computation happens.
 */

// ── The running example ───────────────────────────────────────────────────────
const TOKEN_ID = 9059;
const TOKEN_TEXT = '·cat'; // " cat" with the course's visible-space dot
const VOCAB_ROWS = '248,320';
const HIDDEN_COLS = '1,024';

// Illustrative floats "read out" of row 9059 — deterministic constants, not
// live model output (the chapter's interactive demo fetches the real thing).
const ROW_FLOATS = [0.012, -0.031, 0.118, -0.31, 0.054, 0.207, -0.089, 0.142];

// ── Visible slice of the tall table: 9 labeled rows + 2 ellipsis rows ─────────
type VisRow = { label: string; ellipsis?: boolean; target?: boolean };
const VIS_ROWS: VisRow[] = [
  { label: '0' },
  { label: '1' },
  { label: '2' },
  { label: '⋮', ellipsis: true },
  { label: '9,057' },
  { label: '9,058' },
  { label: '9,059', target: true },
  { label: '9,060' },
  { label: '9,061' },
  { label: '⋮', ellipsis: true },
  { label: '248,319' },
];
const TARGET_VIS_IDX = VIS_ROWS.findIndex((r) => r.target); // 6

// ── Frames ────────────────────────────────────────────────────────────────────
// 0: token chip arrives from the tokenizer (an integer).
// 1: chip slides down beside its row; row 9059 lights up.
// 2: the row's first 8 floats pop out as value cells.
// 3: done — hold the final picture.
const TOTAL_FRAMES = 4;
const DONE_FRAME = TOTAL_FRAMES - 1;
const FRAME_MS = 2100;

// ── Per-locale copy ───────────────────────────────────────────────────────────
//
// Play / Pause / Step are NOT here any more: those verbs now belong to the
// shared `StepControls`, which carries its own en/zh pair so every animated
// widget in the course says the same word.
const COPY = {
  en: {
    title: 'Embedding lookup — one id, one row',
    svgAria:
      'Diagram of an embedding lookup: the token id 9059 for the token cat slides down a tall matrix of 248,320 rows by 1,024 columns to its row, the row lights up, and the first eight of its 1,024 float values pop out as cells — no computation, just reading row 9059.',
    tableLabel: `embedding table · ${VOCAB_ROWS} rows × ${HIDDEN_COLS} columns`,
    rowId: 'row id',
    rowCaption: `row ${TOKEN_ID} · the embedding of "${TOKEN_TEXT}" (illustrative values)`,
    captions: [
      <>
        The tokenizer hands the model one integer: <span className="font-mono">{TOKEN_ID}</span> (the token{' '}
        <span className="font-mono">&quot;{TOKEN_TEXT}&quot;</span>). The embedding table is just a tall matrix —{' '}
        <span className="font-mono">{VOCAB_ROWS}</span> rows, one per vocabulary entry.
      </>,
      <>
        The id is used as a plain <strong>array index</strong>: go straight to row{' '}
        <span className="font-mono text-primary">{TOKEN_ID}</span>. No search, no multiply — row number = token id.
      </>,
      <>
        Read the row off: <span className="font-mono">{HIDDEN_COLS}</span> floats (first 8 shown). That vector{' '}
        <em>is</em> the embedding of <span className="font-mono">&quot;{TOKEN_TEXT}&quot;</span>.
      </>,
      <>
        <strong>No computation happened</strong> — just reading row {TOKEN_ID}. Every forward pass starts with this
        lookup, once per input token.
      </>,
    ] as readonly React.ReactNode[],
    footnote: (
      <>
        The eight floats are illustrative stand-ins — the interactive demo fetches the real {HIDDEN_COLS}-dim rows
        from the loaded model. The mechanism is the lesson: token id in, one row of the table out.
      </>
    ),
  },
  zh: {
    title: '嵌入查询——一个 id，一行',
    svgAria:
      '嵌入查询示意图：token " cat" 的 id 9059 沿着一张 248,320 行 × 1,024 列的高大矩阵滑到自己那一行，该行亮起，随后这行 1,024 个浮点数中的前 8 个以格子的形式弹出——没有任何计算，只是读取第 9059 行。',
    tableLabel: `嵌入表 · ${VOCAB_ROWS} 行 × ${HIDDEN_COLS} 列`,
    rowId: '行号',
    rowCaption: `第 ${TOKEN_ID} 行 · "${TOKEN_TEXT}" 的嵌入（示意值）`,
    captions: [
      <>
        分词器递给模型一个整数：<span className="font-mono">{TOKEN_ID}</span>（token{' '}
        <span className="font-mono">&quot;{TOKEN_TEXT}&quot;</span>）。嵌入表只是一个很高的矩阵——
        <span className="font-mono">{VOCAB_ROWS}</span> 行，每个词表条目一行。
      </>,
      <>
        这个 id 被当作普通的<strong>数组下标</strong>使用：直奔第{' '}
        <span className="font-mono text-primary">{TOKEN_ID}</span> 行。不搜索、不相乘——行号 = token id。
      </>,
      <>
        把这一行读出来：<span className="font-mono">{HIDDEN_COLS}</span> 个浮点数（显示前 8 个）。这个向量
        <em>就是</em> <span className="font-mono">&quot;{TOKEN_TEXT}&quot;</span> 的嵌入。
      </>,
      <>
        <strong>没有发生任何计算</strong>——只是读取第 {TOKEN_ID} 行。每次前向传播都从这次查询开始，每个输入 token
        查一次。
      </>,
    ] as readonly React.ReactNode[],
    footnote: (
      <>
        这 8 个浮点数是示意性替身——实时演示会从已加载的模型里取出真实的 {HIDDEN_COLS}{' '}
        维行向量。机制本身才是课程重点：输入 token id，输出表中的一行。
      </>
    ),
  },
} as const;

// ── SVG geometry ──────────────────────────────────────────────────────────────
const VB_W = 760;
const VB_H = 304;
const MAT_X = 230; // matrix card left edge (leaves room for chip + row labels)
const MAT_Y = 30;
const MAT_W = 180;
const ROW_H = 20;
const MAT_H = VIS_ROWS.length * ROW_H; // 220
const CHIP_W = 130;
const CHIP_H = 30;
const CHIP_X = 20;
const CHIP_TOP_Y = MAT_Y; // frame 0 position
const targetRowY = MAT_Y + TARGET_VIS_IDX * ROW_H; // top of row 9059
const CHIP_ROW_Y = targetRowY + ROW_H / 2 - CHIP_H / 2; // frame ≥1 position
// Value cells popping out of the row.
const CELL_W = 41;
const CELL_H = 24;
const CELL_GAP = 3;
const CELLS_X = MAT_X + MAT_W + 34;
const CELLS_Y = targetRowY + ROW_H / 2 - CELL_H / 2;

function fmtFloat(x: number): string {
  // Keep the minus sign typographically tidy in the tiny cells.
  return (x < 0 ? '−' : '') + Math.abs(x).toFixed(3);
}

export function EmbeddingLookupDiagram() {
  const locale = useLocale();
  const copy = COPY[locale];
  const player = useStepPlayer(TOTAL_FRAMES, { frameMs: FRAME_MS });
  const { frame, reducedMotion } = player;

  const atRow = frame >= 1; // chip beside its row, row lit
  const cellsOut = frame >= 2; // floats popped out

  const chipY = atRow ? CHIP_ROW_Y : CHIP_TOP_Y;

  const caption: React.ReactNode = copy.captions[Math.min(frame, DONE_FRAME)];
  // Every caption this sweep can reach, so DiagramFrame reserves the tallest and
  // the chapter body cannot hop when the beat changes. This one is exhaustive by
  // construction rather than by enumeration: there is a SINGLE caption
  // expression in this file — `copy.captions[Math.min(frame, DONE_FRAME)]` — no
  // branch, no builder, no toggle that swaps caption sets. `useStepPlayer` keeps
  // `frame` in [0, TOTAL_FRAMES) and DONE_FRAME is TOTAL_FRAMES - 1, so the
  // reachable index set is exactly {0,1,2,3}, i.e. all four entries.
  const captions = [...copy.captions];

  return (
    <DiagramFrame
      title={copy.title}
      player={player}
      locale={locale}
      caption={caption}
      captions={captions}
      note={copy.footnote}
    >
      {/* ── Why the svg carries a min-width ──
          A `w-full` svg does not REFLOW in a narrow column, it SCALES — the
          whole canvas shrinks, text and all. At a 375px viewport (a 320-unit
          column) this 760-wide drawing renders at 320, which lands the value
          cells at 3.8 CSS px: eight grey smudges where the embedding is
          supposed to be. The floor stops the shrink and the `PanBox` around
          the svg pans sideways to reach the rest.
          8px legibility floor ÷ the smallest size a reader must READ:
          8 * 760 / 9 = 675.6 → 676. That smallest size is the 9px in the value
          cells, and it is not a superscript or a decoration you can skip — the
          frame-2 caption says "read the row off", and those eight floats ARE
          the row. So it sets the floor rather than ducking under it.
          The floor is the svg's alone, and so is the `PanBox`. This widget has
          no sibling controls today; anything added beside the svg stays OUTSIDE
          the box, where it sizes to the visible width instead of panning out of
          reach. ── */}
      <PanBox locale={locale}>
        <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full min-w-[676px]" role="img" aria-label={copy.svgAria}>
          {/* ── The matrix card ──
              A bare rect, NOT a PanelFrame. The double border is the course's
              "this is a STAGE" marker; the embedding table is a plain list box —
              248,320 repeated rows — and skin.tsx says a list gets a single
              border and no inner frame. It is also a real owned container, so
              SW.OUTER. ── */}
          <rect
            x={MAT_X}
            y={MAT_Y}
            width={MAT_W}
            height={MAT_H}
            rx={RX}
            style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
            strokeWidth={SW.OUTER}
          />
          {VIS_ROWS.map((row, i) => {
            const y = MAT_Y + i * ROW_H;
            const isTarget = !!row.target;
            const lit = isTarget && atRow;
            return (
              <g key={i}>
                {/* row band — drawn INSIDE the matrix card, so SW.INNER. The
                    0.18 fill stays a tint: it sits behind the row's own ticks and
                    beside its label, and the DIM ladder's lowest usable step
                    (IDLE_EDGE 0.4) is a wash, not a highlight. */}
                {lit ? (
                  <rect
                    x={MAT_X + 1.5}
                    y={y + 1}
                    width={MAT_W - 3}
                    height={ROW_H - 2}
                    rx={RX}
                    style={{ fill: 'var(--primary)', fillOpacity: 0.18, stroke: 'var(--primary)' }}
                    strokeWidth={SW.INNER}
                  />
                ) : null}
                {/* row separator — the outline of one repeated slot in the table */}
                {i > 0 ? (
                  <line
                    x1={MAT_X + 6}
                    x2={MAT_X + MAT_W - 6}
                    y1={y}
                    y2={y}
                    style={{ stroke: 'var(--border)' }}
                    strokeOpacity={DIM.EMPTY_SLOT}
                    strokeWidth={SW.INNER}
                  />
                ) : null}
                {/* row label (left of the card) */}
                <text
                  x={MAT_X - 8}
                  y={y + ROW_H / 2 + 3.5}
                  textAnchor="end"
                  style={{ fill: lit ? 'var(--primary)' : 'var(--muted-foreground)' }}
                  className={lit ? 'font-mono text-[11px] font-semibold' : 'font-mono text-[10px]'}
                >
                  {row.ellipsis ? '⋮' : row.label}
                </text>
                {/* faint column ticks to suggest 1,024 columns. Inner marks, so
                    SW.INNER, and the opacity follows PanelFrame's own neutral/hued
                    pair — a hued inner mark sits a step forward of a neutral one. */}
                {!row.ellipsis
                  ? Array.from({ length: 7 }, (_, c) => (
                      <line
                        key={c}
                        x1={MAT_X + 22 + c * 22}
                        x2={MAT_X + 22 + c * 22}
                        y1={y + 5}
                        y2={y + ROW_H - 5}
                        style={{ stroke: lit ? 'var(--primary)' : 'var(--border)' }}
                        strokeOpacity={lit ? DIM.HUED_INNER : DIM.NEUTRAL_INNER}
                        strokeWidth={SW.INNER}
                      />
                    ))
                  : null}
                {row.ellipsis ? (
                  <text
                    x={MAT_X + MAT_W / 2}
                    y={y + ROW_H / 2 + 4}
                    textAnchor="middle"
                    style={{ fill: 'var(--muted-foreground)' }}
                    className="text-[11px]"
                  >
                    ⋯
                  </text>
                ) : null}
              </g>
            );
          })}
          {/* matrix size label */}
          <text
            x={MAT_X + MAT_W / 2}
            y={MAT_Y + MAT_H + 18}
            textAnchor="middle"
            style={{ fill: 'var(--muted-foreground)' }}
            className="text-[11px]"
          >
            {copy.tableLabel}
          </text>
          <text x={MAT_X - 8} y={MAT_Y - 10} textAnchor="end" style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
            {copy.rowId}
          </text>

          {/* ── The token chip (slides down to its row) ── */}
          <g
            style={{
              transform: `translate(${CHIP_X}px, ${chipY}px)`,
              transition: reducedMotion ? undefined : 'transform 600ms ease-in-out',
            }}
          >
            <rect
              x={0}
              y={0}
              width={CHIP_W}
              height={CHIP_H}
              rx={RX}
              style={{
                fill: atRow ? 'var(--primary)' : 'var(--card)',
                fillOpacity: atRow ? 0.14 : 1,
                stroke: atRow ? 'var(--primary)' : 'var(--border)',
              }}
              strokeWidth={SW.OUTER}
            />
            <text
              x={10}
              y={CHIP_H / 2 + 4}
              style={{ fill: atRow ? 'var(--primary)' : 'var(--foreground)' }}
              className="font-mono text-[13px] font-semibold"
            >
              {TOKEN_TEXT} · id {TOKEN_ID}
            </text>
          </g>
          {/* arrow chip → row, once the chip has arrived. A connector, so
              SW.INNER. It is NOT drawn dashed-and-idle from frame 0 the way an
              always-present edge would be: the chip it starts from physically
              moves, so before it lands this line would hang in mid-air beside the
              row, pointing out of nothing. */}
          {atRow ? (
            <g style={{ stroke: 'var(--primary)' }} strokeWidth={SW.INNER}>
              <line x1={CHIP_X + CHIP_W + 4} x2={MAT_X - 56} y1={targetRowY + ROW_H / 2} y2={targetRowY + ROW_H / 2} />
              <path
                d={`M ${MAT_X - 62} ${targetRowY + ROW_H / 2 - 4} l 6 4 l -6 4`}
                fill="none"
              />
            </g>
          ) : null}

          {/* ── Value cells popping out of the row ── */}
          {cellsOut ? (
            <g>
              <g style={{ stroke: 'var(--primary)' }} strokeWidth={SW.INNER}>
                <line
                  x1={MAT_X + MAT_W + 2}
                  x2={CELLS_X - 10}
                  y1={targetRowY + ROW_H / 2}
                  y2={targetRowY + ROW_H / 2}
                />
                <path d={`M ${CELLS_X - 16} ${targetRowY + ROW_H / 2 - 4} l 6 4 l -6 4`} fill="none" />
              </g>
              {ROW_FLOATS.map((v, i) => {
                const col = i % 4;
                const rowIdx = Math.floor(i / 4);
                const x = CELLS_X + col * (CELL_W + CELL_GAP);
                const y = CELLS_Y - (CELL_H + CELL_GAP) / 2 + rowIdx * (CELL_H + CELL_GAP);
                return (
                  <g key={i}>
                    <rect
                      x={x}
                      y={y}
                      width={CELL_W}
                      height={CELL_H}
                      rx={RX}
                      style={{ fill: 'var(--primary)', fillOpacity: 0.1, stroke: 'var(--primary)' }}
                      strokeWidth={SW.INNER}
                    />
                    <text
                      x={x + CELL_W / 2}
                      y={y + CELL_H / 2 + 3.5}
                      textAnchor="middle"
                      style={{ fill: 'var(--foreground)' }}
                      className="font-mono text-[9px]"
                    >
                      {fmtFloat(v)}
                    </text>
                  </g>
                );
              })}
              <text
                x={CELLS_X + 4 * (CELL_W + CELL_GAP) + 6}
                y={targetRowY + ROW_H / 2 + 4}
                style={{ fill: 'var(--muted-foreground)' }}
                className="font-mono text-[11px]"
              >
                … ×{HIDDEN_COLS}
              </text>
              <text
                x={CELLS_X}
                y={CELLS_Y + CELL_H + CELL_GAP + 26}
                style={{ fill: 'var(--muted-foreground)' }}
                className="text-[10px]"
              >
                {copy.rowCaption}
              </text>
            </g>
          ) : null}
        </svg>
      </PanBox>
    </DiagramFrame>
  );
}
