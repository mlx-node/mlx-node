import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

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
 * Animation model mirrors StreamingSoftmaxDiagram: a small set of discrete
 * frames advanced by a setInterval inside a guarded useEffect. Honors
 * prefers-reduced-motion (jumps straight to the final frame, no autoplay).
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
const COPY = {
  en: {
    title: 'Embedding lookup — one id, one row',
    pause: '❚❚ Pause',
    play: '▶ Play',
    step: 'Step ›',
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
        The eight floats are illustrative stand-ins — the live demo on the right fetches the real {HIDDEN_COLS}-dim rows
        from the loaded model. The mechanism is the lesson: token id in, one row of the table out.
      </>
    ),
  },
  zh: {
    title: '嵌入查询——一个 id，一行',
    pause: '❚❚ 暂停',
    play: '▶ 播放',
    step: '单步 ›',
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
        这 8 个浮点数是示意性替身——右侧的实时演示会从已加载的模型里取出真实的 {HIDDEN_COLS}{' '}
        维行向量。机制本身才是课程重点：输入 token id，输出表中的一行。
      </>
    ),
  },
} as const;

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = React.useState<boolean>(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });
  React.useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, []);
  return reduced;
}

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
  const copy = COPY[useLocale()];
  const reducedMotion = usePrefersReducedMotion();
  // Under reduced motion, start on the done frame and never auto-play.
  const [frame, setFrame] = React.useState(reducedMotion ? DONE_FRAME : 0);
  const [playing, setPlaying] = React.useState(!reducedMotion);

  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    const t = window.setInterval(() => setFrame((f) => (f + 1) % TOTAL_FRAMES), FRAME_MS);
    return () => window.clearInterval(t);
  }, [playing, reducedMotion]);

  const step = () => {
    setPlaying(false);
    setFrame((f) => (f + 1) % TOTAL_FRAMES);
  };

  const atRow = frame >= 1; // chip beside its row, row lit
  const cellsOut = frame >= 2; // floats popped out

  const chipY = atRow ? CHIP_ROW_Y : CHIP_TOP_Y;

  const caption: React.ReactNode = copy.captions[Math.min(frame, DONE_FRAME)];

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + controls */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="flex items-center gap-1">
          {!reducedMotion ? (
            <>
              <button
                type="button"
                onClick={() => setPlaying((p) => !p)}
                aria-pressed={playing}
                className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              >
                {playing ? copy.pause : copy.play}
              </button>
              <button
                type="button"
                onClick={step}
                className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              >
                {copy.step}
              </button>
            </>
          ) : null}
        </div>
      </div>

      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="w-full"
        role="img"
        aria-label={copy.svgAria}
      >
        {/* ── The matrix card ── */}
        <rect
          x={MAT_X}
          y={MAT_Y}
          width={MAT_W}
          height={MAT_H}
          rx={6}
          style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
          strokeWidth={1.5}
        />
        {VIS_ROWS.map((row, i) => {
          const y = MAT_Y + i * ROW_H;
          const isTarget = !!row.target;
          const lit = isTarget && atRow;
          return (
            <g key={i}>
              {/* row band */}
              {lit ? (
                <rect
                  x={MAT_X + 1.5}
                  y={y + 1}
                  width={MAT_W - 3}
                  height={ROW_H - 2}
                  rx={3}
                  style={{ fill: 'var(--primary)', fillOpacity: 0.18, stroke: 'var(--primary)' }}
                  strokeWidth={1.5}
                />
              ) : null}
              {/* row separator */}
              {i > 0 ? (
                <line
                  x1={MAT_X + 6}
                  x2={MAT_X + MAT_W - 6}
                  y1={y}
                  y2={y}
                  style={{ stroke: 'var(--border)' }}
                  strokeOpacity={0.6}
                  strokeWidth={1}
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
              {/* faint column ticks to suggest 1,024 columns */}
              {!row.ellipsis
                ? Array.from({ length: 7 }, (_, c) => (
                    <line
                      key={c}
                      x1={MAT_X + 22 + c * 22}
                      x2={MAT_X + 22 + c * 22}
                      y1={y + 5}
                      y2={y + ROW_H - 5}
                      style={{ stroke: lit ? 'var(--primary)' : 'var(--border)' }}
                      strokeOpacity={0.5}
                      strokeWidth={1}
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
            rx={6}
            style={{
              fill: atRow ? 'var(--primary)' : 'var(--card)',
              fillOpacity: atRow ? 0.14 : 1,
              stroke: atRow ? 'var(--primary)' : 'var(--border)',
            }}
            strokeWidth={atRow ? 2 : 1.5}
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
        {/* arrow chip → row, once the chip has arrived */}
        {atRow ? (
          <g style={{ stroke: 'var(--primary)' }} strokeWidth={1.5}>
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
            <g style={{ stroke: 'var(--primary)' }} strokeWidth={1.5}>
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
                    rx={4}
                    style={{ fill: 'var(--primary)', fillOpacity: 0.1, stroke: 'var(--primary)' }}
                    strokeWidth={1}
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

      {/* Caption — what this frame shows */}
      <p className="min-h-[2.5rem] text-[13px] text-foreground/90">{caption}</p>

      <p className="text-[10px] text-muted-foreground/70">{copy.footnote}</p>
    </div>
  );
}
