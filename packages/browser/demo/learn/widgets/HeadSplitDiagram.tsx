import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * HeadSplitDiagram — chapter 4, the mirror of <MultiheadConcat />.
 *
 * MultiheadConcat shows the MERGE at the end of attention (8 head outputs →
 * concat → W_O → residual stream). This widget shows the SPLIT at the start:
 * one 1024-wide token vector goes through W_q to a flat 2048-wide strip, and
 * "splitting into heads" is nothing but cutting that strip into 8 bands of
 * 256 — a free reshape, no math. The K/V path runs the same way but is
 * narrower: 1024 → 512, which cuts into only 2 bands of 256 (that's GQA).
 *
 * Self-contained: NO worker, NO model, NO WASM — pure SVG + React state, so
 * it server-renders for crawlers and works while the model downloads.
 *
 * Animation model mirrors StreamingSoftmaxDiagram: a small set of discrete
 * frames advanced by a setInterval inside a guarded useEffect. Honors
 * prefers-reduced-motion (jumps straight to the final frame, no autoplay).
 *
 * Head colors reuse MultiheadConcat's oklch hue wheel so a band here and a
 * band in the concat diagram with the same color are the same head.
 */

const H = 8; // query heads (Qwen3.5-0.8B)
const KV = 2; // KV heads (GQA)
const D_HEAD = 256;
const HIDDEN = 1024;
const Q_OUT = H * D_HEAD; // 2048
const KV_OUT = KV * D_HEAD; // 512

const FRAME_MS = 2600;

// Frames:
//   0 — input vector only
//   1 — W_q appears, output strip appears (still one flat 2048 strip)
//   2 — the strip is cut into 8 colored bands (the actual "split")
//   3 — K/V path appears: only 2 bands of 256
//   4 — done (repeats the final state)
const TOTAL_FRAMES = 5;
const DONE_FRAME = TOTAL_FRAMES - 1;

// Same hue wheel as MultiheadConcat so head identity colors match across the
// two diagrams (data colors, not theme surfaces — surfaces use CSS tokens).
function headColor(i: number, alpha = 0.7): string {
  const hue = (i * 360) / H;
  return `oklch(0.7 0.13 ${hue} / ${alpha})`;
}

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

// ── SVG geometry ────────────────────────────────────────────────────────────
const VB_W = 760;
const VB_H = 318;
// Width scale: 2048 features → 640 px, so 1024 → 320 and each 256-band → 80.
const PX_PER_FEATURE = 640 / Q_OUT;
const STRIP_H = 28;
const IN_W = HIDDEN * PX_PER_FEATURE; // 320
const IN_X = (VB_W - IN_W) / 2;
const IN_Y = 26;
const PROJ_Y = 88;
const PROJ_W = 150;
const PROJ_H = 30;
const PROJ_X = (VB_W - PROJ_W) / 2;
const Q_W = Q_OUT * PX_PER_FEATURE; // 640
const Q_X = (VB_W - Q_W) / 2;
const Q_Y = 158;
const BAND_W = D_HEAD * PX_PER_FEATURE; // 80
const KV_Y = 252;
const KV_W = KV_OUT * PX_PER_FEATURE; // 160
const KV_X = (VB_W - KV_W) / 2;

// Per-locale copy: every user-visible string lives here so /zh localizes while
// the English output stays byte-identical.
const COPY = {
  en: {
    header: <>Split into heads — one vector in, {H} bands out</>,
    pause: '❚❚ Pause',
    play: '▶ Play',
    step: 'Step ›',
    svgAria: `Diagram of the head split: a ${HIDDEN}-wide token vector is projected by W_q to ${Q_OUT} numbers, then reshaped into ${H} bands of ${D_HEAD} — one per query head. The K and V projections produce only ${KV_OUT} numbers, which reshape into just ${KV} bands of ${D_HEAD}: grouped-query attention.`,
    inputLabel: <>one token&apos;s vector x · [{HIDDEN}]</>,
    inputNumbers: `${HIDDEN} numbers`,
    queriesBanded: `queries · [${Q_OUT}] = ${H} heads × ${D_HEAD}`,
    queriesFlat: `queries · [${Q_OUT}] — one flat strip`,
    headBand: (i: number) => `head ${i}`,
    eachBand: `${D_HEAD} each`,
    kvPathLabel: `same x through W_k (and W_v) · [${HIDDEN} × ${KV_OUT}] →`,
    kvBand: (g: number) => `K/V ${g}`,
    kvOnly: `only ${KV} bands of ${D_HEAD} — GQA`,
    caption0: (
      <>
        One token arrives as a single <span className="font-mono">{HIDDEN}</span>-wide vector. No heads anywhere yet.
      </>
    ),
    caption1: (
      <>
        <span className="font-mono">W_q</span> maps{' '}
        <span className="font-mono">
          {HIDDEN} → {Q_OUT}
        </span>{' '}
        numbers. Still one flat strip — the heads are not in the weights.
      </>
    ),
    caption2: (
      <>
        <strong>The split:</strong> cut the {Q_OUT}-wide strip into {H} bands of {D_HEAD}. That cut — a free reshape, no
        math — is all &quot;splitting into heads&quot; means. Each band is one head&apos;s query.
      </>
    ),
    caption3: (
      <>
        The K and V projections are narrower:{' '}
        <span className="font-mono">
          {HIDDEN} → {KV_OUT}
        </span>
        , which cuts into only <strong>{KV} bands</strong> of {D_HEAD} — that&apos;s GQA. These {KV} K/V heads get
        shared by all {H} query heads.
      </>
    ),
    captionDone: (
      <>
        Split done: {H} query bands, {KV} K/V bands. After attention runs per band, the mirror step — concat the {H}{' '}
        outputs and project back to {HIDDEN} — is the next diagram below.
      </>
    ),
    footnote: (
      <>
        Color = head identity, matching the concat diagram further down — the band that splits off here is the same head
        whose output merges back there. Widths are to scale: {Q_OUT} query features vs {KV_OUT} K/V features.
      </>
    ),
  },
  zh: {
    header: <>拆分成头——一个向量进来，{H} 条带出去</>,
    pause: '❚❚ 暂停',
    play: '▶ 播放',
    step: '单步 ›',
    svgAria: `头拆分示意图：一个 ${HIDDEN} 宽的 token 向量经 W_q 投影成 ${Q_OUT} 个数，再 reshape 成 ${H} 条 ${D_HEAD} 宽的带——每条对应一个 query 头。K 和 V 的投影只产出 ${KV_OUT} 个数，reshape 后只有 ${KV} 条 ${D_HEAD} 宽的带：分组查询注意力。`,
    inputLabel: <>一个 token 的向量 x · [{HIDDEN}]</>,
    inputNumbers: `${HIDDEN} 个数`,
    queriesBanded: `query · [${Q_OUT}] = ${H} 个头 × ${D_HEAD}`,
    queriesFlat: `query · [${Q_OUT}]——一条扁平的带`,
    headBand: (i: number) => `头 ${i}`,
    eachBand: `每条 ${D_HEAD}`,
    kvPathLabel: `同一个 x 过 W_k（和 W_v）· [${HIDDEN} × ${KV_OUT}] →`,
    kvBand: (g: number) => `K/V ${g}`,
    kvOnly: `只有 ${KV} 条 ${D_HEAD} 宽的带——GQA`,
    caption0: (
      <>
        一个 token 以单个 <span className="font-mono">{HIDDEN}</span> 宽的向量到达。此刻还没有任何头。
      </>
    ),
    caption1: (
      <>
        <span className="font-mono">W_q</span> 把{' '}
        <span className="font-mono">
          {HIDDEN} → {Q_OUT}
        </span>{' '}
        个数映射出来。仍然是一条扁平的带——头并不藏在权重里。
      </>
    ),
    caption2: (
      <>
        <strong>拆分：</strong>把 {Q_OUT} 宽的带切成 {H} 条 {D_HEAD} 宽的带。这一刀——一次免费的
        reshape，没有任何数学——就是&ldquo;拆分成头&rdquo;的全部含义。每条带就是一个头的 query。
      </>
    ),
    caption3: (
      <>
        K 和 V 的投影更窄：{' '}
        <span className="font-mono">
          {HIDDEN} → {KV_OUT}
        </span>
        ，只能切出 <strong>{KV} 条</strong> {D_HEAD} 宽的带——这就是 GQA。这 {KV} 个 K/V 头被全部 {H} 个 query 头共享。
      </>
    ),
    captionDone: (
      <>
        拆分完成：{H} 条 query 带，{KV} 条 K/V 带。注意力按带各自运行之后，镜像的一步——把 {H} 个输出拼接起来、投影回{' '}
        {HIDDEN}——就是下方的下一张图。
      </>
    ),
    footnote: (
      <>
        颜色 =
        头的身份，与下方的拼接图一一对应——在这里拆出去的那条带，正是在那里合并回来的同一个头。宽度按真实比例绘制：
        {Q_OUT} 个 query 特征 vs {KV_OUT} 个 K/V 特征。
      </>
    ),
  },
} as const;

export function HeadSplitDiagram() {
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

  const showProj = frame >= 1;
  const showBands = frame >= 2;
  const showKv = frame >= 3;

  let caption: React.ReactNode;
  if (frame === 0) {
    caption = copy.caption0;
  } else if (frame === 1) {
    caption = copy.caption1;
  } else if (frame === 2) {
    caption = copy.caption2;
  } else if (frame === 3) {
    caption = copy.caption3;
  } else {
    caption = copy.captionDone;
  }

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + controls */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
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

      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.svgAria}>
        {/* ── Input vector ── */}
        <text x={IN_X} y={IN_Y - 8} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
          {copy.inputLabel}
        </text>
        <rect
          x={IN_X}
          y={IN_Y}
          width={IN_W}
          height={STRIP_H}
          rx={5}
          style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
          strokeWidth={1.5}
        />
        <text
          x={IN_X + IN_W / 2}
          y={IN_Y + STRIP_H / 2 + 4}
          textAnchor="middle"
          style={{ fill: 'var(--foreground)' }}
          className="font-mono text-[11px]"
        >
          {copy.inputNumbers}
        </text>

        {/* ── Projection block ── */}
        <g style={{ opacity: showProj ? 1 : 0.12 }}>
          <line
            x1={VB_W / 2}
            y1={IN_Y + STRIP_H}
            x2={VB_W / 2}
            y2={PROJ_Y}
            style={{ stroke: 'var(--muted-foreground)' }}
            strokeOpacity={0.5}
            strokeWidth={1.2}
          />
          <rect
            x={PROJ_X}
            y={PROJ_Y}
            width={PROJ_W}
            height={PROJ_H}
            rx={6}
            style={{
              fill: showProj ? 'var(--primary)' : 'var(--card)',
              fillOpacity: showProj ? 0.12 : 1,
              stroke: showProj ? 'var(--primary)' : 'var(--border)',
            }}
            strokeWidth={showProj ? 1.8 : 1.5}
          />
          <text
            x={VB_W / 2}
            y={PROJ_Y + PROJ_H / 2 + 4}
            textAnchor="middle"
            style={{ fill: 'var(--foreground)' }}
            className="font-mono text-[11px]"
          >
            W_q · [{HIDDEN} × {Q_OUT}]
          </text>
          <line
            x1={VB_W / 2}
            y1={PROJ_Y + PROJ_H}
            x2={VB_W / 2}
            y2={Q_Y - 14}
            style={{ stroke: 'var(--muted-foreground)' }}
            strokeOpacity={0.5}
            strokeWidth={1.2}
          />
        </g>

        {/* ── Q output strip: flat at frame 1, banded from frame 2 ── */}
        <g style={{ opacity: showProj ? 1 : 0.12 }}>
          <text x={Q_X} y={Q_Y - 6} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
            {showBands ? copy.queriesBanded : copy.queriesFlat}
          </text>
          {showBands ? (
            <>
              {Array.from({ length: H }, (_, i) => (
                <g key={i}>
                  <rect
                    x={Q_X + i * BAND_W}
                    y={Q_Y}
                    width={BAND_W}
                    height={STRIP_H}
                    fill={headColor(i, 0.65)}
                    style={{ stroke: 'var(--border)' }}
                    strokeWidth={1}
                  />
                  <text
                    x={Q_X + i * BAND_W + BAND_W / 2}
                    y={Q_Y + STRIP_H + 13}
                    textAnchor="middle"
                    style={{ fill: 'var(--muted-foreground)' }}
                    className="text-[10px]"
                  >
                    {copy.headBand(i)}
                  </text>
                </g>
              ))}
            </>
          ) : (
            <rect
              x={Q_X}
              y={Q_Y}
              width={Q_W}
              height={STRIP_H}
              rx={5}
              style={{ fill: 'var(--primary)', fillOpacity: 0.18, stroke: 'var(--primary)' }}
              strokeWidth={1.5}
            />
          )}
          {/* outer border around the banded strip */}
          <rect
            x={Q_X}
            y={Q_Y}
            width={Q_W}
            height={STRIP_H}
            rx={5}
            fill="none"
            style={{ stroke: 'var(--foreground)' }}
            strokeOpacity={0.35}
            strokeWidth={1.2}
          />
          <text
            x={Q_X + Q_W + 8}
            y={Q_Y + STRIP_H / 2 + 4}
            style={{ fill: 'var(--muted-foreground)' }}
            className="font-mono text-[10px]"
          >
            {copy.eachBand}
          </text>
        </g>

        {/* ── K/V path (GQA): only 2 bands ── */}
        <g style={{ opacity: showKv ? 1 : 0.12 }}>
          <text
            x={KV_X - 8}
            y={KV_Y + STRIP_H / 2 + 4}
            textAnchor="end"
            style={{ fill: 'var(--muted-foreground)' }}
            className="text-[11px]"
          >
            {copy.kvPathLabel}
          </text>
          {Array.from({ length: KV }, (_, g) => (
            <g key={g}>
              <rect
                x={KV_X + g * BAND_W}
                y={KV_Y}
                width={BAND_W}
                height={STRIP_H}
                fill={headColor(g * (H / KV), 0.65)}
                style={{ stroke: 'var(--border)' }}
                strokeWidth={1}
              />
              <text
                x={KV_X + g * BAND_W + BAND_W / 2}
                y={KV_Y + STRIP_H + 13}
                textAnchor="middle"
                style={{ fill: 'var(--muted-foreground)' }}
                className="text-[10px]"
              >
                {copy.kvBand(g)}
              </text>
            </g>
          ))}
          <rect
            x={KV_X}
            y={KV_Y}
            width={KV_W}
            height={STRIP_H}
            rx={5}
            fill="none"
            style={{ stroke: 'var(--foreground)' }}
            strokeOpacity={0.35}
            strokeWidth={1.2}
          />
          <text
            x={KV_X + KV_W + 8}
            y={KV_Y + STRIP_H / 2 + 4}
            style={{ fill: 'var(--muted-foreground)' }}
            className="font-mono text-[10px]"
          >
            {copy.kvOnly}
          </text>
        </g>
      </svg>

      {/* Caption — what this frame shows */}
      <p className="min-h-[3.25rem] text-[13px] text-foreground/90">{caption}</p>

      <p className="text-[10px] text-muted-foreground/70">{copy.footnote}</p>
    </div>
  );
}
