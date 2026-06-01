import { cn } from '@/lib/utils';
import * as React from 'react';

/**
 * Chapter 6 supplement — RoPE's RELATIVE-POSITION property, made tangible.
 *
 * The prose claims that the rotated dot product `RoPE(q, m) · RoPE(k, n)`
 * depends ONLY on the offset `Δ = m − n`, never on `m` and `n` separately.
 * This widget makes that single fact physical.
 *
 * Setup: both the query and the key start from the SAME base unit vector at
 * position 0 (pointing along +x). RoPE rotates the query by `m·ω` and the key
 * by `n·ω`, where `ω` is ONE representative angular frequency (rad/token).
 * Because both arrows share the base vector:
 *
 *   • the query is rotated relative to the key by exactly (m − n)·ω = Δ·ω, and
 *   • since both are unit vectors, q·k = cos(Δ·ω).
 *
 * Two teaching beats are made interactive:
 *   1. INVARIANCE (the payoff): the "shift both ±1" buttons bump m and n
 *      together. Δ is unchanged, so both arrows sweep in lockstep, their
 *      relative rotation Δ·ω stays frozen, and the q·k readout does NOT move.
 *   2. OFFSET CHANGES THE SCORE: moving m or n alone changes Δ, opening or
 *      closing the gap between the arrows. q·k rises toward 1 as they align
 *      (Δ → 0) and falls as they pull apart.
 *
 * Illustrative only: ONE frequency, and the content vectors are simplified to
 * a shared base so we isolate the positional part. No model, worker, or
 * network — pure formula-derived SVG.
 */

// One representative angular frequency (radians per token). ω ≈ 0.5 makes a
// single-token step a clearly visible ~28.6° rotation. Across the full 0…16
// slider range the relative rotation Δ·ω can exceed a full turn (16·0.5 = 8 rad
// ≈ 1.27 turns) — that's fine and honest: each token step adds a fixed ω, so it
// accumulates. The score q·k = cos(Δ·ω) is 2π-periodic, so it still depends only
// on the offset Δ. The real model rotates many pairs at many different speeds —
// see the frequency spectrum in the demo.
const OMEGA = 0.5;

const POS_MIN = 0;
const POS_MAX = 16;

// SVG geometry. Square plane so one math unit is the same pixel count on both
// axes (angles stay visually honest). Origin at center.
const SIZE = 320;
const CENTER = SIZE / 2;
// Pixels per unit length. Arrows are unit vectors (length 1), so this keeps the
// arrowheads comfortably inside the frame with room for labels.
const R = 110;

const Q_COLOR = 'oklch(0.72 0.18 350)'; // pink — query at position m
const K_COLOR = 'oklch(0.75 0.12 185)'; // teal — key at position n

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

// Math coords (origin at center, +y up, angle CCW from +x) → SVG pixel coords
// (origin top-left, +y DOWN). The y flip means a positive math angle is drawn
// counter-clockwise, i.e. screen-UP.
function toPx(x: number, y: number): [number, number] {
  return [CENTER + x * R, CENTER - y * R];
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

type Qual = { label: string; tone: string };

// Qualitative tag keyed on cos(Δ·ω) = q·k.
function qualitative(dot: number): Qual {
  if (dot >= 0.85) return { label: 'aligned', tone: 'text-emerald-400' };
  if (dot >= 0.2) return { label: 'partly aligned', tone: 'text-foreground/80' };
  if (dot > -0.2) return { label: 'orthogonal', tone: 'text-muted-foreground' };
  if (dot > -0.85) return { label: 'pulling apart', tone: 'text-foreground/80' };
  return { label: 'opposed', tone: 'text-amber-400' };
}

function Arrow({
  angleRad,
  color,
  label,
  markerId,
}: {
  angleRad: number;
  color: string;
  label: string;
  markerId: string;
}) {
  const tipX = Math.cos(angleRad);
  const tipY = Math.sin(angleRad);
  const [ox, oy] = toPx(0, 0);
  const [tx, ty] = toPx(tipX, tipY);
  // Nudge the label just past the arrowhead, along the arrow direction.
  const [lx, ly] = toPx(tipX * 1.16, tipY * 1.16);
  return (
    <g>
      <line
        x1={ox}
        y1={oy}
        x2={tx}
        y2={ty}
        stroke={color}
        strokeWidth={3}
        strokeLinecap="round"
        markerEnd={`url(#${markerId})`}
      />
      <text
        x={lx}
        y={ly}
        fontSize="12"
        textAnchor={tipX < -0.05 ? 'end' : tipX > 0.05 ? 'start' : 'middle'}
        dominantBaseline="middle"
        className="pointer-events-none select-none font-mono"
        fill={color}
      >
        {label}
      </text>
    </g>
  );
}

export function RopeRelativeDial() {
  const reducedMotion = usePrefersReducedMotion();

  const [m, setM] = React.useState(6); // query position
  const [n, setN] = React.useState(2); // key position

  // RoPE angles. Both arrows start from the same base vector (angle 0) and are
  // rotated by position × ω. The query's rotation RELATIVE to the key is
  // therefore exactly Δ·ω — which accumulates and can exceed a full turn.
  const qAngle = m * OMEGA;
  const kAngle = n * OMEGA;
  const delta = m - n; // offset Δ = m − n
  const relRotation = delta * OMEGA; // signed relative rotation Δ·ω = (m − n)·ω
  const dot = Math.cos(relRotation); // q·k = cos(Δ·ω) for unit vectors
  const qual = qualitative(dot);

  // "Shift both ±1" keeps Δ honest: we bump m and n by the same signed amount,
  // but only when BOTH stay in range — otherwise shifting one and clamping the
  // other would silently change Δ. If either would cross a bound, do nothing
  // (and the button is also disabled below, so this is belt-and-suspenders).
  // Functional updates avoid any stale-closure hazard, but because m and n live
  // in separate state cells we compute the guard from the current render's
  // values and apply both unconditionally inside the handler.
  const canShiftUp = m < POS_MAX && n < POS_MAX;
  const canShiftDown = m > POS_MIN && n > POS_MIN;
  function shiftBoth(by: number) {
    setM((prev) => clamp(prev + by, POS_MIN, POS_MAX));
    setN((prev) => clamp(prev + by, POS_MIN, POS_MAX));
  }

  // Visual gap arc between the two arrows' RENDERED on-screen directions. The
  // arrows are drawn via cos/sin, so they sit at their mod-2π directions; the
  // visible gap is the MINOR (≤π) signed arc from the key to the query. Reducing
  // the raw relative rotation to (−π, π] with atan2 gives exactly that minor-arc
  // sweep, so the arc always matches the gap you see — even when Δ·ω exceeds a
  // full turn. Drawn in math space then mapped through toPx.
  const arc = React.useMemo(() => {
    const aR = 0.46; // arc radius in unit-length terms, inside both arrows
    // Minor-arc sweep in (−π, π]: equivalent to relRotation mod 2π, folded.
    const sweepAngle = Math.atan2(Math.sin(relRotation), Math.cos(relRotation));
    const [sx, sy] = toPx(Math.cos(kAngle) * aR, Math.sin(kAngle) * aR);
    const [ex, ey] = toPx(Math.cos(kAngle + sweepAngle) * aR, Math.sin(kAngle + sweepAngle) * aR);
    const rPx = aR * R;
    // y is flipped in screen space, so a POSITIVE math angle sweep is
    // counter-clockwise on screen → SVG sweep-flag 0; negative → 1.
    const sweep = sweepAngle >= 0 ? 0 : 1;
    // Minor arc is always ≤ π, so the large-arc flag is never set.
    const largeArc = 0;
    return { d: `M ${sx} ${sy} A ${rPx} ${rPx} 0 ${largeArc} ${sweep} ${ex} ${ey}` };
  }, [kAngle, relRotation]);

  const [ringCx, ringCy] = toPx(0, 0);
  const [baseTx, baseTy] = toPx(1, 0); // base unit vector tip (angle 0)

  const ariaLabel =
    `RoPE relative-position dial. Query at position m = ${m} and key at position n = ${n}. ` +
    `Offset Δ = m − n = ${delta}. The query is rotated ${relRotation.toFixed(2)} radians relative to the key; ` +
    `q·k = ${dot.toFixed(2)} (${qual.label}).`;

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          q·k depends only on the offset m − n
        </div>
        <div className="text-[11px] text-muted-foreground">Shift both positions — the score stays put</div>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-[auto_minmax(0,1fr)]">
        {/* ---- The dial ---- */}
        <svg
          role="img"
          aria-label={ariaLabel}
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          className="mx-auto block h-auto w-full max-w-[320px] rounded-md bg-muted/20"
        >
          <defs>
            <marker
              id="rrd-head-q"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6.5"
              markerHeight="6.5"
              orient="auto"
            >
              <path d="M 0 0 L 10 5 L 0 10 Z" fill={Q_COLOR} />
            </marker>
            <marker
              id="rrd-head-k"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6.5"
              markerHeight="6.5"
              orient="auto"
            >
              <path d="M 0 0 L 10 5 L 0 10 Z" fill={K_COLOR} />
            </marker>
          </defs>

          {/* Unit circle + axes through the origin. */}
          <circle cx={ringCx} cy={ringCy} r={R} fill="none" stroke="currentColor" strokeOpacity={0.2} />
          <line
            x1={0}
            y1={CENTER}
            x2={SIZE}
            y2={CENTER}
            stroke="currentColor"
            strokeOpacity={0.1}
            strokeDasharray="2 4"
          />
          <line
            x1={CENTER}
            y1={0}
            x2={CENTER}
            y2={SIZE}
            stroke="currentColor"
            strokeOpacity={0.1}
            strokeDasharray="2 4"
          />

          {/* Shared base unit vector (position 0) — the common origin of both
              rotations, so the gap between the arrows is purely Δ·ω. */}
          <circle
            cx={baseTx}
            cy={baseTy}
            r={2.6}
            fill="none"
            stroke="currentColor"
            strokeOpacity={0.45}
            strokeWidth={1}
          />
          <text x={baseTx + 6} y={baseTy + 12} fontSize="9" className="select-none fill-foreground/45">
            base (pos 0)
          </text>

          {/* Visual gap arc between the two arrows' rendered directions — drawn
              first so arrows sit on top. No numeric label: all numbers (Δ,
              relative rotation Δ·ω, q·k) live in the readout tiles. */}
          <path
            d={arc.d}
            fill="none"
            stroke="var(--foreground)"
            strokeOpacity={0.5}
            strokeWidth={1.5}
            className={cn(!reducedMotion && 'transition-[d] duration-150')}
          />

          <circle cx={ringCx} cy={ringCy} r={2.5} fill="currentColor" fillOpacity={0.4} />

          {/* Key arrow (n) then query arrow (m) on top. */}
          <Arrow angleRad={kAngle} color={K_COLOR} label={`k · n=${n}`} markerId="rrd-head-k" />
          <Arrow angleRad={qAngle} color={Q_COLOR} label={`q · m=${m}`} markerId="rrd-head-q" />
        </svg>

        {/* ---- Controls + readouts ---- */}
        <div className="space-y-3">
          <div className="space-y-1">
            <label htmlFor="rrd-m" className="block text-xs text-muted-foreground">
              query position <span className="font-mono text-foreground/85">m = {m}</span>
            </label>
            <input
              id="rrd-m"
              type="range"
              min={POS_MIN}
              max={POS_MAX}
              step={1}
              value={m}
              onChange={(e) => setM(Number(e.target.value))}
              className="w-full accent-primary"
              style={{ accentColor: Q_COLOR }}
              aria-valuetext={`query position m equals ${m}`}
            />
          </div>

          <div className="space-y-1">
            <label htmlFor="rrd-n" className="block text-xs text-muted-foreground">
              key position <span className="font-mono text-foreground/85">n = {n}</span>
            </label>
            <input
              id="rrd-n"
              type="range"
              min={POS_MIN}
              max={POS_MAX}
              step={1}
              value={n}
              onChange={(e) => setN(Number(e.target.value))}
              className="w-full accent-primary"
              style={{ accentColor: K_COLOR }}
              aria-valuetext={`key position n equals ${n}`}
            />
          </div>

          <div className="space-y-1">
            <div className="text-[11px] text-muted-foreground">
              Shift <strong>both</strong> together (Δ unchanged) — watch the arrows sweep while q·k holds
            </div>
            <div className="flex gap-2" role="group" aria-label="Shift both positions together">
              <button
                type="button"
                onClick={() => shiftBoth(-1)}
                disabled={!canShiftDown}
                className="inline-flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1 text-xs font-medium text-foreground hover:bg-primary/20 disabled:opacity-50"
              >
                <span aria-hidden="true">−1</span>
                shift both
              </button>
              <button
                type="button"
                onClick={() => shiftBoth(1)}
                disabled={!canShiftUp}
                className="inline-flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1 text-xs font-medium text-foreground hover:bg-primary/20 disabled:opacity-50"
              >
                <span aria-hidden="true">+1</span>
                shift both
              </button>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2">
            <div className="rounded-md border border-border/60 bg-muted/20 p-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">offset Δ</div>
              <div className="font-mono text-lg text-foreground/90" aria-live="polite">
                {delta}
              </div>
            </div>
            <div className="rounded-md border border-border/60 bg-muted/20 p-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">relative rotation</div>
              <div className="font-mono text-sm text-foreground/90" aria-live="polite">
                Δ·ω = {relRotation.toFixed(2)} rad
              </div>
            </div>
            <div className="rounded-md border border-border/60 bg-muted/20 p-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">q·k</div>
              <div className="font-mono text-lg text-foreground/90" aria-live="polite">
                {dot.toFixed(2)}
              </div>
            </div>
          </div>

          <div className={cn('text-sm font-medium', qual.tone)} aria-live="polite">
            {qual.label} · q·k = cos(Δ·ω) = cos({relRotation.toFixed(2)}) = {dot.toFixed(2)}
          </div>

          <div className="rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1.5 text-[12px] text-foreground/90">
            <strong>The lesson:</strong> the score <span className="font-mono">q·k = cos(Δ·ω)</span> depends only on the
            offset m − n — shift both positions together and it doesn&apos;t change.
          </div>
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">
        Both arrows start from the same base vector; RoPE turns the query by <span className="font-mono">m·ω</span> and
        the key by <span className="font-mono">n·ω</span>, so the query&apos;s relative rotation is exactly{' '}
        <span className="font-mono">Δ·ω = (m − n)·ω</span> and <span className="font-mono">q·k = cos((m − n)·ω)</span>.
        Increase <span className="font-mono">m</span> and <span className="font-mono">n</span> together and the whole
        picture rotates, but the gap — and the score — stays frozen. Move just one and the gap opens or closes: align
        them (<span className="font-mono">Δ → 0</span>) and <span className="font-mono">q·k → 1</span>.
      </p>

      <p className="text-[10px] text-muted-foreground">
        Illustrative — this uses ONE representative frequency (<span className="font-mono">ω = {OMEGA}</span> rad/token)
        and simplifies the content vectors to a shared base so we isolate the positional part. As positions grow the
        relative rotation <span className="font-mono">Δ·ω</span> can exceed a full turn, but because cosine repeats
        every turn the score still depends only on the offset <span className="font-mono">Δ</span>. The real model
        combines this with the actual query and key content across many frequency pairs at many different speeds (see
        the frequency spectrum below).
      </p>
    </div>
  );
}
