import { cn } from '@/lib/utils';
import * as React from 'react';

/**
 * Chapter 2 supplement — the GEOMETRY behind cosine similarity.
 *
 * The chapter leans on "direction = meaning, angle = similarity" but the only
 * visual is a PCA scatter it repeatedly tells you to distrust. This widget makes
 * the intuition manipulable: a few fixed, labelled "word" arrows from the origin
 * plus one draggable "probe" arrow. As you swing the probe around, the readout
 * shows the angle θ to the selected word and cos θ — the cosine similarity.
 *
 * Two teaching beats are made physical:
 *   1. small angle → cos θ ≈ 1 ("related"); 90° → cos θ ≈ 0 ("unrelated");
 *      opposite → cos θ ≈ −1 ("contrary").
 *   2. MAGNITUDE DOESN'T MATTER — lengthening/shortening the probe (length
 *      slider, or dragging it further out) changes |v| but leaves cos θ fixed.
 *      That is exactly why embeddings use cosine, not Euclidean distance.
 *
 * These are illustrative 2-D teaching vectors — NOT real embeddings, which live
 * in 1024 dimensions. Everything here is formula-derived: no model, no worker,
 * no network. Drag is a bonus on top of fully keyboard-accessible sliders.
 */

// Fixed, illustrative "word" vectors. Angles chosen so the relationships read
// intuitively: cat/dog point similar directions (related); car sits well off to
// the side (unrelated to the animals). Lengths vary on purpose to underline that
// magnitude is irrelevant to the angle. These are NOT embeddings — purely 2-D.
type WordVec = { id: string; label: string; angleDeg: number; length: number; color: string };

const WORDS: WordVec[] = [
  { id: 'cat', label: 'cat', angleDeg: 62, length: 0.92, color: 'oklch(0.72 0.18 350)' }, // pink
  { id: 'dog', label: 'dog', angleDeg: 38, length: 0.7, color: 'oklch(0.80 0.13 80)' }, // orange
  { id: 'car', label: 'car', angleDeg: -28, length: 0.85, color: 'oklch(0.72 0.13 235)' }, // blue
];

const PROBE_COLOR = 'oklch(0.75 0.12 185)'; // teal — distinct from every word vector

// SVG geometry. The plane is square so 1 unit is the same number of pixels on
// both axes (angles stay visually honest). The origin sits at the center.
const SIZE = 360;
const CENTER = SIZE / 2;
// Pixels per unit-length. Word/probe lengths are ~0..1.6, so this keeps the
// longest arrow comfortably inside the frame with room for the label.
const UNIT = 96;

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

// Math coords (origin at center, +y up) → SVG pixel coords (origin top-left, +y down).
function toPx(x: number, y: number): [number, number] {
  return [CENTER + x * UNIT, CENTER - y * UNIT];
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

// Smallest angle between two directions, in [0, 180].
function angleBetween(aDeg: number, bDeg: number): number {
  const d = Math.abs(((aDeg - bDeg) % 360) + 360) % 360;
  return d > 180 ? 360 - d : d;
}

type Qual = { label: string; tone: string };

function qualitative(cos: number): Qual {
  if (cos >= 0.6) return { label: '≈ same direction → related', tone: 'text-emerald-400' };
  if (cos > 0.2) return { label: 'partly aligned → loosely related', tone: 'text-foreground/80' };
  if (cos >= -0.2) return { label: '≈ perpendicular → unrelated', tone: 'text-muted-foreground' };
  if (cos > -0.6) return { label: 'pointing apart → weakly contrary', tone: 'text-foreground/80' };
  return { label: '≈ opposite → contrary', tone: 'text-amber-400' };
}

function Arrow({
  angleDeg,
  length,
  color,
  label,
  dimmed,
  strokeWidth = 2.5,
  markerId,
}: {
  angleDeg: number;
  length: number;
  color: string;
  label: string;
  dimmed?: boolean;
  strokeWidth?: number;
  markerId: string;
}) {
  const rad = (angleDeg * Math.PI) / 180;
  const tipX = Math.cos(rad) * length;
  const tipY = Math.sin(rad) * length;
  const [ox, oy] = toPx(0, 0);
  const [tx, ty] = toPx(tipX, tipY);
  // Nudge the label just past the arrowhead, along the arrow direction.
  const [lx, ly] = toPx(tipX * 1.0 + Math.cos(rad) * 0.16, tipY * 1.0 + Math.sin(rad) * 0.16);
  return (
    <g opacity={dimmed ? 0.4 : 1}>
      <line
        x1={ox}
        y1={oy}
        x2={tx}
        y2={ty}
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        markerEnd={`url(#${markerId})`}
      />
      <text
        x={lx}
        y={ly}
        fontSize="12"
        textAnchor={tipX < -0.05 ? 'end' : 'start'}
        dominantBaseline="middle"
        className="pointer-events-none select-none font-mono"
        fill={color}
      >
        {label}
      </text>
    </g>
  );
}

export function VectorCosine() {
  const reducedMotion = usePrefersReducedMotion();
  const svgRef = React.useRef<SVGSVGElement>(null);

  // Probe state in polar form so the keyboard sliders map directly onto it.
  const [probeAngle, setProbeAngle] = React.useState(12); // degrees, near `dog`/`cat`
  const [probeLen, setProbeLen] = React.useState(1.1);
  const [compareId, setCompareId] = React.useState<string>('cat');
  const [dragging, setDragging] = React.useState(false);

  const compare = WORDS.find((w) => w.id === compareId) ?? WORDS[0]!;

  const theta = angleBetween(probeAngle, compare.angleDeg);
  const cos = Math.cos((theta * Math.PI) / 180);
  const qual = qualitative(cos);

  // Pointer drag: set angle (and length) from the arrowhead position. Keyboard
  // sliders below remain the primary, fully accessible control; this is a bonus.
  function pointTo(evt: React.PointerEvent<SVGSVGElement>) {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    // Convert client px → viewBox px → math units (note the y flip).
    const vx = ((evt.clientX - rect.left) / rect.width) * SIZE;
    const vy = ((evt.clientY - rect.top) / rect.height) * SIZE;
    const mx = (vx - CENTER) / UNIT;
    const my = -(vy - CENTER) / UNIT;
    const len = Math.hypot(mx, my);
    if (len < 1e-3) return;
    const deg = (Math.atan2(my, mx) * 180) / Math.PI;
    setProbeAngle(Math.round(deg));
    setProbeLen(clamp(len, 0.2, 1.6));
  }

  // Geometry for the angle arc drawn between the probe and the compared word.
  const arc = React.useMemo(() => {
    const r = 0.42; // arc radius in math units, inside both arrows
    const a0 = (compare.angleDeg * Math.PI) / 180;
    const a1 = (probeAngle * Math.PI) / 180;
    const [sx, sy] = toPx(Math.cos(a0) * r, Math.sin(a0) * r);
    const [ex, ey] = toPx(Math.cos(a1) * r, Math.sin(a1) * r);
    // Walk the SHORT way around so the arc matches the reported θ.
    let delta = (((a1 - a0) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    if (delta > Math.PI) delta -= 2 * Math.PI;
    const sweep = delta >= 0 ? 1 : 0;
    const largeArc = 0; // θ ≤ 180 by construction
    const rPx = r * UNIT;
    // Midpoint of the arc, for the θ label.
    const aMid = a0 + delta / 2;
    const [mx, my] = toPx(Math.cos(aMid) * (r + 0.16), Math.sin(aMid) * (r + 0.16));
    return { d: `M ${sx} ${sy} A ${rPx} ${rPx} 0 ${largeArc} ${sweep} ${ex} ${ey}`, mx, my };
  }, [compare.angleDeg, probeAngle]);

  const [ringCx, ringCy] = toPx(0, 0);

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Direction = meaning, angle = similarity
        </div>
        <div className="text-[11px] text-muted-foreground">Drag the teal arrow, or use the sliders</div>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-[auto_minmax(0,1fr)]">
        {/* ---- The plane ---- */}
        <svg
          ref={svgRef}
          role="img"
          aria-label={`Vector plane. The probe vector is at ${Math.round(probeAngle)} degrees with length ${probeLen.toFixed(
            2,
          )}. The angle to ${compare.label} is ${Math.round(theta)} degrees, cosine ${cos.toFixed(2)}.`}
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          className="mx-auto block h-auto w-full max-w-[360px] touch-none rounded-md bg-muted/20"
          style={{ cursor: dragging ? 'grabbing' : 'grab' }}
          onPointerDown={(e) => {
            (e.target as Element).setPointerCapture?.(e.pointerId);
            setDragging(true);
            pointTo(e);
          }}
          onPointerMove={(e) => {
            if (e.buttons === 0) return;
            pointTo(e);
          }}
          onPointerUp={(e) => {
            (e.target as Element).releasePointerCapture?.(e.pointerId);
            setDragging(false);
          }}
        >
          <defs>
            {WORDS.map((w) => (
              <marker
                key={w.id}
                id={`vc-head-${w.id}`}
                viewBox="0 0 10 10"
                refX="8"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto"
              >
                <path d="M 0 0 L 10 5 L 0 10 Z" fill={w.color} />
              </marker>
            ))}
            <marker
              id="vc-head-probe"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6.5"
              markerHeight="6.5"
              orient="auto"
            >
              <path d="M 0 0 L 10 5 L 0 10 Z" fill={PROBE_COLOR} />
            </marker>
          </defs>

          {/* Axes through the origin. */}
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
          <circle cx={ringCx} cy={ringCy} r={2.5} fill="currentColor" fillOpacity={0.4} />

          {/* Angle arc between the probe and the compared word — drawn first so
              the arrows sit on top of it. */}
          <path
            d={arc.d}
            fill="none"
            stroke="var(--foreground)"
            strokeOpacity={0.5}
            strokeWidth={1.5}
            className={cn(!reducedMotion && 'transition-[d] duration-150')}
          />
          <text
            x={arc.mx}
            y={arc.my}
            fontSize="11"
            textAnchor="middle"
            dominantBaseline="middle"
            className="pointer-events-none select-none fill-foreground/80"
          >
            {Math.round(theta)}°
          </text>

          {/* Word vectors — the one being compared stays full-strength, the rest dim. */}
          {WORDS.map((w) => (
            <Arrow
              key={w.id}
              angleDeg={w.angleDeg}
              length={w.length}
              color={w.color}
              label={w.label}
              dimmed={w.id !== compareId}
              markerId={`vc-head-${w.id}`}
            />
          ))}

          {/* The draggable probe, on top. */}
          <Arrow
            angleDeg={probeAngle}
            length={probeLen}
            color={PROBE_COLOR}
            label="your word"
            strokeWidth={3}
            markerId="vc-head-probe"
          />
        </svg>

        {/* ---- Controls + readouts ---- */}
        <div className="space-y-3">
          <div className="space-y-1">
            <div className="text-[11px] text-muted-foreground">Compare “your word” against</div>
            <div className="flex flex-wrap gap-1.5" role="group" aria-label="Word vector to compare against">
              {WORDS.map((w) => {
                const active = w.id === compareId;
                return (
                  <button
                    key={w.id}
                    type="button"
                    onClick={() => setCompareId(w.id)}
                    aria-pressed={active}
                    className={cn(
                      'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-xs',
                      active
                        ? 'border-foreground/30 bg-background'
                        : 'border-border/60 bg-muted/40 text-muted-foreground',
                    )}
                  >
                    <span
                      aria-hidden="true"
                      className="inline-block h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: w.color }}
                    />
                    {w.label}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="space-y-1">
            <label htmlFor="vc-angle" className="block text-xs text-muted-foreground">
              probe angle <span className="font-mono text-foreground/85">{Math.round(probeAngle)}°</span>
            </label>
            <input
              id="vc-angle"
              type="range"
              min={-180}
              max={180}
              step={1}
              value={Math.round(probeAngle)}
              onChange={(e) => setProbeAngle(Number(e.target.value))}
              className="w-full accent-primary"
              aria-valuetext={`probe angle ${Math.round(probeAngle)} degrees`}
            />
          </div>

          <div className="space-y-1">
            <label htmlFor="vc-length" className="block text-xs text-muted-foreground">
              probe length <span className="font-mono text-foreground/85">|v| = {probeLen.toFixed(2)}</span>
            </label>
            <input
              id="vc-length"
              type="range"
              min={0.2}
              max={1.6}
              step={0.01}
              value={probeLen}
              onChange={(e) => setProbeLen(Number(e.target.value))}
              className="w-full accent-primary"
              aria-valuetext={`probe length ${probeLen.toFixed(2)}`}
            />
            <p className="text-[11px] text-muted-foreground">
              Slide the length and watch <span className="font-mono">cos θ</span> below stay put — magnitude doesn’t
              change the angle.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div className="rounded-md border border-border/60 bg-muted/20 p-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">angle θ</div>
              <div className="font-mono text-lg text-foreground/90" aria-live="polite">
                {Math.round(theta)}°
              </div>
            </div>
            <div className="rounded-md border border-border/60 bg-muted/20 p-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
                cos θ <span className="normal-case">(similarity)</span>
              </div>
              <div className="font-mono text-lg text-foreground/90" aria-live="polite">
                {cos.toFixed(2)}
              </div>
            </div>
          </div>

          <div className={cn('text-sm font-medium', qual.tone)} aria-live="polite">
            {qual.label}
          </div>

          <div className="text-[11px] text-muted-foreground">
            <span className="font-mono">|your word|</span> = {probeLen.toFixed(2)} ·{' '}
            <span className="font-mono">|{compare.label}|</span> = {compare.length.toFixed(2)} — different lengths, and{' '}
            <span className="font-mono">cos θ</span> only sees the <strong>angle</strong> between them.
          </div>
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">
        Swing the probe to line up with <span className="font-mono">{compare.label}</span> →{' '}
        <span className="font-mono">cos θ → 1</span> (“related”). Turn it to a right angle →{' '}
        <span className="font-mono">cos θ → 0</span> (“unrelated”). Point it the opposite way →{' '}
        <span className="font-mono">cos θ → −1</span> (“contrary”). The length never enters the cosine — which is
        exactly why embedding similarity uses the <strong>angle</strong>, not straight-line (Euclidean) distance.
      </p>

      <p className="text-[10px] text-muted-foreground">
        Illustrative only — these are hand-placed 2-D teaching arrows, NOT real embeddings. Qwen3.5-0.8B&apos;s vectors
        live in <span className="font-mono">1,024</span> dimensions; this flat picture exists to build the intuition.
      </p>
    </div>
  );
}
