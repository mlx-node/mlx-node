import * as React from 'react';

import type { Locale } from '../../lib/i18n';
import { hatchFill, type HatchTone } from './HatchPattern';
import { StepControls } from './StepControls';
import type { StepPlayer } from './use-step-player';

/**
 * skin — the course's diagram style, as constants and primitives instead of as
 * 113 copies of the same magic numbers.
 *
 * Every value here was READ OUT of `learn/widgets/ChunkedPrefillEngine.tsx`,
 * which is the one widget that already ships the look this module codifies. The
 * line cites in the comments below point at it (`CPE:<line>`), so a value that
 * drifts can be checked against the widget rather than argued about.
 *
 * WHAT IS DELIBERATELY *NOT* CPE (the four target-look deltas). Colour and type
 * already match the reference; these four are the whole gap, and each one is
 * marked `TARGET LOOK #n` at its definition so nobody "restores" it later:
 *
 *   #1  square corners        — `RX = 0` replaces CPE's 6/4/3/2/1.5 ladder.
 *   #2  title chips ON the    — `FrameLabel` replaces "title inside the box at
 *       box border               top+17/+20/+26" (CPE:281, 425, 615).
 *   #3  airier canvas         — guidance only, see CANVAS WIDTH below. Geometry
 *                               stays widget-owned; there is nothing to export.
 *   #4  less card chrome      — `DiagramFrame` drops the shell's border/fill/
 *                               radius (CPE:249) in ONE place, not 79.
 *
 * CANVAS WIDTH (#3, unenforceable so it lives in prose). MEASURE FIRST — this
 * is the rule the first pilot got wrong and only a live measurement caught:
 *
 *   a viewBox unit is NOT a CSS pixel. Every diagram is `className="w-full"`, so
 *   the on-screen size of `text-[10px]` is 10 x (columnWidth / VB_W). The
 *   chapter reading column measures 451px at a 1280px viewport (three-column
 *   LessonLayout, `lg:grid-cols-[220px_minmax(0,1fr)_minmax(0,1fr)]`).
 *
 *   So: KEEP VB_W NEAR 460. At 460 the scale is ~0.98 and the type sizes in this
 *   module mean what they say. At VB_W = 700 the same `CHIP_FONT = 10` renders
 *   at 6.4 CSS px, which is not a small label — it is an unreadable one.
 *
 * "Airier" is therefore bought with LAYOUT, never with canvas width: wider
 * gutters and gaps inside a ~460-wide box, and the vertical space that #2 frees
 * by lifting titles out of the boxes. A tall diagram is fine — the column is
 * narrow, so a vertical stack SHOULD come out taller than 16:9. There is no
 * `CANVAS` export because the height is the widget's own business and a wrong
 * shared default is worse than none.
 *
 * WHAT IS NOT HERE, ON PURPOSE:
 *   - Geometry. `P_*`, `F_*`, `K_*` and friends encode ONE diagram's layout.
 *   - Copy. `COPY = { en, zh }` stays with the widget that owns the sentences.
 *   - `FLOW_BLUE`. It is the in-flight-data colour and belongs to FlowDots; it
 *     is never a box colour, so it is not part of the two-hue rule below.
 */

// ── Hues. Exactly two, and neither is ever a state. ──────────────────────────
//
// Hue is the PERMANENT IDENTITY of a box. CPE strokes the forward pass RED in
// every frame (CPE:466) including the idle ones, and the KV cache EMERALD in
// every frame (CPE:598). "Busy right now" is carried by the hatch overlay and
// "not real yet" by the dash — both orthogonal to hue, both colour-blind-safe.
// Swapping a box's hue to show state is the one thing this palette cannot do.
//
// OUT OF SCOPE, and this matters: `HeadSplitDiagram` and `MultiheadConcat` run
// an 8-step oklch hue wheel where hue means WHICH ATTENTION HEAD (documented at
// HeadSplitDiagram.tsx:22-24). They render on the same page as each other and
// the two-hue rule must never be pushed onto them. `PermutationInvariance` is
// likewise out of scope — its whole lesson is that nothing moves.

/** Expensive / busy / compute happening now. 13 widgets hardcode this hex. */
export const RED = '#ef4444'; // red-500 — CPE:153

/** Cached / stored / already done. The second and LAST hue. 21 widgets hardcode it. */
export const EMERALD = '#10b981'; // emerald-500 — CPE:154

/**
 * TARGET LOOK #1 — the single corner-radius knob.
 *
 * CPE runs a nesting ladder: rx 6 outer containers, 4 inner borders, 3 mid-level
 * groups, 2 atomic cells, 1.5 the smallest slot (CPE:264 / 269 / 328 / 299 /
 * 654). The reference we are matching is square, so the ladder collapses to one
 * number and the whole course squares up with ONE edit instead of 113.
 *
 * Kept as a knob rather than deleted because "square" is a taste call that may
 * be revisited; set it to 6 and PanelFrame/FrameLabel go straight back to the
 * CPE look. Widgets pass `rx={RX}` on their own rects so they follow along.
 */
export const RX = 0;

/**
 * The strict two-step stroke ladder. There is no third width.
 *   OUTER — a real, owned container. CPE:266, 421, 467, 599.
 *   INNER — anything drawn inside a container: inner borders, cells, slots,
 *           connectors. CPE:276, 306, 380, 548.
 * Already the house default: 82 files use 1.5, 95 use 1.
 */
export const SW = { OUTER: 1.5, INNER: 1 } as const;

/**
 * Two named dasharrays and nothing else.
 *   BORDER — an inner sub-frame, an idle connector, or a box that is NOT REAL
 *            YET. CPE:277, 479, 508, 549, 611.
 *   SLOT   — an empty repeated container slot, e.g. a KV block holding no
 *            tokens. CPE:634.
 * The corpus currently runs six competing values (`3 3` x15, `2 4` x9, `2 3` x8,
 * `4 4` x6, `4 3` x6, `5 4` x5). These two replace all of them.
 *
 * The "becomes real" transform is all three at once — dash off, SW.INNER ->
 * SW.OUTER, neutral -> hued (CPE:503-508). Never just one of the three.
 */
export const DASH = { BORDER: '4 4', SLOT: '3 3' } as const;

/**
 * Opacity is the dimming channel — the course has two hues and no third colour,
 * so "less important" has to be carried by alpha. These are the steps CPE
 * actually uses; anything outside them is a magic number.
 *
 *   HATCH         hatch stripe alpha. Mirrors `HATCH_OPACITY` in
 *                 HatchPattern.tsx:69, which is private ON PURPOSE (the pattern
 *                 takes no props so every instance stays byte-identical). Listed
 *                 here for reference — do not try to import or override it.
 *   IDLE_EDGE     a static connector: the edge exists, nothing is on it. CPE:547
 *   NEUTRAL_INNER inner dashed border of a NEUTRAL box; sits back further than a
 *                 hued one so it does not read as a second stroke. CPE:275
 *   HUED_INNER    inner dashed border of a RED/EMERALD box. CPE:477, 609
 *   EMPTY_SLOT    an empty repeated slot outline. CPE:632
 *   FILLED        a filled slot. Not 1 — full alpha next to hatch reads as glare.
 *                 CPE:657
 */
export const DIM = {
  HATCH: 0.28,
  IDLE_EDGE: 0.4,
  NEUTRAL_INNER: 0.45,
  HUED_INNER: 0.5,
  EMPTY_SLOT: 0.55,
  FILLED: 0.85,
} as const;

// ── PanelFrame ───────────────────────────────────────────────────────────────

/** Box identity. `neutral` = context, not the lesson (CPE:265, 420). */
type PanelTone = 'neutral' | 'red' | 'emerald';

const TONE_STROKE: Record<PanelTone, string> = {
  neutral: 'var(--border)',
  red: RED,
  emerald: EMERALD,
};

/** The inner dashed border INHERITS the box's hue — never a fixed grey. */
const TONE_INNER: Record<PanelTone, string> = {
  neutral: 'var(--muted-foreground)',
  red: RED,
  emerald: EMERALD,
};

const TONE_HATCH: Record<PanelTone, HatchTone> = {
  neutral: 'muted',
  red: 'red',
  emerald: 'emerald',
};

/**
 * PanelFrame — the double border, the course's "this is a STAGE" marker.
 *
 *   +------------------------------+  outer: solid, SW.OUTER, the tone's hue
 *   | +- - - - - - - - - - - - -+  |  inset 5 (short box) or 6
 *   | :                         :  |  inner: dashed DASH.BORDER, SW.INNER,
 *   | :                         :  |         fill none, DIM.*_INNER
 *   | +- - - - - - - - - - - - -+  |
 *   +------------------------------+
 *
 * Construction is mechanical: `x+I, y+I, w-2I, h-2I`. The inset is 5 on short
 * boxes and 6 otherwise — CPE uses 5 on the 54px prompt strip (CPE:269) and 6 on
 * the 74px and 120px boxes (CPE:471, 603); a 54px box looks choked at 6. The
 * threshold is h <= 60 because that is where the two cases actually fall.
 *
 * A double border means a STAGE box. A plain list box gets a single border and
 * no inner frame — CPE's "other requests" (CPE:414-422) is the worked example.
 * If you are reaching for PanelFrame on a list, draw a bare rect instead.
 *
 * `hatched` stacks the hatch BETWEEN the outer and inner strokes, matching the
 * outer geometry exactly and carrying no stroke of its own (CPE:460 -> 469 ->
 * 470). Requires `<HatchDefs />` somewhere in the same svg.
 *
 * Adoption today is 1 (CPE). 21 files draw a `var(--card)` stage box by hand.
 */
export function PanelFrame({
  x,
  y,
  w,
  h,
  tone,
  hatched = false,
  fill = 'var(--card)',
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  tone: PanelTone;
  /** paint the "busy this step" hatch. Needs `<HatchDefs />` in the svg. */
  hatched?: boolean;
  /**
   * Body fill. Defaults to the shipped `var(--card)`. Pass `"none"` for the
   * airier target-look variant — it also makes the FrameLabel notch vanish,
   * since the chip and the page then paint the same `var(--background)`.
   */
  fill?: string;
}) {
  const inset = h <= 60 ? 5 : 6;
  // The inner radius trails the outer by 2 so the two strokes stay concentric
  // (CPE: 6 -> 4). At RX = 0 both are 0 and the rule is a no-op, but it is kept
  // so flipping RX back to 6 restores the CPE look with no other edit.
  const innerRx = Math.max(0, RX - 2);

  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={RX}
        style={{ fill, stroke: TONE_STROKE[tone] }}
        strokeWidth={SW.OUTER}
      />
      {hatched ? <rect x={x} y={y} width={w} height={h} rx={RX} fill={hatchFill(TONE_HATCH[tone])} /> : null}
      <rect
        x={x + inset}
        y={y + inset}
        width={w - 2 * inset}
        height={h - 2 * inset}
        rx={innerRx}
        fill="none"
        style={{
          stroke: TONE_INNER[tone],
          opacity: tone === 'neutral' ? DIM.NEUTRAL_INNER : DIM.HUED_INNER,
        }}
        strokeWidth={SW.INNER}
        strokeDasharray={DASH.BORDER}
      />
    </g>
  );
}

// ── FrameLabel ───────────────────────────────────────────────────────────────

/**
 * Chip metrics. 10px is the course's box-title size (CPE:284, 428, 517, 618);
 * `tracking-wider` is 0.05em, so 0.5px of extra advance per glyph at that size.
 */
const CHIP_FONT = 10;
const CHIP_H = 14; // clears a 10px face and covers a 1.5 stroke on both sides
const CHIP_PAD_X = 6;
const CHIP_TRACK = 0.5;
/** Vertical centring on the border line: centre + ~0.36em, the same hand-tune CPE uses. */
const CHIP_BASELINE = 3.5;

/** CJK / fullwidth block — these advance a full em where Latin advances ~0.68. */
const FULLWIDTH = /[\u3000-\u9FFF\uFF01-\uFF60]/u;

/**
 * Estimated chip width. SVG cannot measure text without a DOM, and this has to
 * be identical on the prerender server and in the browser, so it is an estimate
 * by construction. Deliberately biased LARGE: an over-wide chip notches a little
 * more border than it needs, which is invisible; an under-wide one lets the
 * stroke run through the letters, which is not.
 */
function chipWidth(label: string): number {
  const chars = Array.from(label);
  let advance = 0;
  for (const ch of chars) {
    if (FULLWIDTH.test(ch)) advance += CHIP_FONT;
    else if (ch === ' ') advance += CHIP_FONT * 0.3;
    // The class applies `uppercase`, so every Latin glyph renders as a capital.
    else advance += CHIP_FONT * 0.68;
  }
  return advance + CHIP_TRACK * chars.length + CHIP_PAD_X * 2;
}

/**
 * TARGET LOOK #2 — FrameLabel: a title chip straddling the top border.
 *
 *        +--[ PAGED KV CACHE ]----------+
 *        |                              |
 *
 * A `var(--background)`-filled rect notched into the stroke, with the label
 * sitting on it. This replaces the course's current rule — title INSIDE the box,
 * baseline at roughly `top + 17 + height/12` (CPE:281, 425, 615) — which costs
 * ~20px of vertical space per box and is the main reason CPE reads as 1.376:1
 * rather than 16:9. Moving the title onto the border gives that space back, so
 * #2 and #3 are the same edit.
 *
 * This is the ONE export with no prior art in the repo; everything else was
 * extracted. Treat the geometry as the first draft it is.
 *
 * `x` is the anchor ON the border line, `y` is the border line itself — pass the
 * panel's own `y`, not `y + something`. `align` says which end of the chip the
 * anchor pins: `start` (default) sits it in from the left corner, which is the
 * reference look; `middle` re-centres it for a symmetric diagram.
 */
export function FrameLabel({
  x,
  y,
  label,
  align = 'start',
}: {
  x: number;
  y: number;
  label: string;
  align?: 'start' | 'middle' | 'end';
}) {
  const w = chipWidth(label);
  const chipX = align === 'start' ? x : align === 'middle' ? x - w / 2 : x - w;
  const textX = align === 'start' ? x + CHIP_PAD_X : align === 'middle' ? x : x - CHIP_PAD_X;

  return (
    <g>
      {/* No stroke: this rect's whole job is to ERASE the border under the text. */}
      <rect x={chipX} y={y - CHIP_H / 2} width={w} height={CHIP_H} rx={RX} style={{ fill: 'var(--background)' }} />
      <text
        x={textX}
        y={y + CHIP_BASELINE}
        textAnchor={align}
        style={{ fill: 'var(--foreground)' }}
        className="text-[10px] font-semibold uppercase tracking-wider"
      >
        {label}
      </text>
    </g>
  );
}

// ── DiagramFrame ─────────────────────────────────────────────────────────────

/**
 * DiagramFrame — the shell every animated diagram repeats: header row, the svg,
 * the caption, the footnote. Pass the `<svg>` as children; this owns everything
 * around it.
 *
 * TARGET LOOK #4 — LESS CARD CHROME. CPE's shell is
 *   `not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3`
 * (CPE:249), copied verbatim into 93 files. A rounded, bordered, filled card
 * around a square-cornered wireframe diagram fights it twice — the radius
 * contradicts RX = 0 and the border draws a fifth box nobody asked for. So the
 * card is gone and the breathing room is dialled up instead (`my-6 space-y-4`).
 * This is the ONE place that decision lives; changing it back is one edit.
 *
 * The rest is not cosmetic — it is the half of the a11y contract that is
 * invisible in a browser and obvious with a screen reader, which is exactly why
 * it belongs in a shared component rather than in a checklist:
 *
 *  - The live region is a SEPARATE `sr-only` span, permanently `aria-live
 *    polite`, holding the caption only while PAUSED. The visible paragraph is
 *    `aria-hidden`. An autoplaying 11-frame sweep at 1400ms would otherwise read
 *    a new sentence at a screen reader every 1.4s, forever, over the top of the
 *    surrounding lesson, with no way to stop it. (CPE:684-694 argues this at
 *    length.) Announcements belong to user intent, so pausing fills the region
 *    and Step/Reset — the reader's own actions — announce normally.
 *
 *    Do NOT collapse this back into one paragraph that toggles `aria-live`
 *    between off and polite: `player.step()` batches `setPlaying(false)` and
 *    `setFrame(...)` into ONE React commit, so the attribute would flip to
 *    polite in the same DOM mutation that changes the text — and a region that
 *    was not already live when its content changed is not reliably announced.
 *    Two nodes sidesteps the race: the span's `aria-live` never changes.
 *  - `aria-atomic` so the whole sentence reads, not the character diff.
 *  - `min-h-[3.75rem]` reserves THREE lines at `text-[12px] leading-relaxed`
 *    (12 x 1.625 = 19.5px each) so the page does not jump as the caption
 *    changes length. Two lines was not enough for the first caller and made the
 *    chapter body hop 38px on every loop. A caption that runs to four lines in
 *    the 451px reading column will still shift the page — shorten the caption.
 *
 * The third part of the contract — a REAL pause control — is `StepControls`,
 * rendered here. The fourth — reduced-motion seeding — is `useStepPlayer`, which
 * the caller owns because it also owns the frame model.
 *
 * The caller still owns `role="img"` + `aria-label` on its own `<svg>`: the
 * description is a paragraph about that specific diagram (CPE:182, 255) and
 * there is nothing generic to say for it here.
 */
export function DiagramFrame({
  title,
  player,
  locale,
  caption,
  note,
  showReset = false,
  children,
}: {
  /** Heading text, upper-cased by CSS. Already localised by the caller. */
  title: string;
  player: StepPlayer;
  locale: Locale;
  /**
   * The sentence for the beat currently on screen. Goes in the live region.
   *
   * `ReactNode`, not `string`: several of the course's captions carry inline
   * emphasis that IS the lesson — the bolded `one number`, `gradient`, the
   * italicised question a block asks itself. Flattening those to plain text to
   * satisfy a prop type would delete teaching, so the type widens instead. A
   * plain string is still the common case and still valid.
   */
  caption: React.ReactNode;
  /** Optional standing footnote — "these numbers are illustrative", etc. */
  note?: React.ReactNode;
  /** Worth it for long sweeps; noise for short loops. See StepControls.tsx:44. */
  showReset?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="not-prose my-6 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{title}</div>
        <StepControls player={player} locale={locale} showReset={showReset} />
      </div>

      {children}

      <p className="min-h-[3.75rem] text-[12px] leading-relaxed text-muted-foreground" aria-hidden="true">
        {caption}
      </p>
      <span className="sr-only" aria-live="polite" aria-atomic="true">
        {player.playing ? '' : caption}
      </span>

      {note ? (
        <p className="border-t border-border/60 pt-2 text-[11px] leading-relaxed text-muted-foreground">{note}</p>
      ) : null}
    </div>
  );
}

// ── elbow ────────────────────────────────────────────────────────────────────

/**
 * elbow — down from `(x1,y1)`, across at `midY`, then into `(x2,y2)`.
 * Right-angle routing only. There are no curved connectors in this course, and
 * a bezier next to these square boxes reads as a different diagram. (CPE:213)
 *
 * THE PAIRED RULE, which is a lint-level obligation and not a preference: every
 * `FlowDots` `d` must be BYTE-IDENTICAL to a static connector `d` already drawn
 * on the diagram. Dots animate an edge that exists; they never invent one. CPE
 * keeps three of its four dot paths character-for-character equal to a connector
 * (CPE:538-541 vs 573/580/587) — copy that shape.
 *
 * The other half of the FlowDots contract: always pass `paused={!player.playing}`.
 * The CSS animation cannot see the JS frame timer, so a Pause button that stops
 * the timer while these keep travelling is a Pause button that visibly does not
 * pause (WCAG 2.2.2). See FlowDots.tsx:45-51.
 */
export function elbow(x1: number, y1: number, x2: number, y2: number, midY: number): string {
  return `M ${x1} ${y1} L ${x1} ${midY} L ${x2} ${midY} L ${x2} ${y2}`;
}
