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
 *                               radius (CPE:249) in ONE place, not 51.
 *
 * CANVAS WIDTH (#3, unenforceable so it lives in prose). MEASURE FIRST — the
 * first pilot got this wrong twice, and only a live measurement caught either:
 *
 *   A viewBox unit is NOT a CSS pixel. Every diagram is `className="w-full"`, so
 *   the on-screen size of `text-[10px]` is 10 x (columnWidth / VB_W). Set VB_W
 *   so that ratio lands near 1 and the type sizes in this module mean what they
 *   say. At scale 0.64 a `CHIP_FONT = 10` chip renders at 6.4 CSS px, which is
 *   not a small label — it is an unreadable one.
 *
 *   THERE IS NO SINGLE COURSE-WIDE COLUMN WIDTH. Measured live at a 1280px
 *   viewport, `document.querySelector('svg').getBoundingClientRect().width`:
 *
 *     chapter WITH a live right-hand pane   451px   /chapters/training
 *     chapter WITHOUT one                   742px   /chapters/kv-cache/batching
 *
 *   The pane is what halves it (`LessonLayout`'s
 *   `lg:grid-cols-[220px_minmax(0,1fr)_minmax(0,1fr)]`), so the number depends
 *   on which chapter mounts the widget, not on the viewport. Look up where the
 *   widget is actually rendered, measure that page, then pick VB_W. 460 is
 *   right for a paned chapter and 1.6x too small for an unpaned one.
 *
 *   The same arithmetic is worth running on any widget you touch, not just the
 *   one you are restyling: `GradientDescent.tsx:347` breaks its controls out at
 *   `md:` (a VIEWPORT query) inside a 425px CONTAINER, so on /chapters/training
 *   the plot is squeezed to 181px — scale 0.335, axis labels at 3.0 CSS px.
 *   Viewport breakpoints cannot express "how wide is my column"; use Tailwind
 *   v4's `@container` for that.
 *
 * "Airier" is therefore bought with LAYOUT, never by inflating VB_W past the
 * column: wider gutters and gaps, plus the vertical space that #2 frees by
 * lifting titles out of the boxes. A tall diagram is fine — in a paned chapter
 * the column is narrow, so a vertical stack SHOULD come out taller than 16:9.
 * There is no `CANVAS` export because the height is the widget's own business
 * and a wrong shared default is worse than none.
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
 * Already the house default: 81 uses of 1.5 across 37 files, 95 of 1 across 38.
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
const CHIP_PAD_X = 6;
const CHIP_TRACK = 0.5;
/** Plate height: face + 4, which clears the face and covers a 1.5 stroke both sides. */
const chipH = (font: number) => font + 4;
/** Vertical centring on the line: centre + 0.35em, the same hand-tune CPE uses. */
const chipBaseline = (font: number) => font * 0.35;

/**
 * Everything that advances a FULL em. CJK and fullwidth ASCII, plus — the part
 * the first draft got wrong — U+2000-U+2BFF, where the zh copy's punctuation
 * actually lives. Measured in Inter 600 (fontTools v4.000): em dash 1.000em,
 * ellipsis 0.956, arrows 0.881-0.954, check mark 0.883, and U+22EF is not in
 * Inter at all, so it falls back to a system face. The old range scored every
 * one of those at 0.68 — NARROW, the direction that puts the stroke through the
 * glyphs; four em dashes alone ate the whole CHIP_PAD_X reserve. Hangul and the
 * CJK compat / vertical-form blocks are in for the same reason. Astral CJK needs
 * the second alternative: a BMP class can never match U+20000+, so Ext-B was
 * scoring 0.68 too.
 *
 * Thin spaces and curly quotes in U+2000-U+2BFF are now over-counted. That is
 * deliberate — over is the safe direction, and they are rare in a box title.
 */
const FULLWIDTH =
  /[\u2000-\u2BFF\u3000-\u9FFF\uAC00-\uD7A3\uF900-\uFAFF\uFE10-\uFE4F\uFF01-\uFF60]|[\u{20000}-\u{3FFFF}]/u;

/**
 * The glyphs whose advance runs past the 0.79em the estimator assumes below.
 * Measured on the RENDERED page, not read out of the font file: a `<text>`
 * carrying these exact classes, differenced over repeats —
 * `(len(ch x 41) - len(ch x 31)) / 10`, which cancels the side bearings — then
 * less the 0.05em tracking, leaving the bare advance.
 *
 *   W 1.043em   %  1.004   M 0.923    <- past 0.79
 *   Q 0.773     O  0.769   N 0.759    <- the widest that are not
 *
 * 1.06 clears all three. fontTools agrees on M/Q/O/N to three places but puts W
 * at 1.020, and an earlier reading of this same page came out ~0.99 — both low
 * enough to under-size a ten-W label, so measure a WARM page
 * (`document.fonts.ready`) and trust it over the file.
 *
 * `%` is not a capital and belongs here anyway: `uppercase` does not shrink it
 * and a caption can plausibly carry one ("30% faster"). Three families are
 * knowingly NOT covered, because no label in this course can contain them — the
 * per-mille signs (U+2030/31, 1.42em), the Latin digraphs (U+01C4-01CC,
 * 1.34-1.42em), and U+00DF, which `uppercase` expands to "SS" (1.35em).
 */
const WIDE_GLYPH = /[MWmw%]/;
const WIDE_ADVANCE = 1.06;

/**
 * Estimated chip width. SVG cannot measure text without a DOM, and this has to
 * be identical on the prerender server and in the browser, so it is an estimate
 * by construction. It MUST be biased large: an over-wide chip notches a little
 * more border than it needs, which is invisible; an under-wide one lets the
 * stroke run through the letters, which is not.
 *
 * The first draft said exactly that and then used 0.68em per capital, which is
 * below 15 of Inter 600's 26 (W 1.020, M 0.922, Q 0.773, O 0.769, N 0.759,
 * G 0.749, H 0.746, C 0.737, U 0.736, A 0.728, V 0.728, D 0.722, X 0.720,
 * Y 0.713, K 0.703). The sign stayed positive only because CHIP_PAD_X is added
 * unconditionally — a fixed reserve, not a bias, and one a four-W label eats
 * outright. 0.79 is an upper bound for the other 24 in Inter AND in the system
 * stack the page paints with before the webfont swaps in.
 */
function chipWidth(label: string, font: number): number {
  const chars = Array.from(label);
  let advance = 0;
  for (const ch of chars) {
    if (FULLWIDTH.test(ch)) advance += font;
    else if (ch === ' ') advance += font * 0.3;
    // The class applies `uppercase`, so every Latin glyph renders as a capital
    // — which is exactly why M and W need a step of their own, and why `%`
    // rides along: uppercasing does not shrink it either.
    else if (WIDE_GLYPH.test(ch)) advance += font * WIDE_ADVANCE;
    else advance += font * 0.79;
  }
  return advance + CHIP_TRACK * chars.length + CHIP_PAD_X * 2;
}

/**
 * TARGET LOOK #2 — FrameLabel: a title chip straddling the top border.
 *
 *        +--[ PAGED KV CACHE ]----------+
 *        | +- -                - - - -+ |
 *        | :                          : |
 *
 * A `var(--background)`-filled rect notched into the stroke, with the label
 * sitting on it. On a short PanelFrame the plate reaches the INNER dashed
 * border too, as drawn — inset is 5 there and the plate half-height is 7. That
 * is correct, not a bug: the 10px face itself crosses both strokes, so erasing
 * only the outer one would leave the dashes running through the descenders. This replaces the course's current rule — title INSIDE the box,
 * baseline at roughly `top + 17 + height/12` (CPE:281, 425, 615) — which costs
 * ~20px of vertical space per box and is the main reason CPE reads as 1.376:1
 * rather than 16:9. Moving the title onto the border gives that space back, so
 * #2 and #3 are the same edit.
 *
 * This is the ONE export with no prior art in the repo; everything else was
 * extracted. Treat the geometry as the first draft it is.
 *
 * `x` is the anchor ON the line, `y` is the line itself — pass the panel's own
 * `y`, not `y + something`. `align` says which end of the chip the anchor pins:
 * `start` (default) sits it in from the left corner, which is the reference
 * look; `middle` re-centres it for a symmetric diagram.
 *
 * The line does not have to be a panel border. Any label sitting ON a stroke
 * needs this — a lane caption centred on its own arrow is the same problem, and
 * BackpropFlowDiagram shipped one frame with the elbow drawn straight through
 * the letters because a bare `<text>` looked fine in the frame that was checked.
 * `font` and `fill` exist for that case; the defaults are the panel-title look.
 */
export function FrameLabel({
  x,
  y,
  label,
  align = 'start',
  font = CHIP_FONT,
  fill = 'var(--foreground)',
}: {
  x: number;
  y: number;
  label: string;
  align?: 'start' | 'middle' | 'end';
  font?: number;
  fill?: string;
}) {
  const w = chipWidth(label, font);
  const h = chipH(font);
  const chipX = align === 'start' ? x : align === 'middle' ? x - w / 2 : x - w;
  const textX = align === 'start' ? x + CHIP_PAD_X : align === 'middle' ? x : x - CHIP_PAD_X;

  return (
    <g>
      {/* No stroke: this rect's whole job is to ERASE the stroke under the text. */}
      <rect x={chipX} y={y - h / 2} width={w} height={h} rx={RX} style={{ fill: 'var(--background)' }} />
      <text
        x={textX}
        y={y + chipBaseline(font)}
        textAnchor={align}
        style={{ fill, fontSize: font }}
        className="font-semibold uppercase tracking-wider"
      >
        {label}
      </text>
    </g>
  );
}

// ── DiagramFrame ─────────────────────────────────────────────────────────────

/**
 * Caption typography. Shared so the invisible sizers and the visible caption
 * cannot drift apart — if they did, the slot would be sized for a different
 * font than the one it holds. 12px at `leading-relaxed` is 19.5px a line.
 */
const CAPTION_CLS = 'text-[12px] leading-relaxed text-muted-foreground';

/**
 * Keys that ACTIVATE a control (Enter/Space) or drive one (arrows, Home/End on a
 * radio group, tablist, or slider). Deliberately excludes Tab — see
 * `noticeKeyIntent`.
 */
const ACTIVATION_KEYS = new Set([
  'Enter',
  ' ',
  'Spacebar', // legacy name, still emitted by some IMEs and older WebKit
  'ArrowLeft',
  'ArrowRight',
  'ArrowUp',
  'ArrowDown',
  'Home',
  'End',
]);

// Names the pan container when the diagram actually overflows. Chrome, not a
// Glossary term, so it translates.
const PAN_LABEL: Record<Locale, string> = {
  en: 'Diagram — scroll sideways to see the rest',
  zh: '示意图 — 左右滑动查看其余部分',
};

/**
 * PanBox — lets a too-wide `<svg>` pan sideways instead of shrinking to fit.
 *
 * Every `<svg>` here is `w-full` + `viewBox` with nothing to stop it, so a
 * narrow column does not reflow the drawing — it SCALES the whole canvas down,
 * text included. Measured at a real 375px viewport (column = 320px): a 760-wide
 * viewBox renders at 320, so a 10px label lands at 4.2px. Ten diagrams were
 * between 2.95px and 6.26px — none of them legible.
 *
 * Wrap the svg in this, then give the svg a `min-w-[Npx]` floor, where
 * `N = ceil(8 * VB_W / smallestFontSize)` (8px being the legibility floor). The
 * svg stops shrinking at N and this container pans to reach the rest.
 *
 * WRAP THE SVG, NOT THE WIDGET. An earlier revision put this on DiagramFrame as
 * a `panWide` flag around ALL its children, which broke twice: the frame's
 * `space-y-4` is a DIRECT-child rule (`> :not(:last-child)`), so a single
 * wrapper swallowed every inter-child gap; and a widget's own controls —
 * MemoryWall's 1K/4K/16K buttons, ResidualStream's mode toggle — sat inside the
 * scroller and slid out of reach exactly when the reader needed them. Siblings
 * belong OUTSIDE, where they size to the visible width.
 *
 * Desktop is untouched: the column is wider than every floor, so nothing
 * overflows, no scrollbar appears, and no tab stop is added.
 */
export function PanBox({
  locale,
  className,
  children,
}: {
  locale: Locale;
  className?: string;
  children: React.ReactNode;
}) {
  // A pannable region has to be reachable by keyboard (WCAG 2.1.1) — Chrome,
  // unlike Firefox, will not focus an overflow container on its own. But the
  // tab stop is only earned when there is actually something off-screen: on
  // desktop the column clears every floor, and 10 diagrams that each swallow a
  // Tab press to scroll nothing is its own kind of broken. So measure.
  const ref = React.useRef<HTMLDivElement | null>(null);
  const [canPan, setCanPan] = React.useState(false);
  React.useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setCanPan(el.scrollWidth > el.clientWidth + 1);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // A tab stop with no visible focus indicator fails WCAG 2.4.7, and this one is
  // easy to miss because the element it lands on is an empty scroll container.
  // The ring is drawn on THIS element, and `overflow` clips descendants rather
  // than the element's own box-shadow, so it survives the scroller it sits on.
  const FOCUS = 'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary';

  return (
    // `group`, not `region`: this needs a name and a tab stop, not a
    // seventeenth landmark in the reader's rotor.
    <div
      ref={ref}
      className={className ? `overflow-x-auto ${FOCUS} ${className}` : `overflow-x-auto ${FOCUS}`}
      tabIndex={canPan ? 0 : undefined}
      role={canPan ? 'group' : undefined}
      aria-label={canPan ? PAN_LABEL[locale] : undefined}
    >
      {children}
    </div>
  );
}

/**
 * DiagramFrame — the shell every animated diagram repeats: header row, the svg,
 * the caption, the footnote. Pass the `<svg>` as children; this owns everything
 * around it.
 *
 * TARGET LOOK #4 — LESS CARD CHROME. CPE's shell is
 *   `not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3`
 * (CPE:249), copied verbatim into 51 files. A rounded, bordered, filled card
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
 *
 *  - `announcing`, not `!player.playing`, gates BOTH nodes. Two reasons, both
 *    measured rather than reasoned:
 *
 *    1. `playing` starts TRUE (use-step-player.ts:60), so the first draft's
 *       `aria-hidden="true"` + empty span put the caption in NO accessible node
 *       at all — confirmed by running this component through the prerender's own
 *       esbuild config: the server emits `<span …aria-live="polite"></span>`,
 *       empty, next to an aria-hidden paragraph. A sighted reader saw a caption
 *       a screen-reader user could not reach, on every page, by default.
 *    2. `useStepPlayer` pauses ITSELF twice — for reduced motion after mount
 *       (:63-67) and when a `loop:false` sweep parks (:83-86). Gating on
 *       `playing` alone turned both into an unsolicited announcement during
 *       load. A MutationObserver caught the reduced-motion one as two commits
 *       21ms apart: span `'' -> caption`, with nothing touched.
 *
 *    So: the paragraph is exposed until the reader presses something, and after
 *    that the two nodes hand the sentence back and forth. Exactly one
 *    accessible copy exists at every instant, which is also why this cannot
 *    double-announce.
 *  - `aria-atomic` so the whole sentence reads, not the character diff.
 *  - The caption slot holds still as the caption changes length, because
 *    `captions` stacks the whole sweep in ONE grid cell and the cell takes the
 *    tallest. A fixed `min-h` cannot do that job: it is one number facing two
 *    locales and a 335px-to-742px column, and it was wrong twice. Two lines
 *    hopped the chapter body 38px per loop; three lines still hopped it 18px at
 *    451px, 38px at 375px and 57px at 335px, because the same caption wraps to
 *    4, 5 and 6 lines there. Callers that omit `captions` keep the three-line
 *    guess, and keep that risk.
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
  captions,
  note,
  showReset = false,
  children,
}: {
  /**
   * Heading text, upper-cased by CSS. Already localised by the caller.
   *
   * `ReactNode` for the same reason `caption` is: a few headings carry inline
   * markup that IS content. BpeMergeWalkthrough's heading ends in the literal
   * input it tokenises, in a mono chip — flattening it to a string turned the
   * one thing the chapter is about into ordinary prose. This slot's className
   * is byte-identical to the header div those widgets used before the
   * conversion, so a node passed here renders exactly as it did.
   */
  title: React.ReactNode;
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
  /**
   * Every caption this sweep can show. Optional, and used ONLY to size the
   * slot — the frame stacks them all in one grid cell, so the cell is as tall
   * as the tallest AT THE CURRENT WIDTH and the page cannot hop when the beat
   * changes. Pass the same list the caller indexes into; order is irrelevant.
   *
   * Cheap to get wrong in the safe direction: a caption missing from the list
   * only costs the hop it would have caused anyway, and a stale extra one only
   * reserves a few px. Omitting it entirely falls back to a three-line guess.
   */
  captions?: React.ReactNode[];
  /**
   * Optional standing footnote — "these numbers are illustrative", a source, a
   * caveat. Renders in a `div`, so several `<p>` are fine; a `p` here forced
   * callers with two trailing paragraphs to fake them as block `<span>`.
   *
   * FOOTNOTE TIER ONLY. This slot is 11px muted, so a body-sized teaching
   * paragraph dropped in here is silently demoted a tier. Match what the
   * paragraph was BEFORE the conversion: footnote-sized text belongs in `note`,
   * body-sized text stays a sibling OUTSIDE the frame (return a fragment).
   * Three of the first eight conversions demoted a teaching paragraph before
   * this rule existed, and each picked a different workaround.
   */
  note?: React.ReactNode;
  /** Worth it for long sweeps; noise for short loops. See StepControls.tsx:44. */
  showReset?: boolean;
  children: React.ReactNode;
}) {
  // A pause the READER asked for. `player.playing` alone cannot tell the two
  // apart — see the `announcing` note above.
  const [asked, setAsked] = React.useState(false);
  const ask = <T,>(fn: T) => ((...args: unknown[]) => {
    setAsked(true);
    (fn as (...a: unknown[]) => void)(...args);
  }) as T;
  const intent: StepPlayer = {
    ...player,
    toggle: ask(player.toggle),
    step: ask(player.step),
    reset: ask(player.reset),
    goTo: ask(player.goTo),
  };
  // `ask` only reaches the copy handed to `StepControls`. `children` gets the
  // caller's OWN player, unwrapped — so a widget with its own phase rail or
  // scrubber (SoftmaxTemperatureSteps) called `goTo`, paused, and left the live
  // region EMPTY: the reader pressed something and heard nothing back. Rather
  // than thread a second player through `children` and make every caller
  // remember which one to use, catch the interaction where it happens.
  //
  // `click`, not `pointerdown`, because a screen reader's virtual click fires
  // no pointer events — and that reader is the whole point of this contract.
  // `keydown` for Space/Enter and for arrow-key controls. `closest` so that
  // selecting the caption text is not mistaken for an ask, which would put the
  // unsolicited-announcement failure (see 2. above) back on the table.
  const controlHit = (e: React.SyntheticEvent) =>
    (e.target as Element | null)?.closest?.(
      'button,input,select,summary,a[href],[role="button"],[role="radio"],[role="tab"],[role="slider"]',
    ) != null;
  const noticeIntent = (e: React.SyntheticEvent) => {
    if (controlHit(e)) setAsked(true);
  };

  // Keyboard is not the same as pointer here: `keydown` also fires for Tab, and
  // Tab merely LANDING on the play button is navigation, not a request. `asked`
  // is a one-way latch wired to a live region, so treating arrival as intent
  // means a reader tabbing past the widget gets the caption read at them — the
  // exact unsolicited announcement this contract exists to prevent.
  const noticeKeyIntent = (e: React.KeyboardEvent) => {
    if (ACTIVATION_KEYS.has(e.key)) noticeIntent(e);
  };

  /** Is the live region holding the caption right now? */
  const announcing = asked && !player.playing;

  return (
    <div className="not-prose my-6 space-y-4" onClickCapture={noticeIntent} onKeyDownCapture={noticeKeyIntent}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{title}</div>
        <StepControls player={intent} locale={locale} showReset={showReset} />
      </div>

      {children}

      {/* The caption slot. Every caption of the sweep sits in the SAME grid
          cell, so the cell is as tall as the tallest one at whatever width the
          column happens to be — no magic number, no per-locale tuning, and
          nothing to keep in sync when a caption is reworded. The sizers are
          `invisible` rather than hidden so they still take up space, and they
          are out of the a11y tree, so only the real one below is readable.

          The cell sizes to the tallest MARGIN box, so it measures ~24px taller
          than any caption in it: the chapter body puts `my-3` on every `p`
          through an `[&_p]` variant, and `.not-prose` does not stop it — this
          course has no typography plugin, so that class is inert here. That is
          the body's own paragraph rhythm, and the same margin the single
          caption carried before this slot existed. Measured, not assumed. Do
          NOT add `m-0` to make the number tidy: it would quietly retune caption
          spacing in every widget that uses this frame. */}
      <div className="grid">
        {(captions ?? []).map((sizer, i) => (
          <p key={i} aria-hidden="true" className={`invisible col-start-1 row-start-1 ${CAPTION_CLS}`}>
            {sizer}
          </p>
        ))}
        {/* Exactly ONE accessible copy, always. Not announcing -> the plain
            paragraph carries it (no live ancestor, so a sweep says nothing).
            Announcing -> the paragraph steps out and the region speaks, once. */}
        <p
          className={`col-start-1 row-start-1 ${captions ? '' : 'min-h-[3.75rem] '}${CAPTION_CLS}`}
          aria-hidden={announcing ? 'true' : undefined}
        >
          {caption}
        </p>
      </div>
      <span className="sr-only" aria-live="polite" aria-atomic="true">
        {announcing ? caption : ''}
      </span>

      {note ? (
        <div className="space-y-2 border-t border-border/60 pt-2 text-[11px] leading-relaxed text-muted-foreground">
          {note}
        </div>
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
