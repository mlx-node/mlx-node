import * as React from 'react';

/**
 * FlowDots — a stream of dots travelling along a path, the "data is moving from
 * here to there, right now" primitive.
 *
 * Implementation note, because the obvious approaches are worse:
 *
 *  - NOT SMIL `<animateMotion>`: deprecated-ish, awkward to stop for reduced
 *    motion, and one element per dot.
 *  - NOT requestAnimationFrame: this is decoration, it should not hold a React
 *    render loop open. The course's one rAF widget is a canvas heatmap.
 *  - INSTEAD: ONE `<path>` whose stroke is a dashed line with round caps and a
 *    zero-length dash — which renders as evenly spaced DOTS — animated by
 *    shifting `stroke-dashoffset`. The browser compositor moves them; React
 *    re-renders nothing. One element, no JS timer, GPU-cheap.
 *
 * The keyframes and the reduced-motion kill live in `demo/styles.css`
 * (`.flow-dots`, `@keyframes flowDots`) alongside the other course animations.
 *
 * Under reduced motion the CSS animation is disabled by the media query, so the
 * dots render as a STATIC dotted line — which still reads correctly as "these
 * two boxes are connected and carrying data", just without the motion.
 *
 * The media query is the ONLY thing CSS can stop by itself, so a widget with a
 * Pause button must pass `paused` — see the prop. Otherwise the frame timer
 * stops while these dots keep travelling, and the Pause control is a lie.
 */

export type FlowDotsProps = {
  /** SVG path `d`. Elbow polylines (`M x y L x y L x y`) are the usual case. */
  d: string;
  /** dot colour; defaults to the course's accent blue */
  color?: string;
  /** dot diameter in viewBox units (stroke width) */
  size?: number;
  /** centre-to-centre spacing between dots, in viewBox units */
  spacing?: number;
  /** seconds for one dot to travel exactly one `spacing` (lower = faster) */
  durationMs?: number;
  /** run the dots backwards along the path */
  reverse?: boolean;
  /** hide entirely — for frames where this edge is idle */
  hidden?: boolean;
  /**
   * Freeze the dots where they are. Pass `!player.playing` — a Pause button
   * that stops the JS frame timer but leaves this CSS animation running is a
   * Pause button that visibly does not pause (WCAG 2.2.2 Pause, Stop, Hide).
   * Freezing rather than hiding keeps the "these boxes are connected" reading.
   */
  paused?: boolean;
};

/** Accent blue for in-flight data. Matches the reference motion language. */
export const FLOW_BLUE = '#3b82f6'; // blue-500

export function FlowDots({
  d,
  color = FLOW_BLUE,
  size = 3.5,
  spacing = 14,
  durationMs = 900,
  reverse = false,
  hidden = false,
  paused = false,
}: FlowDotsProps) {
  if (hidden) return null;

  return (
    <path
      className="flow-dots"
      d={d}
      fill="none"
      stroke={color}
      strokeWidth={size}
      strokeLinecap="round"
      // A near-zero dash with a round cap paints a single dot of diameter
      // `size`; `spacing` is the gap to the next one, so the gap carries the
      // whole period. The dash is 0.01 rather than a true 0 deliberately:
      // WebKit has historically dropped zero-length dashes entirely (nothing
      // renders), while every engine draws the round cap on a tiny non-zero
      // one. 0.01 user units is invisible on its own — the cap is the dot.
      strokeDasharray={`0.01 ${spacing}`}
      style={
        {
          '--flow-dot-period': `${spacing}px`,
          '--flow-dot-duration': `${durationMs}ms`,
          '--flow-dot-direction': reverse ? 'reverse' : 'normal',
          // Inline, so it beats the `.flow-dots` class without a CSS edit.
          animationPlayState: paused ? 'paused' : 'running',
        } as React.CSSProperties
      }
    />
  );
}
