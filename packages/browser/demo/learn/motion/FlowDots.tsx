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
      // A dash of length 0 with a round cap paints a single dot; `spacing` is
      // the gap to the next one. The gap carries the whole period.
      strokeDasharray={`0 ${spacing}`}
      style={
        {
          '--flow-dot-period': `${spacing}px`,
          '--flow-dot-duration': `${durationMs}ms`,
          '--flow-dot-direction': reverse ? 'reverse' : 'normal',
        } as React.CSSProperties
      }
    />
  );
}
