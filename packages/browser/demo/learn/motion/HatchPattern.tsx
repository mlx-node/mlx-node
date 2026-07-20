import * as React from 'react';

/**
 * HatchPattern — diagonal hatching used to mean "this box is BUSY on this
 * step". The course already encodes state with colour (emerald = good/done,
 * red = the expensive part) and with dashed-vs-solid borders (borrowed vs
 * owned). Hatching adds a third, orthogonal channel — *active right now* —
 * that survives being layered on top of either of those.
 *
 * It is also the one state cue here that does not rely on colour at all, so it
 * still reads for a colour-blind reader and in a monochrome print.
 *
 * Usage — render `<HatchDefs />` ONCE inside the svg, then fill any shape with
 * `fill={hatchFill('primary')}`:
 *
 *   <svg …>
 *     <HatchDefs />
 *     <rect … fill={active ? hatchFill('red') : 'var(--card)'} />
 *   </svg>
 *
 * The pattern ids are module constants rather than `useId()` values: `useId()`
 * produces different strings on the server and on the client, and while this
 * course boots with `createRoot` (so a mismatch would not throw), a stable id
 * keeps the prerendered HTML byte-identical between builds. Two widgets on one
 * page emitting the same `<defs>` id is harmless — the definitions are
 * identical, and the first one wins.
 */

export type HatchTone = 'red' | 'emerald' | 'primary' | 'muted';

const IDS: Record<HatchTone, string> = {
  red: 'mlx-hatch-red',
  emerald: 'mlx-hatch-emerald',
  primary: 'mlx-hatch-primary',
  muted: 'mlx-hatch-muted',
};

const STROKES: Record<HatchTone, string> = {
  red: '#ef4444', // red-500 — the expensive/compute box, matches the course's RED
  emerald: '#10b981', // emerald-500 — matches the course's EMERALD
  primary: 'var(--primary)',
  muted: 'var(--muted-foreground)',
};

/** `fill` value that paints a shape with the given hatch. */
export function hatchFill(tone: HatchTone): string {
  return `url(#${IDS[tone]})`;
}

/**
 * The `<defs>` block. Render once per svg, before anything that uses it.
 *
 * `patternUnits="userSpaceOnUse"` keeps the hatch pitch constant in viewBox
 * units, so the stripes stay the same visual weight in every box regardless of
 * that box's size — with the default `objectBoundingBox` they would stretch to
 * each shape and read as a different texture per box.
 */
export function HatchDefs({ opacity = 0.28 }: { opacity?: number }) {
  return (
    <defs>
      {(Object.keys(IDS) as HatchTone[]).map((tone) => (
        <pattern
          key={tone}
          id={IDS[tone]}
          width={7}
          height={7}
          patternUnits="userSpaceOnUse"
          patternTransform="rotate(45)"
        >
          <line x1={0} y1={0} x2={0} y2={7} stroke={STROKES[tone]} strokeWidth={1.6} opacity={opacity} />
        </pattern>
      ))}
    </defs>
  );
}
