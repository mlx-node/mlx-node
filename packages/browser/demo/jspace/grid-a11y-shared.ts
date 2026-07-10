// grid-a11y-shared.ts — neutral primitives shared by the canvas grid widgets and
// the `useCanvasGridA11y` hook.
//
// These live here, not in ArgmaxGridCanvas.tsx, on purpose: the hook needs the
// `SR_ONLY` VALUE, and ArgmaxGridCanvas imports the hook. Homing the value in a
// leaf module keeps that from becoming a value-level import cycle
// (ArgmaxGridCanvas → hook → ArgmaxGridCanvas). `CellRef` is re-exported from
// ArgmaxGridCanvas for back-compat, so existing `./ArgmaxGridCanvas` imports of
// the type keep working.

import type * as React from 'react';

/** A cell coordinate: an index into `slice.layers` and a prompt position. */
export type CellRef = { layerIdx: number; pos: number };

/** Visually-hidden but screen-reader-reachable style (clip, not display:none). */
export const SR_ONLY: React.CSSProperties = {
  position: 'absolute',
  width: 1,
  height: 1,
  overflow: 'hidden',
  clip: 'rect(0,0,0,0)',
  whiteSpace: 'nowrap',
  left: 0,
  top: 0,
};
