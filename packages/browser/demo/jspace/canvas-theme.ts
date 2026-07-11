/**
 * Canvas cannot read CSS custom properties, and the existing canvas widgets
 * (learn/inspector/AttentionHeatmap.tsx) hardcode their colors rather than
 * resolving them. The site is dark-only — `:root` IS the dark palette and
 * `.dark` is never applied — so a literal palette is correct, not a shortcut.
 * Keep these in sync with demo/styles.css `:root`.
 */
export const CANVAS = {
  bg: '#0f0d11',
  ink: '#ece4da', // = --text (warm cream); the grids read as recessed screens in their panels
  inkMuted: '#9e928a', // = --text-dim; warms the ℓ-gutter labels + RankChart y-ticks/axis
  grid: 'rgba(236, 228, 218, 0.07)', // warm, faint hairlines
  selectionRing: '#ec4899', // MUST match .jspace-grid-scroll canvas:focus-visible + RankChart selected-x
  rankSuperscript: '#b91c1c',
  unranked: '#201b26', // grey-out aligned to the warmed surface family
} as const;

export const CELL_W = 68;
export const CELL_H = 26;
export const GUTTER_W = 46; // layer-label column
