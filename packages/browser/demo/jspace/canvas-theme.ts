/**
 * Canvas cannot read CSS custom properties, and the existing canvas widgets
 * (learn/inspector/AttentionHeatmap.tsx) hardcode their colors rather than
 * resolving them. The site is dark-only — `:root` IS the dark palette and
 * `.dark` is never applied — so a literal palette is correct, not a shortcut.
 * Keep these in sync with demo/styles.css `:root`.
 */
export const CANVAS = {
  bg: '#0f0d11',
  ink: '#e8e4ec',
  inkMuted: '#8b8494',
  grid: 'rgba(232, 228, 236, 0.07)',
  selectionRing: '#ec4899',
  rankSuperscript: '#b91c1c',
  unranked: '#241f29',
} as const;

export const CELL_W = 68;
export const CELL_H = 26;
export const GUTTER_W = 46; // layer-label column
