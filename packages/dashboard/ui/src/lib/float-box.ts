/** A box in px from the window's top-left corner. */
export interface Box {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/** A floating box keeps this far from the edge of what can show it. */
const FLOAT_EDGE = 8;

/**
 * The top of a floating box of `height`: `above` when the whole box fits under the top of `view`, else `below` when it
 * fits over its bottom, else the side with more room, moved inside. Always 8 px inside `view`.
 */
export function fitTop(above: number, below: number, height: number, view: Box): { y: number; below: boolean } {
  const top = view.top + FLOAT_EDGE;
  const bottom = view.bottom - FLOAT_EDGE;
  if (above >= top) return { y: above, below: false };
  if (below + height <= bottom) return { y: below, below: true };
  const roomBelow = bottom - below > above + height - top;
  return { y: Math.max(top, Math.min(roomBelow ? below : above, bottom - height)), below: roomBelow };
}

/** The left edge of a floating box of `width` that wants `left`, moved 8 px inside `view` (to its left edge when wider). */
export function fitLeft(left: number, width: number, view: Box): number {
  return Math.max(view.left + FLOAT_EDGE, Math.min(left, view.right - FLOAT_EDGE - width));
}

/** The part of `box` inside `view`, or null when none of it is. */
export function boxInView(box: Box, view: Box): Box | null {
  const left = Math.max(box.left, view.left);
  const top = Math.max(box.top, view.top);
  const right = Math.min(box.right, view.right);
  const bottom = Math.min(box.bottom, view.bottom);
  return right > left && bottom > top ? { left, top, right, bottom } : null;
}

/**
 * `view`: the part of the window a node can show in (the inner box of every clipping ancestor). `limits`: the visible
 * page area (every vertical scroller), for a box drawn on a layer outside those ancestors.
 */
export function clipBoxes(node: Element): { view: Box; limits: Box } {
  const view = { left: 0, top: 0, right: window.innerWidth, bottom: window.innerHeight };
  const limits = { ...view };
  for (let parent = node.parentElement; parent !== null; parent = parent.parentElement) {
    const style = getComputedStyle(parent);
    if (style.display === 'contents' || style.display === 'inline') continue;
    const clipsX = style.overflowX !== 'visible' && style.overflowX !== '';
    const clipsY = style.overflowY !== 'visible' && style.overflowY !== '';
    if (!clipsX && !clipsY) continue;
    const border = parent.getBoundingClientRect();
    const left = border.left + parent.clientLeft;
    const top = border.top + parent.clientTop;
    const right = left + parent.clientWidth;
    const bottom = top + parent.clientHeight;
    if (clipsX) {
      view.left = Math.max(view.left, left);
      view.right = Math.min(view.right, right);
    }
    if (clipsY) {
      view.top = Math.max(view.top, top);
      view.bottom = Math.min(view.bottom, bottom);
    }
    if (style.overflowY === 'auto' || style.overflowY === 'scroll') {
      limits.left = Math.max(limits.left, left);
      limits.right = Math.min(limits.right, right);
      limits.top = Math.max(limits.top, top);
      limits.bottom = Math.min(limits.bottom, bottom);
    }
  }
  return { view, limits };
}
