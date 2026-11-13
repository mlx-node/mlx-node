/**
 * The tooltip text of `node`: its title, or its `data-clip-title` (the full text) while CSS cuts it. null when it has
 * neither.
 */
export function hoverTitle(node: Element): string | null {
  const title = node.getAttribute('title');
  if (title !== null) return title;
  const full = node.getAttribute('data-clip-title');
  return full !== null && node.scrollWidth > node.clientWidth ? full : null;
}

/**
 * A row link's overlay covers every cell, so the pointer never reaches a title inside the row: the link takes the
 * title of whatever lies under the pointer.
 */
export function forwardRowTitle(row: HTMLElement, link: HTMLElement, x: number, y: number): void {
  let title: string | null = null;
  for (const node of document.elementsFromPoint(x, y)) {
    if (node === link) continue;
    if (node === row || !row.contains(node)) break;
    title = hoverTitle(node);
    if (title !== null) break;
  }
  if (title === null) link.removeAttribute('title');
  else link.setAttribute('title', title);
}

/** `full` as the title of `node` while CSS cuts its text; no title while it fits. */
export function titleWhileCut(node: Element, full: string): void {
  if (node.scrollWidth > node.clientWidth) node.setAttribute('title', full);
  else node.removeAttribute('title');
}
