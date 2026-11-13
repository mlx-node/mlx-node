/** The part of a hast node this plugin reads and writes. */
export interface HastNodeLike {
  type: string;
  position?: { start?: { offset?: number } };
  properties?: Record<string, unknown>;
  children?: HastNodeLike[];
}

/**
 * Rehype plugin: marks every TOP-LEVEL element of the tree whose source starts at or after `freshFrom` with
 * `data-fresh`. Nested elements are never marked: a new item of a list that started earlier is part of that block.
 */
export function freshBlocks(freshFrom: number): (tree: HastNodeLike) => void {
  return (tree) => {
    const children = tree.children ?? [];
    for (const child of children) {
      if (child.type !== 'element') continue;
      const offset = child.position?.start?.offset;
      if (offset === undefined || offset < freshFrom) continue;
      child.properties = { ...child.properties, dataFresh: '' };
    }
  };
}
