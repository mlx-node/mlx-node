/**
 * The single pin-derivation predicate, shared by `bake.mts`, both lesson
 * widgets, and the /jspace self-test oracle.
 *
 * A concept is pinned by the first token of its mid-sentence form (` season`).
 * Qwen fuses a leading space into an alphabetic token but splits it off before a
 * digit, so ` 7` tokenizes to [' ', '7'] — pinning the first token there would
 * track bare whitespace, and the rank chip would say nothing about the concept
 * it is labelled with. Fall back to the space-less form in exactly that case.
 *
 * Pure. Async only through the injected `encode`, so the same code runs under
 * node with the native tokenizer and in the browser over the worker RPC.
 */

import { LENS_MAX_PINNED } from '../../src/inspector-types';

export type Encode = (text: string) => Promise<{ id: number; text: string }[]>;

export type DerivedPins = {
  /** Token ids, index-aligned with the accepted concepts. */
  pinnedIds: number[];
  /** True when the concept needed more than one token — the pin tracks only its first. */
  partialFlags: boolean[];
};

export async function derivePins(concepts: string[], encode: Encode): Promise<DerivedPins> {
  const pinnedIds: number[] = [];
  const partialFlags: boolean[] = [];

  for (const concept of concepts) {
    if (pinnedIds.length >= LENS_MAX_PINNED) break;

    let tokens = await encode(` ${concept}`);
    if (tokens.length > 0 && tokens[0]!.text.trim() === '') {
      tokens = await encode(concept);
    }
    if (tokens.length === 0) continue;

    pinnedIds.push(tokens[0]!.id);
    partialFlags.push(tokens.length > 1);
  }

  return { pinnedIds, partialFlags };
}
