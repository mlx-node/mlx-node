import { Buffer } from 'node:buffer';

import type { Entity, RedactOptions, Replacement } from './types.js';

/**
 * Apply entity redaction to `text` using the entities returned by the
 * classifier.
 *
 * - If `opts.labels` is set, only entities whose label is in that list are
 *   redacted (others are left in place verbatim).
 * - Each surviving entity span is replaced via {@link renderReplacement}.
 * - The returned `entities` array is sorted by start offset and only
 *   contains the entities that were actually redacted (post-filter).
 *
 * Entities are assumed to be non-overlapping (the privacy-filter Viterbi
 * decoder produces non-overlapping spans), and we walk them left-to-right
 * with a running cursor so the output is built in a single pass.
 *
 * IMPORTANT: `Entity.start`/`Entity.end` are **UTF-8 byte offsets** (the
 * Hugging Face `tokenizers` convention used by the underlying Rust
 * classifier). JavaScript's `String.prototype.slice` indexes UTF-16 code
 * units, so slicing the original string directly corrupts spans whenever
 * any non-ASCII character (emoji, CJK, accented Latin) appears before an
 * entity. We therefore encode `text` as a UTF-8 Buffer once and slice
 * bytes, decoding each segment back to a string.
 */
export function redactImpl(
  text: string,
  entities: Entity[],
  opts?: RedactOptions,
): { redacted: string; entities: Entity[] } {
  const labelFilter = opts?.labels ? new Set(opts.labels) : null;
  const filtered = labelFilter ? entities.filter((e) => labelFilter.has(e.label)) : entities;
  const sorted = [...filtered].sort((a, b) => a.start - b.start);

  // Fast path: no redactions → return the original text unchanged without
  // a Buffer round-trip.
  if (sorted.length === 0) {
    return { redacted: text, entities: sorted };
  }

  const buf = Buffer.from(text, 'utf8');
  let out = '';
  let cursor = 0;
  for (const e of sorted) {
    out += buf.toString('utf8', cursor, e.start);
    out += renderReplacement(e, opts?.replacement);
    cursor = e.end;
  }
  out += buf.toString('utf8', cursor, buf.length);

  return { redacted: out, entities: sorted };
}

/**
 * Render the replacement string for a single entity. Defaults to
 * `[<label>]` when no replacement is provided.
 */
function renderReplacement(entity: Entity, replacement?: Replacement): string {
  if (typeof replacement === 'function') return replacement(entity);
  if (replacement === 'label' || replacement == null) return `[${entity.label}]`;
  return replacement;
}
