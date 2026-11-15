/**
 * Longest a category (model) label may run before it is shortened.
 *
 * The only bound on how wide a category label grows, so it is the guard that
 * keeps a 60-character model id from eating the plot area.
 */
export const CATEGORY_LABEL_MAX_CHARS = 26;

/**
 * Shorten `text` to `maxChars` by dropping its MIDDLE and keeping both ends.
 *
 * Head truncation is wrong for model ids specifically: the family and size sit
 * at the front while the quantisation recipe sits at the back, so
 * `Gemma-4-31B-IT-UD-Q4_K_XL-mlx` and `Gemma-4-31B-IT-UD-Q4_K_M-mlx` are
 * identical for their first 21 characters. A tail-clipping shortener renders
 * both as the same string on a chart whose entire purpose is to compare them.
 */
export function truncateMiddle(text: string, maxChars: number): string {
  if (maxChars < 2) return text.slice(0, Math.max(0, maxChars));
  if (text.length <= maxChars) return text;
  const keep = maxChars - 1;
  const head = Math.ceil(keep / 2);
  return `${text.slice(0, head)}…${text.slice(text.length - (keep - head))}`;
}

/**
 * Map each key to a display label that is short enough for an axis AND distinct
 * from every other label in the same chart.
 *
 * Distinctness is the point. Shortening alone can map two different models onto
 * one string, and two rows reading the same name on a comparison chart is worse
 * than a long label: the reader cannot tell which model is which, and nothing in
 * the rendering says so. When shortening collides, the later key keeps a
 * numbered suffix so the rows stay tellable apart; the full id is still in the
 * label's `title`/tooltip.
 */
export function categoryLabels(keys: Iterable<string>, maxChars = CATEGORY_LABEL_MAX_CHARS): Map<string, string> {
  const unique = [...new Set(keys)];
  const shortened = unique.map((key) => truncateMiddle(key, maxChars));
  if (new Set(shortened).size === shortened.length) {
    return new Map(unique.map((key, index) => [key, shortened[index]]));
  }
  // Leave room for the ` (2)` discriminator before it is appended, so a
  // disambiguated label is no wider than an ordinary one.
  const bases = unique.map((key) => truncateMiddle(key, Math.max(2, maxChars - 4)));
  const labels = new Map<string, string>();
  const taken = new Set<string>();
  unique.forEach((key, index) => {
    const base = bases[index];
    let label = base;
    for (let nth = 2; taken.has(label); nth++) label = `${base} (${nth})`;
    taken.add(label);
    labels.set(key, label);
  });
  return labels;
}
