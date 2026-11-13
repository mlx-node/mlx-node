/**
 * Cells per segment: largest-remainder rounding of value / capacity × cells; a segment with value > 0 gets at least
 * one cell (taken from the largest). The sum never exceeds `cells`.
 */
export function cellCounts(capacity: number, values: readonly number[], cells: number): number[] {
  const counts = values.map(() => 0);
  if (!(capacity > 0) || cells <= 0) return counts;
  const exact = values.map((value) => (Number.isFinite(value) && value > 0 ? (value / capacity) * cells : 0));
  const floors = exact.map((value) => Math.floor(value));
  const used = floors.reduce((sum, value) => sum + value, 0);
  const wanted = Math.min(cells, Math.round(exact.reduce((sum, value) => sum + value, 0)));
  const order = exact
    .map((value, index) => ({ index, rest: value - Math.floor(value) }))
    .sort((a, b) => b.rest - a.rest || a.index - b.index);
  for (let i = 0; i < floors.length; i += 1) counts[i] = floors[i];
  let extra = Math.max(0, wanted - used);
  for (const entry of order) {
    if (extra <= 0) break;
    if (entry.rest <= 0) continue;
    counts[entry.index] += 1;
    extra -= 1;
  }
  for (let i = 0; i < counts.length; i += 1) {
    if (exact[i] <= 0 || counts[i] > 0) continue;
    let largest = -1;
    for (let j = 0; j < counts.length; j += 1)
      if (counts[j] > 1 && (largest === -1 || counts[j] > counts[largest])) largest = j;
    if (largest !== -1) {
      counts[largest] -= 1;
      counts[i] = 1;
    } else if (counts.reduce((sum, value) => sum + value, 0) < cells) counts[i] = 1;
  }
  let total = counts.reduce((sum, value) => sum + value, 0);
  while (total > cells) {
    let largest = 0;
    for (let j = 1; j < counts.length; j += 1) if (counts[j] > counts[largest]) largest = j;
    counts[largest] -= 1;
    total -= 1;
  }
  return counts;
}
