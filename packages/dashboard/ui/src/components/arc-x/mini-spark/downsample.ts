/** Means of `max` equal buckets, in order; a bucket with no number stays null (a gap, never 0). */
export function downsampleMean(data: readonly (number | null)[], max: number): (number | null)[] {
  if (data.length <= max || max <= 0) return [...data];
  const out: (number | null)[] = [];
  for (let bucket = 0; bucket < max; bucket += 1) {
    const start = Math.floor((bucket * data.length) / max);
    const end = Math.floor(((bucket + 1) * data.length) / max);
    let sum = 0;
    let n = 0;
    for (let i = start; i < end; i += 1) {
      const value = data[i];
      if (typeof value === 'number' && Number.isFinite(value)) {
        sum += value;
        n += 1;
      }
    }
    out.push(n === 0 ? null : sum / n);
  }
  return out;
}
