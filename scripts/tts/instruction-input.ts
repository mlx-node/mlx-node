import type { TtsInputEvent } from '@mlx-node/tts';

/** Schedule control events against the original text, independent of input chunk boundaries. */
export async function* paragraphInstructions(
  source: AsyncIterable<string>,
  text: string,
  instructions: readonly (string | null)[],
): AsyncGenerator<TtsInputEvent> {
  const boundaries = instructions.length
    ? [0, ...Array.from(text.matchAll(/\n\s*\n/gu), (m) => m.index! + m[0].length)]
    : [];
  let at = 0,
    next = 0;
  for await (const chunk of source) {
    let cursor = 0;
    while (next < boundaries.length && boundaries[next] <= at + chunk.length) {
      const end = boundaries[next] - at;
      if (end > cursor) yield chunk.slice(cursor, end);
      yield { type: 'instruct', value: instructions[next % instructions.length] };
      cursor = end;
      next++;
    }
    if (cursor < chunk.length) yield chunk.slice(cursor);
    at += chunk.length;
  }
  if (at !== text.length) throw new Error('Paced input length differs from the instruction schedule');
}
