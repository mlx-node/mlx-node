/** Reasoning close tags: Qwen-style `</think>` and Hy4-flash `</think:X>`. */

function isHy4SuffixChar(ch: string): boolean {
  return ch !== '<' && ch !== '>' && ch.trim().length > 0;
}

export interface Hy4Tag {
  start: number;
  end: number;
  suffix: string;
}

/** The `<name:X>` tag whose `head` (`<name:`) is at `at`, or null. */
export function hy4TagAt(text: string, at: number, head: string): Hy4Tag | null {
  if (!text.startsWith(head, at)) return null;
  let i = at + head.length;
  while (i < text.length && isHy4SuffixChar(text[i]!)) i++;
  if (i === at + head.length || text[i] !== '>') return null;
  return { start: at, end: i + 1, suffix: text.slice(at + head.length - 1, i) };
}

const CLOSE_HEAD = '</think';
const CLOSE = '</think>';
const HY4_CLOSE_HEAD = '</think:';

/**
 * `text` without the reasoning close tags (vLLM `hy_v4_reasoning_parser`
 * drops the close from the reasoning deltas).
 */
export function withoutReasoningClose(text: string): string {
  let out = '';
  let from = 0;
  let at = text.indexOf(CLOSE_HEAD);
  while (at >= 0) {
    const end = text.startsWith(CLOSE, at) ? at + CLOSE.length : (hy4TagAt(text, at, HY4_CLOSE_HEAD)?.end ?? -1);
    if (end < 0) {
      at = text.indexOf(CLOSE_HEAD, at + 1);
      continue;
    }
    out += text.slice(from, at);
    from = end;
    at = text.indexOf(CLOSE_HEAD, end);
  }
  return out + text.slice(from);
}
