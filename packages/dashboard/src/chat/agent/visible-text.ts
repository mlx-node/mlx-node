/**
 * Agent text shown on a permission or plan card: every character that can hide or reorder what is shown (controls
 * other than tab and new line, bidirectional marks and overrides, zero-width and invisible format characters,
 * spaces other than the plain space, blank-looking letters, variation selectors, tag characters, line and paragraph
 * separators) is replaced by its visible code, `\u{202E}`. Nothing else changes, so the card shows exactly what will run.
 */

function hidden(code: number): boolean {
  if (code === 0x09 || code === 0x0a) return false;
  if (code < 0x20 || (code >= 0x7f && code <= 0xa0)) return true;
  if (code === 0x00ad || code === 0x034f || code === 0x061c) return true;
  if (code === 0x115f || code === 0x1160 || code === 0x3164 || code === 0xffa0 || code === 0x2800) return true;
  if (code >= 0x180b && code <= 0x180f) return true;
  if (code >= 0x2000 && code <= 0x200f) return true;
  if (code >= 0x2028 && code <= 0x202f) return true;
  if (code >= 0x205f && code <= 0x206f) return true;
  if (code === 0x3000 || code === 0xfeff) return true;
  if (code >= 0xfe00 && code <= 0xfe0f) return true;
  if (code >= 0xfff9 && code <= 0xfffb) return true;
  if (code >= 0xe0000 && code <= 0xe007f) return true;
  return code >= 0xe0100 && code <= 0xe01ef;
}

export function visibleText(text: string): string {
  let out = '';
  let from = 0;
  for (let i = 0; i < text.length;) {
    const code = text.codePointAt(i)!;
    const size = code > 0xffff ? 2 : 1;
    if (hidden(code)) {
      out += `${text.slice(from, i)}\\u{${code.toString(16).toUpperCase().padStart(4, '0')}}`;
      from = i + size;
    }
    i += size;
  }
  return from === 0 ? text : out + text.slice(from);
}
