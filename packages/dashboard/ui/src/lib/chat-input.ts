/**
 * The composer's copy of the server's slash-text check (`src/chat/route-input.ts`): text whose first non-blank
 * character is `/` runs as an agent slash command, so Send is off for it. The server's check is the law.
 */

/** Unicode `White_Space` (what Rust's `trim` skips) plus U+FEFF (what JavaScript's `trim` also skips). */
function isTrimmed(code: number): boolean {
  return (
    (code >= 0x09 && code <= 0x0d) ||
    code === 0x20 ||
    code === 0x85 ||
    code === 0xa0 ||
    code === 0x1680 ||
    (code >= 0x2000 && code <= 0x200a) ||
    code === 0x2028 ||
    code === 0x2029 ||
    code === 0x202f ||
    code === 0x205f ||
    code === 0x3000 ||
    code === 0xfeff
  );
}

/** Whether the first character after leading white space is `/`. */
export function isSlashText(text: string): boolean {
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (isTrimmed(code)) continue;
    return code === 0x2f;
  }
  return false;
}
