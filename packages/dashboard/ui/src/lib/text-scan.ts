function isWhitespaceCode(code: number): boolean {
  if (code === 0x20 || (code >= 0x09 && code <= 0x0d)) return true;
  if (code < 0x80) return false;
  return (
    code === 0x00a0 ||
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

function isDigitCode(code: number): boolean {
  return code >= 0x30 && code <= 0x39;
}

function isAsciiLetterCode(code: number): boolean {
  return (code >= 0x41 && code <= 0x5a) || (code >= 0x61 && code <= 0x7a);
}

/** Space, \t \n \v \f \r, U+00A0, U+1680, U+2000-200A, U+2028, U+2029, U+202F, U+205F, U+3000, U+FEFF. */
export function isWhitespace(ch: string): boolean {
  return ch.length > 0 && isWhitespaceCode(ch.charCodeAt(0));
}

export function isAsciiDigit(ch: string): boolean {
  return ch.length > 0 && isDigitCode(ch.charCodeAt(0));
}

/** Letter or digit in any script: ASCII [0-9A-Za-z]; or code unit >= 0x80 that is not whitespace and not in U+2000-206F / U+3000-303F. */
export function isWordChar(ch: string): boolean {
  if (ch.length === 0) return false;
  const code = ch.charCodeAt(0);
  if (code < 0x80) return isDigitCode(code) || isAsciiLetterCode(code);
  if (isWhitespaceCode(code)) return false;
  if (code >= 0x2000 && code <= 0x206f) return false;
  if (code >= 0x3000 && code <= 0x303f) return false;
  return true;
}

export function hasNonWhitespace(text: string): boolean {
  for (let i = 0; i < text.length; i += 1) {
    if (!isWhitespaceCode(text.charCodeAt(i))) return true;
  }
  return false;
}

/** Runs of whitespace split; no empty parts. */
export function splitWhitespace(text: string): string[] {
  const parts: string[] = [];
  let start = -1;
  for (let i = 0; i < text.length; i += 1) {
    if (isWhitespaceCode(text.charCodeAt(i))) {
      if (start !== -1) {
        parts.push(text.slice(start, i));
        start = -1;
      }
    } else if (start === -1) {
      start = i;
    }
  }
  if (start !== -1) parts.push(text.slice(start));
  return parts;
}

/** Drops every '<' … '>' run; an unclosed '<' is kept. */
export function stripTags(text: string): string {
  let out = '';
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (ch === '<') {
      const close = text.indexOf('>', i + 1);
      if (close === -1) {
        out += text.slice(i);
        break;
      }
      i = close + 1;
      continue;
    }
    out += ch;
    i += 1;
  }
  return out;
}

export function stripCommas(text: string): string {
  let out = '';
  for (let i = 0; i < text.length; i += 1) {
    if (text[i] !== ',') out += text[i];
  }
  return out;
}

/** A digit first, then only digits . , / : and an optional final %. */
export function isNumericWord(word: string): boolean {
  if (word.length === 0 || !isDigitCode(word.charCodeAt(0))) return false;
  const end = word.endsWith('%') ? word.length - 1 : word.length;
  for (let i = 1; i < end; i += 1) {
    const ch = word[i];
    if (!(isDigitCode(word.charCodeAt(i)) || ch === '.' || ch === ',' || ch === '/' || ch === ':')) return false;
  }
  return true;
}

function isIdentifierStart(code: number): boolean {
  return isAsciiLetterCode(code) || code === 0x5f || code === 0x24;
}

/** JS identifier in ASCII: [A-Za-z_$] then [A-Za-z0-9_$]*. */
export function isIdentifier(text: string): boolean {
  if (text.length === 0 || !isIdentifierStart(text.charCodeAt(0))) return false;
  for (let i = 1; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    if (!(isIdentifierStart(code) || isDigitCode(code))) return false;
  }
  return true;
}

/** Keeps [A-Za-z0-9-] and, when allowUnderscore, '_'. */
export function keepIdChars(text: string, allowUnderscore: boolean): string {
  let out = '';
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    if (isAsciiLetterCode(code) || isDigitCode(code) || code === 0x2d || (allowUnderscore && code === 0x5f)) {
      out += text[i];
    }
  }
  return out;
}

/** "⌘ K" -> ['⌘','K'];  "Shift+D" -> ['Shift','D'];  "Shift++" -> ['Shift','+'].  Splits on whitespace runs and on a '+' that is not the last character. */
export function splitShortcut(shortcut: string): string[] {
  const parts: string[] = [];
  let current = '';
  for (let i = 0; i < shortcut.length; i += 1) {
    const ch = shortcut[i];
    if (isWhitespaceCode(shortcut.charCodeAt(i)) || (ch === '+' && i < shortcut.length - 1)) {
      if (current.length > 0) parts.push(current);
      current = '';
      continue;
    }
    current += ch;
  }
  if (current.length > 0) parts.push(current);
  return parts;
}
