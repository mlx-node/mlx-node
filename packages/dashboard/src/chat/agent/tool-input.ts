/**
 * A web call's `tool_input` that reached the app as text. The agent turns a clean object text into an object before
 * it asks, so a text here is one it could not read. After an allow, `use_tool` reads the text again with strict JSON
 * (serde_json: the last of two equal keys wins, anything after the value is an error) and sends the object it gets; a
 * direct call, and every text that is not an object, sends no arguments and the tool answers `bad-request`. So the app
 * reads the whole text by the same strict grammar: a text that breaks it can never become an object; a text that keeps
 * it is read as an object only with allowed keys, each once, each value a string or a finite number. Every other text
 * that keeps the grammar could become an object the app does not read the same way.
 */

/** The keys each web tool reads; a text with any other key is not read. */
export const WEB_INPUT_KEYS = {
  mlx_web__web_search: ['query', 'count', 'recency'],
  mlx_web__web_fetch: ['url'],
} as const;

/** The keys a tool name may have in a text: none for a name that is not a web tool. */
export function webInputKeys(toolName: string): readonly string[] {
  if (toolName === 'mlx_web__web_search') return WEB_INPUT_KEYS.mlx_web__web_search;
  if (toolName === 'mlx_web__web_fetch') return WEB_INPUT_KEYS.mlx_web__web_fetch;
  return [];
}

/** A longer text is not read. */
export const TOOL_INPUT_TEXT_MAX = 8192;

export type ToolInputText =
  /** One JSON object with only allowed keys, each once, each value a string or a number. */
  | { kind: 'object'; value: Record<string, unknown> }
  /** No strict JSON reader can make an object of it: the tool receives no arguments. */
  | { kind: 'not-object' }
  /** It could become an object that the app does not read the same way. */
  | { kind: 'unreadable' };

/** Characters read; a cost test reads it. */
export interface ReadSteps {
  count: number;
}

const NOT_OBJECT: ToolInputText = { kind: 'not-object' };
const UNREADABLE: ToolInputText = { kind: 'unreadable' };

const OPEN_BRACE = 0x7b;
const CLOSE_BRACE = 0x7d;
const OPEN_BRACKET = 0x5b;
const CLOSE_BRACKET = 0x5d;
const QUOTE = 0x22;
const BACKSLASH = 0x5c;
const COMMA = 0x2c;
const COLON = 0x3a;
const MINUS = 0x2d;

function isJsonSpace(c: number): boolean {
  return c === 0x20 || c === 0x09 || c === 0x0a || c === 0x0d;
}

function isDigit(c: number): boolean {
  return c >= 0x30 && c <= 0x39;
}

function hexValue(c: number): number {
  if (c >= 0x30 && c <= 0x39) return c - 0x30;
  if (c >= 0x61 && c <= 0x66) return c - 0x57;
  if (c >= 0x41 && c <= 0x46) return c - 0x37;
  return -1;
}

/**
 * A strict JSON reader. A method returns null when the text breaks the grammar; `odd` is set for what keeps the grammar
 * but is not read: a lone surrogate, a number that is not finite, a value that is not a string or a number.
 */
class Reader {
  pos = 0;
  odd = false;
  // Explicit fields, not parameter properties: the db worker runs this file from source under Node's type stripping.
  private readonly text: string;
  private readonly steps: ReadSteps;
  constructor(text: string, steps: ReadSteps) {
    this.text = text;
    this.steps = steps;
  }

  at(): number {
    return this.pos < this.text.length ? this.text.charCodeAt(this.pos) : -1;
  }

  next(): number {
    return this.pos + 1 < this.text.length ? this.text.charCodeAt(this.pos + 1) : -1;
  }

  step(): void {
    this.pos += 1;
    this.steps.count += 1;
  }

  space(): void {
    while (isJsonSpace(this.at())) this.step();
  }

  hex4(): number {
    let value = 0;
    for (let i = 0; i < 4; i += 1) {
      const h = hexValue(this.at());
      if (h < 0) return -1;
      value = value * 16 + h;
      this.step();
    }
    return value;
  }

  /** A string after its opening quote. */
  string(): string | null {
    let out = '';
    for (;;) {
      const c = this.at();
      if (c < 0x20) return null;
      this.step();
      if (c === QUOTE) return out;
      if (c === BACKSLASH) {
        const e = this.at();
        this.step();
        if (e === QUOTE) out += '"';
        else if (e === BACKSLASH) out += '\\';
        else if (e === 0x2f) out += '/';
        else if (e === 0x62) out += '\b';
        else if (e === 0x66) out += '\f';
        else if (e === 0x6e) out += '\n';
        else if (e === 0x72) out += '\r';
        else if (e === 0x74) out += '\t';
        else if (e === 0x75) {
          const unit = this.hex4();
          if (unit < 0) return null;
          if (unit >= 0xdc00 && unit <= 0xdfff) this.odd = true;
          else if (unit >= 0xd800 && unit <= 0xdbff) {
            if (this.at() !== BACKSLASH || this.next() !== 0x75) {
              this.odd = true;
              continue;
            }
            this.step();
            this.step();
            const low = this.hex4();
            if (low < 0) return null;
            if (low < 0xdc00 || low > 0xdfff) this.odd = true;
            else out += String.fromCharCode(unit, low);
          } else out += String.fromCharCode(unit);
        } else return null;
      } else if (c >= 0xd800 && c <= 0xdbff) {
        const low = this.at();
        if (low < 0xdc00 || low > 0xdfff) this.odd = true;
        else {
          this.step();
          out += String.fromCharCode(c, low);
        }
      } else if (c >= 0xdc00 && c <= 0xdfff) {
        this.odd = true;
      } else out += String.fromCharCode(c);
    }
  }

  /** A number; one that is not finite is odd. */
  number(): number | null {
    const start = this.pos;
    if (this.at() === MINUS) this.step();
    if (this.at() === 0x30) this.step();
    else if (isDigit(this.at())) while (isDigit(this.at())) this.step();
    else return null;
    if (this.at() === 0x2e) {
      this.step();
      if (!isDigit(this.at())) return null;
      while (isDigit(this.at())) this.step();
    }
    if (this.at() === 0x65 || this.at() === 0x45) {
      this.step();
      if (this.at() === 0x2b || this.at() === MINUS) this.step();
      if (!isDigit(this.at())) return null;
      while (isDigit(this.at())) this.step();
    }
    const value = Number(this.text.slice(start, this.pos));
    if (!Number.isFinite(value)) this.odd = true;
    return value;
  }

  word(text: string): boolean {
    for (let i = 0; i < text.length; i += 1) {
      if (this.at() !== text.charCodeAt(i)) return false;
      this.step();
    }
    return true;
  }

  /** A key with its colon, inside an object. */
  key(): boolean {
    this.space();
    if (this.at() !== QUOTE) return false;
    this.step();
    if (this.string() === null) return false;
    this.space();
    if (this.at() !== COLON) return false;
    this.step();
    return true;
  }

  /** Any JSON value, with no recursion: the open objects and arrays are a stack of their closing characters. */
  value(): boolean {
    const open: number[] = [];
    for (;;) {
      this.space();
      const c = this.at();
      if (c === QUOTE) {
        this.step();
        if (this.string() === null) return false;
      } else if (c === MINUS || isDigit(c)) {
        if (this.number() === null) return false;
      } else if (c === 0x74) {
        if (!this.word('true')) return false;
      } else if (c === 0x66) {
        if (!this.word('false')) return false;
      } else if (c === 0x6e) {
        if (!this.word('null')) return false;
      } else if (c === OPEN_BRACKET || c === OPEN_BRACE) {
        this.step();
        this.space();
        const close = c === OPEN_BRACKET ? CLOSE_BRACKET : CLOSE_BRACE;
        if (this.at() === close) this.step();
        else {
          open.push(close);
          if (close === CLOSE_BRACE && !this.key()) return false;
          continue;
        }
      } else return false;
      for (;;) {
        const close = open.at(-1);
        if (close === undefined) return true;
        this.space();
        const sep = this.at();
        if (sep === close) {
          this.step();
          open.pop();
          continue;
        }
        if (sep !== COMMA) return false;
        this.step();
        if (close === CLOSE_BRACE && !this.key()) return false;
        break;
      }
    }
  }
}

/** The object a `use_tool` text gives the tool, by the agent's own rules; see the file comment. */
export function readToolInputText(
  text: string,
  keys: readonly string[],
  steps: ReadSteps = { count: 0 },
): ToolInputText {
  let first = 0;
  while (first < text.length && (isJsonSpace(text.charCodeAt(first)) || text.charCodeAt(first) === 0xfeff)) {
    first += 1;
    steps.count += 1;
  }
  let last = text.length - 1;
  while (last > first && isJsonSpace(text.charCodeAt(last))) {
    last -= 1;
    steps.count += 1;
  }
  if (
    first >= text.length ||
    text.charCodeAt(first) !== OPEN_BRACE ||
    last <= first ||
    text.charCodeAt(last) !== CLOSE_BRACE
  ) {
    return NOT_OBJECT;
  }
  if (text.length > TOOL_INPUT_TEXT_MAX) return UNREADABLE;
  const r = new Reader(text, steps);
  r.space();
  if (r.at() !== OPEN_BRACE) return UNREADABLE;
  r.step();
  const value: Record<string, unknown> = Object.create(null) as Record<string, unknown>;
  const seen: string[] = [];
  r.space();
  if (r.at() === CLOSE_BRACE) r.step();
  else {
    for (;;) {
      if (r.at() !== QUOTE) return NOT_OBJECT;
      r.step();
      const key = r.string();
      if (key === null) return NOT_OBJECT;
      const fresh = keys.includes(key) && !seen.includes(key);
      if (fresh) seen.push(key);
      else r.odd = true;
      r.space();
      if (r.at() !== COLON) return NOT_OBJECT;
      r.step();
      r.space();
      const c = r.at();
      let item: string | number | null = null;
      if (c === QUOTE) {
        r.step();
        item = r.string();
        if (item === null) return NOT_OBJECT;
      } else if (c === MINUS || isDigit(c)) {
        item = r.number();
        if (item === null) return NOT_OBJECT;
      } else {
        if (!r.value()) return NOT_OBJECT;
        r.odd = true;
      }
      if (fresh && item !== null) value[key] = item;
      r.space();
      const sep = r.at();
      r.step();
      if (sep === CLOSE_BRACE) break;
      if (sep !== COMMA) return NOT_OBJECT;
      r.space();
    }
  }
  r.space();
  if (r.pos !== text.length) return NOT_OBJECT;
  return r.odd ? UNREADABLE : { kind: 'object', value };
}
