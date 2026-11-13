/**
 * Line breaks and tabs become spaces; other control, bidi override and isolate characters, direction marks and invisible
 * spaces (zero width space, word joiner, invisible operators, BOM) are dropped. ZWJ and ZWNJ stay: emoji and scripts
 * need them.
 */
function hidden(cp: number): boolean {
  return (
    cp <= 0x1f ||
    (cp >= 0x7f && cp <= 0x9f) ||
    cp === 0x061c ||
    cp === 0x200b ||
    cp === 0x200e ||
    cp === 0x200f ||
    (cp >= 0x2060 && cp <= 0x2064) ||
    cp === 0xfeff ||
    (cp >= 0x202a && cp <= 0x202e) ||
    (cp >= 0x2066 && cp <= 0x2069)
  );
}

/**
 * A title the agent wrote, as one plain line: it is shown in the list, the heading, the window title and the delete
 * sentence, where a bidi override would turn the rest of the text around.
 */
export function agentTitle(raw: string): string {
  let out = '';
  for (const ch of raw) {
    const cp = ch.codePointAt(0)!;
    if (ch === '\n' || ch === '\r' || ch === '\t') out += ' ';
    else if (!hidden(cp)) out += ch;
  }
  return out;
}

/** A prompt title is at most this many characters, `…` included. */
export const PROMPT_TITLE_MAX = 60;
/** A cut at a space is used only when the space is at least this far in. */
const PROMPT_TITLE_WORD_MIN = 30;

function isSpace(ch: string): boolean {
  return ch === ' ' || ch === ' ' || ch === '　' || (ch >= ' ' && ch <= ' ');
}

/**
 * The title of a session the agent did not name: the first non-empty line of the first typed prompt, cleaned, on one
 * line, cut at a word to at most {@link PROMPT_TITLE_MAX} characters.
 */
export function promptTitle(text: string): string {
  let line = '';
  for (const raw of text.split('\n')) {
    if (raw.trim() !== '') {
      line = raw;
      break;
    }
  }
  let out = '';
  let gap = false;
  for (const ch of agentTitle(line)) {
    if (isSpace(ch)) {
      gap = out !== '';
      continue;
    }
    if (gap) out += ' ';
    gap = false;
    out += ch;
  }
  const chars = Array.from(out);
  if (chars.length <= PROMPT_TITLE_MAX) return out;
  const room = PROMPT_TITLE_MAX - 1;
  let cut = room;
  for (let i = room; i >= PROMPT_TITLE_WORD_MIN; i--) {
    if (chars[i] === ' ') {
      cut = i;
      break;
    }
  }
  return `${chars.slice(0, cut).join('').trimEnd()}…`;
}

/** The one title rule: the agent's title when it has one, else the first prompt's title, else ''. */
export function resolveTitle(agent: string, firstPrompt: string | null): string {
  const named = agentTitle(agent).trim();
  if (named !== '') return named;
  return firstPrompt === null ? '' : promptTitle(firstPrompt);
}
