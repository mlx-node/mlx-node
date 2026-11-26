/**
 * The composer's reading of slash text, from the host's own rule (`src/chat/commands.ts`): text whose first non-blank
 * character is `/` is a command, run by the app or sent for the agent to run. The server's check is the law.
 */

import { startsWithSlash } from '../../../src/chat/commands.js';

export {
  ALLOWED_BUILTINS,
  APP_COMMANDS,
  appCommandOf,
  parseSlash,
  refusedName,
  type AppCommand,
} from '../../../src/chat/commands.js';

/** Whether the first character after leading white space is `/`. */
export const isSlashText = startsWithSlash;

/** The first word of a draft that starts with `/`: where it starts and ends, and the name typed after the `/`. */
export interface SlashToken {
  start: number;
  end: number;
  /** What follows the `/`, as typed. */
  typed: string;
}

/** The space characters the agent splits a command's name on (Unicode `White_Space`). */
function isSpace(code: number): boolean {
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
    code === 0x3000
  );
}

/** The draft's first word when it starts with `/` after only white space, else null. */
export function slashToken(draft: string): SlashToken | null {
  let start = 0;
  while (start < draft.length && isSpace(draft.charCodeAt(start))) start += 1;
  if (start >= draft.length || draft.charCodeAt(start) !== 0x2f) return null;
  let end = start + 1;
  while (end < draft.length && !isSpace(draft.charCodeAt(end))) end += 1;
  return { start, end, typed: draft.slice(start + 1, end) };
}

/** The draft after its `/` word and the white space that follows it: what an app command, which takes no words, leaves. */
export function slashRest(draft: string): string {
  const token = slashToken(draft);
  if (token === null) return draft;
  let at = token.end;
  while (at < draft.length && isSpace(draft.charCodeAt(at))) at += 1;
  return draft.slice(at);
}
