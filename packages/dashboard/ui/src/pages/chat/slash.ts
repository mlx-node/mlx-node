/**
 * The page's side of slash commands: the rows of the `/` list for the mode, and what a typed draft is. App commands
 * run here through `historyActions` and are never sent; agent commands (Work) are sent as typed and the host checks
 * them again. Only the app commands whose op exists are offered.
 */

import type { CommandRow } from '../../components/arc-x/command-menu/command-menu';
import {
  ALLOWED_BUILTINS,
  APP_COMMANDS,
  appCommandOf,
  isSlashText,
  parseSlash,
  refusedName,
  slashToken,
  type AppCommand,
} from '../../lib/chat-input';
import { CHAT } from '../../lib/chat-strings';
import type { ChatMode, CommandView } from '../../lib/chat-types';
import { matchCommands } from '../../lib/command-match';

/** The app commands the page runs, in list order, with their words. */
const APP_ROWS: ReadonlyArray<{ op: AppCommand; about: string; workOnly?: true }> = [
  { op: 'compact', about: CHAT.slash.about.compact },
  { op: 'context', about: CHAT.slash.about.context },
  { op: 'fork', about: CHAT.slash.about.fork },
  { op: 'plan', about: CHAT.slash.about.plan, workOnly: true },
  { op: 'recap', about: CHAT.slash.about.recap },
  { op: 'rewind', about: CHAT.slash.about.rewind },
];

/** Our words for the built-ins Work sends. */
const BUILTIN_WORDS: Readonly<Record<string, { about: string; words: string; tip?: string }>> = {
  goal: { about: CHAT.slash.about.goal, words: CHAT.slash.hint.goal, tip: CHAT.goal.commandTip },
  loop: { about: CHAT.slash.about.loop, words: CHAT.slash.hint.loop },
  'deep-research': { about: CHAT.slash.about.deepResearch, words: CHAT.slash.hint.deepResearch },
  workflow: { about: CHAT.slash.about.workflow, words: CHAT.slash.hint.workflow },
};

function builtinRow(name: string): CommandRow | null {
  const words = BUILTIN_WORDS[name];
  if (words === undefined) return null;
  return {
    name,
    group: 'builtin',
    about: words.about,
    words: words.words,
    ...(words.tip === undefined ? {} : { tip: words.tip }),
  };
}

/**
 * The `/` list: the app commands, then in Work the agent's rows the host lists (its built-ins in our words, skills and
 * workflows in the agent's). Before the conversation's session is loaded Work shows its four built-ins and says more
 * come after the first message.
 */
export function slashRows(mode: ChatMode, list: readonly CommandView[] | null): { rows: CommandRow[]; later: boolean } {
  const rows: CommandRow[] = APP_ROWS.filter((row) => mode === 'work' || row.workOnly !== true).map((row) => ({
    name: row.op,
    group: 'builtin',
    about: row.about,
    words: null,
    aliases: APP_COMMANDS[row.op],
  }));
  if (mode === 'chat') return { rows, later: false };
  const agent =
    list === null
      ? ALLOWED_BUILTINS.work.map(builtinRow)
      : list.map((view) =>
          view.kind === 'builtin'
            ? builtinRow(view.name)
            : { name: view.name, group: view.kind, about: view.description, words: view.hint },
        );
  for (const row of agent) if (row !== null) rows.push(row);
  return { rows, later: list === null };
}

/** What the draft is: not a command, a command this page cannot send, an app command, or an agent command. */
export type SlashDraft =
  | { kind: 'none' }
  | { kind: 'unknown' }
  | { kind: 'app'; op: AppCommand; args: string }
  | { kind: 'agent'; name: string; args: string; words: string | null };

/**
 * The draft read as the host reads it: refused names, app commands this page does not run, every agent command in
 * Chat and one not in the list (a built-in too) are unknown. While the Work list is not known any other name may go:
 * the host checks it once the session is loaded.
 */
export function slashDraft(draft: string, mode: ChatMode, list: readonly CommandView[] | null): SlashDraft {
  if (!isSlashText(draft)) return { kind: 'none' };
  const slash = parseSlash(draft);
  if (slash === null) return { kind: 'unknown' };
  const op = appCommandOf(slash.name);
  if (op !== null)
    return APP_ROWS.some((row) => row.op === op && (mode === 'work' || row.workOnly !== true))
      ? { kind: 'app', op, args: slash.args }
      : { kind: 'unknown' };
  if (mode === 'chat' || refusedName(slash.name)) return { kind: 'unknown' };
  const builtin = ALLOWED_BUILTINS.work.includes(slash.name) ? builtinRow(slash.name) : null;
  if (builtin !== null && (list === null || list.some((view) => view.kind === 'builtin' && view.name === slash.name)))
    return { kind: 'agent', name: slash.name, args: slash.args, words: builtin.words };
  if (builtin !== null) return { kind: 'unknown' };
  if (list === null) return { kind: 'agent', name: slash.name, args: slash.args, words: null };
  const row = list.find((view) => view.kind !== 'builtin' && view.name === slash.name);
  return row === undefined ? { kind: 'unknown' } : { kind: 'agent', name: row.name, args: slash.args, words: row.hint };
}

/** The reason slot's hint for a command with no words typed yet: its words, else "Enter to run". */
export function slashHint(slash: SlashDraft): string | undefined {
  if (slash.kind === 'app') return slash.args === '' ? CHAT.slash.run : undefined;
  if (slash.kind === 'agent') return slash.args === '' ? (slash.words ?? CHAT.slash.run) : undefined;
  return undefined;
}

/** The `/` list has nothing to offer for the draft: no row matches its first word, or words follow that word. */
export function slashListClosed(draft: string, rows: readonly CommandRow[]): boolean {
  const token = slashToken(draft);
  if (token === null || draft.slice(token.end).trim() !== '') return true;
  return matchCommands(rows, token.typed).length === 0;
}
