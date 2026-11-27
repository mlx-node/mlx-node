/**
 * @vitest-environment happy-dom
 */

/**
 * The Rewind dialog in each state: first focus on Cancel (never the confirm), what the rewind removes and ends, the
 * Work line about files, Esc and a press outside blocked while it runs, the long phase's choice, and the error.
 */

import { act, createElement, useState, type ReactElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  confirmText,
  defaultChoice,
  REWIND_FILES_SHOWN,
  RewindDialog,
  type RewindDialogState,
} from '../ui/src/components/arc-x/rewind-dialog/rewind-dialog.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import type { ChatMode, RewindEndMark, RewindMode, RewindPlan } from '../ui/src/lib/chat-types.js';
import { toast } from '../ui/src/lib/toast.js';
import type { TranscriptEntry } from '../ui/src/lib/types.js';
import { useRewind } from '../ui/src/pages/chat/use-rewind.js';
import { mount, rerender, unmountAll } from './helpers/mount.js';
import {
  accessibleName,
  allByRole,
  byRole,
  deferred,
  sequence,
  stubApi,
  stubError,
  type StubbedApi,
  waitFor,
} from './render.js';

afterEach(unmountAll);

const H = CHAT.history;

const PLAN: RewindPlan = {
  token: 't',
  removes: { messages: 2, steps: 4 },
  background: { commands: 0, schedules: 0, runs: 0 },
  helpers: false,
  attachments: false,
  files: [],
  filesMore: 0,
};

const handlers = {
  onClose: vi.fn(),
  onConfirm: vi.fn(),
  onKeepWaiting: vi.fn(),
  onRestart: vi.fn(),
  onChoice: vi.fn(),
  onRetry: vi.fn(),
};

afterEach(() => {
  for (const fn of Object.values(handlers)) fn.mockClear();
});

function view(state: RewindDialogState, mode: ChatMode = 'chat', commands = 0, choice: RewindMode = 'conversation') {
  return createElement(RewindDialog, {
    open: true,
    state,
    mode,
    message: 'fix the build\nand run the tests',
    commands,
    choice,
    ...handlers,
  });
}

function dialog(): HTMLElement {
  const found = byRole(document.body, 'dialog', H.rewindTitle);
  if (found === null) throw new Error('no dialog');
  return found;
}

function button(name: string): HTMLButtonElement {
  const found = allByRole(dialog(), 'button').find((b) => accessibleName(b) === name);
  if (found === undefined) throw new Error(`no button ${name}`);
  return found as HTMLButtonElement;
}

const buttons = (): string[] =>
  allByRole(dialog(), 'button')
    .map(accessibleName)
    .filter((name) => name !== 'Close dialog');

function key(name: string): void {
  act(() => {
    document.activeElement!.dispatchEvent(new KeyboardEvent('keydown', { key: name, bubbles: true, cancelable: true }));
  });
}

const text = (): string => dialog().textContent ?? '';

describe('RewindDialog', () => {
  it('loading: the opening line and Cancel, with focus on Cancel', async () => {
    mount(view({ phase: 'loading' }));
    await waitFor(() => document.activeElement === button(CHAT.cancel));
    expect(text()).toContain(H.opening);
    expect(buttons()).toEqual([CHAT.cancel]);
  });

  // Mutation: focus the confirm button first → the Cancel focus check fails.
  it('ready (Chat): the first line of the message, what it removes, the refill and no undo; Cancel has focus', async () => {
    mount(view({ phase: 'ready', plan: PLAN }));
    await waitFor(() => document.activeElement === button(CHAT.cancel));
    const quote = dialog().querySelector('blockquote');
    expect(quote?.textContent).toBe('fix the build');
    const said = document.getElementById(dialog().getAttribute('aria-describedby')!);
    expect(said?.textContent).toBe(fill(H.removes, { n: 4 }));
    expect(text()).toContain(H.refill);
    expect(text()).toContain(H.noUndo);
    expect(text()).not.toContain(H.filesStay);
    expect(text()).not.toContain(H.filesNone);
    expect(buttons()).toEqual([CHAT.cancel, H.confirm]);
    act(() => button(H.confirm).click());
    expect(handlers.onConfirm).toHaveBeenCalledTimes(1);
    key('Escape');
    expect(handlers.onClose).toHaveBeenCalledTimes(1);
  });

  it('ready (Work, no files): the agent changed none, no choices; the commands, schedules and attachments it ends', () => {
    const plan: RewindPlan = {
      ...PLAN,
      removes: { messages: 1, steps: 1 },
      background: { commands: 2, schedules: 1, runs: 0 },
      attachments: true,
    };
    mount(view({ phase: 'ready', plan }, 'work'));
    expect(text()).toContain(H.removesStep);
    expect(text()).toContain(H.filesNone);
    expect(text()).not.toContain(H.filesStay);
    expect(allByRole(dialog(), 'radiogroup')).toEqual([]);
    const warnings = [...dialog().querySelectorAll('li')].map((li) => li.textContent);
    expect(warnings).toEqual([fill(H.background, { n: 2 }), H.schedulesOne, H.attachments]);
    rerender(
      view(
        {
          phase: 'ready',
          plan: { ...plan, removes: { messages: 1, steps: 0 }, background: { commands: 1, schedules: 2, runs: 0 } },
        },
        'work',
      ),
    );
    expect(text()).toContain(H.removesOne);
    expect([...dialog().querySelectorAll('li')].map((li) => li.textContent)).toEqual([
      H.backgroundOne,
      fill(H.schedules, { n: 2 }),
      H.attachments,
    ]);
  });

  // Mutation: let Escape close while it runs → onClose is called.
  it('working: the confirm is busy, Cancel is off, and Esc does nothing; after 20 s it says it is still restoring', async () => {
    mount(view({ phase: 'ready', plan: PLAN }));
    await waitFor(() => document.activeElement === button(CHAT.cancel));
    rerender(view({ phase: 'working', plan: PLAN, slow: false }));
    await waitFor(() => buttons().includes(H.working));
    const busy = button(H.working);
    expect(busy.getAttribute('aria-busy')).toBe('true');
    expect(button(CHAT.cancel).disabled).toBe(true);
    act(() => busy.click());
    expect(handlers.onConfirm).not.toHaveBeenCalled();
    act(() => {
      dialog().dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    });
    expect(handlers.onClose).not.toHaveBeenCalled();
    rerender(view({ phase: 'working', plan: PLAN, slow: true }));
    await waitFor(() => buttons().includes(H.stillRestoring));
  });

  it('long: wait or restart the agent; the restart says how many commands it stops; Esc does nothing', async () => {
    mount(view({ phase: 'long', plan: PLAN }, 'work', 3));
    await waitFor(() => document.activeElement === button(H.keepWaiting));
    expect(text()).toContain(H.slow);
    expect(text()).toContain(fill(H.restartStops, { n: 3 }));
    expect(buttons()).toEqual([H.keepWaiting, H.restartAgent]);
    act(() => button(H.keepWaiting).click());
    act(() => button(H.restartAgent).click());
    expect(handlers.onKeepWaiting).toHaveBeenCalledTimes(1);
    expect(handlers.onRestart).toHaveBeenCalledTimes(1);
    key('Escape');
    expect(handlers.onClose).not.toHaveBeenCalled();
    rerender(view({ phase: 'long', plan: PLAN }, 'work', 1));
    await waitFor(() => text().includes(H.restartStopsOne) && !text().includes(fill(H.restartStops, { n: 3 })));
    rerender(view({ phase: 'long', plan: PLAN }, 'work', 0));
    await waitFor(() => !text().includes(H.restartStopsOne));
  });

  it('error: the reason and Close', async () => {
    mount(view({ phase: 'error', tone: 'warning', text: H.errChanged }));
    await waitFor(() => document.activeElement === button(H.close));
    expect(text()).toContain(H.errChanged);
    act(() => button(H.close).click());
    expect(handlers.onClose).toHaveBeenCalledTimes(1);
  });
});

const FILES: RewindPlan = {
  ...PLAN,
  background: { commands: 1, schedules: 0, runs: 0 },
  files: [
    { path: 'src/upload.ts', state: 'clean' },
    { path: 'src/upload.test.ts', state: 'clean' },
    { path: 'docs/notes.md', state: 'clean' },
  ],
};
const CONFLICTS: RewindPlan = {
  ...FILES,
  files: [
    { path: 'src/upload.test.ts', state: 'modified' },
    { path: 'docs/notes.md', state: 'deleted' },
    { path: 'src/upload.ts', state: 'clean' },
  ],
};

/** The dialog in Work with its choice held as the page holds it. */
function Choosing({ plan, phase = 'ready' }: { plan: RewindPlan; phase?: 'ready' | 'working' }): ReactElement {
  const [choice, setChoice] = useState<RewindMode>(defaultChoice('work', plan));
  return createElement(RewindDialog, {
    open: true,
    state: phase === 'ready' ? { phase, plan } : { phase, plan, slow: false },
    mode: 'work',
    message: 'Add a retry to the upload step',
    commands: 0,
    ...handlers,
    choice,
    onChoice: (mode: RewindMode) => {
      handlers.onChoice(mode);
      setChoice(mode);
    },
  });
}

const radio = (name: string): HTMLElement => {
  const found = allByRole(dialog(), 'radio').find((r) => (r.textContent ?? '').startsWith(name));
  if (found === undefined) throw new Error(`no radio ${name}`);
  return found;
};
const checked = (): string =>
  allByRole(dialog(), 'radio').find((r) => r.getAttribute('aria-checked') === 'true')?.textContent ?? '';
const fileList = (): HTMLElement | null => dialog().querySelector('ul[aria-label]');
/** Each file row as a screen reader reads it: the parts hidden from it left out. */
const rows = (scope: ParentNode = dialog()): string[] =>
  [...scope.querySelectorAll('li[data-file-state]')].map((li) => {
    const copy = li.cloneNode(true) as HTMLElement;
    for (const hidden of copy.querySelectorAll('[aria-hidden="true"]')) hidden.remove();
    return copy.textContent ?? '';
  });
const buttonsAre = (names: string[]): Promise<void> =>
  waitFor(() => JSON.stringify(buttons()) === JSON.stringify(names));
const lead = (): string => document.getElementById(dialog().getAttribute('aria-describedby')!)?.textContent ?? '';

describe('RewindDialog: Work with files (#7 files)', () => {
  // Mutation: open on Cancel when the plan has choices → the focus check fails.
  it('no conflict: three choices, Conversation and files checked and focused, the files and the limits; Enter on a choice does not confirm', async () => {
    mount(createElement(Choosing, { plan: FILES }));
    await waitFor(() => document.activeElement === radio(H.modeAll));
    expect(allByRole(dialog(), 'radiogroup').map(accessibleName)).toEqual([H.what]);
    expect(checked()).toBe(H.modeAll + fill(H.modeAllNote, { n: 3 }));
    expect(radio(H.modeConversation).textContent).toBe(H.modeConversation + H.modeConversationNote);
    expect(radio(H.modeFiles).textContent).toBe(H.modeFiles + H.modeFilesNote);
    expect(dialog().querySelectorAll('[data-conflict-dot]')).toHaveLength(0);
    expect(lead()).toBe(fill(H.removes, { n: 4 }));
    expect(text()).toContain(H.refill);
    expect(text()).toContain(fill(H.files, { n: 3 }));
    expect(accessibleName(fileList()!)).toBe(fill(H.filesLabel, { n: 3 }));
    expect(rows()).toEqual(['src/upload.ts', 'src/upload.test.ts', 'docs/notes.md']);
    expect(text()).toContain(H.limits);
    expect([...dialog().querySelectorAll('li:not([data-file-state])')].map((li) => li.textContent)).toEqual([
      H.backgroundOne,
    ]);
    expect(buttons()).toEqual([CHAT.cancel, H.confirm]);
    key('Enter');
    expect(handlers.onConfirm).not.toHaveBeenCalled();
    expect(handlers.onChoice).not.toHaveBeenCalled();
  });

  // Mutation: keep the file list for Conversation only → the list is still there.
  it('Conversation only folds the files away; Files only says the conversation stays, ends no command and restores', async () => {
    mount(createElement(Choosing, { plan: FILES }));
    await waitFor(() => document.activeElement === radio(H.modeAll));
    act(() => radio(H.modeConversation).click());
    expect(handlers.onChoice).toHaveBeenLastCalledWith('conversation');
    await waitFor(() => fileList() === null);
    expect(text()).toContain(H.filesStay);
    expect(text()).not.toContain(H.limits);
    expect(lead()).toBe(fill(H.removes, { n: 4 }));
    await buttonsAre([CHAT.cancel, H.confirm]);

    act(() => radio(H.modeFiles).click());
    await waitFor(() => fileList() !== null);
    expect(lead()).toBe(fill(H.filesOnlyLead, { n: 3 }));
    expect(text()).not.toContain(H.refill);
    expect(text()).not.toContain(H.filesStay);
    expect(text()).not.toContain(H.backgroundOne);
    expect(text()).toContain(H.limits);
    await buttonsAre([CHAT.cancel, H.confirmFiles]);
    act(() => button(H.confirmFiles).click());
    expect(handlers.onConfirm).toHaveBeenCalledTimes(1);
  });

  // Mutation: open on Conversation and files with conflicts → the checked choice is wrong.
  it('conflicts: opens on Conversation only, the file choices carry a dot; a file choice flags each file and overwrites', async () => {
    mount(createElement(Choosing, { plan: CONFLICTS }));
    await waitFor(() => document.activeElement === radio(H.modeConversation));
    expect(checked()).toBe(H.modeConversation + H.modeConversationNote);
    expect(radio(H.modeAll).textContent).toBe(
      H.modeAll + fill(H.modeAllConflict, { files: fill(H.nFiles, { n: 3 }), c: 2 }),
    );
    expect(dialog().querySelectorAll('[data-conflict-dot]')).toHaveLength(2);
    expect(radio(H.modeConversation).querySelector('[data-conflict-dot]')).toBeNull();
    expect(fileList()).toBeNull();
    expect(dialog().querySelector('[data-rewind-conflicts]')).toBeNull();
    expect(buttons()).toEqual([CHAT.cancel, H.confirm]);

    act(() => radio(H.modeAll).click());
    await waitFor(() => fileList() !== null);
    expect(rows()).toEqual([
      `src/upload.test.ts, ${H.fileModified}`,
      `docs/notes.md, ${H.fileDeleted}`,
      'src/upload.ts',
    ]);
    expect(dialog().querySelector('[data-rewind-conflicts]')?.textContent).toBe(
      fill(H.conflicts, { files: fill(H.nFiles, { n: 2 }) }),
    );
    await buttonsAre([CHAT.cancel, H.confirmOver]);
    act(() => radio(H.modeFiles).click());
    await waitFor(() => buttons().includes(H.confirmOver) && lead() === fill(H.filesOnlyLead, { n: 3 }));
    const made: RewindPlan = { ...FILES, files: [{ path: 'gen.ts', state: 'created' }] };
    expect(confirmText('files', made)).toBe(H.confirmOver);
    expect(confirmText('conversation', made)).toBe(H.confirm);
    expect(confirmText('files', FILES)).toBe(H.confirmFiles);
  });

  it('one file: the singular words; a made-again file says so', async () => {
    const one: RewindPlan = { ...FILES, files: [{ path: 'src/a.ts', state: 'clean' }] };
    mount(createElement(Choosing, { plan: one }));
    await waitFor(() => checked() === H.modeAll + H.modeAllNoteOne);
    act(() => radio(H.modeFiles).click());
    await waitFor(() => lead() === H.filesOnlyLeadOne);
    const created: RewindPlan = { ...FILES, files: [{ path: 'src/a.ts', state: 'created' }] };
    unmountAll();
    mount(createElement(Choosing, { plan: created }));
    act(() => radio(H.modeFiles).click());
    await waitFor(() => rows().length === 1);
    expect(rows()).toEqual([`src/a.ts, ${H.fileCreated}`]);
    expect(dialog().querySelector('[data-rewind-conflicts]')?.textContent).toBe(
      fill(H.conflicts, { files: H.oneFile }),
    );
  });

  // Mutation: show every file in the list → more than 8 rows before the fold.
  it('more than 8 files: 8 rows, the rest behind "N more"; past the host limit the count goes on', async () => {
    const many: RewindPlan = {
      ...FILES,
      files: Array.from({ length: 12 }, (_, i) => ({
        path: `src/f${String(i).padStart(2, '0')}.ts`,
        state: 'clean' as const,
      })),
    };
    mount(createElement(Choosing, { plan: many }));
    await waitFor(() => fileList() !== null);
    expect(rows(fileList()!)).toHaveLength(REWIND_FILES_SHOWN);
    const more = button(fill(H.filesMore, { n: 4 }));
    act(() => more.click());
    await waitFor(() => rows().length === 12);
    expect(rows().at(-1)).toBe('src/f11.ts');

    unmountAll();
    const capped: RewindPlan = { ...many, files: many.files.slice(0, 10), filesMore: 5 };
    mount(createElement(Choosing, { plan: capped }));
    await waitFor(() => fileList() !== null);
    expect(text()).toContain(fill(H.files, { n: 15 }));
    act(() => button(fill(H.filesMore, { n: 7 })).click());
    await waitFor(() => rows().length === 10 && text().includes(fill(H.filesMore, { n: 5 })));
    expect(checked()).toBe(H.modeAll + fill(H.modeAllNote, { n: 15 }));
  });

  it('while it runs the choices are off; loading says it checks the files; a slow plan offers Try again', async () => {
    mount(createElement(Choosing, { plan: FILES, phase: 'working' }));
    await waitFor(() => buttons().includes(H.working));
    expect(allByRole(dialog(), 'radiogroup')[0]!.getAttribute('aria-disabled')).toBe('true');
    act(() => radio(H.modeFiles).click());
    expect(handlers.onChoice).not.toHaveBeenCalled();
    unmountAll();
    mount(view({ phase: 'loading', checking: true }, 'work'));
    await waitFor(() => text().includes(H.checking));
    expect(text()).not.toContain(H.opening);
    unmountAll();
    mount(view({ phase: 'error', tone: 'warning', text: H.errSlow, retry: true }, 'work'));
    await waitFor(() => document.activeElement === button(CHAT.cancel));
    expect(buttons()).toEqual([CHAT.cancel, H.tryAgain]);
    act(() => button(H.tryAgain).click());
    expect(handlers.onRetry).toHaveBeenCalledTimes(1);
  });
});

describe('useRewind: the marks of a slow rewind and an end with no rewind', () => {
  let api: StubbedApi | undefined;
  afterEach(() => {
    api?.();
    api = undefined;
    vi.useRealTimers();
  });

  interface HarnessProps {
    end: RewindEndMark | null;
    entries?: TranscriptEntry[];
    /** Short marks, for the test that runs them on a fake clock; else the app's, which no test here reaches. */
    marks?: { slowMs: number; longMs: number };
  }

  const FIX: TranscriptEntry[] = [{ kind: 'user', ts: 1, text: 'fix the build', promptIndex: 0 }];

  function Harness({ end, entries = FIX, marks }: HarnessProps): ReactElement {
    const rewind = useRewind({
      conversationId: 'c1',
      mode: 'chat',
      hot: false,
      folder: null,
      entries,
      end,
      off: null,
      commands: 2,
      readDraft: () => '',
      putBack: () => {},
      keepText: () => {},
      openConversation: () => {},
      titleOf: () => 'Title',
      composer: () => null,
      focusComposer: () => {},
      announce: () => {},
      ...(marks !== undefined ? { marks } : {}),
    });
    return createElement(
      'div',
      null,
      createElement('button', { type: 'button', onClick: () => rewind.onRewind(0, null) }, 'go'),
      rewind.dialog,
    );
  }

  /** The fake clock `ms` on, React flushed after. */
  async function advance(ms: number): Promise<void> {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ms);
    });
  }

  /** Waits with the clock held: promises and zero-delay timers run, a mark never fires. */
  async function settle(predicate: () => boolean): Promise<void> {
    for (let n = 0; n < 50 && !predicate(); n += 1) await advance(0);
    expect(predicate()).toBe(true);
  }

  // The marks run on a fake clock: on the real one a busy worker can fire both in one tick, and "Still restoring…"
  // is never drawn.
  // Mutation: drop the long timer → the Keep waiting choice never shows; do not arm it again → no way out after it.
  it('working, still restoring after the first mark, the choice after the second; Keep waiting goes back and the choice comes again; a stopped end says so', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const posted: string[] = [];
    api = stubApi(
      { 'POST /chat/conversations/c1/rewind/plan': PLAN, 'POST /chat/conversations/c1/rewind': {} },
      { onCall: (call) => void (call.method === 'POST' && posted.push(call.path)) },
    );
    const marks = { slowMs: 40, longMs: 120 };
    mount(createElement(Harness, { end: null, marks }));
    act(() => document.querySelector<HTMLButtonElement>('button')!.click());
    await settle(() => byRole(document.body, 'dialog', H.rewindTitle) !== null && buttons().includes(H.confirm));
    act(() => button(H.confirm).click());
    await settle(() => buttons().includes(H.working));
    await advance(40);
    await settle(() => buttons().includes(H.stillRestoring));
    await advance(80);
    await settle(() => buttons().includes(H.keepWaiting));
    expect(text()).toContain(fill(H.restartStops, { n: 2 }));
    act(() => button(H.keepWaiting).click());
    await settle(() => buttons().includes(H.stillRestoring));
    await advance(120);
    await settle(() => buttons().includes(H.restartAgent));
    act(() => button(H.keepWaiting).click());
    await settle(() => buttons().includes(H.stillRestoring));
    rerender(createElement(Harness, { end: { n: 1, end: { outcome: 'stopped' } }, marks }));
    await settle(() => text().includes(H.errStopped));
    await settle(() => JSON.stringify(buttons()) === JSON.stringify([H.close]));
    expect(posted.filter((p) => p.endsWith('/rewind'))).toHaveLength(1);
  });

  // Mutation: take any new end object (not a higher number) → a resync's copy of the old end closes the new dialog.
  it('an end from before the confirm, or a resync copy of it, leaves the dialog; a newer one from a sync closes it', async () => {
    api = stubApi({ 'POST /chat/conversations/c1/rewind/plan': PLAN, 'POST /chat/conversations/c1/rewind': {} });
    const old: RewindEndMark = { n: 1, end: { outcome: 'stopped' } };
    mount(createElement(Harness, { end: old }));
    act(() => document.querySelector<HTMLButtonElement>('button')!.click());
    await waitFor(() => byRole(document.body, 'dialog', H.rewindTitle) !== null && buttons().includes(H.confirm));
    act(() => button(H.confirm).click());
    await waitFor(() => buttons().includes(H.working));
    rerender(createElement(Harness, { end: { n: 1, end: { outcome: 'stopped' } } }));
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(text()).not.toContain(H.errStopped);
    expect(byRole(document.body, 'dialog', H.rewindTitle)).not.toBeNull();
    rerender(
      createElement(Harness, {
        end: { n: 2, end: { outcome: 'done', mode: 'conversation', restored: 0, entry: 0 } },
      }),
    );
    await waitFor(() => byRole(document.body, 'dialog', H.rewindTitle) === null);
  });

  // Mutation: accept the plan by conversation and index alone → the dialog quotes the old message with Rewind on.
  it('a sync that puts another message at the place: the plan names the time it was opened on, the dialog says it changed', async () => {
    const bodies: unknown[] = [];
    const held = deferred(PLAN);
    api = stubApi(
      {
        'POST /chat/conversations/c1/rewind/plan': held.body,
        'POST /chat/conversations/c1/rewind': {},
      },
      { onCall: (call) => void (call.path.endsWith('/plan') && bodies.push(call.body)) },
    );
    mount(createElement(Harness, { end: null }));
    act(() => document.querySelector<HTMLButtonElement>('button')!.click());
    await waitFor(() => bodies.length === 1);
    expect(bodies).toEqual([{ entry: 0, ts: 1 }]);
    rerender(
      createElement(Harness, {
        end: null,
        entries: [{ kind: 'user', ts: 7, text: 'drop the cache', promptIndex: 0 }],
      }),
    );
    act(() => held.release());
    await waitFor(() => text().includes(H.errChanged));
    await buttonsAre([H.close]);
    expect(dialog().querySelector('blockquote')).toBeNull();
  });

  interface WorkProps {
    end: RewindEndMark | null;
    draft: { value: string; id?: string };
    said: string[];
  }

  function WorkHarness({ end, draft, said }: WorkProps): ReactElement {
    const rewind = useRewind({
      conversationId: 'c1',
      mode: 'work',
      hot: true,
      folder: 'app',
      entries: [{ kind: 'user', ts: 1, text: 'Add a retry', promptIndex: 0 }],
      end,
      off: null,
      commands: 0,
      readDraft: () => draft.value,
      putBack: (id, text) => {
        draft.value = text;
        draft.id = id;
      },
      keepText: () => {},
      openConversation: () => {},
      titleOf: () => 'Title',
      composer: () => null,
      focusComposer: () => {},
      announce: (text) => said.push(text),
    });
    return createElement(
      'div',
      null,
      createElement('button', { type: 'button', onClick: () => rewind.onRewind(0, null) }, 'go'),
      rewind.dialog,
    );
  }

  // Mutation: post the plan's default instead of the picked choice → the body says `all`.
  it('Work: a slow plan, Try again, Files only sent as `files`; its end keeps the box and says the files came back', async () => {
    const bodies: unknown[] = [];
    api = stubApi(
      {
        'POST /chat/conversations/c1/rewind/plan': sequence(stubError('E_UNAVAILABLE', 'slow', 'agent-slow'), FILES),
        'POST /chat/conversations/c1/rewind': {},
      },
      { onCall: (call) => void (call.path.endsWith('/rewind') && bodies.push(call.body)) },
    );
    const success = vi.spyOn(toast, 'success');
    const draft = { value: '' };
    const said: string[] = [];
    mount(createElement(WorkHarness, { end: null, draft, said }));
    act(() => document.querySelector<HTMLButtonElement>('button')!.click());
    await waitFor(() => byRole(document.body, 'dialog', H.rewindTitle) !== null && buttons().includes(H.tryAgain));
    expect(text()).toContain(H.errSlow);
    act(() => button(H.tryAgain).click());
    await waitFor(() => allByRole(dialog(), 'radio').length === 3);
    expect(checked()).toBe(H.modeAll + fill(H.modeAllNote, { n: 3 }));
    act(() => radio(H.modeFiles).click());
    await waitFor(() => buttons().includes(H.confirmFiles));
    act(() => button(H.confirmFiles).click());
    await waitFor(() => bodies.length === 1);
    expect(bodies).toEqual([{ token: 't', mode: 'files' }]);
    rerender(
      createElement(WorkHarness, {
        end: { n: 1, end: { outcome: 'done', mode: 'files', restored: 3, entry: 0 } },
        draft,
        said,
      }),
    );
    await waitFor(() => byRole(document.body, 'dialog', H.rewindTitle) === null);
    expect(draft.value).toBe('');
    expect(success).toHaveBeenLastCalledWith(fill(H.doneFilesOnly, { files: fill(H.nFiles, { n: 3 }) }));
    expect(said).toEqual([fill(H.statusFilesOnly, { files: fill(H.nFiles, { n: 3 }) })]);
    success.mockRestore();
  });

  // Mutation: focus the checked choice once per plan token → a second Rewind on the same message leaves it on Cancel.
  it('Work: Rewind opened again on the same message focuses the checked choice again', async () => {
    api = stubApi({ 'POST /chat/conversations/c1/rewind/plan': FILES, 'POST /chat/conversations/c1/rewind': {} });
    mount(createElement(WorkHarness, { end: null, draft: { value: '' }, said: [] }));
    for (let round = 0; round < 2; round++) {
      act(() => document.querySelector<HTMLButtonElement>('button')!.click());
      await waitFor(
        () => byRole(document.body, 'dialog', H.rewindTitle) !== null && allByRole(dialog(), 'radio').length === 3,
      );
      await waitFor(() => document.activeElement === radio(H.modeAll));
      act(() => button(CHAT.cancel).click());
      await waitFor(() => byRole(document.body, 'dialog', H.rewindTitle) === null);
    }
  });

  it('Work: Conversation and files refills the box and counts the files; an agent that stops names the folder', async () => {
    api = stubApi({ 'POST /chat/conversations/c1/rewind/plan': FILES, 'POST /chat/conversations/c1/rewind': {} });
    const success = vi.spyOn(toast, 'success');
    const draft = { value: '' };
    const said: string[] = [];
    mount(createElement(WorkHarness, { end: null, draft, said }));
    act(() => document.querySelector<HTMLButtonElement>('button')!.click());
    await waitFor(() => byRole(document.body, 'dialog', H.rewindTitle) !== null && buttons().includes(H.confirm));
    act(() => button(H.confirm).click());
    await waitFor(() => buttons().includes(H.working));
    rerender(createElement(WorkHarness, { end: { n: 1, end: { outcome: 'stopped' } }, draft, said }));
    await waitFor(() => text().includes(fill(H.errStoppedFiles, { folder: 'app' })));
    unmountAll();
    mount(createElement(WorkHarness, { end: null, draft, said }));
    act(() => document.querySelector<HTMLButtonElement>('button')!.click());
    await waitFor(() => byRole(document.body, 'dialog', H.rewindTitle) !== null && buttons().includes(H.confirm));
    act(() => button(H.confirm).click());
    await waitFor(() => buttons().includes(H.working));
    rerender(
      createElement(WorkHarness, {
        end: { n: 1, end: { outcome: 'done', mode: 'all', restored: 3, entry: 0 } },
        draft,
        said,
      }),
    );
    await waitFor(() => byRole(document.body, 'dialog', H.rewindTitle) === null);
    expect(draft.value).toBe('Add a retry');
    expect(success).toHaveBeenLastCalledWith(fill(H.doneFiles, { files: fill(H.nFiles, { n: 3 }) }));
    expect(said).toEqual([fill(H.statusFiles, { files: fill(H.nFiles, { n: 3 }) })]);
    success.mockRestore();
  });

  // Mutation: show a `changed` end as an error → the user never sees the new conflicts before overwriting.
  it('Work: a file changed after the check: the dialog plans again, says so, and opens on Conversation only', async () => {
    const plans: unknown[] = [];
    api = stubApi(
      {
        'POST /chat/conversations/c1/rewind/plan': sequence(FILES, { ...CONFLICTS, token: 't2' }),
        'POST /chat/conversations/c1/rewind': {},
      },
      { onCall: (call) => void (call.method === 'POST' && plans.push(call.path)) },
    );
    const draft = { value: '' };
    mount(createElement(WorkHarness, { end: null, draft, said: [] }));
    act(() => document.querySelector<HTMLButtonElement>('button')!.click());
    await waitFor(() => byRole(document.body, 'dialog', H.rewindTitle) !== null && buttons().includes(H.confirm));
    expect(text()).not.toContain(H.planChanged);
    act(() => button(H.confirm).click());
    await waitFor(() => buttons().includes(H.working));
    rerender(createElement(WorkHarness, { end: { n: 1, end: { outcome: 'changed' } }, draft, said: [] }));
    await waitFor(() => text().includes(H.planChanged));
    expect(checked()).toBe(H.modeConversation + H.modeConversationNote);
    expect(plans.filter((p) => String(p).endsWith('/plan'))).toHaveLength(2);
    expect(draft.value).toBe('');
  });

  // Mutation: treat a `partial` end as `done` → the dialog closes with a success toast and names no file.
  it('Work: files the restore could not put back: the dialog stays with the error and the files; the message goes back', async () => {
    api = stubApi({ 'POST /chat/conversations/c1/rewind/plan': FILES, 'POST /chat/conversations/c1/rewind': {} });
    const success = vi.spyOn(toast, 'success');
    const draft: { value: string; id?: string } = { value: '' };
    const said: string[] = [];
    mount(createElement(WorkHarness, { end: null, draft, said }));
    act(() => document.querySelector<HTMLButtonElement>('button')!.click());
    await waitFor(() => byRole(document.body, 'dialog', H.rewindTitle) !== null && buttons().includes(H.confirm));
    act(() => button(H.confirm).click());
    await waitFor(() => buttons().includes(H.working));
    const end: RewindEndMark = {
      n: 1,
      end: {
        outcome: 'partial',
        mode: 'all',
        restored: 1,
        entry: 0,
        failed: ['src/upload.ts', 'docs/notes.md'],
        failedMore: 0,
      },
    };
    rerender(createElement(WorkHarness, { end, draft, said }));
    const words = fill(H.errPartial, { files: fill(H.nFiles, { n: 2 }) });
    await waitFor(() => text().includes(words));
    const list = dialog().querySelector(`ul[aria-label="${fill(H.failedLabel, { n: 2 })}"]`);
    expect(
      [...(list?.querySelectorAll('li') ?? [])].map((li) => {
        const copy = li.cloneNode(true) as HTMLElement;
        for (const hidden of copy.querySelectorAll('[aria-hidden="true"]')) hidden.remove();
        return copy.textContent;
      }),
    ).toEqual(['src/upload.ts', 'docs/notes.md']);
    await buttonsAre([H.close]);
    expect(success).not.toHaveBeenCalled();
    expect(draft).toEqual({ value: 'Add a retry', id: 'c1' });
    expect(said).toEqual([words]);
    success.mockRestore();
  });
});
