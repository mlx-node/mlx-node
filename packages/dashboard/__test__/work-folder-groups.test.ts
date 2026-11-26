import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import {
  folderHints,
  groupByFolder,
  groupLive,
  matchesQuery,
  NO_FOLDER_KEY,
  placeName,
  shownRows,
  spokenFolder,
} from '../ui/src/lib/conversations.js';
import {
  defaultPick,
  firstModeDraft,
  type ModeDraft,
  type NewPlace,
  newPlace,
  nextModeDraft,
  placeBlock,
  recentFolders,
} from '../ui/src/pages/chat/drafts.js';
import { nextFolded, readFolded, revealed, toReveal, writeFolded } from '../ui/src/pages/chat/work/folds.js';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

const DAY = new Date(2026, 10, 26, 0, 0, 0).getTime();
const at = (hour: number, dayOffset = 0): number => DAY + dayOffset * 86_400_000 + hour * 3_600_000;

type Row = { id: string; updatedAt: number; folder?: string | null; noFolder?: boolean };
const row = (id: string, folder: string | null, updatedAt: number, noFolder?: boolean): Row => ({
  id,
  updatedAt,
  folder,
  ...(noFolder === undefined ? {} : { noFolder }),
});

const shape = (rows: readonly Row[]) =>
  groupByFolder(rows).map((group) => [group.key, group.name, group.items.map((item) => item.id)]);

describe('groupByFolder', () => {
  // Mutations: group by folder name instead of path; keep rows in input order.
  it('one group per folder, rows newest first', () => {
    const rows = [
      row('n1', '/w/napi-rs', at(1)),
      row('o1', '/w/oxc-project', at(10)),
      row('n2', '/w/napi-rs', at(13)),
      row('t1', '/w/tar', at(9)),
      row('n3', '/w/napi-rs', at(5)),
      row('f1', '/w/woff-build', at(8)),
      row('n4', '/w/napi-rs', at(3)),
    ];
    expect(shape(rows)).toEqual([
      ['/w/napi-rs', 'napi-rs', ['n2', 'n3', 'n4', 'n1']],
      ['/w/oxc-project', 'oxc-project', ['o1']],
      ['/w/tar', 'tar', ['t1']],
      ['/w/woff-build', 'woff-build', ['f1']],
    ]);
  });

  // Mutation: key = folder name.
  it('two folders with one name are two groups', () => {
    const groups = groupByFolder([row('a', '/x/2026-10-29/task', at(2)), row('b', '/x/2026-10-30/task', at(1))]);
    expect(groups.map((group) => group.key)).toEqual(['/x/2026-10-29/task', '/x/2026-10-30/task']);
    expect(groups.map((group) => group.hint)).toEqual(['2026-10-29', '2026-10-30']);
  });

  // Mutation: a case fold of the key.
  it('/a/Repo and /a/repo are two groups', () => {
    expect(groupByFolder([row('a', '/a/Repo', at(2)), row('b', '/a/repo', at(1))])).toHaveLength(2);
  });

  // Mutation: plain newest-first group order.
  it('a group last used yesterday sorts under one used today; one day sorts by name, not by time', () => {
    const rows = [row('y', '/w/aaa', at(23, -1)), row('late', '/w/zed', at(20)), row('early', '/w/bar', at(1))];
    expect(shape(rows).map((group) => group[1])).toEqual(['bar', 'zed', 'aaa']);
  });

  // Mutations: a text compare without `numeric`; an unstable tie.
  it('v2 sorts before v10; Tar and tar tie-break by path', () => {
    expect(shape([row('a', '/w/v10', at(2)), row('b', '/w/v2', at(1))]).map((group) => group[1])).toEqual([
      'v2',
      'v10',
    ]);
    const tie = [row('a', '/w/tar', at(2)), row('b', '/w/Tar', at(1))];
    expect(shape(tie).map((group) => group[0])).toEqual(['/w/Tar', '/w/tar']);
    expect(shape([...tie].reverse()).map((group) => group[0])).toEqual(['/w/Tar', '/w/tar']);
  });

  // Mutation: No folder pinned first, or pinned last.
  it('No folder is last among the groups of its day and above an older day', () => {
    const rows = [row('none', null, at(12)), row('today', '/w/zzz', at(2)), row('old', '/w/aaa', at(12, -2))];
    expect(shape(rows).map((group) => group[0])).toEqual(['/w/zzz', NO_FOLDER_KEY, '/w/aaa']);
  });

  // Mutation: a group named after the dated folder of a No folder row.
  it('noFolder, a null and an empty folder all land in the No folder group', () => {
    const groups = groupByFolder([
      row('flag', '/Users/me/Portato/2026-11-26-2057', at(3), true),
      row('null', null, at(2)),
      row('empty', '', at(1)),
    ]);
    expect(groups.map((group) => [group.key, group.name, group.hint, group.items.map((item) => item.id)])).toEqual([
      [NO_FOLDER_KEY, null, null, ['flag', 'null', 'empty']],
    ]);
  });

  // Mutation: an order tied to the clock.
  it('the order is the same at any time of day', () => {
    const rows = [row('a', '/w/b', at(23)), row('b', '/w/a', at(0, -1)), row('c', '/w/c', at(1))];
    vi.useFakeTimers();
    vi.setSystemTime(at(23, 0));
    const before = shape(rows);
    vi.setSystemTime(at(1, 3));
    expect(shape(rows)).toEqual(before);
  });
});

describe('folderHints', () => {
  // Mutations: a hint on every group; a hint that does not tell them apart.
  it('only shared names get a hint: the fewest parent parts that tell them apart', () => {
    expect(folderHints(['/w/napi-rs', '/w/tar']).size).toBe(0);
    expect(Object.fromEntries(folderHints(['/d/2026-10-29/task', '/d/2026-10-30/task', '/w/tar']))).toEqual({
      '/d/2026-10-29/task': '2026-10-29',
      '/d/2026-10-30/task': '2026-10-30',
    });
    expect(Object.fromEntries(folderHints(['/x/a/task', '/y/a/task', '/z/b/task']))).toEqual({
      '/x/a/task': 'x/a',
      '/y/a/task': 'y/a',
      '/z/b/task': 'z/b',
    });
  });
});

describe('spokenFolder', () => {
  // Mutation: the hint left out → two folders named task give two "+" buttons with one name.
  it('a shared name is read with its hint, a lone one alone', () => {
    const groups = groupByFolder([row('a', '/d/2026-10-29/task', at(1)), row('b', '/d/2026-10-30/task', at(2))]);
    const spoken = groups.map((group) => spokenFolder({ name: group.name!, hint: group.hint }));
    expect(spoken).toEqual(['task, 2026-10-29', 'task, 2026-10-30']);
    expect(fill(CHAT.list.newIn, { folder: spoken[0]! })).toBe('New conversation in task, 2026-10-29');
    expect(spokenFolder({ name: 'tar', hint: null })).toBe('tar');
  });
});

type Live = { id: string; state: 'idle' | 'running' | 'waiting' };
const live = (n: number, states: Record<number, Live['state']> = {}): Live[] =>
  Array.from({ length: n }, (_, index) => ({ id: `r${index}`, state: states[index] ?? 'idle' }));
const ids = (rows: readonly Live[]) => rows.map((entry) => entry.id);

describe('shownRows', () => {
  // Mutations: "Show 1 more"; a hidden live row; lifted rows moved to the top.
  it('a short group shows whole; a long one its newest 5, its live rows and the open one, in time order', () => {
    expect(shownRows(live(6), false, null)).toEqual({ shown: live(6), hidden: 0 });
    const seven = shownRows(live(7), false, null);
    expect([ids(seven.shown), seven.hidden]).toEqual([['r0', 'r1', 'r2', 'r3', 'r4'], 2]);
    const lifted = shownRows(live(10, { 6: 'running', 8: 'waiting' }), false, 'r9');
    expect([ids(lifted.shown), lifted.hidden]).toEqual([['r0', 'r1', 'r2', 'r3', 'r4', 'r6', 'r8', 'r9'], 2]);
    expect(shownRows(live(10), true, null)).toEqual({ shown: live(10), hidden: 0 });
  });
});

describe('groupLive', () => {
  // Mutation: running wins over waiting.
  it('waiting wins over running; idle rows give none', () => {
    expect(groupLive([{ state: 'running' }, { state: 'waiting' }])).toBe('waiting');
    expect(groupLive([{ state: 'idle' }, { state: 'running' }])).toBe('running');
    expect(groupLive([{ state: 'idle' }])).toBeNull();
  });
});

describe('matchesQuery', () => {
  const napi = { mode: 'work' as const, updatedAt: 0, folder: '/Users/me/workspace/github/napi-rs' };
  const hit = (entry: Parameters<typeof placeName>[0] & { title: string }, query: string): boolean =>
    matchesQuery({ title: entry.title, folderName: placeName(entry) }, query);

  // Mutation: the whole path still matched.
  it('the folder name matches, the folders above it do not; "no folder" matches a No folder row', () => {
    expect(hit({ ...napi, title: 'Fix the build' }, 'workspace')).toBe(false);
    expect(hit({ ...napi, title: 'Fix the build' }, 'NAPI')).toBe(true);
    expect(hit({ ...napi, title: 'Fix the build' }, 'build')).toBe(true);
    expect(hit({ mode: 'work', updatedAt: 0, folder: null, title: 'Plan' }, 'no folder')).toBe(true);
    expect(hit({ mode: 'work', updatedAt: 0, folder: '/p/x', noFolder: true, title: 'Plan' }, 'No folder')).toBe(true);
    expect(hit({ mode: 'chat', updatedAt: 0, folder: null, title: 'Plan' }, CHAT.folderNone)).toBe(false);
  });
});

describe('folds', () => {
  // Mutations: folds that never clear; a toggle that wipes other folds; keys of gone folders kept.
  it('a fold adds its key, an open removes it, keys not in the list are dropped', () => {
    const keys = new Set(['/a', '/b', '/c']);
    expect([...nextFolded(new Set(['/a']), '/b', false, keys)].sort()).toEqual(['/a', '/b']);
    expect([...nextFolded(new Set(['/a', '/b']), '/a', true, keys)]).toEqual(['/b']);
    expect([...nextFolded(new Set(['/gone', '/c']), '/a', false, keys)].sort()).toEqual(['/a', '/c']);
  });

  // Mutation: a failed or odd read that throws, or folds every group.
  it('the store reads back what was written; a broken or failing read means all open', () => {
    const store = new Map<string, string>();
    vi.stubGlobal('localStorage', {
      getItem: (key: string) => store.get(key) ?? null,
      setItem: (key: string, value: string) => store.set(key, value),
    });
    expect(readFolded().size).toBe(0);
    writeFolded(new Set(['/a', NO_FOLDER_KEY]));
    expect([...readFolded()].sort()).toEqual([NO_FOLDER_KEY, '/a']);
    store.set('mlx-node:work-folded:v1', '{"not":"a list"}');
    expect(readFolded().size).toBe(0);
    vi.stubGlobal('localStorage', {
      getItem: () => {
        throw new Error('blocked');
      },
      setItem: () => {
        throw new Error('blocked');
      },
    });
    expect(readFolded().size).toBe(0);
    expect(() => writeFolded(new Set(['/a']))).not.toThrow();
  });

  // Mutation: the window's first look opens the group of the conversation that is open then.
  it('the first look reveals nothing; a new open id is revealed until it is marked done', () => {
    expect(toReveal('a')).toBeNull();
    expect(toReveal('a')).toBeNull();
    expect(toReveal('b')).toBe('b');
    expect(toReveal('b')).toBe('b');
    revealed();
    expect(toReveal('b')).toBeNull();
  });
});

describe('the place of a new Work conversation', () => {
  const work = (cwd: string | null, updatedAt: number, noFolder?: true) => ({
    cwd,
    updatedAt,
    ...(noFolder === undefined ? {} : { noFolder }),
  });

  // Mutations: No folder never chosen; the dated folder of a No folder conversation offered as a pick.
  it('the newest row gives its folder, or none when it has no folder; no rows give none', () => {
    expect(defaultPick([work('/p/a', 1), work('/p/b', 3), work('/p/c', 2)])).toEqual({ kind: 'folder', path: '/p/b' });
    expect(defaultPick([work('/p/a', 1), work('/h/Portato/2026-11-26-2057', 2, true)])).toEqual({ kind: 'none' });
    expect(defaultPick([])).toEqual({ kind: 'none' });
  });

  // Mutation: the folders the app made offered as recent folders.
  it('recent folders leave out No folder conversations', () => {
    const rows = [work('/h/Portato/2026-11-26-2057', 3, true), work('/p/a', 2), work('/p/a', 1), work('/p/b', 0)];
    expect(recentFolders(rows)).toEqual(['/p/a', '/p/b']);
  });
  // Mutation: the default read from an empty list while it loads → No folder, and a send makes a ~/Portato folder.
  // Mutation: a failed list read as loading → Send waits for a list that is not coming.
  // Mutation: the window's pick read after the list → a pick made while the list loads is lost.
  it('no place while the Work list loads or after it failed; the window pick wins at any time', () => {
    const rows = [work('/p/a', 1), work('/p/b', 3)];
    const loading = newPlace(null, { rows: undefined, failed: false });
    const failed = newPlace(null, { rows: undefined, failed: true });
    expect([loading, failed]).toEqual([{ kind: 'loading' }, { kind: 'unset' }]);
    expect([placeBlock(loading), placeBlock(failed)]).toEqual([CHAT.folderLoading, CHAT.folderUnsetBlock]);
    expect(newPlace(null, { rows, failed: false })).toEqual({ kind: 'folder', path: '/p/b' });
    // A list that failed on a refresh keeps the rows it had.
    expect(newPlace(null, { rows, failed: true })).toEqual({ kind: 'folder', path: '/p/b' });
    expect(newPlace(null, { rows: [], failed: false })).toEqual({ kind: 'none' });
    for (const list of [
      { rows: undefined, failed: false },
      { rows: undefined, failed: true },
      { rows, failed: false },
    ]) {
      expect(newPlace({ kind: 'none' }, list)).toEqual({ kind: 'none' });
      expect(newPlace({ kind: 'folder', path: '/p/a' }, list)).toEqual({ kind: 'folder', path: '/p/a' });
    }
    expect(placeBlock({ kind: 'none' })).toBeUndefined();
    expect(placeBlock({ kind: 'folder', path: '/p/a' })).toBeUndefined();
  });

  // Mutation: a draft made while the list loads gets a new id when the list comes in → a mode picked then is lost.
  // Mutation: no new id when a known folder changes → a mode raised for one folder reaches another.
  it('the mode draft keeps its id when the list comes in; a new visit, mode or folder makes a new one', () => {
    let n = 0;
    const id = () => `d${++n}`;
    const at = (place: NewPlace, visit = 'new:1') => ({ place, visit, mode: 'work' as const });
    const start: ModeDraft = firstModeDraft(at({ kind: 'loading' }), id());
    expect(start).toEqual({ id: 'd1', folder: undefined, visit: 'new:1', mode: 'work' });
    expect(nextModeDraft(start, at({ kind: 'loading' }), id)).toBe(start);
    const placed = nextModeDraft(start, at({ kind: 'folder', path: '/p/a' }), id);
    expect(placed).toEqual({ id: 'd1', folder: '/p/a', visit: 'new:1', mode: 'work' });
    expect(nextModeDraft(placed, at({ kind: 'folder', path: '/p/a' }), id)).toBe(placed);
    // A reconnect empties the list for a moment: the draft keeps its folder.
    expect(nextModeDraft(placed, at({ kind: 'loading' }), id)).toBe(placed);
    expect(nextModeDraft(placed, at({ kind: 'none' }), id)).toEqual({
      id: 'd2',
      folder: null,
      visit: 'new:1',
      mode: 'work',
    });
    expect(nextModeDraft(placed, at({ kind: 'folder', path: '/p/a' }, 'new:2'), id).id).toBe('d3');
    expect(nextModeDraft(placed, { ...at({ kind: 'folder', path: '/p/a' }), mode: 'chat' }, id).id).toBe('d4');
  });
});
