import { describe, expect, it } from 'vite-plus/test';

import {
  diffEdits,
  lineDiff,
  MAX_BODY_CHARS,
  MAX_DISTANCE,
  MAX_ROW_CHARS,
  MAX_ROWS,
  type RawEdit,
  splitLines,
} from '../src/agent-sessions/line-diff.js';

function file(lines: number, change: (i: number) => string | null = () => null): string {
  let out = '';
  for (let i = 1; i <= lines; i += 1) out += `${change(i) ?? `line ${i}`}\n`;
  return out;
}

function edit(oldText: string, newText: string, overrides: Partial<RawEdit> = {}): RawEdit {
  return { path: '/w/a.ts', oldText, newText, details: null, line: [1, 1], create: false, ...overrides };
}

/** Each op as its letter: `=` unchanged, `-` removed, `+` added. */
function ops(a: string, b: string, maxDistance?: number): string {
  return [...lineDiff(splitLines(a), splitLines(b), maxDistance)].map((op) => '=-+'[op]).join('');
}

describe('splitLines', () => {
  it('ends the last line at a last newline and drops a \\r before a newline', () => {
    expect(splitLines('')).toEqual([]);
    expect(splitLines('a\nb')).toEqual(['a', 'b']);
    expect(splitLines('a\nb\n')).toEqual(['a', 'b']);
    expect(splitLines('a\r\nb\r\n')).toEqual(['a', 'b']);
    expect(splitLines('\n\n')).toEqual(['', '']);
    expect(splitLines('a\rb\n')).toEqual(['a\rb']);
  });
});

describe('lineDiff', () => {
  it('finds the shortest edit', () => {
    expect(ops('a\nb\nc\n', 'a\nx\nc\n')).toBe('=-+=');
    expect(ops('a\nb\nc\nd\n', 'b\nc\nd\ne\n')).toBe('-===+');
    expect(ops('', 'a\nb\n')).toBe('++');
    expect(ops('a\nb\n', '')).toBe('--');
  });

  it('past the distance cap removes every old line, then adds every new line, between the common ends', () => {
    const kept = (cap: number) => ops('k\na\nb\nc\nk\n', 'k\nx\ny\nb\nk\n', cap);
    expect(kept(4).split('=')).toHaveLength(4);
    expect(kept(3)).toBe('=---+++=');
    const big = file(MAX_DISTANCE + 10);
    const changed = file(MAX_DISTANCE + 10, (i) => (i % 2 === 0 ? `changed ${i}` : null));
    const rows = lineDiff(splitLines(big), splitLines(changed));
    // Line 1 is the common start; the 1,005 changed lines need an edit distance of 2,010.
    const n = MAX_DISTANCE + 10 - 1;
    expect(rows).toHaveLength(1 + 2 * n);
    expect(rows[0]).toBe(0);
    expect(rows.slice(1, 1 + n).every((op) => op === 1)).toBe(true);
    expect(rows.slice(1 + n).every((op) => op === 2)).toBe(true);
  });
});

describe('diffEdits', () => {
  it('makes no hunk for equal texts', () => {
    const [one] = diffEdits([edit(file(10), file(10))]);
    expect(one).toEqual({ path: '/w/a.ts', added: 0, removed: 0, hunks: [] });
  });

  it('keeps 3 unchanged lines on each side of one changed line', () => {
    const [one] = diffEdits([
      edit(
        file(20),
        file(20, (i) => (i === 10 ? 'ten' : null)),
      ),
    ]);
    expect(one!.added).toBe(1);
    expect(one!.removed).toBe(1);
    expect(one!.hunks).toEqual([
      {
        start: [7, 7],
        body: [' line 7', ' line 8', ' line 9', '-line 10', '+ten', ' line 11', ' line 12', ' line 13'].join('\n'),
      },
    ]);
  });

  it('merges two changes 5 lines apart into one hunk, and keeps 8 apart as two', () => {
    const near = diffEdits([
      edit(
        file(30),
        file(30, (i) => (i === 10 || i === 16 ? 'x' : null)),
      ),
    ])[0]!;
    expect(near.hunks).toHaveLength(1);
    expect(near.hunks[0]!.start).toEqual([7, 7]);
    const far = diffEdits([
      edit(
        file(30),
        file(30, (i) => (i === 10 || i === 19 ? 'x' : null)),
      ),
    ])[0]!;
    const six = diffEdits([
      edit(
        file(30),
        file(30, (i) => (i === 10 || i === 17 ? 'x' : null)),
      ),
    ])[0]!;
    expect(six.hunks).toHaveLength(1);
    const seven = diffEdits([
      edit(
        file(30),
        file(30, (i) => (i === 10 || i === 18 ? 'x' : null)),
      ),
    ])[0]!;
    expect(seven.hunks).toHaveLength(2);
    expect(far.hunks.map((h) => h.start)).toEqual([
      [7, 7],
      [16, 16],
    ]);
  });

  it('numbers a hunk after an insertion on each side', () => {
    const old = file(30);
    const lines = splitLines(old);
    lines.splice(2, 0, 'new A', 'new B');
    lines[24] = 'changed';
    const [one] = diffEdits([edit(old, `${lines.join('\n')}\n`)]);
    expect(one!.hunks.map((h) => h.start)).toEqual([
      [1, 1],
      [20, 22],
    ]);
    expect(one!.hunks[1]!.body.split('\n')[3]).toBe('-line 23');
  });

  it('a change in line endings or the last newline only is endsOnly, with no hunk', () => {
    expect(diffEdits([edit('a\r\nb\r\n', 'a\nb\n')])[0]).toMatchObject({ hunks: [], endsOnly: true });
    expect(diffEdits([edit('a\nb', 'a\nb\n')])[0]).toMatchObject({ hunks: [], endsOnly: true });
    expect(diffEdits([edit('a\nb\n', 'a\nc\n')])[0]).not.toHaveProperty('endsOnly');
  });

  it('a result detail is diffed old string against new string, with its context lines around it', () => {
    const [one] = diffEdits([
      edit('    let a = 1;\n    let b = 2;\n', '    let a = 1;\n    let b = 3;\n', {
        details: [
          {
            oldString: 'let a = 1;\n    let b = 2;\n',
            newString: 'let a = 1;\n    let b = 3;\n',
            line: 12,
            before: 'fn f() {\n',
            after: '}\n',
            prefix: '    ',
          },
        ],
        oldText: 'let a = 1;\n    let b = 2;\n',
        newText: 'let a = 1;\n    let b = 3;\n',
        line: null,
      }),
    ]);
    expect(one!.hunks).toEqual([
      { start: [11, 11], body: ' fn f() {\n     let a = 1;\n-    let b = 2;\n+    let b = 3;\n }' },
    ]);
    expect([one!.added, one!.removed]).toEqual([1, 1]);
  });

  it('numbers each replaced match of replace_all on both sides', () => {
    const detail = (line: number) => ({ oldString: 'a', newString: 'b\nc', line, before: '', after: '', prefix: '' });
    const [one] = diffEdits([
      edit('a', 'b\nc', { details: [detail(4), detail(9)], oldText: 'a', newText: 'b\nc', line: null }),
    ]);
    expect(one!.hunks.map((h) => h.start)).toEqual([
      [4, 4],
      [8, 9],
    ]);
  });

  it('replace_all matches whose context lines meet share one hunk, and no line shows twice', () => {
    const detail = (line: number, before: string, after: string) => ({
      oldString: 'x\n',
      newString: 'X\n',
      line,
      before,
      after,
      prefix: '',
    });
    // New file a b c d X e f X g h i j: each match's context holds the other match's new line.
    const [one] = diffEdits([
      edit('x\n', 'X\n', {
        details: [detail(5, 'b\nc\nd\n', 'e\nf\nX\n'), detail(8, 'X\ne\nf\n', 'g\nh\ni\n')],
        oldText: 'x\n',
        newText: 'X\n',
        line: null,
      }),
    ]);
    expect(one!.hunks).toEqual([
      { start: [2, 2], body: [' b', ' c', ' d', '-x', '+X', ' e', ' f', '-x', '+X', ' g', ' h', ' i'].join('\n') },
    ]);
    expect([one!.added, one!.removed]).toEqual([2, 2]);
  });

  it('a deleted block shows no after context: the agent’s starts one line past the block', () => {
    // Old a b c x y d e f g; the agent's after context skips d, the line that took the block's place.
    const [one] = diffEdits([
      edit('x\ny\n', '', {
        details: [{ oldString: 'x\ny\n', newString: '', line: 4, before: 'a\nb\nc\n', after: 'e\nf\ng\n', prefix: '' }],
        oldText: 'x\ny\n',
        newText: '',
        line: null,
      }),
    ]);
    expect(one!.hunks).toEqual([{ start: [1, 1], body: ' a\n b\n c\n-x\n-y' }]);
  });

  it('a new file has an empty old side: old number 0, every row added', () => {
    const [one] = diffEdits([edit('', 'a\nb\n', { create: true, line: null })]);
    expect(one).toEqual({
      path: '/w/a.ts',
      create: true,
      added: 2,
      removed: 0,
      hunks: [{ start: [0, 1], body: '+a\n+b' }],
    });
  });

  it('an asked-for change with no line named stays one hunk with no numbers', () => {
    const [one] = diffEdits([
      edit(
        file(12),
        file(12, (i) => (i === 1 || i === 12 ? 'x' : null)),
        { line: null },
      ),
    ]);
    expect(one!.hunks).toHaveLength(1);
    expect(one!.hunks[0]!.start).toBeNull();
    expect(one!.hunks[0]!.body.split('\n')).toHaveLength(14);
  });

  it('caps rows across the entry and counts the changed lines it leaves out in cut', () => {
    const old = file(300);
    const changed = file(300, () => 'x');
    const [first, second] = diffEdits([edit(old, changed), edit(old, changed, { path: '/w/b.ts' })]);
    const rows = (e: typeof first) => e!.hunks.reduce((n, h) => n + h.body.split('\n').length, 0);
    expect(rows(first)).toBe(MAX_ROWS);
    expect(first!.cut).toBe(600 - MAX_ROWS);
    expect(second!.hunks).toEqual([]);
    expect(second!.cut).toBe(600);
    expect([second!.added, second!.removed]).toEqual([300, 300]);
  });

  it('caps the body size of the entry and each row at its length', () => {
    const long = 'y'.repeat(1_000);
    const [one] = diffEdits([
      edit(
        '',
        file(200, () => long),
        { line: null },
      ),
    ]);
    const body = one!.hunks.map((h) => h.body).join('\n');
    expect(body.length).toBeLessThanOrEqual(MAX_BODY_CHARS);
    const rows = body.split('\n');
    expect(rows[0]).toBe(`+${'y'.repeat(MAX_ROW_CHARS)}…`);
    expect(one!.cut).toBe(200 - rows.length);
    expect(one!.added).toBe(200);
  });
});
