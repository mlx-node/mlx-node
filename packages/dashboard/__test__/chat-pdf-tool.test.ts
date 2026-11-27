/**
 * The PDF tool's two files in the agent home: the saved workflow with its page root written in as a Rhai string, the
 * reader agent with `read_file` alone, the marker that makes a file the app's, a file of the person's left alone (the
 * tool is then off), and the command a host start sends.
 */

import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { workflowCommand } from '../src/agent-sessions/transcript.js';
import {
  installPdfTool,
  PDF_READER_MARKER,
  PDF_READER_TEXT,
  pdfToolBlocked,
  pdfToolShadowed,
  READ_PDF_MARKER,
  READ_PDF_QUESTION_MAX,
  readPdfCommand,
  readPdfScript,
  rhaiString,
} from '../src/chat/agent/pdf-tool.js';
import { checkCommand, parseSlash } from '../src/chat/commands.js';

let home: string;
const ROOT = '/Users/me/Library/Application Support/mlx/chat/work-pdf';

beforeEach(() => {
  home = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-pdf-tool-')));
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

const script = () => join(home, 'workflows', 'read-pdf.rhai');
const agent = () => join(home, 'agents', 'pdf-reader.md');

/** The value of a Rhai `"…"` literal at the start of `text`, read back as Rhai reads `\\` and `\"`. */
function readRhaiString(text: string): string {
  if (!text.startsWith('"')) throw new Error('not a string');
  let out = '';
  for (let i = 1; i < text.length; i++) {
    const ch = text.charAt(i);
    if (ch === '"') return out;
    if (ch === '\\') {
      i += 1;
      out += text.charAt(i);
    } else out += ch;
  }
  throw new Error('open string');
}

/** The value of `const ROOT` in a script. */
function rootOf(text: string): string {
  const head = 'const ROOT = ';
  const line = text.split('\n').find((l) => l.startsWith(head));
  if (line === undefined) throw new Error('no ROOT');
  return readRhaiString(line.slice(head.length));
}

describe('the script', () => {
  // Mutation: the path joined raw into the script → red.
  it('holds its page root as a Rhai string: a quote and a backslash are escaped, a control character refused', () => {
    expect(rhaiString('a"b\\c')).toBe('"a\\"b\\\\c"');
    for (const root of [ROOT, '/tmp/odd"dir\\x/work-pdf', '/tmp/naïve 日本/work-pdf']) {
      expect(rootOf(readPdfScript(root))).toBe(root);
    }
    expect(() => rhaiString('/tmp/a\nb')).toThrow();
    expect(() => rhaiString('/tmp/a\u007fb')).toThrow();
  });

  it('starts with its marker, and the listing text stays under the agent cap with the args shape in it', () => {
    const text = readPdfScript(ROOT);
    expect(text.split('\n')[0]).toBe(READ_PDF_MARKER);
    const field = (key: string) => {
      const line = text.split('\n').find((l) => l.trimStart().startsWith(`${key}: `))!;
      return readRhaiString(line.trimStart().slice(key.length + 2));
    };
    expect(field('name')).toBe('read-pdf');
    const listed = Buffer.byteLength(field('description')) + Buffer.byteLength(field('when_to_use'));
    expect(listed).toBeLessThanOrEqual(400);
    expect(field('when_to_use')).toContain('{"pdf": "<id>", "question": "<what to find out>"');
    for (const line of ['const PER = 4;', 'const MAX_PAGES = 320;', 'const RETRIES = 16;', 'const ANSWER = 3500;']) {
      expect(text.split('\n')).toContain(line);
    }
  });

  it('the reader agent has read_file alone, and its marker inside its frontmatter', () => {
    const lines = PDF_READER_TEXT.split('\n');
    expect(lines.slice(0, 8)).toEqual([
      '---',
      PDF_READER_MARKER,
      'name: pdf-reader',
      'description: Reads PDF page images for the read-pdf workflow.',
      'tools: read_file',
      'disallowedTools: search_tool, use_tool',
      'injectDefaultTools: false',
      '---',
    ]);
  });
});

describe('the files in the agent home', () => {
  it('are written when missing, 0600 in folders it makes 0700, and left as they are when the same', async () => {
    expect(await pdfToolBlocked(home)).toBe(false);
    expect(await installPdfTool(home, ROOT)).toBe(true);
    expect(readFileSync(script(), 'utf8')).toBe(readPdfScript(ROOT));
    expect(readFileSync(agent(), 'utf8')).toBe(PDF_READER_TEXT);
    expect(statSync(script()).mode & 0o777).toBe(0o600);
    expect(statSync(join(home, 'agents')).mode & 0o777).toBe(0o700);
    const before = statSync(script()).ino;
    expect(await installPdfTool(home, ROOT)).toBe(true);
    expect(statSync(script()).ino).toBe(before);
    expect(await pdfToolBlocked(home)).toBe(false);
  });

  it("an app file that differs (another root, an older text) is written again; the person's folder modes stay", async () => {
    mkdirSync(join(home, 'workflows'), { mode: 0o755 });
    writeFileSync(script(), `${READ_PDF_MARKER}\nold text\n`);
    expect(await installPdfTool(home, '/elsewhere/work-pdf')).toBe(true);
    expect(rootOf(readFileSync(script(), 'utf8'))).toBe('/elsewhere/work-pdf');
    expect(await installPdfTool(home, ROOT)).toBe(true);
    expect(rootOf(readFileSync(script(), 'utf8'))).toBe(ROOT);
    expect(statSync(join(home, 'workflows')).mode & 0o777).toBe(0o755);
  });

  // Mutation: a person's file replaced → red.
  it("a file of either name without the marker is the person's: nothing is written and the tool is off", async () => {
    mkdirSync(join(home, 'agents'));
    const theirs = '---\nname: pdf-reader\ndescription: mine\n---\nMy own reader.\n';
    writeFileSync(agent(), theirs);
    expect(await pdfToolBlocked(home)).toBe(true);
    expect(await installPdfTool(home, ROOT)).toBe(false);
    expect(readFileSync(agent(), 'utf8')).toBe(theirs);
    expect(existsSync(script())).toBe(false);
    rmSync(agent());
    mkdirSync(join(home, 'workflows'));
    writeFileSync(script(), `// my read-pdf\n${READ_PDF_MARKER}\n`);
    expect(await installPdfTool(home, ROOT)).toBe(false);
    expect(readFileSync(script(), 'utf8')).toBe(`// my read-pdf\n${READ_PDF_MARKER}\n`);
    expect(existsSync(agent())).toBe(false);
  });

  it('a link or a folder of that name is not the app file either', async () => {
    mkdirSync(join(home, 'workflows'));
    const target = join(home, 'target.rhai');
    writeFileSync(target, readPdfScript(ROOT));
    symlinkSync(target, script());
    expect(await installPdfTool(home, ROOT)).toBe(false);
    rmSync(script());
    mkdirSync(script());
    expect(await pdfToolBlocked(home)).toBe(true);
  });
});

// Mutations: the project workflows not read; the walk stopped at the Work folder; the walk past the git root; the app's
// own agent home taken for a project folder; a folder that cannot be read taken as none; a home that is a repo taken
// as the project → red.
describe('a file of the Work folder that takes a PDF tool name first', () => {
  it('is any project workflow script or a pdf-reader agent from the Work folder up to the git root', async () => {
    const user = join(home, 'u');
    const agentHome = join(user, '.mlx-agent');
    const repo = join(user, 'repo');
    const cwd = join(repo, 'sub', 'work');
    mkdirSync(cwd, { recursive: true });
    mkdirSync(join(repo, '.git'));
    const put = (path: string): string => {
      mkdirSync(join(path, '..'), { recursive: true });
      writeFileSync(path, 'x');
      return path;
    };
    const shadowed = (at = cwd) => pdfToolShadowed(at, agentHome, user);
    put(join(agentHome, 'workflows', 'read-pdf.rhai'));
    put(join(agentHome, 'agents', 'pdf-reader.md'));
    expect(await shadowed()).toBe(false);

    const found = [
      join(repo, '.mlx-agent', 'workflows', 'any-name.rhai'),
      join(repo, 'sub', '.claude', 'agents', 'pdf-reader.md'),
      join(cwd, '.mlx-agent', 'agents', 'pdf-reader.md'),
      join(repo, '.claude', 'agents', 'pdf-reader.md'),
    ];
    for (const path of found) {
      put(path);
      expect([path, await shadowed()]).toEqual([path, true]);
      rmSync(path);
    }
    put(join(repo, '.mlx-agent', 'workflows', 'notes.txt'));
    put(join(user, '.claude', 'agents', 'pdf-reader.md'));
    expect(await shadowed()).toBe(false);
    rmSync(join(user, '.claude'), { recursive: true });
    expect(await shadowed(user)).toBe(false);

    const locked = join(repo, '.mlx-agent', 'workflows');
    chmodSync(locked, 0o000);
    try {
      expect(await shadowed()).toBe(true);
    } finally {
      chmodSync(locked, 0o700);
    }

    rmSync(join(repo, '.git'), { recursive: true });
    rmSync(join(repo, '.mlx-agent'), { recursive: true });
    mkdirSync(join(user, '.git'));
    put(join(repo, 'sub', '.claude', 'agents', 'pdf-reader.md'));
    expect(await shadowed()).toBe(false);
  });
});

describe('the host start', () => {
  it("is the agent's own /workflow command with the args as JSON, which Work sends as typed", () => {
    const text = readPdfCommand('k3f9a2c1-212', 'What does "p. 4" say?\nAnd the table?', 'report.pdf');
    expect(text.startsWith('/workflow read-pdf {')).toBe(true);
    const slash = parseSlash(text)!;
    expect(slash.name).toBe('workflow');
    expect(JSON.parse(slash.args.slice('read-pdf '.length))).toEqual({
      pdf: 'k3f9a2c1-212',
      question: 'What does "p. 4" say?\nAnd the table?',
      objective: 'Read report.pdf',
    });
    expect(text.includes('\n')).toBe(false);
    expect(workflowCommand(text)).toBeNull();
    expect(
      checkCommand('work', text, [
        { name: 'workflow', kind: 'builtin', description: '', hint: null, qualifiedName: null },
      ]),
    ).toBe('allowed');
  });

  it('cuts a long question, never inside a surrogate pair', () => {
    const long = `${'q'.repeat(READ_PDF_QUESTION_MAX - 1)}😀 and more`;
    const args = JSON.parse(parseSlash(readPdfCommand('k3f9a2c1-212', long, 'a.pdf'))!.args.slice(9)) as {
      question: string;
    };
    expect(args.question).toBe('q'.repeat(READ_PDF_QUESTION_MAX - 1));
  });
});
