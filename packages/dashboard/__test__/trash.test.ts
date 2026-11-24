import { describe, expect, it, vi } from 'vite-plus/test';

// The real tool is never run here: `execFile` is replaced, and the test checks only how it is called.
const calls = vi.hoisted(() => ({
  seen: [] as Array<{ file: string; args: string[]; options: Record<string, unknown> }>,
  error: null as Error | null,
  errors: [] as Array<Error | null>,
}));

vi.mock('node:child_process', () => ({
  execFile: (
    file: string,
    args: string[],
    options: Record<string, unknown>,
    done: (error: Error | null, stdout: string, stderr: string) => void,
  ) => {
    calls.seen.push({ file, args, options });
    const nextError = calls.errors.length > 0 ? (calls.errors.shift() ?? null) : calls.error;
    queueMicrotask(() => done(nextError, '', ''));
  },
}));

const { moveToSystemTrash } = await import('../src/trash.js');

describe('moveToSystemTrash', () => {
  it('runs /usr/bin/trash -s with the path as one argument, no shell, and a 60 s timeout', async () => {
    calls.seen = [];
    calls.error = null;
    calls.errors = [];
    await moveToSystemTrash('/Volumes/Disk/models/a folder; with $(words)', 'darwin');
    expect(calls.seen).toEqual([
      {
        file: '/usr/bin/trash',
        args: ['-s', '/Volumes/Disk/models/a folder; with $(words)'],
        options: { timeout: 60_000 },
      },
    ]);
  });

  it('rejects when the tool fails', async () => {
    calls.seen = [];
    calls.errors = [];
    calls.error = Object.assign(new Error('Command failed: /usr/bin/trash'), { code: 5 });
    await expect(moveToSystemTrash('/tmp/x', 'darwin')).rejects.toThrow('Command failed');
  });

  it('rejects paths containing NUL bytes before spawning anything', async () => {
    calls.seen = [];
    calls.error = null;
    calls.errors = [];
    await expect(moveToSystemTrash('/tmp/bad\0path', 'darwin')).rejects.toThrow('NUL');
    await expect(moveToSystemTrash('C:\\bad\0path', 'win32')).rejects.toThrow('NUL');
    expect(calls.seen).toEqual([]);
  });

  it('uses PowerShell -STA + Win32 IFileOperation recycle-only sink on Windows, passing the path only via env', async () => {
    calls.seen = [];
    calls.error = null;
    calls.errors = [];
    const hostile = "C:\\models\\a'b; $(calc.exe) [1]";
    await moveToSystemTrash(hostile, 'win32');
    expect(calls.seen).toHaveLength(1);
    const call = calls.seen[0]!;
    expect(call.file.toLowerCase().endsWith('\\system32\\windowspowershell\\v1.0\\powershell.exe')).toBe(true);
    expect(call.args.slice(0, 4)).toEqual(['-STA', '-NoProfile', '-NonInteractive', '-Command']);
    const script = call.args[4]!;
    expect(call.args.join('\n')).not.toContain(hostile);
    expect(script).toContain('$env:MLX_TRASH_PATH');
    expect(script).toContain('IFileOperation');
    expect(script).toContain('IFileOperationProgressSink');
    expect(script).toContain('0x00080000');
    expect(script).toContain('0x00000080');
    expect(script).toContain('0x80004004');
    expect(script).toContain('SawRecycledPostDelete');
    expect(script).toContain('psiNewlyCreated != IntPtr.Zero');
    expect(script).toContain('GetAnyOperationsAborted');
    expect(script).not.toContain('Microsoft.VisualBasic');
    expect(script).not.toContain('DeleteDirectory');
    expect(script).not.toContain('DeleteFile');
    expect(call.options.timeout).toBe(60_000);
    expect(call.options.windowsHide).toBe(true);
    expect((call.options.env as Record<string, string>).MLX_TRASH_PATH).toBe(hostile);
  });

  it('uses gio trash on Linux, falling back to trash-put only on ENOENT', async () => {
    calls.seen = [];
    calls.error = null;
    calls.errors = [];
    await moveToSystemTrash('/home/me/models/-dash', 'linux');
    expect(calls.seen).toEqual([
      {
        file: 'gio',
        args: ['trash', '--', '/home/me/models/-dash'],
        options: { timeout: 60_000 },
      },
    ]);

    calls.seen = [];
    calls.errors = [Object.assign(new Error('spawn gio ENOENT'), { code: 'ENOENT' }), null];
    await moveToSystemTrash('/home/me/models/m', 'linux');
    expect(calls.seen).toEqual([
      { file: 'gio', args: ['trash', '--', '/home/me/models/m'], options: { timeout: 60_000 } },
      { file: 'trash-put', args: ['--', '/home/me/models/m'], options: { timeout: 60_000 } },
    ]);

    // Non-ENOENT errors from gio must NOT fall back to trash-put.
    calls.seen = [];
    calls.errors = [Object.assign(new Error('gio permission denied'), { code: 'EACCES' })];
    await expect(moveToSystemTrash('/home/me/models/m', 'linux')).rejects.toThrow('gio permission denied');
    expect(calls.seen).toHaveLength(1);

    // Both missing produces a clear message.
    calls.seen = [];
    calls.errors = [
      Object.assign(new Error('spawn gio ENOENT'), { code: 'ENOENT' }),
      Object.assign(new Error('spawn trash-put ENOENT'), { code: 'ENOENT' }),
    ];
    await expect(moveToSystemTrash('/home/me/models/m', 'linux')).rejects.toThrow('gio');
    expect(calls.seen).toHaveLength(2);
  });
});
