import { describe, expect, it, vi } from 'vite-plus/test';

// The real tool is never run here: `execFile` is replaced, and the test checks only how it is called.
const calls = vi.hoisted(() => ({
  seen: [] as Array<{ file: string; args: string[]; options: Record<string, unknown> }>,
  error: null as Error | null,
}));

vi.mock('node:child_process', () => ({
  execFile: (
    file: string,
    args: string[],
    options: Record<string, unknown>,
    done: (error: Error | null, stdout: string, stderr: string) => void,
  ) => {
    calls.seen.push({ file, args, options });
    queueMicrotask(() => done(calls.error, '', ''));
  },
}));

const { moveToSystemTrash } = await import('../src/trash.js');

describe('moveToSystemTrash', () => {
  it('runs /usr/bin/trash -s with the path as one argument, no shell, and a 60 s timeout', async () => {
    calls.seen = [];
    calls.error = null;
    await moveToSystemTrash('/Volumes/Disk/models/a folder; with $(words)');
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
    calls.error = Object.assign(new Error('Command failed: /usr/bin/trash'), { code: 5 });
    await expect(moveToSystemTrash('/tmp/x')).rejects.toThrow('Command failed');
  });
});
