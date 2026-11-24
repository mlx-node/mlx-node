import { describe, expect, it } from 'vite-plus/test';

import { readPermissionRequest } from '../src/chat/acp/messages.js';
import { autoScan } from '../src/chat/agent/auto-scan.js';
import { autoFacts, changesFolder, decideAuto, protectedLanding } from '../src/chat/agent/auto.js';
import { agentPath, commandFlags, pathFlags, type WorkPlace } from '../src/chat/agent/command-flags.js';
import { workInteraction } from '../src/chat/agent/permissions.js';
import { workGuardHookName } from '../src/chat/agent/profiles.js';
import { below, isWinPath, rootOf, rootText } from '../src/chat/agent/roots.js';

const WIN_PLACE: WorkPlace = {
  cwd: 'C:\\Users\\me\\code\\app',
  home: 'C:\\Users\\me',
};

const POSIX_PLACE: WorkPlace = {
  cwd: '/Users/me/code/app',
  home: '/Users/me',
};

describe('Windows paths in Chat/Work roots, flags, permissions, and auto', () => {
  it('detects Windows drive and UNC paths without misclassifying POSIX backslash filenames', () => {
    expect(isWinPath('C:\\Users\\me\\app')).toBe(true);
    expect(isWinPath('C:/Users/me/app')).toBe(true);
    expect(isWinPath('\\\\server\\share\\app')).toBe(true);
    expect(isWinPath('/Users/me/code/app')).toBe(false);
    expect(isWinPath('/Users/me/code/app/src\\n')).toBe(false);
  });

  it('below, rootText, and rootOf handle Windows backslash and forward-slash paths', () => {
    expect(below('C:\\Users\\me\\code\\app', 'C:\\Users\\me\\code\\app')).toBe(true);
    expect(below('C:\\Users\\me\\code\\app\\src\\a.ts', 'C:\\Users\\me\\code\\app')).toBe(true);
    expect(below('C:/Users/me/code/app/src/a.ts', 'C:\\Users\\me\\code\\app')).toBe(true);
    expect(below('C:\\Users\\me\\code\\app-other\\a.ts', 'C:\\Users\\me\\code\\app')).toBe(false);
    expect(below('C:\\Users\\me', 'C:\\')).toBe(true);

    expect(rootText('C:\\Users\\me\\code\\app\\src\\a.ts', WIN_PLACE)?.root.kind).toBe('work');
    expect(rootOf('C:/Users/me/code/app/src/a.ts', WIN_PLACE)?.kind).toBe('work');
    expect(rootOf('D:\\other\\a.ts', WIN_PLACE)).toBeNull();
  });

  it('agentPath and pathFlags resolve Windows drive, UNC, relative, and ~\\ paths', () => {
    expect(agentPath('src\\a.ts', WIN_PLACE)).toBe('C:\\Users\\me\\code\\app\\src\\a.ts');
    expect(agentPath('~\\Documents\\a.txt', WIN_PLACE)).toBe('C:\\Users\\me\\Documents\\a.txt');
    expect(agentPath('~/.gitconfig', WIN_PLACE)).toBe('C:\\Users\\me\\.gitconfig');
    expect(agentPath('~', WIN_PLACE)).toBe('C:\\Users\\me');
    expect(agentPath('C:\\Windows\\win.ini', WIN_PLACE)).toBe('C:\\Windows\\win.ini');
    expect(agentPath('\\\\server\\share\\a.txt', WIN_PLACE)).toBe('\\\\server\\share\\a.txt');

    expect(pathFlags('src\\a.ts', WIN_PLACE)).toEqual([]);
    expect(pathFlags('src/a.ts', WIN_PLACE)).toEqual([]);
    expect(pathFlags('C:/Users/me/code/app/src/a.ts', WIN_PLACE)).toEqual([]);
    expect(pathFlags('..\\other\\a.ts', WIN_PLACE)).toEqual(['outside-folder']);
    expect(pathFlags('~\\.gitconfig', WIN_PLACE)).toEqual(['outside-folder']);
    expect(pathFlags('C:\\Windows\\win.ini', WIN_PLACE)).toEqual(['outside-folder']);

    // POSIX backslash filenames remain literal.
    expect(agentPath('src\\n', POSIX_PLACE)).toBe('/Users/me/code/app/src\\n');
  });

  it('commandFlags detects writes inside vs outside a Windows work folder', () => {
    expect(commandFlags('echo hi > C:/Users/me/code/app/notes.txt', WIN_PLACE)).toEqual([]);
    expect(commandFlags("echo hi > 'C:\\Users\\me\\code\\app\\notes.txt'", WIN_PLACE)).toEqual([]);
    expect(commandFlags('echo hi > C:/Windows/win.ini', WIN_PLACE)).toEqual(['outside-folder']);
    expect(commandFlags("echo hi > 'C:\\Users\\me\\Desktop\\out.txt'", WIN_PLACE)).toEqual(['outside-folder']);
    expect(commandFlags("echo hi > '~\\.gitconfig'", WIN_PLACE)).toEqual(['outside-folder']);
    expect(commandFlags('touch ../other/a.txt', WIN_PLACE)).toEqual(['outside-folder']);
    expect(commandFlags('echo hi > $HOME/.profile', WIN_PLACE)).toEqual(['outside-folder']);
  });

  it('workInteraction compares Windows edit landings with win32.normalize', () => {
    const req = readPermissionRequest({
      sessionId: 's',
      toolCall: {
        toolCallId: 't1',
        kind: 'edit',
        title: 'Write',
        rawInput: { variant: 'Write', file_path: 'src/a.ts' },
        _meta: { 'x.ai/tool': { name: 'write', read_only: false } },
      },
      options: [
        { optionId: 'allow-once', name: 'Allow', kind: 'allow_once' },
        { optionId: 'reject-once', name: 'Reject', kind: 'reject_once' },
      ],
    })!;

    const same = workInteraction('i1', req, [], WIN_PLACE, 'C:\\Users\\me\\code\\app\\src\\a.ts');
    expect(same.flags).toEqual([]);
    expect(same.resolvedPath).toBeNull();

    const redirected = workInteraction('i2', req, [], WIN_PLACE, 'C:\\Users\\me\\outside\\a.ts');
    expect(redirected.flags).toEqual(['outside-folder']);
    expect(redirected.resolvedPath).toBe('C:\\Users\\me\\outside\\a.ts');
  });

  it('protectedLanding and autoFacts split Windows paths on both separators', async () => {
    expect(protectedLanding('C:\\Users\\me\\code\\app\\.npmrc', WIN_PLACE)).toBe(true);
    expect(protectedLanding('C:\\Users\\me\\code\\app\\src\\.git\\config', WIN_PLACE)).toBe(true);
    expect(protectedLanding('C:/Users/me/code/app/.claude/settings.json', WIN_PLACE)).toBe(true);
    expect(protectedLanding('C:\\Users\\me\\code\\app\\src\\index.ts', WIN_PLACE)).toBe(false);

    const req = readPermissionRequest({
      sessionId: 's',
      toolCall: {
        toolCallId: 't2',
        kind: 'edit',
        title: 'Write',
        rawInput: { variant: 'Write', file_path: 'C:\\Temp\\sub\\..\\a.txt' },
        _meta: { 'x.ai/tool': { name: 'write', read_only: false } },
      },
      options: [{ optionId: 'allow-once', name: 'Allow', kind: 'allow_once' }],
    })!;
    const placeWithTemp: WorkPlace = {
      ...WIN_PLACE,
      temp: [{ top: 'C:\\Temp', spellings: ['C:\\Temp'], kind: 'temp' }],
    };
    const facts = await autoFacts(req, placeWithTemp, 'C:\\Temp\\a.txt', { cdpath: false });
    expect(facts.temp).toEqual({ path: 'C:\\Temp\\sub\\..\\a.txt', cause: 'place' });
  });

  it('fails closed for Windows shell commands in autoScan, autoFacts, decideAuto, and changesFolder', async () => {
    for (const command of ['mv "*.txt" dst', 'ls']) {
      const scan = autoScan(command, WIN_PLACE, { cdpath: false });
      expect(scan.unread).toBe(true);
      expect(scan.checks).toBeNull();
      expect(scan.targets).toBeNull();
    }

    const guardHook = workGuardHookName('mlx-work-guard-0123456789abcdef');
    for (const command of ['ls', 'mv "*.txt" dst']) {
      const req = {
        ...readPermissionRequest({
          sessionId: 's',
          toolCall: {
            toolCallId: 't3',
            kind: 'execute',
            title: 'Run shell',
            rawInput: { variant: 'RunTerminalCommand', command },
            _meta: { 'x.ai/tool': { name: 'run_terminal_command', read_only: false } },
          },
          options: [
            { optionId: 'allow-once', name: 'Allow', kind: 'allow_once' },
            { optionId: 'reject-once', name: 'Reject', kind: 'reject_once' },
          ],
        })!,
        hookAsk: { hookName: guardHook, reason: null },
      };
      const interaction = workInteraction('i3', req, [], WIN_PLACE, null);
      expect(interaction.access).toBe('bash');
      expect(interaction.command).toBe(command);

      const facts = await autoFacts(req, WIN_PLACE, null, { cdpath: false });
      expect(facts.scan?.unread).toBe(true);
      expect(facts.scan?.checks).toBeNull();
      expect(facts.scan?.targets).toBeNull();
      expect(
        decideAuto({
          auto: true,
          watched: true,
          request: req,
          interaction,
          guardHook,
          elsewhere: false,
          facts,
          allowed: 0,
        }),
      ).toEqual({ allow: false, why: 'unread' });
      expect(changesFolder(facts, 'bash')).toBe(true);
    }
  });

  it('resolves drive-less rooted paths against the Work drive and forces manual approval for drive-relative edits', async () => {
    const workOnD: WorkPlace = {
      cwd: 'D:\\work\\app',
      home: 'C:\\Users\\me',
    };
    expect(agentPath('\\Windows\\win.ini', workOnD)).toBe('D:\\Windows\\win.ini');
    expect(pathFlags('\\Windows\\win.ini', workOnD)).toEqual(['outside-folder']);
    expect(agentPath('/work/app/src/a.ts', workOnD)).toBe('D:/work/app/src/a.ts');
    expect(pathFlags('/work/app/src/a.ts', workOnD)).toEqual([]);

    // Drive-relative paths (`C:secret.txt`) must NEVER be concatenated under `D:\work\app`.
    expect(agentPath('C:secret.txt', workOnD)).toBe('C:secret.txt');
    expect(pathFlags('C:secret.txt', workOnD)).toEqual(['outside-folder']);

    const req = readPermissionRequest({
      sessionId: 's',
      toolCall: {
        toolCallId: 't4',
        kind: 'edit',
        title: 'Write',
        rawInput: { variant: 'Write', file_path: 'C:secret.txt' },
        _meta: { 'x.ai/tool': { name: 'write', read_only: false } },
      },
      options: [
        { optionId: 'allow-once', name: 'Allow', kind: 'allow_once' },
        { optionId: 'reject-once', name: 'Reject', kind: 'reject_once' },
      ],
    })!;
    const interaction = workInteraction('i4', req, [], workOnD, 'D:\\work\\app\\C:secret.txt');
    const facts = await autoFacts(req, workOnD, 'D:\\work\\app\\C:secret.txt', { cdpath: false });
    expect(facts.landing).toBeNull();
    expect(facts.landingOutside).toBe(true);
    expect(
      decideAuto({
        auto: true,
        watched: true,
        request: req,
        interaction,
        guardHook: null,
        elsewhere: false,
        facts,
        allowed: 0,
      }).allow,
    ).toBe(false);
  });

  it('catches Win32 trailing-dot/space/ADS protected names and refuses ambiguous Windows edits in Auto', async () => {
    expect(protectedLanding('C:\\Users\\me\\code\\app\\.npmrc.', WIN_PLACE)).toBe(true);
    expect(protectedLanding('C:\\Users\\me\\code\\app\\.npmrc ', WIN_PLACE)).toBe(true);
    expect(protectedLanding('C:\\Users\\me\\code\\app\\.git.\\config', WIN_PLACE)).toBe(true);
    expect(protectedLanding('C:\\Users\\me\\code\\app\\.npmrc:stream', WIN_PLACE)).toBe(true);

    for (const filePath of ['.npmrc.', 'notes.txt.', 'notes.txt:ads']) {
      const req = readPermissionRequest({
        sessionId: 's',
        toolCall: {
          toolCallId: `t-ambig-${filePath}`,
          kind: 'edit',
          title: 'Write',
          rawInput: { variant: 'Write', file_path: filePath },
          _meta: { 'x.ai/tool': { name: 'write', read_only: false } },
        },
        options: [
          { optionId: 'allow-once', name: 'Allow', kind: 'allow_once' },
          { optionId: 'reject-once', name: 'Reject', kind: 'reject_once' },
        ],
      })!;
      const landing = `C:\\Users\\me\\code\\app\\${filePath}`;
      const interaction = workInteraction('i-ambig', req, [], WIN_PLACE, landing);
      const facts = await autoFacts(req, WIN_PLACE, landing, { cdpath: false });
      expect(facts.landing).toBeNull();
      expect(
        decideAuto({
          auto: true,
          watched: true,
          request: req,
          interaction,
          guardHook: null,
          elsewhere: false,
          facts,
          allowed: 0,
        }).allow,
      ).toBe(false);
    }
  });
});
