import { execFileSync } from 'node:child_process';
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  claudeSpawnSpec,
  findClaudeOnPath,
  resolveClaudeLaunch,
  resolveWindowsClaudeShim,
} from '../src/commands/launch-claude/helpers.js';

let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'mlx-launch-claude-'));
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

function dir(name: string): string {
  const full = join(root, name);
  mkdirSync(full, { recursive: true });
  return full;
}

describe('findClaudeOnPath', () => {
  it('finds an executable regular file on POSIX and skips directories and non-executable files', () => {
    const dirNamed = dir('dir-named');
    mkdirSync(join(dirNamed, 'claude'));
    const plain = dir('plain');
    writeFileSync(join(plain, 'claude'), '#!/bin/sh\n');
    chmodSync(join(plain, 'claude'), 0o644);
    const good = dir('good');
    writeFileSync(join(good, 'claude'), '#!/bin/sh\n');
    chmodSync(join(good, 'claude'), 0o755);

    expect(findClaudeOnPath({ PATH: [dirNamed, plain, good].join(':') }, 'darwin')).toBe(join(good, 'claude'));
    expect(findClaudeOnPath({ PATH: plain }, 'linux')).toBeNull();
    expect(findClaudeOnPath({}, 'darwin')).toBeNull();
  });

  it('on win32 reads Path case-insensitively, splits on ;, honors PATHEXT, and skips directories and shims', () => {
    const shim = dir('shim');
    writeFileSync(join(shim, 'claude'), '#!/bin/sh\n');
    const dirNamed = dir('dir-named');
    mkdirSync(join(dirNamed, 'claude.exe'));
    const npmBin = dir('npm bin');
    writeFileSync(join(npmBin, 'claude.cmd'), '@echo off\r\n');
    const nativeBin = dir('native');
    writeFileSync(join(nativeBin, 'claude.exe'), 'mz');

    const path = [shim, dirNamed, `"${npmBin}"`, nativeBin].join(';');
    expect(findClaudeOnPath({ Path: path }, 'win32')).toBe(join(npmBin, 'claude.cmd'));
    expect(findClaudeOnPath({ Path: path, pathext: '.EXE' }, 'win32')).toBe(join(nativeBin, 'claude.exe'));
    expect(findClaudeOnPath({ Path: shim, PATHEXT: '' }, 'win32')).toBe(join(shim, 'claude'));
  });

  it('on win32 prefers native claude.exe over claude.cmd even when PATHEXT lists .CMD before .EXE', () => {
    const mixed = dir('mixed');
    writeFileSync(join(mixed, 'claude.cmd'), '@echo off\r\n');
    writeFileSync(join(mixed, 'claude.exe'), 'mz');

    expect(findClaudeOnPath({ PATH: mixed, PATHEXT: '.CMD;.EXE' }, 'win32')).toBe(join(mixed, 'claude.exe'));
    expect(findClaudeOnPath({ PATH: mixed, PATHEXT: '.BAT;.CMD;.COM;.EXE' }, 'win32')).toBe(join(mixed, 'claude.exe'));

    // Preserves relative order inside the direct-executable group (.COM before .EXE)
    // and never promotes non-PE scripting extensions (.VBS/.JS) ahead of .CMD.
    writeFileSync(join(mixed, 'claude.com'), 'mz');
    expect(findClaudeOnPath({ PATH: mixed, PATHEXT: '.CMD;.COM;.EXE' }, 'win32')).toBe(join(mixed, 'claude.com'));
    expect(findClaudeOnPath({ PATH: mixed, PATHEXT: '.CMD;.EXE;.COM' }, 'win32')).toBe(join(mixed, 'claude.exe'));

    const scriptDir = dir('script-vs-cmd');
    writeFileSync(join(scriptDir, 'claude.cmd'), '@echo off\r\n');
    writeFileSync(join(scriptDir, 'claude.vbs'), 'WScript.Quit 0\r\n');
    expect(findClaudeOnPath({ PATH: scriptDir, PATHEXT: '.COM;.EXE;.BAT;.CMD;.VBS' }, 'win32')).toBe(
      join(scriptDir, 'claude.cmd'),
    );
  });
});

describe('resolveWindowsClaudeShim and claudeSpawnSpec', () => {
  it('resolves standard npm claude.cmd to node + cli.js (sibling node.exe or fallback execPath) without cmd.exe', () => {
    const npmBin = dir('npm-global');
    const cliDir = dir('npm-global/node_modules/@anthropic-ai/claude-code');
    const cliJs = join(cliDir, 'cli.js');
    writeFileSync(cliJs, 'console.log("ok");\n');

    const shim = join(npmBin, 'claude.cmd');
    writeFileSync(
      shim,
      [
        '@ECHO off',
        'GOTO start',
        ':find_dp0',
        'SET dp0=%~dp0',
        'EXIT /b',
        ':start',
        'SETLOCAL',
        'CALL :find_dp0',
        '',
        'IF EXIST "%dp0%\\node.exe" (',
        '  SET "_prog=%dp0%\\node.exe"',
        ') ELSE (',
        '  SET "_prog=node"',
        '  SET PATHEXT=%PATHEXT:;.JS;=;%',
        ')',
        '',
        'endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"  "%dp0%\\node_modules\\@anthropic-ai\\claude-code\\cli.js" %*',
        '',
      ].join('\r\n'),
    );

    const fallbackNode = '/custom/bin/node';
    expect(resolveWindowsClaudeShim(shim, fallbackNode)).toEqual({
      file: fallbackNode,
      args: [cliJs],
    });
    expect(
      claudeSpawnSpec(shim, ['--resume', 'explain a & calc.exe'], 'win32', { ComSpec: 'C:\\Windows\\cmd.exe' }),
    ).toEqual({
      file: process.execPath,
      args: [cliJs, '--resume', 'explain a & calc.exe'],
    });

    // When a sibling node.exe exists next to claude.cmd, prefer it over fallback nodeExecPath.
    const siblingNode = join(npmBin, 'node.exe');
    writeFileSync(siblingNode, 'mz');
    expect(resolveWindowsClaudeShim(shim, fallbackNode)).toEqual({
      file: siblingNode,
      args: [cliJs],
    });
    expect(claudeSpawnSpec(shim, ['--resume'], 'win32', {}, fallbackNode)).toEqual({
      file: siblingNode,
      args: [cliJs, '--resume'],
    });
  });

  it('resolves minimal npm, legacy IF/ELSE, and pnpm global .cmd shims', () => {
    const minimalDir = dir('minimal');
    const minimalCli = join(dir('minimal/node_modules/@anthropic-ai/claude-code'), 'cli.js');
    writeFileSync(minimalCli, '');
    const minimalShim = join(minimalDir, 'claude.cmd');
    writeFileSync(minimalShim, '"%_prog%"  "%dp0%\\node_modules\\@anthropic-ai\\claude-code\\cli.js" %*\r\n');
    expect(resolveWindowsClaudeShim(minimalShim, '/node')).toEqual({
      file: '/node',
      args: [minimalCli],
    });

    const legacyDir = dir('legacy');
    const legacyCli = join(dir('legacy/node_modules/@anthropic-ai/claude-code'), 'cli.js');
    writeFileSync(legacyCli, '');
    const legacyShim = join(legacyDir, 'claude.cmd');
    writeFileSync(
      legacyShim,
      [
        '@IF EXIST "%~dp0\\node.exe" (',
        '  "%~dp0\\node.exe"  "%~dp0\\node_modules\\@anthropic-ai\\claude-code\\cli.js" %*',
        ') ELSE (',
        '  @SETLOCAL',
        '  @SET PATHEXT=%PATHEXT:;.JS;=;%',
        '  node  "%~dp0\\node_modules\\@anthropic-ai\\claude-code\\cli.js" %*',
        ')',
      ].join('\r\n'),
    );
    expect(resolveWindowsClaudeShim(legacyShim, '/node')).toEqual({
      file: '/node',
      args: [legacyCli],
    });

    const pnpmDir = dir('pnpm');
    const pnpmCli = join(dir('pnpm/global/5/node_modules/@anthropic-ai/claude-code'), 'cli.js');
    writeFileSync(pnpmCli, '');
    const pnpmShim = join(pnpmDir, 'claude.cmd');
    writeFileSync(
      pnpmShim,
      [
        '@ECHO off',
        'SETLOCAL',
        'CALL :find_dp0',
        '"%_prog%"  "%dp0%\\global\\5\\node_modules\\@anthropic-ai\\claude-code\\cli.js" %*',
        'ENDLOCAL',
        'EXIT /b %errorlevel%',
        ':find_dp0',
        'SET dp0=%~dp0',
        'EXIT /b',
      ].join('\r\n'),
    );
    expect(resolveWindowsClaudeShim(pnpmShim, '/node')).toEqual({
      file: '/node',
      args: [pnpmCli],
    });

    // cmd-shim 9.0.2 combines `set PATHEXT=%PATHEXT:;.JS;=;% & "%_prog%" ... %*` on the endLocal line.
    const shim9Dir = dir('cmd-shim-9');
    const shim9Cli = join(dir('cmd-shim-9/node_modules/@anthropic-ai/claude-code'), 'cli.js');
    writeFileSync(shim9Cli, '');
    const shim9 = join(shim9Dir, 'claude.cmd');
    writeFileSync(
      shim9,
      [
        '@ECHO off',
        'GOTO start',
        ':find_dp0',
        'SET dp0=%~dp0',
        'EXIT /b',
        ':start',
        'SETLOCAL',
        'CALL :find_dp0',
        '',
        'IF EXIST "%dp0%\\node.exe" (',
        '  SET "_prog=%dp0%\\node.exe"',
        ') ELSE (',
        '  SET "_prog=node"',
        ')',
        '',
        'endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & set PATHEXT=%PATHEXT:;.JS;=;% & "%_prog%"  "%dp0%\\node_modules\\@anthropic-ai\\claude-code\\cli.js" %*',
        '',
      ].join('\r\n'),
    );
    expect(resolveWindowsClaudeShim(shim9, '/node')).toEqual({
      file: '/node',
      args: [shim9Cli],
    });
  });

  it('resolves direct .exe wrapper .cmd shims and keeps native .exe direct', () => {
    const wrapperDir = dir('wrapper');
    const exeDir = dir('wrapper/bin');
    const targetExe = join(exeDir, 'claude.exe');
    writeFileSync(targetExe, 'mz');

    const relShim = join(wrapperDir, 'claude.cmd');
    writeFileSync(relShim, '@echo off\r\n"%~dp0\\bin\\claude.exe" %*\r\n');
    expect(resolveWindowsClaudeShim(relShim)).toEqual({ file: targetExe, args: [] });
    expect(claudeSpawnSpec(relShim, ['--resume'], 'win32', {})).toEqual({
      file: targetExe,
      args: ['--resume'],
    });

    const absShim = join(wrapperDir, 'claude-abs.cmd');
    writeFileSync(absShim, `@echo off\r\n"${targetExe}" %*\r\n`);
    expect(resolveWindowsClaudeShim(absShim)).toEqual({ file: targetExe, args: [] });

    expect(claudeSpawnSpec('C:\\Users\\me\\.local\\bin\\claude.exe', ['--resume'], 'win32', {})).toEqual({
      file: 'C:\\Users\\me\\.local\\bin\\claude.exe',
      args: ['--resume'],
    });
  });

  it('refuses cmd.exe fallback and throws a clear error for unrecognized .cmd/.bat wrappers', () => {
    const customDir = dir('custom');

    const wslBat = join(customDir, 'claude.bat');
    writeFileSync(wslBat, '@echo off\r\nwsl claude %*\r\n');
    expect(() => claudeSpawnSpec(wslBat, ['hi'], 'win32', { ComSpec: 'C:\\Windows\\cmd.exe' })).toThrow(
      /refusing to forward arguments through cmd\.exe/i,
    );
    expect(() => resolveWindowsClaudeShim(wslBat)).toThrow(wslBat);

    const chainedCmd = join(customDir, 'chained.cmd');
    writeFileSync(chainedCmd, '@echo off\r\ncall "%~dp0\\other.cmd" %*\r\n');
    expect(() => resolveWindowsClaudeShim(chainedCmd)).toThrow(/cmd\.exe/i);

    const missingTargetCmd = join(customDir, 'missing.cmd');
    writeFileSync(missingTargetCmd, '"%_prog%"  "%dp0%\\node_modules\\missing\\cli.js" %*\r\n');
    expect(() => resolveWindowsClaudeShim(missingTargetCmd)).toThrow(/cmd\.exe/i);

    const extraCmdScript = join(customDir, 'extra-setup.cmd');
    const targetExe = join(customDir, 'claude.exe');
    writeFileSync(targetExe, 'mz');
    writeFileSync(extraCmdScript, '@echo off\r\npowershell -File setup.ps1\r\n"%~dp0\\claude.exe" %*\r\n');
    expect(() => resolveWindowsClaudeShim(extraCmdScript)).toThrow(/cmd\.exe/i);

    const nonNodeProg = join(customDir, 'python-prog.cmd');
    const cliJs = join(customDir, 'cli.js');
    writeFileSync(cliJs, '');
    writeFileSync(nonNodeProg, '@echo off\r\nSET "_prog=python"\r\n"%_prog%" "%~dp0\\cli.js" %*\r\n');
    expect(() => resolveWindowsClaudeShim(nonNodeProg)).toThrow(/cmd\.exe/i);

    const oversized = join(customDir, 'huge.cmd');
    writeFileSync(oversized, `${'REM pad\r\n'.repeat(10000)}"%~dp0\\claude.exe" %*\r\n`);
    expect(() => resolveWindowsClaudeShim(oversized)).toThrow(/cmd\.exe/i);

    const profileWrapper = join(customDir, 'profile.cmd');
    writeFileSync(profileWrapper, '@echo off\r\nSET CLAUDE_CONFIG_DIR=C:\\work-claude\r\n"%~dp0\\claude.exe" %*\r\n');
    expect(() => resolveWindowsClaudeShim(profileWrapper)).toThrow(/unsupported environment assignment/i);

    const customBranch = join(customDir, 'branch.cmd');
    writeFileSync(
      customBranch,
      '@echo off\r\nIF "%1"=="--work" SET CLAUDE_CONFIG_DIR=C:\\work\r\n"%~dp0\\claude.exe" %*\r\n',
    );
    expect(() => resolveWindowsClaudeShim(customBranch)).toThrow(/cmd\.exe/i);
  });

  it('forwards hostile shell metacharacters verbatim when executing a resolved npm .cmd shim', () => {
    const binDir = dir('live-shim');
    const pkgDir = dir('live-shim/node_modules/@anthropic-ai/claude-code');
    const cliMjs = join(pkgDir, 'cli.mjs');
    writeFileSync(cliMjs, 'process.stdout.write(JSON.stringify(process.argv.slice(2)));\n');

    const shim = join(binDir, 'claude.cmd');
    writeFileSync(
      shim,
      [
        '@ECHO off',
        'GOTO start',
        ':find_dp0',
        'SET dp0=%~dp0',
        'EXIT /b',
        ':start',
        'SETLOCAL',
        'CALL :find_dp0',
        'endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"  "%dp0%\\node_modules\\@anthropic-ai\\claude-code\\cli.mjs" %*',
        '',
      ].join('\r\n'),
    );

    const hostileArgs = ['--', 'explain a & calc.exe && %USERPROFILE% | "quoted" ^caret'];
    const spec = claudeSpawnSpec(shim, hostileArgs, 'win32', {}, process.execPath);
    expect(spec.file).toBe(process.execPath);
    expect(spec.args).toEqual([cliMjs, ...hostileArgs]);

    const stdout = execFileSync(spec.file, spec.args, { encoding: 'utf8' });
    expect(JSON.parse(stdout)).toEqual(hostileArgs);
  });

  it('runs directly on POSIX regardless of filename extension', () => {
    const args = ['--resume'];
    const spec = claudeSpawnSpec('/usr/local/bin/claude', args, 'darwin', {});
    expect(spec).toEqual({ file: '/usr/local/bin/claude', args: ['--resume'] });
    expect(spec.args).not.toBe(args);
    expect(claudeSpawnSpec('/tmp/claude.cmd', ['x'], 'linux', {})).toEqual({ file: '/tmp/claude.cmd', args: ['x'] });
  });

  it('resolveClaudeLaunch skips stale or unsupported Windows shims to find a working later candidate', () => {
    const staleDir = dir('stale-npm');
    const staleShim = join(staleDir, 'claude.cmd');
    writeFileSync(staleShim, '"%_prog%"  "%dp0%\\node_modules\\@anthropic-ai\\claude-code\\cli.js" %*\r\n');

    const profileDir = dir('profile-wrapper');
    const profileExe = join(profileDir, 'real-claude.exe');
    writeFileSync(profileExe, 'mz');
    const profileShim = join(profileDir, 'claude.cmd');
    writeFileSync(profileShim, '@echo off\r\nSET CLAUDE_CONFIG_DIR=C:\\work\r\n"%~dp0\\real-claude.exe" %*\r\n');

    const nativeDir = dir('native-claude');
    const nativeExe = join(nativeDir, 'claude.exe');
    writeFileSync(nativeExe, 'mz');

    const path = [staleDir, profileDir, nativeDir].join(';');
    expect(resolveClaudeLaunch(['--resume'], { Path: path }, 'win32')).toEqual({
      claudeBin: nativeExe,
      spawnSpec: { file: nativeExe, args: ['--resume'] },
    });

    // When only broken/unsupported shims exist on PATH, reports the rejected wrappers.
    expect(() => resolveClaudeLaunch(['--resume'], { Path: [staleDir, profileDir].join(';') }, 'win32')).toThrow(
      staleShim,
    );
    expect(() => resolveClaudeLaunch(['--resume'], { Path: [staleDir, profileDir].join(';') }, 'win32')).toThrow(
      profileShim,
    );

    // When no candidate exists at all, returns null.
    expect(resolveClaudeLaunch(['--resume'], { Path: dir('empty') }, 'win32')).toBeNull();
  });
});
