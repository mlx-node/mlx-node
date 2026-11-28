import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { delimiter, dirname, join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { PayloadError } from '../scripts/payload.js';
import { ReleaseVersionError } from '../scripts/release-version.js';
import {
  collectInstallManifest,
  computeDirSizeKiB,
  escapeNsisString,
  findMakensis,
  NSIS_AUMID_REG_KEY,
  NSIS_MANIFEST_FILENAME,
  NSIS_MANIFEST_HEADER,
  NSIS_MAX_INSTALLER_BYTES,
  NSIS_MAX_STRLEN,
  NSIS_MIN_WINDOWS_BUILD,
  NSIS_UNINSTALL_REG_KEY,
  nsisVersionQuad,
  PS_CHECK_RUNNING_SCRIPT,
  PS_KILL_RUNNING_SCRIPT,
  renderInstallManifestBuffer,
  renderNsisScript,
  verifyWindowsInstaller,
} from '../scripts/windows-nsis.js';

const NSIS_MAGIC = Buffer.from([
  0xef, 0xbe, 0xad, 0xde, 0x4e, 0x75, 0x6c, 0x6c, 0x73, 0x6f, 0x66, 0x74, 0x49, 0x6e, 0x73, 0x74,
]);

describe('NSIS version quad and string escaping', () => {
  it('maps semver (including prereleases) to 4-part uint16 Windows version quads', () => {
    expect(nsisVersionQuad('0.0.25')).toBe('0.0.25.0');
    expect(nsisVersionQuad('0.0.25-rc.1')).toBe('0.0.25.0');
    expect(nsisVersionQuad('1.2.3+build.9')).toBe('1.2.3.0');
    expect(nsisVersionQuad('65535.65535.65535')).toBe('65535.65535.65535.0');
  });

  it('rejects invalid semver or components over uint16 max', () => {
    expect(() => nsisVersionQuad('nightly')).toThrow(ReleaseVersionError);
    expect(() => nsisVersionQuad('65536.0.0')).toThrow(ReleaseVersionError);
    expect(() => nsisVersionQuad('0.70000.0')).toThrow(ReleaseVersionError);
  });

  it('escapes NSIS $ and quote characters and rejects control chars', () => {
    expect(escapeNsisString('C:\\a\\$dir\\"q"')).toBe('C:\\a\\$$dir\\$\\"q$\\"');
    expect(() => escapeNsisString('line1\nline2')).toThrow(PayloadError);
    expect(() => escapeNsisString('bad\0null')).toThrow(PayloadError);
  });
});

describe('renderNsisScript', () => {
  const base = {
    version: '0.0.25',
    appPath: 'C:\\build\\Portato-win32-arm64',
    iconPath: 'C:\\repo\\packages\\desktop\\build\\icon.ico',
    outFile: 'C:\\build\\Portato-0.0.25-win32-arm64-setup.exe',
    estimatedSizeKiB: 2_500_000,
  };

  function assertNsExecArgsFitNsisStringLimit(script: string): void {
    const matches = [...script.matchAll(/nsExec::Exec\s+'([^']+)'/g)];
    expect(matches.length).toBeGreaterThan(0);
    for (const match of matches) {
      // Simulate expanded runtime length with a generous 64-char $SYSDIR path.
      const expanded = match[1]!.replaceAll('$SYSDIR', 'C:\\Windows\\System32'.padEnd(64, 'x'));
      expect(expanded.length).toBeLessThan(NSIS_MAX_STRLEN - 100);
    }
  }

  it('renders Pass 1 uninstaller generator with validated manifest cleanup, ownership check, and no recursive RMDir', () => {
    const script = renderNsisScript({
      ...base,
      outFile: 'C:\\stage\\uninst-gen.exe',
      mode: { kind: 'build-uninstaller', uninstallerOutFile: 'C:\\out\\.nsis-uninst\\Uninstall.exe' },
    });

    assertNsExecArgsFitNsisStringLimit(script);
    expect(script).toContain('SilentInstall silent');
    expect(script).toContain('WriteUninstaller "C:\\out\\.nsis-uninst\\Uninstall.exe"');
    expect(script).toContain('VIAddVersionKey "OriginalFilename" "Uninstall.exe"');
    expect(script).toContain('!define MUI_ICON "C:\\repo\\packages\\desktop\\build\\icon.ico"');
    expect(script).toContain('!define MUI_UNICON "C:\\repo\\packages\\desktop\\build\\icon.ico"');
    expect(script).toContain('${UnStrStr}');
    expect(script).not.toContain('${StrStr}');
    expect(script).toContain('Section "Uninstall"');
    expect(script).toContain('!insertmacro StopRunningApp uninstall');
    expect(script).toContain(`$3 != "${NSIS_MANIFEST_HEADER}"`);
    expect(script).toContain('${UnStrStr} $7 $5 ".."');
    expect(script).toContain('${UnStrStr} $7 $5 "*"');
    expect(script).toContain('ReadRegStr $0 HKCU "${UNINSTALL_KEY}" "InstallLocation"');
    expect(script).toContain('${If} $0 == "$INSTDIR"');
    expect(script).not.toContain('${If} $0 == ""');
    expect(script).toContain('Delete "$INSTDIR\\Uninstall.exe"');
    expect(script).toContain('RMDir "$INSTDIR"');
    expect(script).not.toContain('RMDir /r');
    expect(script).not.toContain('File /r');

    // Manifest cleanup must run BEFORE deleting shortcuts/ARP/Uninstall.exe so failures remain retryable.
    expect(script.indexOf('manifest_loop_uninstall:')).toBeLessThan(
      script.indexOf('DeleteRegKey HKCU "${UNINSTALL_KEY}"'),
    );
  });

  it('renders Pass 2 signed installer with build 26100 check, safe dir guard, compact path-scoped stop, shortcut AUMID, and manifest validation', () => {
    const script = renderNsisScript({
      ...base,
      mode: {
        kind: 'installer',
        signedUninstallerPath: 'C:\\out\\.nsis-uninst\\Uninstall.exe',
        manifestPath: 'C:\\stage\\.portato-manifest.txt',
      },
    });

    assertNsExecArgsFitNsisStringLimit(script);
    expect(script).toContain('Unicode True');
    expect(script).toContain('RequestExecutionLevel user');
    expect(script).toContain('SetCompressor /FINAL zlib');
    expect(script).toContain('!include "StrFunc.nsh"');
    expect(script).toContain('${StrStr}');
    expect(script).not.toContain('${UnStrStr}');
    expect(script).toContain('!include "Win\\COM.nsh"');
    expect(script).toContain('!include "Win\\Propkey.nsh"');
    expect(script).toContain('InstallDir "$LOCALAPPDATA\\Programs\\${APP_NAME}"');
    expect(script).toContain(`!define UNINSTALL_KEY "${NSIS_UNINSTALL_REG_KEY}"`);
    expect(NSIS_UNINSTALL_REG_KEY).toBe('Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Portato');
    expect(script).toContain(`!define AUMID_KEY "${NSIS_AUMID_REG_KEY}"`);
    expect(script).toContain(`!define MANIFEST_FILE "${NSIS_MANIFEST_FILENAME}"`);
    expect(script).toContain('${IfNot} ${IsNativeARM64}');
    expect(script).toContain(
      'ReadRegStr $0 HKLM "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion" "CurrentBuildNumber"',
    );
    expect(script).toContain(`IntCmp $0 ${NSIS_MIN_WINDOWS_BUILD} build_ok build_too_old build_ok`);
    expect(script).toContain('SetRegView 64');
    expect(script).toContain('VIAddVersionKey "OriginalFilename" "Portato-0.0.25-win32-arm64-setup.exe"');
    expect(script).toContain('!insertmacro AssertSafeInstallDir');
    expect(script).toContain('SetErrorLevel 3');
    expect(script).toContain('!insertmacro StopRunningApp install');
    expect(script).toContain('${If} $0 == 42');
    expect(script).toContain('${ElseIf} $0 != 43');
    expect(script).toContain('GetFileAttributes');
    expect(script).toContain('${StrStr} $7 $5 ".."');
    expect(script).toContain('${StrStr} $7 $5 "*"');
    expect(script).toContain('${StrStr} $7 $5 "?"');
    expect(script).toContain('${StrStr} $7 $5 ":"');
    expect(script).toContain('IDOK stop_ok_${LABEL_TAG}');
    expect(script).toContain('stop_ok_${LABEL_TAG}:');
    expect(script).not.toContain('__LINE__');
    expect(script).toContain('File /r "C:\\build\\Portato-win32-arm64\\*.*"');
    expect(script).toContain('File "/oname=$INSTDIR\\${MANIFEST_FILE}" "C:\\stage\\.portato-manifest.txt"');
    expect(script).toContain('File "/oname=$INSTDIR\\Uninstall.exe" "C:\\out\\.nsis-uninst\\Uninstall.exe"');
    expect(script).toContain('!insertmacro SetLnkAppUserModelId "$SMPROGRAMS\\${APP_NAME}.lnk"');
    expect(script).toContain('!insertmacro SetLnkAppUserModelId "$DESKTOP\\${APP_NAME}.lnk"');
    expect(script).toContain('${PKEY_AppUserModel_ID}');
    expect(script).toContain('WriteRegDWORD HKCU "${UNINSTALL_KEY}" "EstimatedSize" 2500000');
    expect(script).not.toContain('WriteUninstaller');
    expect(script).not.toContain('Section "Uninstall"');
    expect(script).not.toContain('MUI_UNICON');
    expect(script).not.toContain('RMDir /r');
    expect(script).not.toContain('Delete "$INSTDIR\\*.dll"');
    expect(script).not.toContain('taskkill.exe" /F /IM');

    // Verify embedded PowerShell commands scope termination strictly to $INSTDIR PIDs.
    expect(PS_CHECK_RUNNING_SCRIPT).toContain('$env:PI');
    expect(PS_CHECK_RUNNING_SCRIPT).toContain('StartsWith($d,5)');
    expect(PS_KILL_RUNNING_SCRIPT).toContain('/F /PID $_.ProcessId /T');
    expect(PS_KILL_RUNNING_SCRIPT).not.toContain('/IM');

    const codeLines = script
      .split(/\r?\n/)
      .filter((line) => !line.trim().startsWith(';'))
      .join('\n');
    expect(codeLines).not.toContain('$APPDATA\\mlx-node');
    expect(codeLines).not.toContain('.mlx-node');
  });

  it('renders single-pass unsigned installer with both StrStr and UnStrStr declarations', () => {
    const script = renderNsisScript({
      ...base,
      mode: {
        kind: 'installer',
        signedUninstallerPath: null,
        manifestPath: 'C:\\stage\\.portato-manifest.txt',
      },
    });

    assertNsExecArgsFitNsisStringLimit(script);
    expect(script).toContain('${StrStr}');
    expect(script).toContain('${UnStrStr}');
    expect(script).toContain('!define MUI_UNICON "C:\\repo\\packages\\desktop\\build\\icon.ico"');
    expect(script).toContain('WriteUninstaller "$INSTDIR\\Uninstall.exe"');
    expect(script).toContain('Section "Uninstall"');
    expect(script).toContain('manifest_loop_uninstall:');
    expect(script).toContain('Delete "$SMPROGRAMS\\${APP_NAME}.lnk"');
    expect(script).toContain('Delete "$DESKTOP\\${APP_NAME}.lnk"');
    expect(script).toContain('DeleteRegKey HKCU "${AUMID_KEY}"');
    expect(script).toContain('DeleteRegKey HKCU "${UNINSTALL_KEY}"');
    expect(script).toContain('RMDir "$INSTDIR"');
    expect(script).not.toContain('RMDir /r');
  });
});

describe('collectInstallManifest, findMakensis, computeDirSizeKiB, and verifyWindowsInstaller', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'mlx-windows-nsis-'));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  function write(rel: string, contents: string | Buffer): string {
    const full = join(root, rel);
    mkdirSync(dirname(full), { recursive: true });
    writeFileSync(full, contents);
    return full;
  }

  it('collects files lexicographically and subdirectories deepest-first as UTF-16LE without BOM', () => {
    write('app/Portato.exe', 'exe');
    write('app/resources/app/package.json', '{}');
    write('app/resources/native/openblas.dll', 'dll');
    write('app/locales/en-US.pak', 'pak');

    const entries = collectInstallManifest(join(root, 'app'));
    expect(entries.files).toEqual([
      'locales\\en-US.pak',
      'Portato.exe',
      'resources\\app\\package.json',
      'resources\\native\\openblas.dll',
    ]);
    expect(entries.dirs).toEqual(['resources\\native', 'resources\\app', 'resources', 'locales']);

    const buf = renderInstallManifestBuffer(entries);
    expect(buf[0]).not.toBe(0xff);
    const decoded = buf.toString('utf16le');
    expect(decoded.split('\r\n')).toEqual([
      NSIS_MANIFEST_HEADER,
      'F:locales\\en-US.pak',
      'F:Portato.exe',
      'F:resources\\app\\package.json',
      'F:resources\\native\\openblas.dll',
      'D:resources\\native',
      'D:resources\\app',
      'D:resources',
      'D:locales',
      '',
    ]);
  });

  it('rejects symbolic links when building an install manifest', () => {
    const target = write('outside.txt', 'data');
    mkdirSync(join(root, 'app'), { recursive: true });
    try {
      symlinkSync(target, join(root, 'app', 'link.txt'));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'EPERM') return;
      throw error;
    }
    expect(() => collectInstallManifest(join(root, 'app'))).toThrow(PayloadError);
  });

  it('resolves makensis from MAKENSIS, default candidates, or PATH, and fails closed when missing', () => {
    const custom = write('custom/makensis.exe', 'stub');
    expect(findMakensis({ MAKENSIS: custom }, { defaultCandidates: [] })).toBe(custom);
    expect(() => findMakensis({ MAKENSIS: join(root, 'missing.exe') }, { defaultCandidates: [] })).toThrow(
      PayloadError,
    );

    const candidate = write('nsis/makensis.exe', 'stub');
    expect(findMakensis({}, { defaultCandidates: [candidate] })).toBe(candidate);

    const pathBin = write('bin/makensis.exe', 'stub');
    expect(
      findMakensis({ PATH: `${join(root, 'empty')}${delimiter}${join(root, 'bin')}` }, { defaultCandidates: [] }),
    ).toBe(pathBin);

    expect(() => findMakensis({ PATH: '' }, { defaultCandidates: [] })).toThrow(PayloadError);
  });

  it('computes recursive directory size in KiB', () => {
    write('app/a.bin', Buffer.alloc(1500, 1));
    write('app/sub/b.bin', Buffer.alloc(2048, 2));
    expect(computeDirSizeKiB(join(root, 'app'))).toBe(4);
  });

  it('accepts a valid MZ + NullsoftInst installer matching the expected version and rejects corrupt/mismatched ones', () => {
    const valid = write(
      'out/Portato-0.0.25-win32-arm64-setup.exe',
      Buffer.concat([Buffer.from([0x4d, 0x5a]), Buffer.alloc(1022, 0), NSIS_MAGIC, Buffer.alloc(64, 0)]),
    );
    expect(verifyWindowsInstaller({ installerPath: valid, expectedVersion: '0.0.25' }).sizeBytes).toBe(1024 + 16 + 64);
    expect(NSIS_MAX_INSTALLER_BYTES).toBe(2 * 1024 * 1024 * 1024 - 1);

    expect(() => verifyWindowsInstaller({ installerPath: valid, expectedVersion: '0.0.26' })).toThrow(
      ReleaseVersionError,
    );

    const badMz = write('bad-mz/Portato-0.0.25-win32-arm64-setup.exe', Buffer.concat([Buffer.from('PK'), NSIS_MAGIC]));
    expect(() => verifyWindowsInstaller({ installerPath: badMz, expectedVersion: '0.0.25' })).toThrow(/MZ header/);

    const badMagic = write('bad-magic/Portato-0.0.25-win32-arm64-setup.exe', Buffer.from([0x4d, 0x5a, 0x90, 0x00]));
    expect(() => verifyWindowsInstaller({ installerPath: badMagic, expectedVersion: '0.0.25' })).toThrow(
      /NullsoftInst/,
    );

    const empty = write('empty/Portato-0.0.25-win32-arm64-setup.exe', Buffer.alloc(0));
    expect(() => verifyWindowsInstaller({ installerPath: empty, expectedVersion: '0.0.25' })).toThrow(/empty/i);
  });
});
