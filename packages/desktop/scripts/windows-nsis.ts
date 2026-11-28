/**
 * Build and verify the Windows ARM64 NSIS setup installer (`Portato-<version>-win32-arm64-setup.exe`).
 *
 * Supports two-pass Authenticode signing so the embedded `Uninstall.exe` is
 * signed alongside the app binaries before the final setup `.exe` is compiled:
 *
 *   1. `buildWindowsUninstaller`: compiles the `.nsi` with `BUILD_UNINSTALLER`
 *      into a tiny temporary generator, runs it to write `Uninstall.exe`
 *      OUTSIDE `appPath` (so Store MSIX packaging never sees it), and cleans up.
 *   2. Caller signs all PE files in `appPath` plus `Uninstall.exe`.
 *   3. `buildWindowsInstaller`: writes a UTF-16LE install manifest of the
 *      exact packaged files/dirs, then compiles the `.nsi` with
 *      `SIGNED_UNINSTALLER` (or single-pass without it for unsigned local
 *      builds) using non-solid `zlib` compression so builds stay fast, fit
 *      comfortably under NSIS's 2 GiB datablock ceiling, and parse cleanly in
 *      Komac / WinGet.
 */

import { execFileSync } from 'node:child_process';
import {
  closeSync,
  existsSync,
  lstatSync,
  mkdirSync,
  openSync,
  readdirSync,
  readSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { basename, delimiter, dirname, join, resolve } from 'node:path';

import { APP_ID, APP_NAME, APP_PUBLISHER } from '../src/app-name.js';
import { PayloadError } from './payload.js';
import { assertSemver, ReleaseVersionError, versionFromWindowsInstallerPath } from './release-version.js';

/** Hard 32-bit signed offset ceiling of NSIS 3.x installers (2 GiB - 1 byte). */
export const NSIS_MAX_INSTALLER_BYTES = 2 * 1024 * 1024 * 1024 - 1;
/** Warn once the setup executable crosses 1.75 GiB so we have runway before 2 GiB. */
export const NSIS_WARN_BYTES = Math.floor(1.75 * 1024 * 1024 * 1024);

/** Stock NSIS 3.x string buffer ceiling (`NSIS_MAX_STRLEN`). */
export const NSIS_MAX_STRLEN = 1024;

/** Minimum supported Windows 11 OS build (24H2 / CUDA 13.4 ARM64 floor, matching MSIX MinVersion). */
export const NSIS_MIN_WINDOWS_BUILD = 26100;

/** Per-installation manifest listing every file and subdirectory installed by Portato. */
export const NSIS_MANIFEST_FILENAME = '.portato-manifest.txt';
export const NSIS_MANIFEST_HEADER = '# portato-manifest-v1';

/**
 * Exact Add/Remove Programs subkey casing expected by Komac v2.17.2's NSIS
 * registry analyzer (`Software\Microsoft\Windows\CurrentVersion\Uninstall`).
 */
export const NSIS_UNINSTALL_REG_KEY = `Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}`;
export const NSIS_AUMID_REG_KEY = `Software\\Classes\\AppUserModelId\\${APP_ID}`;
export const WINDOWS_EXECUTABLE_NAME = `${APP_NAME}.exe`;

/**
 * Compact PowerShell script (passed via `-EncodedCommand` with `$env:PI` = `$INSTDIR`
 * and `$env:PE` = `${APP_EXE}`) that exits 42 iff at least one running instance of
 * the target executable resides inside `$INSTDIR`, or exits 43 when none do.
 * Uses `-EA Stop` and non-1 sentinel codes so PowerShell's default error code `1`
 * can never be mistaken for "not running".
 * Kept compact so the full `nsExec::Exec` argument stays well below stock NSIS's
 * 1,024-character `NSIS_MAX_STRLEN` limit.
 * (`StartsWith($d, 5)` uses `[StringComparison]::OrdinalIgnoreCase`.)
 */
export const PS_CHECK_RUNNING_SCRIPT =
  "if(!$env:PI){exit 43};$d=[IO.Path]::GetFullPath($env:PI).TrimEnd('\\')+'\\';if(@(gcim Win32_Process -EA Stop -f \"Name='$env:PE'\"|?{$_.ExecutablePath -and [IO.Path]::GetFullPath($_.ExecutablePath).StartsWith($d,5)})){exit 42};exit 43";

/**
 * Compact PowerShell script (passed via `-EncodedCommand`) that terminates only
 * the PIDs (and their child trees) of `$env:PE` running from `$env:PI`.
 * Kept compact so the full `nsExec::Exec` argument stays well below 1,024 chars.
 */
export const PS_KILL_RUNNING_SCRIPT =
  "if(!$env:PI){exit 0};$d=[IO.Path]::GetFullPath($env:PI).TrimEnd('\\')+'\\';gcim Win32_Process -f \"Name='$env:PE'\"|?{$_.ExecutablePath -and [IO.Path]::GetFullPath($_.ExecutablePath).StartsWith($d,5)}|%{taskkill /F /PID $_.ProcessId /T}";

const PS_CHECK_RUNNING_B64 = Buffer.from(PS_CHECK_RUNNING_SCRIPT, 'utf16le').toString('base64');
const PS_KILL_RUNNING_B64 = Buffer.from(PS_KILL_RUNNING_SCRIPT, 'utf16le').toString('base64');

const NSIS_HEADER_SIGNATURE = Buffer.from([
  0xef, 0xbe, 0xad, 0xde, 0x4e, 0x75, 0x6c, 0x6c, 0x73, 0x6f, 0x66, 0x74, 0x49, 0x6e, 0x73, 0x74,
]);
const NSIS_HEADER_SCAN_BYTES = 512 * 1024;
const NSIS_MAX_UINT16 = 65535;
const NSIS_MAX_DWORD_SIGNED = 0x7fffffff;

export interface FindMakensisOptions {
  defaultCandidates?: readonly string[];
}

const DEFAULT_MAKENSIS_CANDIDATES = [
  'C:\\Program Files (x86)\\NSIS\\makensis.exe',
  'C:\\Program Files\\NSIS\\makensis.exe',
] as const;

function isExistingFile(filePath: string): boolean {
  try {
    return statSync(filePath).isFile();
  } catch {
    return false;
  }
}

/** Locate `makensis.exe` from `MAKENSIS`, standard NSIS install directories, or `PATH`. */
export function findMakensis(env: NodeJS.ProcessEnv = process.env, options: FindMakensisOptions = {}): string {
  const explicit = env.MAKENSIS?.trim();
  if (explicit) {
    if (isExistingFile(explicit)) return explicit;
    throw new PayloadError(`MAKENSIS points to a missing file: ${explicit}`, 'install NSIS 3.x or fix MAKENSIS');
  }

  for (const candidate of options.defaultCandidates ?? DEFAULT_MAKENSIS_CANDIDATES) {
    if (isExistingFile(candidate)) return candidate;
  }

  const pathVar = env.PATH ?? env.Path ?? '';
  for (const dir of pathVar.split(delimiter).filter(Boolean)) {
    const candidate = join(dir, 'makensis.exe');
    if (isExistingFile(candidate)) return candidate;
  }

  throw new PayloadError(
    'NSIS compiler makensis.exe was not found',
    'install NSIS 3.x (or set MAKENSIS to makensis.exe)',
  );
}

/** Convert semver `MAJOR.MINOR.PATCH[-pre][+build]` to 4-part `MAJOR.MINOR.PATCH.0` for Windows PE version info. */
export function nsisVersionQuad(version: string): string {
  assertSemver('version', version);
  const core = version.split(/[-+]/, 1)[0]!;
  const [major, minor, patch] = core.split('.').map(Number);
  if (
    !Number.isInteger(major) ||
    !Number.isInteger(minor) ||
    !Number.isInteger(patch) ||
    major < 0 ||
    minor < 0 ||
    patch < 0 ||
    major > NSIS_MAX_UINT16 ||
    minor > NSIS_MAX_UINT16 ||
    patch > NSIS_MAX_UINT16
  ) {
    throw new ReleaseVersionError(
      `NSIS version component out of range (0..${NSIS_MAX_UINT16}): ${JSON.stringify(version)}`,
      `each MAJOR, MINOR, and PATCH component must be between 0 and ${NSIS_MAX_UINT16}`,
    );
  }
  return `${major}.${minor}.${patch}.0`;
}

/** Sum regular file sizes under `dir` and return the value in KiB (clamped to `[1, 0x7fffffff]`). */
export function computeDirSizeKiB(dir: string): number {
  let totalBytes = 0;
  const stack = [dir];
  for (let current = stack.pop(); current !== undefined; current = stack.pop()) {
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const fullPath = join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
      } else if (entry.isFile()) {
        totalBytes += statSync(fullPath).size;
      }
    }
  }
  const kib = Math.max(1, Math.ceil(totalBytes / 1024));
  return Math.min(kib, NSIS_MAX_DWORD_SIGNED);
}

export interface InstallManifestEntries {
  /** Relative file paths (Windows `\` separators), sorted lexicographically. */
  files: string[];
  /** Relative subdirectory paths (Windows `\` separators), sorted deepest-first. */
  dirs: string[];
}

/**
 * Walk `appPath` and collect every packaged file and subdirectory so upgrade and
 * uninstall remove ONLY files installed by Portato (never unrelated files).
 */
export function collectInstallManifest(appPath: string): InstallManifestEntries {
  const files: string[] = [];
  const dirs: string[] = [];

  function walk(absDir: string, relPrefix: string): void {
    const entries = readdirSync(absDir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      if (entry.name === '.' || entry.name === '..' || entry.name.includes('..') || /[\r\n\0*?:\\/]/.test(entry.name)) {
        throw new PayloadError(
          `Unsafe filename in Windows app payload: ${JSON.stringify(entry.name)}`,
          'clean and re-stage the Windows app directory',
        );
      }
      const fullPath = join(absDir, entry.name);
      const stat = lstatSync(fullPath);
      if (stat.isSymbolicLink()) {
        throw new PayloadError(
          `Unexpected symbolic link in Windows app payload: ${fullPath}`,
          'remove symbolic links from the staged Windows app directory',
        );
      }
      const relPath = relPrefix ? `${relPrefix}\\${entry.name}` : entry.name;
      if (stat.isDirectory()) {
        dirs.push(relPath);
        walk(fullPath, relPath);
      } else if (stat.isFile()) {
        files.push(relPath);
      }
    }
  }

  walk(appPath, '');
  files.sort((a, b) => a.localeCompare(b));
  dirs.sort((a, b) => {
    const depthDiff = b.split('\\').length - a.split('\\').length;
    return depthDiff !== 0 ? depthDiff : b.localeCompare(a);
  });
  return { files, dirs };
}

/**
 * Serialize `InstallManifestEntries` to UTF-16LE (no BOM) for NSIS
 * `FileReadUTF16LE` consumption.
 */
export function renderInstallManifestBuffer(entries: InstallManifestEntries): Buffer {
  const lines = [NSIS_MANIFEST_HEADER, ...entries.files.map((f) => `F:${f}`), ...entries.dirs.map((d) => `D:${d}`), ''];
  return Buffer.from(lines.join('\r\n'), 'utf16le');
}

/** Escape a string for safe interpolation inside NSIS double-quoted literals. */
export function escapeNsisString(value: string): string {
  if (/[\r\n\0]/.test(value)) {
    throw new PayloadError(
      `Invalid control character in NSIS string: ${JSON.stringify(value)}`,
      'use a single-line path/value',
    );
  }
  return value.replace(/[$"]/g, (ch) => (ch === '$' ? '$$' : '$\\"'));
}

function fileNameOnly(filePath: string): string {
  return filePath.split(/[\\/]/).pop() ?? filePath;
}

function emitCleanManifestedFiles(phase: 'install' | 'uninstall'): string[] {
  const strStr = phase === 'install' ? 'StrStr' : 'UnStrStr';
  const tag = phase;
  return [
    `  ; Validate header and every entry in $INSTDIR\\\${MANIFEST_FILE} before deleting.`,
    '  ${If} ${FileExists} "$INSTDIR\\${MANIFEST_FILE}"',
    '    ClearErrors',
    '    FileOpen $2 "$INSTDIR\\${MANIFEST_FILE}" r',
    '    ${If} ${Errors}',
    '      ${IfNot} ${Silent}',
    '        MessageBox MB_ICONSTOP "Could not read ${APP_NAME} installation manifest. Please check folder permissions and try again."',
    '      ${EndIf}',
    '      SetErrorLevel 1',
    '      Abort',
    '    ${EndIf}',
    '    ClearErrors',
    '    FileReadUTF16LE $2 $3',
    '    ${If} ${Errors}',
    '      FileClose $2',
    '      ${IfNot} ${Silent}',
    '        MessageBox MB_ICONSTOP "The ${APP_NAME} installation manifest is empty or unreadable. Aborting to protect files in $INSTDIR."',
    '      ${EndIf}',
    '      SetErrorLevel 1',
    '      Abort',
    '    ${EndIf}',
    `    manifest_hdr_trim_${tag}:`,
    '      StrCpy $4 $3 1 -1',
    '      ${If} $4 == "$\\r"',
    '      ${OrIf} $4 == "$\\n"',
    '        StrCpy $3 $3 -1',
    `        Goto manifest_hdr_trim_${tag}`,
    '      ${EndIf}',
    `    \${If} $3 != "${NSIS_MANIFEST_HEADER}"`,
    '      FileClose $2',
    '      ${IfNot} ${Silent}',
    '        MessageBox MB_ICONSTOP "Unrecognized ${APP_NAME} installation manifest. Aborting to protect files in $INSTDIR."',
    '      ${EndIf}',
    '      SetErrorLevel 1',
    '      Abort',
    '    ${EndIf}',
    '    StrCpy $6 0',
    `    manifest_loop_${tag}:`,
    '      ClearErrors',
    '      FileReadUTF16LE $2 $3',
    '      ${If} ${Errors}',
    `        Goto manifest_done_${tag}`,
    '      ${EndIf}',
    `      manifest_trim_${tag}:`,
    '        StrCpy $4 $3 1 -1',
    '        ${If} $4 == "$\\r"',
    '        ${OrIf} $4 == "$\\n"',
    '          StrCpy $3 $3 -1',
    `          Goto manifest_trim_${tag}`,
    '        ${EndIf}',
    '      ${If} $3 == ""',
    `        Goto manifest_loop_${tag}`,
    '      ${EndIf}',
    '      StrCpy $4 $3 2',
    '      StrCpy $5 $3 "" 2',
    '      ${If} $5 == ""',
    '        FileClose $2',
    '        SetErrorLevel 1',
    '        Abort',
    '      ${EndIf}',
    '      StrCpy $7 $5 1',
    '      ${If} $7 == "\\"',
    '      ${OrIf} $7 == "/"',
    '        FileClose $2',
    '        SetErrorLevel 1',
    '        Abort',
    '      ${EndIf}',
    `      \${${strStr}} $7 $5 ".."`,
    '      ${If} $7 != ""',
    '        FileClose $2',
    '        SetErrorLevel 1',
    '        Abort',
    '      ${EndIf}',
    `      \${${strStr}} $7 $5 "*"`,
    '      ${If} $7 != ""',
    '        FileClose $2',
    '        SetErrorLevel 1',
    '        Abort',
    '      ${EndIf}',
    `      \${${strStr}} $7 $5 "?"`,
    '      ${If} $7 != ""',
    '        FileClose $2',
    '        SetErrorLevel 1',
    '        Abort',
    '      ${EndIf}',
    `      \${${strStr}} $7 $5 ":"`,
    '      ${If} $7 != ""',
    '        FileClose $2',
    '        SetErrorLevel 1',
    '        Abort',
    '      ${EndIf}',
    '      ${If} $4 == "F:"',
    '        ClearErrors',
    '        Delete "$INSTDIR\\$5"',
    '        ${If} ${FileExists} "$INSTDIR\\$5"',
    '          StrCpy $6 1',
    '        ${EndIf}',
    '      ${ElseIf} $4 == "D:"',
    '        RMDir "$INSTDIR\\$5"',
    '      ${Else}',
    '        FileClose $2',
    '        SetErrorLevel 1',
    '        Abort',
    '      ${EndIf}',
    `      Goto manifest_loop_${tag}`,
    `    manifest_done_${tag}:`,
    '    FileClose $2',
    '    ${If} $6 != 0',
    '      ${IfNot} ${Silent}',
    '        MessageBox MB_ICONSTOP "Some ${APP_NAME} files could not be removed because they are still in use. Please close any running applications and try again."',
    '      ${EndIf}',
    '      SetErrorLevel 1',
    '      Abort',
    '    ${EndIf}',
    '    Delete "$INSTDIR\\${MANIFEST_FILE}"',
    '  ${EndIf}',
  ];
}

export interface RenderNsisScriptOptions {
  version: string;
  appPath: string;
  iconPath: string;
  outFile: string;
  estimatedSizeKiB: number;
  publisher?: string;
  mode:
    | { kind: 'build-uninstaller'; uninstallerOutFile: string }
    | { kind: 'installer'; signedUninstallerPath: string | null; manifestPath: string };
}

/**
 * Pure `.nsi` script renderer so every NSIS invariant (ARM64 + build 26100 check,
 * 64-bit registry view, exact ARP key casing, install-path-scoped process stop,
 * non-empty custom dir overwrite guard, validated manifest-only cleanup,
 * shortcut AUMID property, two-pass signed uninstaller embedding) is unit-tested
 * on all OSes.
 */
export function renderNsisScript(options: RenderNsisScriptOptions): string {
  const version = assertSemver('version', options.version);
  const versionQuad = nsisVersionQuad(version);
  const publisher = options.publisher ?? APP_PUBLISHER;
  const estimatedSizeKiB = Math.max(1, Math.min(Math.trunc(options.estimatedSizeKiB), NSIS_MAX_DWORD_SIGNED));

  const isBuildUninstaller = options.mode.kind === 'build-uninstaller';
  const signedUninstallerPath = options.mode.kind === 'installer' ? options.mode.signedUninstallerPath : null;
  const hasInstallSection = !isBuildUninstaller;
  const hasUninstallSection = signedUninstallerPath === null;
  const originalFilename = isBuildUninstaller ? 'Uninstall.exe' : fileNameOnly(options.outFile);

  const lines: string[] = [
    'Unicode True',
    'ManifestDPIAware true',
    'ManifestSupportedOS all',
    'RequestExecutionLevel user',
    '',
    '!include "MUI2.nsh"',
    '!include "LogicLib.nsh"',
    '!include "x64.nsh"',
    '!include "StrFunc.nsh"',
  ];

  if (hasInstallSection) {
    lines.push('!include "Win\\COM.nsh"', '!include "Win\\Propkey.nsh"');
    lines.push('${StrStr}');
  }
  if (hasUninstallSection) {
    lines.push('${UnStrStr}');
  }

  lines.push(
    '',
    `!define APP_NAME "${escapeNsisString(APP_NAME)}"`,
    `!define APP_EXE "${escapeNsisString(WINDOWS_EXECUTABLE_NAME)}"`,
    `!define APP_ID "${escapeNsisString(APP_ID)}"`,
    `!define APP_PUBLISHER "${escapeNsisString(publisher)}"`,
    `!define APP_VERSION "${escapeNsisString(version)}"`,
    `!define APP_VERSION_QUAD "${escapeNsisString(versionQuad)}"`,
    `!define UNINSTALL_KEY "${escapeNsisString(NSIS_UNINSTALL_REG_KEY)}"`,
    `!define AUMID_KEY "${escapeNsisString(NSIS_AUMID_REG_KEY)}"`,
    `!define MANIFEST_FILE "${escapeNsisString(NSIS_MANIFEST_FILENAME)}"`,
    '',
    'Name "${APP_NAME}"',
    'InstallDir "$LOCALAPPDATA\\Programs\\${APP_NAME}"',
    'InstallDirRegKey HKCU "${UNINSTALL_KEY}" "InstallLocation"',
    '',
    '; Non-solid zlib keeps the ~1 GB CUDA payload fast to build/install,',
    '; avoids 32-bit makensis LZMA OOM, and avoids Komac solid-LZMA seek bugs.',
    'SetCompressor /FINAL zlib',
    '',
    `!define MUI_ICON "${escapeNsisString(options.iconPath)}"`,
  );

  if (hasUninstallSection) {
    lines.push(`!define MUI_UNICON "${escapeNsisString(options.iconPath)}"`);
  }

  lines.push(
    '',
    'VIProductVersion "${APP_VERSION_QUAD}"',
    'VIFileVersion "${APP_VERSION_QUAD}"',
    'VIAddVersionKey "ProductName" "${APP_NAME}"',
    'VIAddVersionKey "CompanyName" "${APP_PUBLISHER}"',
    `VIAddVersionKey "FileDescription" "${escapeNsisString(isBuildUninstaller ? `${APP_NAME} Uninstaller` : `${APP_NAME} Setup`)}"`,
    'VIAddVersionKey "FileVersion" "${APP_VERSION}"',
    'VIAddVersionKey "ProductVersion" "${APP_VERSION}"',
    `VIAddVersionKey "OriginalFilename" "${escapeNsisString(originalFilename)}"`,
    '',
  );

  if (isBuildUninstaller) {
    lines.push(
      'SilentInstall silent',
      `OutFile "${escapeNsisString(options.outFile)}"`,
      '!insertmacro MUI_PAGE_INSTFILES',
      '!insertmacro MUI_UNPAGE_CONFIRM',
      '!insertmacro MUI_UNPAGE_INSTFILES',
      '!insertmacro MUI_LANGUAGE "English"',
    );
  } else {
    lines.push(
      `OutFile "${escapeNsisString(options.outFile)}"`,
      '!define MUI_ABORTWARNING',
      '!define MUI_FINISHPAGE_RUN "$INSTDIR\\${APP_EXE}"',
      '!define MUI_FINISHPAGE_RUN_TEXT "Launch ${APP_NAME}"',
      '!insertmacro MUI_PAGE_WELCOME',
      '!insertmacro MUI_PAGE_DIRECTORY',
      '!insertmacro MUI_PAGE_INSTFILES',
      '!insertmacro MUI_PAGE_FINISH',
    );
    if (hasUninstallSection) {
      lines.push('!insertmacro MUI_UNPAGE_CONFIRM', '!insertmacro MUI_UNPAGE_INSTFILES');
    }
    lines.push('!insertmacro MUI_LANGUAGE "English"');
  }

  lines.push(
    '',
    '; Stop only Portato.exe processes whose ExecutablePath resides inside $INSTDIR',
    '; so installing or uninstalling standalone Portato never touches a Microsoft Store copy.',
    '; EncodedCommand arguments are kept well under stock NSIS 1,024-char string limits.',
    '!macro StopRunningApp LABEL_TAG',
    '  System::Call \'Kernel32::SetEnvironmentVariable(t "PI", t "$INSTDIR")\'',
    '  System::Call \'Kernel32::SetEnvironmentVariable(t "PE", t "${APP_EXE}")\'',
    `  nsExec::Exec '"$SYSDIR\\WindowsPowerShell\\v1.0\\powershell.exe" -NoP -NonI -EP Bypass -Enc ${PS_CHECK_RUNNING_B64}'`,
    '  Pop $0',
    '  ${If} $0 == 42',
    '    ${IfNot} ${Silent}',
    '      MessageBox MB_OKCANCEL|MB_ICONEXCLAMATION "${APP_NAME} is currently running from this folder and must be closed before continuing.$\\r$\\n$\\r$\\nClick OK to close ${APP_NAME} and continue, or Cancel to abort." /SD IDOK IDOK stop_ok_${LABEL_TAG}',
    '      Abort',
    '      stop_ok_${LABEL_TAG}:',
    '    ${EndIf}',
    `    nsExec::Exec '"$SYSDIR\\WindowsPowerShell\\v1.0\\powershell.exe" -NoP -NonI -EP Bypass -Enc ${PS_KILL_RUNNING_B64}'`,
    '    Pop $0',
    '    StrCpy $1 0',
    '    ${DoWhile} $1 < 10',
    '      Sleep 500',
    `      nsExec::Exec '"$SYSDIR\\WindowsPowerShell\\v1.0\\powershell.exe" -NoP -NonI -EP Bypass -Enc ${PS_CHECK_RUNNING_B64}'`,
    '      Pop $0',
    '      ${If} $0 == 43',
    '        ${ExitDo}',
    '      ${EndIf}',
    '      IntOp $1 $1 + 1',
    '    ${Loop}',
    '    ${If} $0 != 43',
    '      ${IfNot} ${Silent}',
    '        MessageBox MB_ICONSTOP "Could not close ${APP_NAME}. Please exit ${APP_NAME} manually and try again."',
    '      ${EndIf}',
    '      SetErrorLevel 1',
    '      Abort',
    '    ${EndIf}',
    '  ${ElseIf} $0 != 43',
    '    ${IfNot} ${Silent}',
    '      MessageBox MB_ICONSTOP "Could not verify whether ${APP_NAME} is running. Please close ${APP_NAME} manually and try again."',
    '    ${EndIf}',
    '    SetErrorLevel 1',
    '    Abort',
    '  ${EndIf}',
    '!macroend',
    '',
  );

  if (hasInstallSection) {
    lines.push(
      '; Refuse to overwrite an existing non-empty or unreadable folder that lacks our manifest.',
      '!macro AssertSafeInstallDir',
      '  ${IfNot} ${FileExists} "$INSTDIR\\${MANIFEST_FILE}"',
      '    System::Call \'Kernel32::GetFileAttributes(t "$INSTDIR") i.r0\'',
      '    ${If} $0 != -1',
      '      ClearErrors',
      '      FindFirst $0 $1 "$INSTDIR\\*.*"',
      '      ${If} ${Errors}',
      '        ${IfNot} ${Silent}',
      '          MessageBox MB_ICONSTOP "Could not inspect the selected folder. Please choose an empty folder or a new subfolder."',
      '        ${EndIf}',
      '        SetErrorLevel 3',
      '        Abort',
      '      ${EndIf}',
      '      safe_dir_scan:',
      '        ${If} $1 != ""',
      '        ${AndIf} $1 != "."',
      '        ${AndIf} $1 != ".."',
      '          FindClose $0',
      '          ${IfNot} ${Silent}',
      '            MessageBox MB_ICONSTOP "The selected folder is not empty and does not contain an existing ${APP_NAME} installation.$\\r$\\n$\\r$\\nPlease choose an empty folder or a new subfolder."',
      '          ${EndIf}',
      '          SetErrorLevel 3',
      '          Abort',
      '        ${EndIf}',
      '        ClearErrors',
      '        FindNext $0 $1',
      '        ${IfNot} ${Errors}',
      '          Goto safe_dir_scan',
      '        ${EndIf}',
      '      FindClose $0',
      '    ${EndIf}',
      '  ${EndIf}',
      '!macroend',
      '',
      '; Set System.AppUserModel.ID on .lnk shortcuts so pinned icons group with Portato.exe.',
      '!macro SetLnkAppUserModelId shortcut',
      '  !insertmacro ComHlpr_CreateInProcInstance ${CLSID_ShellLink} ${IID_IShellLink} r0 ""',
      '  ${If} $0 P<> 0',
      '    ${IUnknown::QueryInterface} $0 \'("${IID_IPersistFile}",.r1)\'',
      '    ${If} $1 P<> 0',
      '      ${IPersistFile::Load} $1 \'("${shortcut}", ${STGM_READWRITE})\'',
      '      ${IUnknown::QueryInterface} $0 \'("${IID_IPropertyStore}",.r2)\'',
      '      ${If} $2 P<> 0',
      '        System::Call \'Oleaut32::SysAllocString(w "${APP_ID}") p.r3\'',
      "        System::Call '*${SYSSTRUCT_PROPERTYKEY}(${PKEY_AppUserModel_ID})p.r4'",
      "        System::Call '*${SYSSTRUCT_PROPVARIANT}(${VT_BSTR},,&i${NSIS_PTR_SIZE} $3)p.r5'",
      "        ${IPropertyStore::SetValue} $2 '($4,$5)'",
      "        System::Call 'Oleaut32::SysFreeString(p r3)'",
      '        System::Free $4',
      '        System::Free $5',
      '        ${IPropertyStore::Commit} $2 ""',
      '        ${IUnknown::Release} $2 ""',
      '        ${IPersistFile::Save} $1 \'("${shortcut}",1)\'',
      '      ${EndIf}',
      '      ${IUnknown::Release} $1 ""',
      '    ${EndIf}',
      '    ${IUnknown::Release} $0 ""',
      '  ${EndIf}',
      '!macroend',
      '',
    );
  }

  if (options.mode.kind === 'build-uninstaller') {
    lines.push(
      'Function .onInit',
      `  WriteUninstaller "${escapeNsisString(options.mode.uninstallerOutFile)}"`,
      '  SetErrorLevel 0',
      '  Quit',
      'FunctionEnd',
      '',
      'Section "Install"',
      'SectionEnd',
    );
  } else {
    lines.push(
      'Function .onInit',
      '  SetRegView 64',
      '  ${IfNot} ${IsNativeARM64}',
      '    ${IfNot} ${Silent}',
      `      MessageBox MB_ICONSTOP "\${APP_NAME} requires Windows 11 (build ${NSIS_MIN_WINDOWS_BUILD} or newer) on ARM64."`,
      '    ${EndIf}',
      '    SetErrorLevel 2',
      '    Abort',
      '  ${EndIf}',
      '  ReadRegStr $0 HKLM "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion" "CurrentBuildNumber"',
      '  ${If} $0 == ""',
      '    StrCpy $0 0',
      '  ${EndIf}',
      `  IntCmp $0 ${NSIS_MIN_WINDOWS_BUILD} build_ok build_too_old build_ok`,
      '  build_too_old:',
      '    ${IfNot} ${Silent}',
      `      MessageBox MB_ICONSTOP "\${APP_NAME} requires Windows 11 24H2 (OS build ${NSIS_MIN_WINDOWS_BUILD}) or newer."`,
      '    ${EndIf}',
      '    SetErrorLevel 2',
      '    Abort',
      '  build_ok:',
      'FunctionEnd',
      '',
      'Section "Install"',
      '  SetRegView 64',
      '  !insertmacro AssertSafeInstallDir',
      '  !insertmacro StopRunningApp install',
      ...emitCleanManifestedFiles('install'),
      '',
      '  SetOutPath "$INSTDIR"',
      `  File /r "${escapeNsisString(options.appPath)}\\*.*"`,
      `  File "/oname=$INSTDIR\\\${MANIFEST_FILE}" "${escapeNsisString(options.mode.manifestPath)}"`,
      '',
    );

    if (signedUninstallerPath !== null) {
      lines.push(`  File "/oname=$INSTDIR\\Uninstall.exe" "${escapeNsisString(signedUninstallerPath)}"`);
    } else {
      lines.push('  WriteUninstaller "$INSTDIR\\Uninstall.exe"');
    }

    lines.push(
      '',
      '  CreateShortcut "$SMPROGRAMS\\${APP_NAME}.lnk" "$INSTDIR\\${APP_EXE}" "" "$INSTDIR\\${APP_EXE}" 0',
      '  !insertmacro SetLnkAppUserModelId "$SMPROGRAMS\\${APP_NAME}.lnk"',
      '  CreateShortcut "$DESKTOP\\${APP_NAME}.lnk" "$INSTDIR\\${APP_EXE}" "" "$INSTDIR\\${APP_EXE}" 0',
      '  !insertmacro SetLnkAppUserModelId "$DESKTOP\\${APP_NAME}.lnk"',
      '',
      '  WriteRegStr HKCU "${AUMID_KEY}" "DisplayName" "${APP_NAME}"',
      '  WriteRegStr HKCU "${AUMID_KEY}" "IconUri" "$INSTDIR\\${APP_EXE}"',
      '',
      '  WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayName" "${APP_NAME}"',
      '  WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayVersion" "${APP_VERSION}"',
      '  WriteRegStr HKCU "${UNINSTALL_KEY}" "Publisher" "${APP_PUBLISHER}"',
      '  WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayIcon" "$INSTDIR\\${APP_EXE},0"',
      '  WriteRegStr HKCU "${UNINSTALL_KEY}" "InstallLocation" "$INSTDIR"',
      '  WriteRegStr HKCU "${UNINSTALL_KEY}" "UninstallString" \'"$INSTDIR\\Uninstall.exe"\'',
      '  WriteRegStr HKCU "${UNINSTALL_KEY}" "QuietUninstallString" \'"$INSTDIR\\Uninstall.exe" /S\'',
      '  WriteRegStr HKCU "${UNINSTALL_KEY}" "URLInfoAbout" "https://github.com/mlx-node/mlx-node"',
      '  WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoModify" 1',
      '  WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoRepair" 1',
      `  WriteRegDWORD HKCU "\${UNINSTALL_KEY}" "EstimatedSize" ${estimatedSizeKiB}`,
      '',
      "  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'",
      'SectionEnd',
    );
  }

  if (hasUninstallSection) {
    lines.push(
      '',
      'Function un.onInit',
      '  SetRegView 64',
      'FunctionEnd',
      '',
      'Section "Uninstall"',
      '  SetRegView 64',
      '  !insertmacro StopRunningApp uninstall',
      ...emitCleanManifestedFiles('uninstall'),
      '',
      '  ; Only remove shared shortcuts and registry entries if they point to this installation.',
      '  ReadRegStr $0 HKCU "${UNINSTALL_KEY}" "InstallLocation"',
      '  ${If} $0 == "$INSTDIR"',
      '    Delete "$SMPROGRAMS\\${APP_NAME}.lnk"',
      '    Delete "$DESKTOP\\${APP_NAME}.lnk"',
      '    DeleteRegKey HKCU "${AUMID_KEY}"',
      '    DeleteRegKey HKCU "${UNINSTALL_KEY}"',
      '  ${EndIf}',
      '',
      '  Delete "$INSTDIR\\Uninstall.exe"',
      '  RMDir "$INSTDIR"',
      '',
      "  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'",
      'SectionEnd',
    );
  }

  return `${lines.join('\r\n')}\r\n`;
}

export interface BuildWindowsUninstallerOptions {
  appPath: string;
  iconPath: string;
  uninstallerPath: string;
  version: string;
  publisher?: string;
  env?: NodeJS.ProcessEnv;
}

/**
 * Pass 1 of two-pass NSIS signing: generate `Uninstall.exe` outside `appPath`
 * so it can be Authenticode-signed alongside the app binaries before Pass 2.
 */
export function buildWindowsUninstaller(options: BuildWindowsUninstallerOptions): { uninstallerPath: string } {
  const makensis = findMakensis(options.env);
  const uninstallerPath = resolve(options.uninstallerPath);
  const outDir = dirname(uninstallerPath);
  const stageDir = join(outDir, '.nsis-uninst-stage');

  mkdirSync(outDir, { recursive: true });
  rmSync(uninstallerPath, { force: true });
  rmSync(stageDir, { recursive: true, force: true });
  mkdirSync(stageDir, { recursive: true });

  const genExePath = join(stageDir, 'uninst-gen.exe');
  const scriptPath = join(stageDir, 'uninstaller.nsi');

  try {
    const script = renderNsisScript({
      version: options.version,
      appPath: resolve(options.appPath),
      iconPath: resolve(options.iconPath),
      outFile: genExePath,
      estimatedSizeKiB: computeDirSizeKiB(options.appPath),
      publisher: options.publisher,
      mode: { kind: 'build-uninstaller', uninstallerOutFile: uninstallerPath },
    });
    writeFileSync(scriptPath, script, 'utf8');
    execFileSync(makensis, ['/V3', '/INPUTCHARSET', 'UTF8', scriptPath], { stdio: 'inherit' });
    execFileSync(genExePath, [], { stdio: 'inherit' });
    if (!isExistingFile(uninstallerPath) || statSync(uninstallerPath).size === 0) {
      throw new PayloadError(
        `NSIS uninstaller generator did not produce ${uninstallerPath}`,
        'check makensis output and permissions',
      );
    }
    return { uninstallerPath };
  } finally {
    rmSync(stageDir, { recursive: true, force: true });
  }
}

export interface BuildWindowsInstallerOptions {
  appPath: string;
  iconPath: string;
  installerPath: string;
  version: string;
  signedUninstallerPath: string | null;
  publisher?: string;
  env?: NodeJS.ProcessEnv;
}

/** Pass 2 (or single-pass unsigned): compile the NSIS setup executable. */
export function buildWindowsInstaller(options: BuildWindowsInstallerOptions): {
  installerPath: string;
  sizeBytes: number;
} {
  const makensis = findMakensis(options.env);
  const appPath = resolve(options.appPath);
  const iconPath = resolve(options.iconPath);
  const installerPath = resolve(options.installerPath);
  const signedUninstallerPath = options.signedUninstallerPath === null ? null : resolve(options.signedUninstallerPath);

  if (!isExistingFile(join(appPath, WINDOWS_EXECUTABLE_NAME))) {
    throw new PayloadError(
      `Packaged Windows app executable not found: ${join(appPath, WINDOWS_EXECUTABLE_NAME)}`,
      'run package-windows.ts before building the NSIS installer',
    );
  }
  if (!isExistingFile(iconPath) || statSync(iconPath).size === 0) {
    throw new PayloadError(`Missing or empty Windows icon: ${iconPath}`, 'yarn workspace @mlx-node/desktop icons');
  }
  if (signedUninstallerPath !== null) {
    if (!isExistingFile(signedUninstallerPath) || statSync(signedUninstallerPath).size === 0) {
      throw new PayloadError(
        `Signed uninstaller not found or empty: ${signedUninstallerPath}`,
        'build and sign Uninstall.exe in Pass 1 before calling package-windows.ts --from-app --signed --uninstaller',
      );
    }
  }

  const outDir = dirname(installerPath);
  const stageDir = join(outDir, '.nsis-stage');
  mkdirSync(outDir, { recursive: true });
  rmSync(installerPath, { force: true });
  rmSync(stageDir, { recursive: true, force: true });
  mkdirSync(stageDir, { recursive: true });

  const manifestPath = join(stageDir, NSIS_MANIFEST_FILENAME);
  const scriptPath = join(stageDir, 'installer.nsi');
  try {
    const manifestEntries = collectInstallManifest(appPath);
    writeFileSync(manifestPath, renderInstallManifestBuffer(manifestEntries));
    const script = renderNsisScript({
      version: options.version,
      appPath,
      iconPath,
      outFile: installerPath,
      estimatedSizeKiB: computeDirSizeKiB(appPath),
      publisher: options.publisher,
      mode: { kind: 'installer', signedUninstallerPath, manifestPath },
    });
    writeFileSync(scriptPath, script, 'utf8');
    execFileSync(makensis, ['/V3', '/INPUTCHARSET', 'UTF8', scriptPath], { stdio: 'inherit' });
  } finally {
    rmSync(stageDir, { recursive: true, force: true });
  }

  const verified = verifyWindowsInstaller({ installerPath, expectedVersion: options.version });
  if (verified.sizeBytes > NSIS_WARN_BYTES) {
    console.warn(
      `WARNING: Windows NSIS installer is ${(verified.sizeBytes / 1024 ** 3).toFixed(2)} GiB ` +
        `(approaching NSIS's 2.00 GiB compressed datablock ceiling).`,
    );
  }
  return { installerPath, sizeBytes: verified.sizeBytes };
}

/** Verify the built Windows NSIS installer exists, matches expected version, stays under 2 GiB, and carries MZ + NullsoftInst headers. */
export function verifyWindowsInstaller(options: { installerPath: string; expectedVersion: string }): {
  sizeBytes: number;
} {
  const expectedVersion = assertSemver('expectedVersion', options.expectedVersion);
  const installerPath = options.installerPath;
  if (!existsSync(installerPath) || !isExistingFile(installerPath)) {
    throw new PayloadError(`Windows installer not found: ${installerPath}`, 'repackage the Windows installer');
  }
  const parsedVersion = versionFromWindowsInstallerPath(installerPath);
  if (parsedVersion !== expectedVersion) {
    throw new ReleaseVersionError(
      `Windows installer filename ${basename(installerPath)} does not match expected version ${expectedVersion}`,
      'name the Windows installer Portato-<version>-win32-arm64-setup.exe',
    );
  }
  const sizeBytes = statSync(installerPath).size;
  if (sizeBytes === 0) {
    throw new PayloadError(`Windows installer is empty: ${installerPath}`, 'repackage the Windows installer');
  }
  if (sizeBytes > NSIS_MAX_INSTALLER_BYTES) {
    throw new PayloadError(
      `Windows installer exceeds NSIS 2 GiB limit (${sizeBytes} bytes): ${installerPath}`,
      'switch NSIS compression to non-solid lzma or use a web/external-payload installer',
    );
  }

  const readLen = Math.min(sizeBytes, NSIS_HEADER_SCAN_BYTES);
  const head = Buffer.allocUnsafe(readLen);
  const fd = openSync(installerPath, 'r');
  try {
    let offset = 0;
    while (offset < readLen) {
      const n = readSync(fd, head, offset, readLen - offset, offset);
      if (n === 0) break;
      offset += n;
    }
    const slice = head.subarray(0, offset);
    if (slice.length < 2 || slice[0] !== 0x4d || slice[1] !== 0x5a) {
      throw new PayloadError(
        `Windows installer is not a PE executable (missing MZ header): ${installerPath}`,
        'repackage the Windows installer with makensis.exe',
      );
    }
    if (slice.indexOf(NSIS_HEADER_SIGNATURE) === -1) {
      throw new PayloadError(
        `Windows installer is missing the NSIS NullsoftInst header: ${installerPath}`,
        'repackage the Windows installer with makensis.exe',
      );
    }
  } finally {
    closeSync(fd);
  }

  return { sizeBytes };
}
