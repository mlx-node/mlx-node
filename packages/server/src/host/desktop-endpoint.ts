/** Native-free discovery of the desktop's existing inference host. */
import { execFile } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { chmod, lstat, mkdir, open, readFile, rename, unlink, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { promisify } from 'node:util';

import { resolveMlxNodeHome } from './paths.js';

const execFileAsync = promisify(execFile);

const WINDOWS_SYSTEM_SID = 'S-1-5-18';
const WINDOWS_ADMINS_SID = 'S-1-5-32-544';
const WINDOWS_ACL_TIMEOUT_MS = 10_000;
const NUMERIC_SID_RE = /^S-1-\d+(?:-\d+)*$/;
const DOMAIN_SID_PREFIX_RE = /^(S-1-5-21-\d+-\d+-\d+)-\d+$/;

/** Standard well-known SDDL 2-letter trustee aliases (`[MS-DTYP] 2.5.1.1`) mapped to numeric SIDs. */
const SDDL_TRUSTEE_ALIASES: Readonly<Record<string, string>> = {
  WD: 'S-1-1-0',
  CO: 'S-1-3-0',
  CG: 'S-1-3-1',
  OW: 'S-1-3-4',
  NU: 'S-1-5-2',
  IU: 'S-1-5-4',
  SU: 'S-1-5-6',
  AN: 'S-1-5-7',
  ED: 'S-1-5-9',
  PS: 'S-1-5-10',
  AU: 'S-1-5-11',
  RC: 'S-1-5-12',
  SY: WINDOWS_SYSTEM_SID,
  LS: 'S-1-5-19',
  NS: 'S-1-5-20',
  WR: 'S-1-5-33',
  BA: WINDOWS_ADMINS_SID,
  BU: 'S-1-5-32-545',
  BG: 'S-1-5-32-546',
  PU: 'S-1-5-32-547',
  AO: 'S-1-5-32-548',
  SO: 'S-1-5-32-549',
  PO: 'S-1-5-32-550',
  BO: 'S-1-5-32-551',
  RE: 'S-1-5-32-552',
  RU: 'S-1-5-32-554',
  RD: 'S-1-5-32-555',
  NO: 'S-1-5-32-556',
  MU: 'S-1-5-32-558',
  LU: 'S-1-5-32-559',
  IS: 'S-1-5-32-568',
  CY: 'S-1-5-32-569',
  ER: 'S-1-5-32-573',
  CD: 'S-1-5-32-574',
  RA: 'S-1-5-32-575',
  ES: 'S-1-5-32-576',
  MS: 'S-1-5-32-577',
  HA: 'S-1-5-32-578',
  AA: 'S-1-5-32-579',
  RM: 'S-1-5-32-580',
  UD: 'S-1-5-84-0-0-0-0-0',
  AC: 'S-1-15-2-1',
  LW: 'S-1-16-4096',
  ME: 'S-1-16-8192',
  MP: 'S-1-16-8448',
  HI: 'S-1-16-12288',
  SI: 'S-1-16-16384',
};

/** Domain-relative SDDL 2-letter aliases (`[MS-DTYP] 2.5.1.1`) mapped to their RID. */
const SDDL_DOMAIN_RELATIVE_RIDS: Readonly<Record<string, number>> = {
  RO: 498,
  LA: 500,
  LG: 501,
  DA: 512,
  DU: 513,
  DG: 514,
  DC: 515,
  DD: 516,
  CA: 517,
  SA: 518,
  EA: 519,
  PA: 520,
  CN: 522,
  RS: 553,
};

/** Result of validating a Windows SDDL security descriptor for per-user privacy. */
export interface WindowsSddlCheck {
  ok: boolean;
  /** True when the DACL has the SDDL Protected (`P`) flag blocking parent inheritance. */
  protectedDacl: boolean;
}

function normalizeWindowsSid(sid: string, domainPrefix?: string): string {
  const upper = sid.trim().toUpperCase();
  const wellKnown = SDDL_TRUSTEE_ALIASES[upper];
  if (wellKnown !== undefined) return wellKnown;
  if (domainPrefix !== undefined) {
    const rid = SDDL_DOMAIN_RELATIVE_RIDS[upper];
    if (rid !== undefined) return `${domainPrefix}-${rid}`;
  }
  return upper;
}

function windowsDomainPrefix(sid: string): string | undefined {
  return DOMAIN_SID_PREFIX_RE.exec(sid.trim().toUpperCase())?.[1];
}

function parseDaclFlags(raw: string): { valid: boolean; protectedDacl: boolean } {
  let i = 0;
  let protectedDacl = false;
  while (i < raw.length) {
    if (raw.startsWith('AI', i) || raw.startsWith('AR', i)) {
      i += 2;
    } else if (raw[i] === 'P') {
      protectedDacl = true;
      i += 1;
    } else {
      return { valid: false, protectedDacl: false };
    }
  }
  return { valid: true, protectedDacl };
}

interface ParsedWindowsSddlOwner {
  owner: string;
  explicitTrustees: string[];
}

function parseWindowsSddlOwnerAndExplicitTrustees(sddl: string, userSid: string): ParsedWindowsSddlOwner | null {
  const text = sddl.trim();
  if (!text.startsWith('O:')) return null;
  const domainPrefix = windowsDomainPrefix(userSid);

  let i = 2;
  const ownerStart = i;
  while (i < text.length && !(text[i + 1] === ':' && (text[i] === 'G' || text[i] === 'D' || text[i] === 'S'))) {
    if (text[i] === '(' || text[i] === ')') return null;
    i += 1;
  }
  const owner = normalizeWindowsSid(text.slice(ownerStart, i), domainPrefix);
  if (!owner) return null;

  if (text.startsWith('G:', i)) {
    i += 2;
    while (i < text.length && !(text[i + 1] === ':' && (text[i] === 'D' || text[i] === 'S'))) {
      if (text[i] === '(' || text[i] === ')') return null;
      i += 1;
    }
  }

  const explicitTrustees = new Set<string>();
  if (text.startsWith('D:', i)) {
    i += 2;
    while (i < text.length && text[i] !== '(' && !(text[i] === 'S' && text[i + 1] === ':')) {
      i += 1;
    }
    while (i < text.length && text[i] === '(') {
      i += 1;
      const aceStart = i;
      let depth = 1;
      while (i < text.length && depth > 0) {
        if (text[i] === '(') depth += 1;
        else if (text[i] === ')') depth -= 1;
        if (depth > 0) i += 1;
      }
      if (depth !== 0) return null;
      const ace = text.slice(aceStart, i);
      i += 1;

      const parts = ace.split(';');
      if (parts.length >= 6) {
        const flags = parts[1] ?? '';
        const rawTrustee = parts[5] ?? '';
        // Only explicit (non-inherited) ACEs need `/remove`; `/inheritance:r` drops inherited ACEs.
        if (!flags.includes('ID')) {
          const resolved = normalizeWindowsSid(rawTrustee, domainPrefix);
          if (NUMERIC_SID_RE.test(resolved)) {
            explicitTrustees.add(resolved);
          }
        }
      }
    }
  }

  return { owner, explicitTrustees: [...explicitTrustees] };
}

/**
 * Pure SDDL validator for per-user Windows credential files and directories.
 *
 * Requires:
 * - An `O:` owner in `{ currentUserSid, SYSTEM (SY), Administrators (BA) }` (unprivileged local accounts cannot
 *   create objects owned by `SY` or `BA`, and accepting `BA`/`SY` ensures an elevated launch followed by a normal
 *   unelevated launch or reader does not lock out the same user).
 * - A non-empty `D:` DACL with no unknown flags or ACE types.
 * - Every Allow (`A`) ACE restricted strictly to `{ currentUserSid, SYSTEM (SY), Administrators (BA) }`.
 *
 * Fails closed on null/empty DACLs, unknown ACE types (including conditional/object ACEs), malformed syntax, or any
 * grant to another local account or group (`WD`, `AU`, `BU`, `CodexSandboxUsers`, etc.).
 */
export function isPrivateWindowsSddl(sddl: string, currentUserSid: string): WindowsSddlCheck {
  const text = sddl.trim();
  const userSid = normalizeWindowsSid(currentUserSid);
  const domainPrefix = windowsDomainPrefix(userSid);
  if (!userSid || !text.startsWith('O:')) {
    return { ok: false, protectedDacl: false };
  }

  let i = 2;
  const ownerStart = i;
  while (i < text.length && !(text[i + 1] === ':' && (text[i] === 'G' || text[i] === 'D' || text[i] === 'S'))) {
    if (text[i] === '(' || text[i] === ')') return { ok: false, protectedDacl: false };
    i += 1;
  }
  const actualOwner = normalizeWindowsSid(text.slice(ownerStart, i), domainPrefix);
  const allowedTrustees = new Set([userSid, WINDOWS_SYSTEM_SID, WINDOWS_ADMINS_SID]);
  if (!actualOwner || !allowedTrustees.has(actualOwner)) {
    return { ok: false, protectedDacl: false };
  }

  if (text.startsWith('G:', i)) {
    i += 2;
    while (i < text.length && !(text[i + 1] === ':' && (text[i] === 'D' || text[i] === 'S'))) {
      if (text[i] === '(' || text[i] === ')') return { ok: false, protectedDacl: false };
      i += 1;
    }
  }

  if (!text.startsWith('D:', i)) {
    return { ok: false, protectedDacl: false };
  }
  i += 2;

  const flagsStart = i;
  while (i < text.length && text[i] !== '(' && !(text[i] === 'S' && text[i + 1] === ':')) {
    i += 1;
  }
  const flagStr = text.slice(flagsStart, i);
  const { valid: flagsValid, protectedDacl } = parseDaclFlags(flagStr);
  if (!flagsValid) return { ok: false, protectedDacl: false };

  let hasAllowedAce = false;

  while (i < text.length && text[i] === '(') {
    i += 1;
    const aceStart = i;
    while (i < text.length && text[i] !== ')') {
      // Reject nested parentheses (e.g. conditional ACEs).
      if (text[i] === '(') return { ok: false, protectedDacl: false };
      i += 1;
    }
    if (i >= text.length || text[i] !== ')') return { ok: false, protectedDacl: false };
    const ace = text.slice(aceStart, i);
    i += 1;

    const parts = ace.split(';');
    if (parts.length !== 6) return { ok: false, protectedDacl: false };
    const [type, , , , , rawTrustee] = parts as [string, string, string, string, string, string];
    const trustee = normalizeWindowsSid(rawTrustee, domainPrefix);
    if (!trustee) return { ok: false, protectedDacl: false };

    if (type === 'A') {
      if (!allowedTrustees.has(trustee)) return { ok: false, protectedDacl: false };
      hasAllowedAce = true;
    } else if (type === 'D') {
      // Explicit deny ACEs only restrict access.
    } else {
      return { ok: false, protectedDacl: false };
    }
  }

  if (!hasAllowedAce) return { ok: false, protectedDacl: false };
  if (i < text.length && !text.startsWith('S:', i)) {
    return { ok: false, protectedDacl: false };
  }

  return { ok: true, protectedDacl };
}

function windowsSystemRoot(): string {
  for (const [key, value] of Object.entries(process.env)) {
    if (key.toUpperCase() === 'SYSTEMROOT' && value) return value;
  }
  return 'C:\\Windows';
}

function windowsSystem32Exe(name: string): string {
  return join(windowsSystemRoot(), 'System32', name);
}

function windowsPowerShellPath(): string {
  return join(windowsSystemRoot(), 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe');
}

function buildWindowsToolEnv(extraEnv?: Record<string, string>): NodeJS.ProcessEnv {
  if (!extraEnv) return process.env;
  const env: NodeJS.ProcessEnv = { ...process.env };
  for (const [key, value] of Object.entries(extraEnv)) {
    const upper = key.toUpperCase();
    for (const existing of Object.keys(env)) {
      if (existing !== key && existing.toUpperCase() === upper) {
        delete env[existing];
      }
    }
    env[key] = value;
  }
  return env;
}

async function execWindowsTool(
  exe: string,
  args: string[],
  extraEnv?: Record<string, string>,
): Promise<string> {
  const { stdout } = await execFileAsync(exe, args, {
    env: buildWindowsToolEnv(extraEnv),
    windowsHide: true,
    timeout: WINDOWS_ACL_TIMEOUT_MS,
    maxBuffer: 1024 * 1024,
  });
  return stdout;
}

async function runWindowsPowerShell(script: string, extraEnv: Record<string, string>): Promise<string> {
  const encoded = Buffer.from(script, 'utf16le').toString('base64');
  // Pin PSModulePath strictly to Windows PowerShell 5.1's built-in system modules so an inherited PowerShell 7
  // (`pwsh`) PSModulePath cannot break `Get-Acl` with `FormatXmlUpdateException` (`CouldNotAutoloadMatchingModule`),
  // and no user or OneDrive module directory can shadow built-in cmdlets.
  const systemModules = join(windowsSystemRoot(), 'System32', 'WindowsPowerShell', 'v1.0', 'Modules');
  return execWindowsTool(
    windowsPowerShellPath(),
    ['-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
    { ...extraEnv, PSModulePath: systemModules },
  );
}

let cachedWindowsUserSid: Promise<string> | null = null;

/**
 * Resolve the current Windows process user SID via native `whoami.exe`
 * (zero PowerShell / `.NET` dependency, pure ASCII SID extraction). Cached per process.
 */
function getWindowsUserSid(): Promise<string> {
  if (cachedWindowsUserSid !== null) return cachedWindowsUserSid;
  const pending = (async (): Promise<string> => {
    const userOut = await execWindowsTool(windowsSystem32Exe('whoami.exe'), ['/user', '/fo', 'csv', '/nh']);
    const userSidMatch = /(?:^|,)"(S-1-\d+(?:-\d+)*)"\s*$/m.exec(userOut.trim());
    const userSid = userSidMatch?.[1]?.toUpperCase();
    if (!userSid) {
      throw new Error('Failed to resolve current Windows user SID.');
    }
    return userSid;
  })().catch((error: unknown) => {
    cachedWindowsUserSid = null;
    throw error;
  });
  cachedWindowsUserSid = pending;
  return pending;
}

/**
 * Constrained-Language-safe Windows PowerShell inspection scripts.
 *
 * Uses only built-in signed cmdlets (`Get-Item`, `Get-Acl`, `Write-Output`) and property reads
 * (zero `.NET` method invocations or `New-Object` calls) so ACL inspection works under WDAC / AppLocker
 * PowerShell Constrained Language mode as well as Full Language mode.
 *
 * @internal Exported so Windows tests can verify execution directly under `ConstrainedLanguage`.
 */
export const WINDOWS_ACL_INSPECT_SCRIPTS = Object.freeze({
  path: `
$ErrorActionPreference = 'Stop'
$target = $env:MLX_ENDPOINT_ACL_PATH
$item = Get-Item -LiteralPath $target -Force
if ($item.LinkType -or ($item.Attributes -like '*ReparsePoint*')) {
  throw 'Target is a reparse point.'
}
$acl = Get-Acl -LiteralPath $target
Write-Output $acl.Sddl
`.trim(),
  pair: `
$ErrorActionPreference = 'Stop'
$filePath = $env:MLX_ENDPOINT_ACL_FILE
$dirPath = $env:MLX_ENDPOINT_ACL_DIR
$fileItem = Get-Item -LiteralPath $filePath -Force
$dirItem = Get-Item -LiteralPath $dirPath -Force
if ($fileItem.LinkType -or ($fileItem.Attributes -like '*ReparsePoint*')) {
  throw 'Endpoint file is a reparse point.'
}
if ($dirItem.LinkType -or ($dirItem.Attributes -like '*ReparsePoint*')) {
  throw 'Endpoint directory is a reparse point.'
}
$fileAcl = Get-Acl -LiteralPath $filePath
$dirAcl = Get-Acl -LiteralPath $dirPath
Write-Output $fileAcl.Sddl
Write-Output $dirAcl.Sddl
`.trim(),
});

async function inspectWindowsPathSddl(resolvedPath: string): Promise<string> {
  const stdout = await runWindowsPowerShell(WINDOWS_ACL_INSPECT_SCRIPTS.path, {
    MLX_ENDPOINT_ACL_PATH: resolvedPath,
  });
  const sddl = stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find(Boolean);
  if (!sddl) {
    throw new Error('Failed to read Windows SDDL for desktop endpoint path.');
  }
  return sddl;
}

async function hardenWindowsPath(targetPath: string, kind: 'dir' | 'file', verifyAfter = true): Promise<void> {
  const resolvedPath = resolve(targetPath);
  const [userSid, beforeSddl] = await Promise.all([getWindowsUserSid(), inspectWindowsPathSddl(resolvedPath)]);
  const parsed = parseWindowsSddlOwnerAndExplicitTrustees(beforeSddl, userSid);
  if (
    parsed === null ||
    (parsed.owner !== userSid && parsed.owner !== WINDOWS_SYSTEM_SID && parsed.owner !== WINDOWS_ADMINS_SID)
  ) {
    throw new Error(`Desktop endpoint ${kind} is not owned by the current user.`);
  }

  const allowedSids = [WINDOWS_SYSTEM_SID, WINDOWS_ADMINS_SID];
  if (userSid !== WINDOWS_SYSTEM_SID && userSid !== WINDOWS_ADMINS_SID) {
    allowedSids.push(userSid);
  }
  const grantPerm = kind === 'dir' ? '(OI)(CI)(F)' : '(F)';
  // Mutate the DACL via native `icacls.exe /L` (does not follow symlinks/junctions, works for non-admin
  // accounts and under PowerShell Constrained Language policy). Removing explicit trustees before `/grant:r`
  // wipes any pre-existing Allow or Deny ACEs for other accounts/groups as well as stale Deny ACEs.
  const args = [resolvedPath, '/L', '/inheritance:r'];
  for (const sid of parsed.explicitTrustees) {
    args.push('/remove', `*${sid}`);
  }
  for (const sid of allowedSids) {
    args.push('/grant:r', `*${sid}:${grantPerm}`);
  }
  args.push('/q');
  await execWindowsTool(windowsSystem32Exe('icacls.exe'), args);

  if (verifyAfter) {
    const afterSddl = await inspectWindowsPathSddl(resolvedPath);
    const check = isPrivateWindowsSddl(afterSddl, userSid);
    if (!check.ok || !check.protectedDacl) {
      throw new Error(`Hardened Windows ACL verification failed on ${kind}.`);
    }
  }
}

async function inspectWindowsPrivatePair(
  resolvedFile: string,
  resolvedDir: string,
): Promise<{ fileCheck: WindowsSddlCheck; dirCheck: WindowsSddlCheck } | null> {
  const [userSid, stdout] = await Promise.all([
    getWindowsUserSid(),
    runWindowsPowerShell(WINDOWS_ACL_INSPECT_SCRIPTS.pair, {
      MLX_ENDPOINT_ACL_FILE: resolvedFile,
      MLX_ENDPOINT_ACL_DIR: resolvedDir,
    }),
  ]);
  const lines = stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length < 2) return null;
  const [fileSddl, dirSddl] = lines as [string, string];
  return {
    fileCheck: isPrivateWindowsSddl(fileSddl, userSid),
    dirCheck: isPrivateWindowsSddl(dirSddl, userSid),
  };
}

async function verifyWindowsProtectedPair(filePath: string, dirPath: string): Promise<boolean> {
  try {
    const result = await inspectWindowsPrivatePair(resolve(filePath), resolve(dirPath));
    return (
      result !== null &&
      result.fileCheck.ok &&
      result.fileCheck.protectedDacl &&
      result.dirCheck.ok &&
      result.dirCheck.protectedDacl
    );
  } catch {
    return false;
  }
}

async function verifyWindowsEndpointAcl(filePath: string, dirPath: string): Promise<boolean> {
  try {
    const result = await inspectWindowsPrivatePair(resolve(filePath), resolve(dirPath));
    return result !== null && result.fileCheck.ok && result.dirCheck.ok;
  } catch {
    return false;
  }
}

export interface DesktopEndpoint {
  version: 1;
  pid: number;
  url: string;
  token: string;
  models: DesktopEndpointModel[];
}

/** A served model: its id, its path, and the base and build the app's model picker groups by. */
export interface DesktopEndpointModel {
  name: string;
  path: string;
  base: string;
  variant: string;
  format: 'gguf' | 'mlx';
  modelType?: string;
  kind?: 'chat' | 'decision';
  /** Discovery's image support: before a load, the best guess the app has. */
  images?: boolean;
}

function validModel(model: unknown): boolean {
  if (typeof model !== 'object' || model === null) return false;
  const { name, path, base, variant, format, modelType, kind, images } = model as Record<string, unknown>;
  return (
    typeof name === 'string' &&
    typeof path === 'string' &&
    isAbsolute(path) &&
    typeof base === 'string' &&
    base !== '' &&
    typeof variant === 'string' &&
    variant !== '' &&
    (format === 'gguf' || format === 'mlx') &&
    (modelType === undefined || (typeof modelType === 'string' && modelType !== '')) &&
    (kind === undefined || kind === 'chat' || kind === 'decision') &&
    (images === undefined || typeof images === 'boolean')
  );
}

export function desktopEndpointPath(): string {
  return join(resolveMlxNodeHome(), 'desktop', 'inference.json');
}

function validate(value: DesktopEndpoint): void {
  const url = new URL(value.url);
  if (
    value.version !== 1 ||
    !Number.isSafeInteger(value.pid) ||
    value.pid < 1 ||
    !Array.isArray(value.models) ||
    !value.models.every(validModel) ||
    value.url !== url.origin ||
    typeof value.token !== 'string' ||
    !value.token ||
    url.protocol !== 'http:' ||
    url.hostname !== '127.0.0.1' ||
    !url.port ||
    url.username ||
    url.password ||
    url.pathname !== '/' ||
    url.search ||
    url.hash
  )
    throw new Error('Invalid desktop inference endpoint. Restart Portato.');
}

/** Atomic publication; an older generation must never remove a newer one's credentials. */
export async function publishDesktopEndpoint(
  endpoint: DesktopEndpoint,
  path = desktopEndpointPath(),
): Promise<() => Promise<void>> {
  validate(endpoint);
  const dir = dirname(path);
  await mkdir(dir, { recursive: true, mode: 0o700 });
  let dirBefore: { dev: bigint; ino: bigint } | undefined;
  if (process.platform === 'win32') {
    const dirInfo = await lstat(dir, { bigint: true });
    if (!dirInfo.isDirectory()) {
      throw new Error('Desktop inference endpoint directory is not a regular directory.');
    }
    await hardenWindowsPath(dir, 'dir');
    const hardenedDir = await lstat(dir, { bigint: true });
    if (!hardenedDir.isDirectory() || hardenedDir.dev !== dirInfo.dev || hardenedDir.ino !== dirInfo.ino) {
      throw new Error('Desktop inference endpoint directory changed during hardening.');
    }
    dirBefore = hardenedDir;
  } else {
    await chmod(dir, 0o700);
  }
  const temp = `${path}.${randomUUID()}.tmp`;
  try {
    if (process.platform === 'win32') {
      // Hold an open handle to the newly created empty temp file across ACL hardening and verification.
      // On Windows NTFS, an open handle pins the directory and every ancestor against rename/delete while
      // we harden `temp`, verify both `temp` and `dir` have Protected private DACLs, and confirm the path
      // still resolves to this exact handle before writing the bearer token directly to that handle.
      const handle = await open(temp, 'wx+', 0o600);
      try {
        const created = await handle.stat({ bigint: true });
        await hardenWindowsPath(temp, 'file', false);
        const [tempAfter, dirAfter] = await Promise.all([lstat(temp, { bigint: true }), lstat(dir, { bigint: true })]);
        if (
          !tempAfter.isFile() ||
          tempAfter.dev !== created.dev ||
          tempAfter.ino !== created.ino ||
          !dirAfter.isDirectory() ||
          dirBefore === undefined ||
          dirAfter.dev !== dirBefore.dev ||
          dirAfter.ino !== dirBefore.ino
        ) {
          throw new Error('Desktop inference endpoint path changed before write.');
        }
        if (!(await verifyWindowsProtectedPair(temp, dir))) {
          throw new Error('Hardened Windows ACL verification failed on desktop endpoint.');
        }
        const [tempFinal, dirFinal, handleFinal] = await Promise.all([
          lstat(temp, { bigint: true }),
          lstat(dir, { bigint: true }),
          handle.stat({ bigint: true }),
        ]);
        if (
          !tempFinal.isFile() ||
          tempFinal.dev !== created.dev ||
          tempFinal.ino !== created.ino ||
          tempFinal.ctimeNs !== tempAfter.ctimeNs ||
          // Note: do not require `dirFinal.ctimeNs === dirAfter.ctimeNs` because sibling files in the same
          // `desktop/` directory (such as `supervisor.json`) legitimately advance the directory's `ctimeNs`.
          !dirFinal.isDirectory() ||
          dirFinal.dev !== dirBefore.dev ||
          dirFinal.ino !== dirBefore.ino ||
          handleFinal.dev !== created.dev ||
          handleFinal.ino !== created.ino ||
          handleFinal.ctimeNs !== tempFinal.ctimeNs ||
          handleFinal.size !== 0n
        ) {
          throw new Error('Desktop inference endpoint temp file changed before write.');
        }
        await handle.writeFile(JSON.stringify(endpoint), 'utf8');
      } finally {
        await handle.close();
      }
    } else {
      await writeFile(temp, JSON.stringify(endpoint), { mode: 0o600, flag: 'wx' });
    }
    await rename(temp, path);
  } finally {
    await unlink(temp).catch(() => {});
  }
  return async () => {
    try {
      const current = JSON.parse(await readFile(path, 'utf8')) as DesktopEndpoint;
      if (current.token === endpoint.token && current.pid === endpoint.pid) await unlink(path);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    }
  };
}

function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === 'EPERM';
  }
}

/**
 * The published endpoint, or null unless the file is a regular file owned by this user with private permissions
 * (POSIX `0o600` on Unix; verified per-user NTFS owner + DACL on file and containing directory on Windows), holds a
 * valid endpoint, and names a live process. Never throws.
 */
export async function readDesktopEndpoint(path = desktopEndpointPath()): Promise<DesktopEndpoint | null> {
  try {
    let raw: string;
    if (process.platform === 'win32') {
      const dir = dirname(path);
      const [fileBefore, dirBefore] = await Promise.all([lstat(path, { bigint: true }), lstat(dir, { bigint: true })]);
      if (!fileBefore.isFile() || !dirBefore.isDirectory()) return null;
      const handle = await open(path, 'r');
      try {
        const [opened, dirPinned] = await Promise.all([handle.stat({ bigint: true }), lstat(dir, { bigint: true })]);
        if (
          opened.dev !== fileBefore.dev ||
          opened.ino !== fileBefore.ino ||
          opened.ctimeNs !== fileBefore.ctimeNs ||
          opened.mtimeNs !== fileBefore.mtimeNs ||
          opened.size !== fileBefore.size ||
          !dirPinned.isDirectory() ||
          dirPinned.dev !== dirBefore.dev ||
          dirPinned.ino !== dirBefore.ino
        ) {
          return null;
        }
        if (!(await verifyWindowsEndpointAcl(path, dir))) return null;
        const [fileAfter, dirAfter, handleAfter] = await Promise.all([
          lstat(path, { bigint: true }),
          lstat(dir, { bigint: true }),
          handle.stat({ bigint: true }),
        ]);
        if (
          !fileAfter.isFile() ||
          fileAfter.dev !== opened.dev ||
          fileAfter.ino !== opened.ino ||
          fileAfter.ctimeNs !== opened.ctimeNs ||
          fileAfter.mtimeNs !== opened.mtimeNs ||
          fileAfter.size !== opened.size ||
          !dirAfter.isDirectory() ||
          dirAfter.dev !== dirPinned.dev ||
          dirAfter.ino !== dirPinned.ino ||
          handleAfter.dev !== opened.dev ||
          handleAfter.ino !== opened.ino ||
          handleAfter.ctimeNs !== opened.ctimeNs ||
          handleAfter.mtimeNs !== opened.mtimeNs ||
          handleAfter.size !== opened.size
        ) {
          return null;
        }
        raw = await handle.readFile('utf8');
      } finally {
        await handle.close();
      }
    } else {
      const info = await lstat(path);
      if (!info.isFile() || (info.mode & 0o077) !== 0) return null;
      if (typeof process.getuid === 'function' && info.uid !== process.getuid()) return null;
      raw = await readFile(path, 'utf8');
    }
    const value = JSON.parse(raw) as DesktopEndpoint;
    if (typeof value !== 'object' || value === null) return null;
    validate(value);
    return pidAlive(value.pid) ? value : null;
  } catch {
    return null;
  }
}
