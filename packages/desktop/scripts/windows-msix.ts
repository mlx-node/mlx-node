/**
 * Windows ARM64 MSIX packaging (`MakePri.exe` + `MakeAppx.exe`) and verification.
 *
 * Produces an unsigned `.msix` ready for Microsoft Store Partner Center upload.
 * Partner Center signs the package with Microsoft's Store certificate during
 * ingestion, so no local Authenticode certificate is needed here.
 */

import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { delimiter, dirname, join, relative } from 'node:path';

import { PayloadError } from './payload.js';
import { msixPackageVersion } from './release-version.js';
import { WINDOWS_APP_RUNTIME_DIRS, WINDOWS_APP_RUNTIME_FILES } from './windows-payload.js';

export interface MsixIdentity {
  identityName: string;
  publisher: string;
  publisherDisplayName: string;
  displayName: string;
  description: string;
}

/**
 * Verified Microsoft Partner Center identity for the reserved Store app `Portato`.
 * Individual fields can be overridden through `MSIX_*` environment variables.
 */
export const DEFAULT_MSIX_IDENTITY: Readonly<MsixIdentity> = {
  identityName: 'LongYinan.Portato',
  publisher: 'CN=F04DA088-C2D4-46E8-921E-82D147E19F3F',
  publisherDisplayName: 'LongYinan',
  displayName: 'Portato',
  description: 'Local AI inference and control panel for Apple Silicon and NVIDIA CUDA',
};

/**
 * Microsoft Store Windows App Certification Kit (WACK) Desktop Bridge test 2.1
 * rejects any manifest visual asset that is 204,800 bytes (200 KiB) or larger.
 */
export const MAX_MSIX_ASSET_BYTES = 204_800;

/**
 * Minimum MSIX visual assets expected in `packages/desktop/build/msix/`:
 * every base file referenced by `AppxManifest.xml` plus key scale and unplated
 * taskbar variants indexed into `resources.pri`.
 */
export const REQUIRED_MSIX_ASSETS = [
  'StoreLogo.png',
  'StoreLogo.scale-100.png',
  'StoreLogo.scale-200.png',
  'Square44x44Logo.png',
  'Square44x44Logo.scale-100.png',
  'Square44x44Logo.scale-200.png',
  'Square44x44Logo.targetsize-24.png',
  'Square44x44Logo.targetsize-24_altform-unplated.png',
  'Square44x44Logo.targetsize-48_altform-unplated.png',
  'Square44x44Logo.targetsize-24_altform-lightunplated.png',
  'Square44x44Logo.targetsize-48_altform-lightunplated.png',
  'Square71x71Logo.png',
  'Square71x71Logo.scale-100.png',
  'Square71x71Logo.scale-200.png',
  'Square150x150Logo.png',
  'Square150x150Logo.scale-100.png',
  'Square150x150Logo.scale-200.png',
  'Wide310x150Logo.png',
  'Wide310x150Logo.scale-100.png',
  'Wide310x150Logo.scale-200.png',
  'Square310x310Logo.png',
  'Square310x310Logo.scale-100.png',
  'Square310x310Logo.scale-200.png',
  'SplashScreen.png',
  'SplashScreen.scale-100.png',
  'SplashScreen.scale-200.png',
] as const;

const MANIFEST_BASE_ASSETS = [
  'StoreLogo.png',
  'Square44x44Logo.png',
  'Square71x71Logo.png',
  'Square150x150Logo.png',
  'Wide310x150Logo.png',
  'Square310x310Logo.png',
  'SplashScreen.png',
] as const;

const IDENTITY_NAME_PATTERN = /^[A-Za-z0-9.-]{3,50}$/;
const VERSION_DIR_PATTERN = /^\d+\.\d+\.\d+\.\d+$/;
const DEFAULT_WINDOWS_KITS_BIN = 'C:\\Program Files (x86)\\Windows Kits\\10\\bin';

function hasControlChars(value: string): boolean {
  for (let i = 0; i < value.length; i += 1) {
    const code = value.charCodeAt(i);
    if (code <= 0x1f || code === 0x7f) return true;
  }
  return false;
}

export type WindowsSdkTool = 'makeappx.exe' | 'makepri.exe';

export interface AppxManifestConfig {
  /** Stable semver (e.g. `0.0.21`), mapped to 4-part Store version `(MAJOR + 1).MINOR.PATCH.0`. */
  version: string;
  identity?: MsixIdentity;
}

export interface MakeAppxMappingEntry {
  source: string;
  target: string;
}

export interface ParsedAppxManifest {
  identityName: string;
  publisher: string;
  version: string;
  processorArchitecture: string;
  displayName: string;
  publisherDisplayName: string;
}

export interface BuildWindowsMsixOptions {
  /** Packaged Electron Windows app directory (e.g. `out/mlx-node-win32-arm64`). */
  appPath: string;
  /** Destination `.msix` archive path. */
  msixPath: string;
  /** Stable semver version (e.g. `0.0.21`). */
  version: string;
  /** Directory containing generated MSIX visual assets (`packages/desktop/build/msix`). */
  assetsDir: string;
  identity?: MsixIdentity;
  env?: NodeJS.ProcessEnv;
}

export interface BuildWindowsMsixResult {
  msixPath: string;
  msixVersion: string;
  identity: MsixIdentity;
}

export interface VerifyWindowsMsixOptions {
  msixPath: string;
  /** Expected stable semver version (e.g. `0.0.21`). */
  expectedVersion: string;
  runtimeFiles: readonly string[];
  identity?: MsixIdentity;
  env?: NodeJS.ProcessEnv;
}

export function escapeXml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

function unescapeXml(value: string): string {
  return value
    .replaceAll('&apos;', "'")
    .replaceAll('&quot;', '"')
    .replaceAll('&gt;', '>')
    .replaceAll('&lt;', '<')
    .replaceAll('&amp;', '&');
}

function resolveIdentityField(env: NodeJS.ProcessEnv, envKey: string, fallback: string, label: string): string {
  const raw = env[envKey];
  const value = raw === undefined ? fallback : raw.trim();
  if (value === '') {
    throw new PayloadError(
      `MSIX ${label} (${envKey}) must be a non-empty string`,
      `unset ${envKey} to use the default or provide a valid value`,
    );
  }
  if (hasControlChars(value)) {
    throw new PayloadError(
      `MSIX ${label} (${envKey}) contains invalid control characters`,
      `provide a single-line ${label} without control characters`,
    );
  }
  return value;
}

/** Resolve Store identity values from environment overrides or verified Partner Center defaults. */
export function resolveMsixIdentity(env: NodeJS.ProcessEnv = process.env): MsixIdentity {
  const identityName = resolveIdentityField(
    env,
    'MSIX_IDENTITY_NAME',
    DEFAULT_MSIX_IDENTITY.identityName,
    'identity name',
  );
  if (!IDENTITY_NAME_PATTERN.test(identityName)) {
    throw new PayloadError(
      `Invalid MSIX identity name ${JSON.stringify(identityName)}`,
      'MSIX_IDENTITY_NAME must be 3-50 characters using only ASCII letters, digits, dots, and hyphens',
    );
  }

  const publisher = resolveIdentityField(env, 'MSIX_PUBLISHER', DEFAULT_MSIX_IDENTITY.publisher, 'publisher');
  if (!publisher.includes('=')) {
    throw new PayloadError(
      `Invalid MSIX publisher ${JSON.stringify(publisher)}`,
      'MSIX_PUBLISHER must be an X.500 distinguished name such as CN=F04DA088-C2D4-46E8-921E-82D147E19F3F',
    );
  }

  const publisherDisplayName = resolveIdentityField(
    env,
    'MSIX_PUBLISHER_DISPLAY_NAME',
    DEFAULT_MSIX_IDENTITY.publisherDisplayName,
    'publisher display name',
  );
  const displayName = resolveIdentityField(env, 'MSIX_DISPLAY_NAME', DEFAULT_MSIX_IDENTITY.displayName, 'display name');

  return {
    identityName,
    publisher,
    publisherDisplayName,
    displayName,
    description: DEFAULT_MSIX_IDENTITY.description,
  };
}

/** Render a schema-valid UTF-8 `AppxManifest.xml` for Windows ARM64 Store packaging. */
export function renderAppxManifest(config: AppxManifestConfig): string {
  const msixVersion = msixPackageVersion(config.version);
  const identity = config.identity ?? DEFAULT_MSIX_IDENTITY;

  return (
    [
      '<?xml version="1.0" encoding="utf-8"?>',
      '<Package',
      '  xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"',
      '  xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"',
      '  xmlns:uap5="http://schemas.microsoft.com/appx/manifest/uap/windows10/5"',
      '  xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"',
      '  IgnorableNamespaces="uap uap5 rescap">',
      '  <Identity',
      `    Name="${escapeXml(identity.identityName)}"`,
      `    Publisher="${escapeXml(identity.publisher)}"`,
      `    Version="${escapeXml(msixVersion)}"`,
      '    ProcessorArchitecture="arm64" />',
      '  <Properties>',
      `    <DisplayName>${escapeXml(identity.displayName)}</DisplayName>`,
      `    <PublisherDisplayName>${escapeXml(identity.publisherDisplayName)}</PublisherDisplayName>`,
      '    <Logo>Assets\\StoreLogo.png</Logo>',
      '  </Properties>',
      '  <Resources>',
      '    <Resource Language="en-US" />',
      '  </Resources>',
      '  <Dependencies>',
      '    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.26100.0" MaxVersionTested="10.0.26100.0" />',
      '    <PackageDependency',
      '      Name="Microsoft.VCLibs.140.00.UWPDesktop"',
      '      MinVersion="14.0.30704.0"',
      '      Publisher="CN=Microsoft Corporation, O=Microsoft Corporation, L=Redmond, S=Washington, C=US" />',
      '  </Dependencies>',
      '  <Capabilities>',
      '    <rescap:Capability Name="runFullTrust" />',
      '  </Capabilities>',
      '  <Applications>',
      '    <Application',
      '      Id="Portato"',
      '      Executable="app\\mlx-node.exe"',
      '      EntryPoint="Windows.FullTrustApplication">',
      '      <uap:VisualElements',
      `        DisplayName="${escapeXml(identity.displayName)}"`,
      `        Description="${escapeXml(identity.description)}"`,
      '        BackgroundColor="transparent"',
      '        Square150x150Logo="Assets\\Square150x150Logo.png"',
      '        Square44x44Logo="Assets\\Square44x44Logo.png">',
      '        <uap:DefaultTile',
      '          Wide310x150Logo="Assets\\Wide310x150Logo.png"',
      '          Square71x71Logo="Assets\\Square71x71Logo.png"',
      '          Square310x310Logo="Assets\\Square310x310Logo.png" />',
      '        <uap:SplashScreen Image="Assets\\SplashScreen.png" />',
      '      </uap:VisualElements>',
      '      <Extensions>',
      '        <uap5:Extension',
      '          Category="windows.appExecutionAlias"',
      '          Executable="app\\mlx-node.exe"',
      '          EntryPoint="Windows.FullTrustApplication">',
      '          <uap5:AppExecutionAlias>',
      '            <uap5:ExecutionAlias Alias="mlx-node.exe" />',
      '          </uap5:AppExecutionAlias>',
      '        </uap5:Extension>',
      '      </Extensions>',
      '    </Application>',
      '  </Applications>',
      '</Package>',
    ].join('\n') + '\n'
  );
}

/** Render a `MakeAppx.exe pack /f` mapping file with safe path quoting. */
export function renderMakeAppxMapping(entries: readonly MakeAppxMappingEntry[]): string {
  const lines = ['[Files]'];
  for (const entry of entries) {
    if (entry.source === '' || entry.target === '') {
      throw new PayloadError('MakeAppx mapping entry has an empty source or target path', 'check MSIX staging inputs');
    }
    if (
      entry.source.includes('"') ||
      entry.target.includes('"') ||
      hasControlChars(entry.source) ||
      hasControlChars(entry.target)
    ) {
      throw new PayloadError(
        `MakeAppx mapping path contains forbidden quote or newline characters: ${JSON.stringify(entry)}`,
        'remove quotes and control characters from staged filenames',
      );
    }
    const normalizedTarget = entry.target.replaceAll('/', '\\');
    const segments = normalizedTarget.split('\\');
    if (
      normalizedTarget.startsWith('\\') ||
      /^[A-Za-z]:/.test(normalizedTarget) ||
      segments.some((segment) => segment === '' || segment === '.' || segment === '..')
    ) {
      throw new PayloadError(
        `Invalid MSIX package target path: ${JSON.stringify(entry.target)}`,
        'MSIX target paths must be clean relative paths inside the package',
      );
    }
    lines.push(`"${entry.source}" "${normalizedTarget}"`);
  }
  return `${lines.join('\r\n')}\r\n`;
}

function isExistingFile(path: string): boolean {
  try {
    return statSync(path).isFile();
  } catch {
    return false;
  }
}

function compareVersionsDesc(a: string, b: string): number {
  const aParts = a.split('.').map(Number);
  const bParts = b.split('.').map(Number);
  const length = Math.max(aParts.length, bParts.length);
  for (let i = 0; i < length; i += 1) {
    const diff = (bParts[i] ?? 0) - (aParts[i] ?? 0);
    if (diff !== 0) return diff;
  }
  return 0;
}

function findToolInKitsBin(kitsBinDir: string, toolName: WindowsSdkTool, preferredVersion?: string): string | null {
  if (!existsSync(kitsBinDir)) return null;
  let versionDirs: string[];
  try {
    versionDirs = readdirSync(kitsBinDir, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && VERSION_DIR_PATTERN.test(entry.name))
      .map((entry) => entry.name)
      .sort(compareVersionsDesc);
  } catch {
    return null;
  }

  const trimmedPreferred = preferredVersion?.replace(/[\\/]+$/, '');
  const orderedVersions =
    trimmedPreferred && versionDirs.includes(trimmedPreferred)
      ? [trimmedPreferred, ...versionDirs.filter((ver) => ver !== trimmedPreferred)]
      : versionDirs;

  // Prefer native ARM64 binaries (highest SDK version first), then x64 emulation fallback.
  for (const arch of ['arm64', 'x64'] as const) {
    for (const ver of orderedVersions) {
      const candidate = join(kitsBinDir, ver, arch, toolName);
      if (isExistingFile(candidate)) return candidate;
    }
    const direct = join(kitsBinDir, arch, toolName);
    if (isExistingFile(direct)) return direct;
  }

  return null;
}

/**
 * Locate `makeappx.exe` or `makepri.exe` from the MSVC/Windows SDK environment,
 * `PATH`, or the standard `Windows Kits\10\bin` installation tree.
 */
export function findWindowsSdkTool(
  toolName: WindowsSdkTool,
  env: NodeJS.ProcessEnv = process.env,
  options: { defaultKitsBinDir?: string } = {},
): string {
  const sdkVerBin = env.WindowsSdkVerBinPath?.trim();
  if (sdkVerBin) {
    for (const candidate of [
      join(sdkVerBin, 'arm64', toolName),
      join(sdkVerBin, 'x64', toolName),
      join(sdkVerBin, toolName),
    ]) {
      if (isExistingFile(candidate)) return candidate;
    }
  }

  const sdkDir = env.WindowsSdkDir?.trim();
  if (sdkDir) {
    const found = findToolInKitsBin(join(sdkDir, 'bin'), toolName, env.WindowsSDKVersion?.trim());
    if (found !== null) return found;
  }

  const pathEnv = env.PATH ?? env.Path ?? '';
  for (const rawDir of pathEnv.split(delimiter)) {
    const dir = rawDir.trim().replace(/^"(.*)"$/, '$1');
    if (!dir) continue;
    const candidate = join(dir, toolName);
    if (isExistingFile(candidate)) return candidate;
  }

  const fallbackRoot = options.defaultKitsBinDir ?? DEFAULT_WINDOWS_KITS_BIN;
  const fallback = findToolInKitsBin(fallbackRoot, toolName, env.WindowsSDKVersion?.trim());
  if (fallback !== null) return fallback;

  throw new PayloadError(
    `Windows SDK tool not found: ${toolName}`,
    'install the Windows 10/11 SDK (MSIX packaging tools) or run inside an ARM64 MSVC Developer Command Prompt',
  );
}

function collectFilesRecursive(rootDir: string): string[] {
  const files: string[] = [];
  const visit = (dir: string): void => {
    const entries = readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      const fullPath = join(dir, entry.name);
      if (entry.isDirectory()) {
        visit(fullPath);
      } else if (entry.isFile()) {
        files.push(fullPath);
      } else if (entry.isSymbolicLink()) {
        const stats = statSync(fullPath);
        if (stats.isDirectory()) visit(fullPath);
        else if (stats.isFile()) files.push(fullPath);
      }
    }
  };
  visit(rootDir);
  return files;
}

const SDK_TOOL_MAX_BUFFER = 64 * 1024 * 1024;
const MAX_ERROR_OUTPUT_LINES = 200;

function formatToolOutput(output: string | Buffer | null | undefined): string {
  const text = (typeof output === 'string' ? output : (output?.toString('utf8') ?? '')).trim();
  if (text === '') return '';
  const lines = text.split(/\r?\n/);
  if (lines.length <= MAX_ERROR_OUTPUT_LINES) return text;
  const omitted = lines.length - MAX_ERROR_OUTPUT_LINES;
  return `... (${omitted} earlier lines omitted)\n${lines.slice(-MAX_ERROR_OUTPUT_LINES).join('\n')}`;
}

function runWindowsSdkTool(toolPath: string, toolLabel: string, args: readonly string[]): void {
  try {
    execFileSync(toolPath, args, {
      stdio: 'pipe',
      encoding: 'utf8',
      maxBuffer: SDK_TOOL_MAX_BUFFER,
    });
  } catch (error) {
    const execError =
      typeof error === 'object' && error !== null
        ? (error as NodeJS.ErrnoException & { stdout?: string | Buffer | null; stderr?: string | Buffer | null })
        : null;
    const stderr = formatToolOutput(execError?.stderr);
    const stdout = formatToolOutput(execError?.stdout);
    const details = [stderr ? `stderr:\n${stderr}` : '', stdout ? `stdout:\n${stdout}` : ''].filter(Boolean).join('\n');
    const summary = error instanceof Error ? error.message : String(error);
    throw new PayloadError(
      details ? `${toolLabel} failed: ${summary}\n${details}` : `${toolLabel} failed: ${summary}`,
      `inspect ${toolLabel} output and verify the Windows SDK MSIX packaging inputs`,
    );
  }
}

function assertValidMsixPngAssets(dir: string, files: readonly string[]): void {
  const present = new Set(files);
  const missingAssets = REQUIRED_MSIX_ASSETS.filter((file) => !present.has(file));
  if (missingAssets.length > 0) {
    throw new PayloadError(
      `Missing required regular-file MSIX visual assets in ${dir}: ${missingAssets.join(', ')}`,
      'yarn workspace @mlx-node/desktop icons',
    );
  }
  const pngAssetSizes = files.map((file) => ({
    file,
    size: statSync(join(dir, file)).size,
  }));
  const emptyPngAssets = pngAssetSizes.filter(({ size }) => size === 0);
  if (emptyPngAssets.length > 0) {
    throw new PayloadError(
      `Empty MSIX visual assets in ${dir}: ${emptyPngAssets.map(({ file }) => `${file} (0 bytes)`).join(', ')}`,
      'yarn workspace @mlx-node/desktop icons',
    );
  }
  const oversizedPngAssets = pngAssetSizes.filter(({ size }) => size >= MAX_MSIX_ASSET_BYTES);
  if (oversizedPngAssets.length > 0) {
    const details = oversizedPngAssets.map(({ file, size }) => `${file} (${size} bytes)`).join(', ');
    throw new PayloadError(
      `MSIX visual assets in ${dir} must be strictly under ${MAX_MSIX_ASSET_BYTES} bytes (Microsoft Store WACK limit): ${details}`,
      'yarn workspace @mlx-node/desktop icons',
    );
  }
}

function listRegularPngFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith('.png'))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b));
}

/** Build an unsigned Windows ARM64 `.msix` using `MakePri.exe` and `MakeAppx.exe`. */
export function buildWindowsMsix(options: BuildWindowsMsixOptions): BuildWindowsMsixResult {
  const env = options.env ?? process.env;
  const msixVersion = msixPackageVersion(options.version);
  const identity = options.identity ?? resolveMsixIdentity(env);

  if (!existsSync(options.assetsDir)) {
    throw new PayloadError(
      `Missing MSIX visual assets directory: ${options.assetsDir}`,
      'yarn workspace @mlx-node/desktop icons',
    );
  }
  const pngAssets = listRegularPngFiles(options.assetsDir);
  assertValidMsixPngAssets(options.assetsDir, pngAssets);

  if (!isExistingFile(join(options.appPath, 'mlx-node.exe'))) {
    throw new PayloadError(
      `Packaged Windows app executable not found in ${options.appPath}`,
      'run package-windows.ts before building the MSIX',
    );
  }

  const makepri = findWindowsSdkTool('makepri.exe', env);
  const makeappx = findWindowsSdkTool('makeappx.exe', env);

  const outDir = dirname(options.msixPath);
  mkdirSync(outDir, { recursive: true });
  const stageRoot = join(outDir, '.msix-stage');
  rmSync(stageRoot, { recursive: true, force: true });

  try {
    // Keep only `Assets/*.png` under `priRoot` so MakePri indexes visual assets
    // cleanly without scanning the multi-gigabyte Electron + CUDA app tree.
    const priRoot = join(stageRoot, 'pri-root');
    const stagedAssets = join(priRoot, 'Assets');
    mkdirSync(stagedAssets, { recursive: true });

    for (const asset of pngAssets) {
      cpSync(join(options.assetsDir, asset), join(stagedAssets, asset));
    }
    // Re-verify the exact staged regular-file copies consumed by MakePri and MakeAppx.
    const stagedPngAssets = listRegularPngFiles(stagedAssets);
    assertValidMsixPngAssets(stagedAssets, stagedPngAssets);

    const manifestPath = join(stageRoot, 'AppxManifest.xml');
    writeFileSync(manifestPath, renderAppxManifest({ version: options.version, identity }), 'utf8');

    const priConfigPath = join(stageRoot, 'priconfig.xml');
    runWindowsSdkTool(makepri, 'makepri.exe createconfig', [
      'createconfig',
      '/cf',
      priConfigPath,
      '/dq',
      'en-US',
      '/pv',
      '10.0.0',
      '/o',
    ]);

    // Strip `<packaging>` autoResourcePackage rules so MakePri writes ONE single
    // `resources.pri` containing every scale (100..400), targetsize, and unplated
    // variant instead of splitting scale-125/150/200/400 into separate `.pri` files.
    const priConfig = readFileSync(priConfigPath, 'utf8').replace(/<packaging>[\s\S]*?<\/packaging>/, '');
    writeFileSync(priConfigPath, priConfig, 'utf8');

    const priPath = join(stageRoot, 'resources.pri');
    runWindowsSdkTool(makepri, 'makepri.exe new', [
      'new',
      '/pr',
      priRoot,
      '/cf',
      priConfigPath,
      '/mn',
      manifestPath,
      '/in',
      identity.identityName,
      '/o',
      '/of',
      priPath,
    ]);
    if (!isExistingFile(priPath) || statSync(priPath).size === 0) {
      throw new PayloadError(`MakePri produced no resources.pri at ${priPath}`, 'inspect MakePri output');
    }

    const mappingEntries: MakeAppxMappingEntry[] = [
      { source: manifestPath, target: 'AppxManifest.xml' },
      { source: priPath, target: 'resources.pri' },
      ...stagedPngAssets.map((asset) => ({
        source: join(stagedAssets, asset),
        target: `Assets\\${asset}`,
      })),
      ...collectFilesRecursive(options.appPath).map((file) => ({
        source: file,
        target: `app\\${relative(options.appPath, file).replaceAll('/', '\\')}`,
      })),
    ];

    const mappingPath = join(stageRoot, 'mapping.txt');
    writeFileSync(mappingPath, renderMakeAppxMapping(mappingEntries), 'utf8');

    rmSync(options.msixPath, { force: true });
    runWindowsSdkTool(makeappx, 'makeappx.exe pack', ['pack', '/f', mappingPath, '/p', options.msixPath, '/o']);

    return {
      msixPath: options.msixPath,
      msixVersion,
      identity,
    };
  } finally {
    rmSync(stageRoot, { recursive: true, force: true });
  }
}

function tarExecutable(): string {
  return process.platform === 'win32' ? join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'tar.exe') : 'tar';
}

function parseXmlAttributes(rawAttrs: string): Record<string, string> {
  const attrs: Record<string, string> = {};
  const attrPattern = /\b([A-Za-z_][A-Za-z0-9:._-]*)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;
  for (const match of rawAttrs.matchAll(attrPattern)) {
    attrs[match[1]] = unescapeXml(match[2] ?? match[3] ?? '');
  }
  return attrs;
}

/** Parse `<Identity>` and `<Properties>` fields out of an `AppxManifest.xml` string. */
export function parseAppxManifest(xml: string): ParsedAppxManifest {
  const stripped = xml.replace(/^\uFEFF/, '').replace(/<!--[\s\S]*?-->/g, '');
  const identityMatch = /<Identity\b([^>]*?)\/?>/.exec(stripped);
  if (identityMatch === null) {
    throw new PayloadError('AppxManifest.xml is missing an <Identity> element', 'repackage the MSIX');
  }
  const attrs = parseXmlAttributes(identityMatch[1]);
  const identityName = attrs.Name;
  const publisher = attrs.Publisher;
  const version = attrs.Version;
  const processorArchitecture = attrs.ProcessorArchitecture;
  if (!identityName || !publisher || !version || !processorArchitecture) {
    throw new PayloadError(
      'AppxManifest.xml <Identity> is missing Name, Publisher, Version, or ProcessorArchitecture',
      'repackage the MSIX',
    );
  }

  const propertiesMatch = /<Properties\b[^>]*>([\s\S]*?)<\/Properties>/.exec(stripped);
  if (propertiesMatch === null) {
    throw new PayloadError('AppxManifest.xml is missing a <Properties> element', 'repackage the MSIX');
  }
  const propertiesXml = propertiesMatch[1];
  const displayNameMatch = /<DisplayName>([^<]*)<\/DisplayName>/.exec(propertiesXml);
  const publisherDisplayNameMatch = /<PublisherDisplayName>([^<]*)<\/PublisherDisplayName>/.exec(propertiesXml);
  const displayName = displayNameMatch ? unescapeXml(displayNameMatch[1].trim()) : '';
  const publisherDisplayName = publisherDisplayNameMatch ? unescapeXml(publisherDisplayNameMatch[1].trim()) : '';
  if (!displayName || !publisherDisplayName) {
    throw new PayloadError(
      'AppxManifest.xml <Properties> is missing <DisplayName> or <PublisherDisplayName>',
      'repackage the MSIX',
    );
  }

  return {
    identityName,
    publisher,
    version,
    processorArchitecture,
    displayName,
    publisherDisplayName,
  };
}

/** Read and parse `AppxManifest.xml` directly from a packaged `.msix` archive. */
export function readMsixManifest(msixPath: string): ParsedAppxManifest {
  if (!isExistingFile(msixPath)) {
    throw new PayloadError(`MSIX package not found: ${msixPath}`, 'run package-windows.ts to build the MSIX');
  }
  const xml = execFileSync(tarExecutable(), ['-xOf', msixPath, 'AppxManifest.xml'], {
    encoding: 'utf8',
    maxBuffer: 4 * 1024 * 1024,
  });
  return parseAppxManifest(xml);
}

/**
 * Encode a package-relative POSIX path to the canonical OPC part name written
 * by `MakeAppx.exe` (Microsoft MSIX SDK `Encoding.cpp`: percent-encodes `@` as
 * `%40`, `%` as `%25`, spaces as `%20`, etc., while keeping `/` and unreserved
 * characters intact).
 */
export function encodeMsixPartName(path: string): string {
  return path
    .split('/')
    .map((segment) =>
      encodeURIComponent(segment).replace(/[!'()]/g, (ch) => `%${ch.charCodeAt(0).toString(16).toUpperCase()}`),
    )
    .join('/');
}

/** Return any required archive members missing from an MSIX OPC entry list. */
export function missingWindowsMsixEntries(entries: readonly string[], runtimeFiles: readonly string[]): string[] {
  const normalizedList = entries.map((entry) => entry.replaceAll('\\', '/').replace(/^\.\//, ''));
  const normalized = new Set(normalizedList);
  const required = [
    'AppxManifest.xml',
    'resources.pri',
    ...MANIFEST_BASE_ASSETS.map((asset) => `Assets/${asset}`),
    'app/mlx-node.exe',
    'app/resources/app/package.json',
    ...WINDOWS_APP_RUNTIME_FILES.map((file) => `app/resources/app/${file}`),
    ...runtimeFiles.map((file) => `app/resources/native/${file}`),
    'app/resources/native/include/cuda_runtime.h',
    'app/resources/www/index.html',
  ];
  const missing = required.filter((entry) => !normalized.has(encodeMsixPartName(entry)));
  for (const dir of WINDOWS_APP_RUNTIME_DIRS) {
    const encodedPrefix = `${encodeMsixPartName(`app/resources/app/${dir}`)}/`;
    const hasFile = normalizedList.some(
      (entry) => entry.startsWith(encodedPrefix) && entry.length > encodedPrefix.length && !entry.endsWith('/'),
    );
    if (!hasFile) {
      missing.push(`app/resources/app/${dir}`);
    }
  }
  return missing;
}

/**
 * Inspect the real `.msix` archive without extracting gigabytes of native runtime,
 * checking both required payload members and the embedded `AppxManifest.xml`.
 */
export function verifyWindowsMsix(options: VerifyWindowsMsixOptions): ParsedAppxManifest {
  if (!isExistingFile(options.msixPath) || statSync(options.msixPath).size === 0) {
    throw new PayloadError(
      `Missing or empty Windows MSIX package: ${options.msixPath}`,
      'run package-windows.ts to build the MSIX',
    );
  }
  const expectedMsixVersion = msixPackageVersion(options.expectedVersion);
  const expectedIdentity = options.identity ?? resolveMsixIdentity(options.env ?? process.env);

  const entries = execFileSync(tarExecutable(), ['-tf', options.msixPath], {
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
  })
    .split(/\r?\n/)
    .filter(Boolean);

  const missing = missingWindowsMsixEntries(entries, options.runtimeFiles);
  if (missing.length > 0) {
    throw new PayloadError(
      `Windows MSIX is missing required entries:\n${missing.join('\n')}`,
      'repackage the Windows MSIX',
    );
  }

  const manifest = readMsixManifest(options.msixPath);
  const mismatches: string[] = [];
  if (manifest.version !== expectedMsixVersion) {
    mismatches.push(`Identity/@Version is ${manifest.version}, expected ${expectedMsixVersion}`);
  }
  if (manifest.processorArchitecture !== 'arm64') {
    mismatches.push(`Identity/@ProcessorArchitecture is ${manifest.processorArchitecture}, expected arm64`);
  }
  if (manifest.identityName !== expectedIdentity.identityName) {
    mismatches.push(`Identity/@Name is ${manifest.identityName}, expected ${expectedIdentity.identityName}`);
  }
  if (manifest.publisher !== expectedIdentity.publisher) {
    mismatches.push(`Identity/@Publisher is ${manifest.publisher}, expected ${expectedIdentity.publisher}`);
  }
  if (manifest.displayName !== expectedIdentity.displayName) {
    mismatches.push(`Properties/DisplayName is ${manifest.displayName}, expected ${expectedIdentity.displayName}`);
  }
  if (manifest.publisherDisplayName !== expectedIdentity.publisherDisplayName) {
    mismatches.push(
      `Properties/PublisherDisplayName is ${manifest.publisherDisplayName}, expected ${expectedIdentity.publisherDisplayName}`,
    );
  }
  if (mismatches.length > 0) {
    throw new PayloadError(
      `Windows MSIX AppxManifest.xml mismatch:\n${mismatches.join('\n')}`,
      'check MSIX identity settings and --app-version, then repackage',
    );
  }

  return manifest;
}
