import { existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { delimiter, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { PayloadError } from '../scripts/payload.js';
import { ReleaseVersionError } from '../scripts/release-version.js';
import {
  buildWindowsMsix,
  DEFAULT_MSIX_IDENTITY,
  escapeXml,
  findWindowsSdkTool,
  MAX_MSIX_ASSET_BYTES,
  missingWindowsMsixEntries,
  parseAppxManifest,
  readMsixManifest,
  renderAppxManifest,
  renderMakeAppxMapping,
  REQUIRED_MSIX_ASSETS,
  resolveMsixIdentity,
  verifyWindowsMsix,
} from '../scripts/windows-msix.js';
import { WINDOWS_ADDON, WINDOWS_RUNTIME_DLLS } from '../scripts/windows-payload.js';

const COMMITTED_MSIX_ASSETS_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'build', 'msix');

const runtimeFiles = [WINDOWS_ADDON, ...WINDOWS_RUNTIME_DLLS, 'nvrtc-builtins64_134.dll'];

// Minimal valid 1x1 PNG so MakePri.exe accepts the visual asset files in end-to-end Windows tests.
const TINY_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
  'base64',
);

function hasWindowsSdkTools(): boolean {
  if (process.platform !== 'win32') return false;
  try {
    findWindowsSdkTool('makepri.exe');
    findWindowsSdkTool('makeappx.exe');
    return true;
  } catch {
    return false;
  }
}

describe('MSIX identity resolution', () => {
  it('defaults to the verified Partner Center identity for Portato', () => {
    expect(resolveMsixIdentity({})).toEqual(DEFAULT_MSIX_IDENTITY);
    expect(DEFAULT_MSIX_IDENTITY.identityName).toBe('LongYinan.Portato');
    expect(DEFAULT_MSIX_IDENTITY.publisher).toBe('CN=F04DA088-C2D4-46E8-921E-82D147E19F3F');
    expect(DEFAULT_MSIX_IDENTITY.publisherDisplayName).toBe('LongYinan');
    expect(DEFAULT_MSIX_IDENTITY.displayName).toBe('Portato');
  });

  it('supports environment overrides and trims surrounding whitespace', () => {
    const resolved = resolveMsixIdentity({
      MSIX_IDENTITY_NAME: '  Acme.CustomApp  ',
      MSIX_PUBLISHER: ' CN=Acme & Co, O=Acme ',
      MSIX_PUBLISHER_DISPLAY_NAME: ' Acme & Sons ',
      MSIX_DISPLAY_NAME: ' Portato <Preview> ',
    });
    expect(resolved).toEqual({
      identityName: 'Acme.CustomApp',
      publisher: 'CN=Acme & Co, O=Acme',
      publisherDisplayName: 'Acme & Sons',
      displayName: 'Portato <Preview>',
      description: DEFAULT_MSIX_IDENTITY.description,
    });
  });

  it('rejects empty overrides, control characters, and malformed identity/publisher values', () => {
    expect(() => resolveMsixIdentity({ MSIX_IDENTITY_NAME: '' })).toThrow(PayloadError);
    expect(() => resolveMsixIdentity({ MSIX_IDENTITY_NAME: '   ' })).toThrow(PayloadError);
    expect(() => resolveMsixIdentity({ MSIX_IDENTITY_NAME: 'bad name with spaces' })).toThrow(PayloadError);
    expect(() => resolveMsixIdentity({ MSIX_PUBLISHER: 'not-a-dn' })).toThrow(PayloadError);
    expect(() => resolveMsixIdentity({ MSIX_DISPLAY_NAME: 'Line1\nLine2' })).toThrow(PayloadError);
    expect(() => resolveMsixIdentity({ MSIX_PUBLISHER_DISPLAY_NAME: '\t' })).toThrow(PayloadError);
  });
});

describe('AppxManifest rendering and parsing', () => {
  it('renders a schema-complete ARM64 desktop manifest and parses it back', () => {
    const xml = renderAppxManifest({ version: '0.0.21' });
    expect(xml).toContain('xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"');
    expect(xml).toContain('xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"');
    expect(xml).toContain('xmlns:uap5="http://schemas.microsoft.com/appx/manifest/uap/windows10/5"');
    expect(xml).toContain(
      'xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"',
    );
    expect(xml).toContain('IgnorableNamespaces="uap uap5 rescap"');
    expect(xml).toContain('Name="LongYinan.Portato"');
    expect(xml).toContain('Publisher="CN=F04DA088-C2D4-46E8-921E-82D147E19F3F"');
    expect(xml).toContain('Version="1.0.21.0"');
    expect(xml).toContain('ProcessorArchitecture="arm64"');
    expect(xml).toContain(
      '<TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.26100.0" MaxVersionTested="10.0.26100.0" />',
    );
    expect(xml).toContain('Name="Microsoft.VCLibs.140.00.UWPDesktop"');
    expect(xml).toContain('<rescap:Capability Name="runFullTrust" />');
    expect(xml).toContain('Executable="app\\mlx-node.exe"');
    expect(xml).toContain('<uap5:ExecutionAlias Alias="mlx-node.exe" />');

    expect(parseAppxManifest(xml)).toEqual({
      identityName: DEFAULT_MSIX_IDENTITY.identityName,
      publisher: DEFAULT_MSIX_IDENTITY.publisher,
      version: '1.0.21.0',
      processorArchitecture: 'arm64',
      displayName: DEFAULT_MSIX_IDENTITY.displayName,
      publisherDisplayName: DEFAULT_MSIX_IDENTITY.publisherDisplayName,
    });
  });

  it('XML-escapes special characters and round-trips through parseAppxManifest', () => {
    const identity = {
      ...DEFAULT_MSIX_IDENTITY,
      publisher: 'CN=Dev & "Test" <1>',
      publisherDisplayName: "O'Reilly & <Co>",
      displayName: 'Portato "Beta" & AI',
      description: 'Runs <local> models & "CUDA"',
    };
    const xml = renderAppxManifest({ version: '1.2.3', identity });
    expect(xml).toContain(escapeXml(identity.publisher));
    expect(xml).toContain(`<DisplayName>${escapeXml(identity.displayName)}</DisplayName>`);
    expect(xml).toContain(`<PublisherDisplayName>${escapeXml(identity.publisherDisplayName)}</PublisherDisplayName>`);
    expect(xml).toContain(`Description="${escapeXml(identity.description)}"`);

    const parsed = parseAppxManifest(`\uFEFF<!-- comment --><Root>${xml}</Root>`);
    expect(parsed).toEqual({
      identityName: identity.identityName,
      publisher: identity.publisher,
      version: '2.2.3.0',
      processorArchitecture: 'arm64',
      displayName: identity.displayName,
      publisherDisplayName: identity.publisherDisplayName,
    });
  });

  it('rejects prerelease versions when rendering AppxManifest', () => {
    expect(() => renderAppxManifest({ version: '0.0.22-rc.1' })).toThrow(ReleaseVersionError);
  });

  it('fails closed when parsing a malformed AppxManifest', () => {
    expect(() => parseAppxManifest('<Package></Package>')).toThrow(PayloadError);
    expect(() => parseAppxManifest('<Package><Identity Name="A" Version="1.0.0.0" /><Properties /></Package>')).toThrow(
      PayloadError,
    );
    expect(() =>
      parseAppxManifest(
        '<Package><Identity Name="A" Publisher="CN=B" Version="1.0.0.0" ProcessorArchitecture="arm64" /><Properties><DisplayName>App</DisplayName></Properties></Package>',
      ),
    ).toThrow(PayloadError);
  });
});

describe('MakeAppx mapping generation', () => {
  it('renders [Files] with CRLF line endings and normalized Windows package paths', () => {
    const mapping = renderMakeAppxMapping([
      { source: 'C:\\stage\\AppxManifest.xml', target: 'AppxManifest.xml' },
      { source: 'C:\\my app\\resources\\app\\package.json', target: 'app/resources/app/package.json' },
    ]);
    expect(mapping).toBe(
      '[Files]\r\n' +
        '"C:\\stage\\AppxManifest.xml" "AppxManifest.xml"\r\n' +
        '"C:\\my app\\resources\\app\\package.json" "app\\resources\\app\\package.json"\r\n',
    );
  });

  it('rejects embedded quotes, newlines, empty paths, and invalid target paths', () => {
    expect(() => renderMakeAppxMapping([{ source: '', target: 'a.txt' }])).toThrow(PayloadError);
    expect(() => renderMakeAppxMapping([{ source: 'C:\\bad"file.txt', target: 'a.txt' }])).toThrow(PayloadError);
    expect(() => renderMakeAppxMapping([{ source: 'C:\\bad\nfile.txt', target: 'a.txt' }])).toThrow(PayloadError);
    expect(() => renderMakeAppxMapping([{ source: 'C:\\ok.txt', target: '..\\escape.txt' }])).toThrow(PayloadError);
    expect(() => renderMakeAppxMapping([{ source: 'C:\\ok.txt', target: '\\rooted.txt' }])).toThrow(PayloadError);
    expect(() => renderMakeAppxMapping([{ source: 'C:\\ok.txt', target: 'C:\\abs.txt' }])).toThrow(PayloadError);
  });
});

describe('MSIX entry verification', () => {
  const completeEntries = [
    'AppxManifest.xml',
    'resources.pri',
    'Assets/StoreLogo.png',
    'Assets/Square44x44Logo.png',
    'Assets/Square71x71Logo.png',
    'Assets/Square150x150Logo.png',
    'Assets/Wide310x150Logo.png',
    'Assets/Square310x310Logo.png',
    'Assets/SplashScreen.png',
    'app/mlx-node.exe',
    'app/resources/app/package.json',
    'app/resources/native/include/cuda_runtime.h',
    'app/resources/www/index.html',
    ...runtimeFiles.map((file) => `app/resources/native/${file}`),
  ];

  it('accepts a complete MSIX member list with either slash direction', () => {
    expect(missingWindowsMsixEntries(completeEntries, runtimeFiles)).toEqual([]);
    expect(
      missingWindowsMsixEntries(
        completeEntries.map((entry) => entry.replaceAll('/', '\\')),
        runtimeFiles,
      ),
    ).toEqual([]);
  });

  it('reports every missing manifest, PRI, asset, app, or native entry', () => {
    const pruned = completeEntries.filter(
      (entry) =>
        entry !== 'resources.pri' && entry !== 'Assets/StoreLogo.png' && entry !== 'app/resources/native/openblas.dll',
    );
    expect(missingWindowsMsixEntries(pruned, runtimeFiles)).toEqual([
      'resources.pri',
      'Assets/StoreLogo.png',
      'app/resources/native/openblas.dll',
    ]);
  });
});

describe('Windows SDK tool discovery', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'mlx-windows-sdk-'));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  function touch(relativePath: string): string {
    const fullPath = join(root, relativePath);
    mkdirSync(dirname(fullPath), { recursive: true });
    writeFileSync(fullPath, 'stub');
    return fullPath;
  }

  it('prefers arm64 over x64 in WindowsSdkVerBinPath', () => {
    const x64Tool = touch('verbin/x64/makeappx.exe');
    expect(
      findWindowsSdkTool(
        'makeappx.exe',
        { WindowsSdkVerBinPath: join(root, 'verbin') },
        { defaultKitsBinDir: join(root, 'none') },
      ),
    ).toBe(x64Tool);

    const arm64Tool = touch('verbin/arm64/makeappx.exe');
    expect(
      findWindowsSdkTool(
        'makeappx.exe',
        { WindowsSdkVerBinPath: join(root, 'verbin') },
        { defaultKitsBinDir: join(root, 'none') },
      ),
    ).toBe(arm64Tool);
  });

  it('selects the highest numeric SDK version under WindowsSdkDir and prefers arm64 before x64', () => {
    touch('sdk/bin/10.0.19041.0/arm64/makepri.exe');
    const newestArm64 = touch('sdk/bin/10.0.100000.0/arm64/makepri.exe');
    touch('sdk/bin/10.0.26100.0/arm64/makepri.exe');
    touch('sdk/bin/10.0.100000.0/x64/makepri.exe');

    expect(
      findWindowsSdkTool(
        'makepri.exe',
        { WindowsSdkDir: join(root, 'sdk') },
        { defaultKitsBinDir: join(root, 'none') },
      ),
    ).toBe(newestArm64);
  });

  it('falls back to PATH and then to the default Windows Kits bin scan (arm64 then x64)', () => {
    const pathTool = touch('path-bin/makeappx.exe');
    expect(
      findWindowsSdkTool(
        'makeappx.exe',
        { PATH: `${join(root, 'empty')}${delimiter}${join(root, 'path-bin')}` },
        { defaultKitsBinDir: join(root, 'none') },
      ),
    ).toBe(pathTool);

    const fallbackX64 = touch('kits/10.0.26100.0/x64/makepri.exe');
    expect(findWindowsSdkTool('makepri.exe', {}, { defaultKitsBinDir: join(root, 'kits') })).toBe(fallbackX64);

    const fallbackArm64 = touch('kits/10.0.22621.0/arm64/makepri.exe');
    expect(findWindowsSdkTool('makepri.exe', {}, { defaultKitsBinDir: join(root, 'kits') })).toBe(fallbackArm64);
  });

  it('throws PayloadError when the tool cannot be found anywhere', () => {
    expect(() =>
      findWindowsSdkTool('makeappx.exe', { PATH: '' }, { defaultKitsBinDir: join(root, 'missing-kits') }),
    ).toThrow(PayloadError);
  });
});

describe('MSIX visual asset size validation', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'mlx-windows-msix-assets-'));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  function writeFixture(relativePath: string, contents: string | Buffer): void {
    const fullPath = join(root, relativePath);
    mkdirSync(dirname(fullPath), { recursive: true });
    writeFileSync(fullPath, contents);
  }

  it('rejects any MSIX PNG asset >= MAX_MSIX_ASSET_BYTES (204,800 bytes) with filename and byte size', () => {
    expect(MAX_MSIX_ASSET_BYTES).toBe(204_800);

    const appDir = join(root, 'mlx-node-win32-arm64');
    const assetsDir = join(root, 'msix-assets');
    const msixPath = join(root, 'out', 'mlx-node-0.0.21-win32-arm64.msix');

    writeFixture('mlx-node-win32-arm64/mlx-node.exe', Buffer.from([0x4d, 0x5a]));
    for (const asset of REQUIRED_MSIX_ASSETS) {
      writeFixture(`msix-assets/${asset}`, TINY_PNG);
    }

    // Exact boundary: 204,800 bytes is rejected by WACK test 2.1.
    writeFixture('msix-assets/Square310x310Logo.scale-200.png', Buffer.alloc(MAX_MSIX_ASSET_BYTES, 1));
    expect(() =>
      buildWindowsMsix({
        appPath: appDir,
        msixPath,
        version: '0.0.21',
        assetsDir,
      }),
    ).toThrow(PayloadError);
    expect(() =>
      buildWindowsMsix({
        appPath: appDir,
        msixPath,
        version: '0.0.21',
        assetsDir,
      }),
    ).toThrow(/Square310x310Logo\.scale-200\.png \(204800 bytes\)/);

    // Non-required scale PNG over the limit is also caught before staging.
    writeFixture('msix-assets/Square310x310Logo.scale-200.png', TINY_PNG);
    writeFixture('msix-assets/Square150x150Logo.scale-400.png', Buffer.alloc(MAX_MSIX_ASSET_BYTES + 123, 1));
    expect(() =>
      buildWindowsMsix({
        appPath: appDir,
        msixPath,
        version: '0.0.21',
        assetsDir,
      }),
    ).toThrow(/Square150x150Logo\.scale-400\.png \(204923 bytes\)/);

    // Empty extra PNG in assetsDir is also rejected.
    writeFixture('msix-assets/Square150x150Logo.scale-400.png', Buffer.alloc(0));
    expect(() =>
      buildWindowsMsix({
        appPath: appDir,
        msixPath,
        version: '0.0.21',
        assetsDir,
      }),
    ).toThrow(/Square150x150Logo\.scale-400\.png \(0 bytes\)/);
  });

  it('ensures every committed PNG in packages/desktop/build/msix/ is non-empty and strictly < 204,800 bytes', () => {
    expect(existsSync(COMMITTED_MSIX_ASSETS_DIR)).toBe(true);
    const pngFiles = readdirSync(COMMITTED_MSIX_ASSETS_DIR, { withFileTypes: true })
      .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith('.png'))
      .map((entry) => entry.name)
      .sort((a, b) => a.localeCompare(b));

    expect(pngFiles.length).toBeGreaterThanOrEqual(REQUIRED_MSIX_ASSETS.length);
    for (const required of REQUIRED_MSIX_ASSETS) {
      expect(pngFiles).toContain(required);
    }

    for (const file of pngFiles) {
      const { size } = statSync(join(COMMITTED_MSIX_ASSETS_DIR, file));
      expect(size, `${file} must be non-empty`).toBeGreaterThan(0);
      expect(size, `${file} (${size} bytes) must be strictly under ${MAX_MSIX_ASSET_BYTES} bytes`).toBeLessThan(
        MAX_MSIX_ASSET_BYTES,
      );
    }
  });
});

describe.runIf(hasWindowsSdkTools())('MSIX end-to-end packaging with Windows SDK', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'mlx-windows-msix-e2e-'));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  function writeFixture(relativePath: string, contents: string | Buffer = 'x'): void {
    const fullPath = join(root, relativePath);
    mkdirSync(dirname(fullPath), { recursive: true });
    writeFileSync(fullPath, contents);
  }

  it('builds a real MSIX with MakePri + MakeAppx and verifies its archive contents and AppxManifest', () => {
    const appDir = join(root, 'mlx-node-win32-arm64');
    const assetsDir = join(root, 'msix-assets');
    const outDir = join(root, 'out');
    const msixPath = join(outDir, 'mlx-node-0.0.21-win32-arm64.msix');

    writeFixture('mlx-node-win32-arm64/mlx-node.exe', Buffer.from([0x4d, 0x5a]));
    writeFixture('mlx-node-win32-arm64/resources/app/package.json', JSON.stringify({ version: '0.0.21' }));
    writeFixture('mlx-node-win32-arm64/resources/native/include/cuda_runtime.h', '// header');
    writeFixture('mlx-node-win32-arm64/resources/www/index.html', '<!doctype html>');
    for (const file of runtimeFiles) {
      writeFixture(`mlx-node-win32-arm64/resources/native/${file}`, file);
    }
    for (const asset of REQUIRED_MSIX_ASSETS) {
      writeFixture(`msix-assets/${asset}`, TINY_PNG);
    }

    const built = buildWindowsMsix({
      appPath: appDir,
      msixPath,
      version: '0.0.21',
      assetsDir,
    });

    expect(built.msixPath).toBe(msixPath);
    expect(built.msixVersion).toBe('1.0.21.0');
    expect(existsSync(msixPath)).toBe(true);
    expect(existsSync(join(outDir, '.msix-stage'))).toBe(false);

    const manifest = verifyWindowsMsix({
      msixPath,
      expectedVersion: '0.0.21',
      runtimeFiles,
    });
    expect(manifest).toEqual(readMsixManifest(msixPath));
    expect(manifest.version).toBe('1.0.21.0');
    expect(manifest.processorArchitecture).toBe('arm64');
    expect(manifest.identityName).toBe(DEFAULT_MSIX_IDENTITY.identityName);
    expect(manifest.displayName).toBe(DEFAULT_MSIX_IDENTITY.displayName);

    expect(() =>
      verifyWindowsMsix({
        msixPath,
        expectedVersion: '0.0.22',
        runtimeFiles,
      }),
    ).toThrow(/Identity\/@Version/);
  });

  it('builds and verifies a real MSIX against the actual committed packages/desktop/build/msix/ assets', () => {
    const appDir = join(root, 'mlx-node-win32-arm64');
    const outDir = join(root, 'out');
    const msixPath = join(outDir, 'mlx-node-0.0.21-win32-arm64-real-assets.msix');

    writeFixture('mlx-node-win32-arm64/mlx-node.exe', Buffer.from([0x4d, 0x5a]));
    writeFixture('mlx-node-win32-arm64/resources/app/package.json', JSON.stringify({ version: '0.0.21' }));
    writeFixture('mlx-node-win32-arm64/resources/native/include/cuda_runtime.h', '// header');
    writeFixture('mlx-node-win32-arm64/resources/www/index.html', '<!doctype html>');
    for (const file of runtimeFiles) {
      writeFixture(`mlx-node-win32-arm64/resources/native/${file}`, file);
    }

    const built = buildWindowsMsix({
      appPath: appDir,
      msixPath,
      version: '0.0.21',
      assetsDir: COMMITTED_MSIX_ASSETS_DIR,
    });

    expect(built.msixPath).toBe(msixPath);
    expect(existsSync(msixPath)).toBe(true);

    const manifest = verifyWindowsMsix({
      msixPath,
      expectedVersion: '0.0.21',
      runtimeFiles,
    });
    expect(manifest.version).toBe('1.0.21.0');
    expect(manifest.identityName).toBe(DEFAULT_MSIX_IDENTITY.identityName);
  });
});
