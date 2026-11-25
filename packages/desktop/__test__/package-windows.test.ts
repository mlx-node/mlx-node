import { describe, expect, it } from 'vite-plus/test';

import { parsePackageWindowsArgs } from '../scripts/package-windows.js';
import { PayloadError } from '../scripts/payload.js';
import { ReleaseVersionError } from '../scripts/release-version.js';

describe('parsePackageWindowsArgs', () => {
  it('returns default flags and falls back to the manifest version', () => {
    expect(parsePackageWindowsArgs([], '0.0.21')).toEqual({
      skipMsix: false,
      appOnly: false,
      signed: false,
      fromApp: null,
      appVersion: '0.0.21',
    });
  });

  it('parses --app-only and explicit --app-version forms', () => {
    expect(parsePackageWindowsArgs(['--app-version', '0.0.22', '--app-only'], '0.0.21')).toEqual({
      skipMsix: false,
      appOnly: true,
      signed: false,
      fromApp: null,
      appVersion: '0.0.22',
    });

    expect(parsePackageWindowsArgs(['--app-version=0.0.22-rc.1', '--app-only'], '0.0.21')).toEqual({
      skipMsix: false,
      appOnly: true,
      signed: false,
      fromApp: null,
      appVersion: '0.0.22-rc.1',
    });
  });

  it('parses --from-app with --signed and --skip-msix', () => {
    expect(
      parsePackageWindowsArgs(
        ['--app-version', '0.0.22', '--from-app', 'out/mlx-node-win32-arm64', '--signed', '--skip-msix'],
        '0.0.21',
      ),
    ).toEqual({
      skipMsix: true,
      appOnly: false,
      signed: true,
      fromApp: 'out/mlx-node-win32-arm64',
      appVersion: '0.0.22',
    });

    expect(parsePackageWindowsArgs(['--from-app=C:\\build\\mlx-node-win32-arm64', '--signed'], '0.0.21')).toEqual({
      skipMsix: false,
      appOnly: false,
      signed: true,
      fromApp: 'C:\\build\\mlx-node-win32-arm64',
      appVersion: '0.0.21',
    });
  });

  it('rejects combining --app-only and --from-app', () => {
    expect(() => parsePackageWindowsArgs(['--app-only', '--from-app', 'out/app'], '0.0.21')).toThrow(PayloadError);
    expect(() => parsePackageWindowsArgs(['--from-app', 'out/app', '--app-only'], '0.0.21')).toThrow(PayloadError);
  });

  it('rejects --signed without --from-app', () => {
    expect(() => parsePackageWindowsArgs(['--signed'], '0.0.21')).toThrow(PayloadError);
    expect(() => parsePackageWindowsArgs(['--app-only', '--signed'], '0.0.21')).toThrow(PayloadError);
  });

  it('rejects missing, blank, or flag-like values for --from-app', () => {
    expect(() => parsePackageWindowsArgs(['--from-app'], '0.0.21')).toThrow(PayloadError);
    expect(() => parsePackageWindowsArgs(['--from-app', ''], '0.0.21')).toThrow(PayloadError);
    expect(() => parsePackageWindowsArgs(['--from-app', '   '], '0.0.21')).toThrow(PayloadError);
    expect(() => parsePackageWindowsArgs(['--from-app', '--signed'], '0.0.21')).toThrow(PayloadError);
    expect(() => parsePackageWindowsArgs(['--from-app='], '0.0.21')).toThrow(PayloadError);
    expect(() => parsePackageWindowsArgs(['--from-app=--signed'], '0.0.21')).toThrow(PayloadError);
  });

  it('rejects missing, empty, or flag-like values for --app-version', () => {
    expect(() => parsePackageWindowsArgs(['--app-version'], '0.0.21')).toThrow(PayloadError);
    expect(() => parsePackageWindowsArgs(['--app-version', ''], '0.0.21')).toThrow(PayloadError);
    expect(() => parsePackageWindowsArgs(['--app-version', '--app-only'], '0.0.21')).toThrow(PayloadError);
    expect(() => parsePackageWindowsArgs(['--app-version='], '0.0.21')).toThrow(PayloadError);
    expect(() => parsePackageWindowsArgs(['--app-version=--app-only'], '0.0.21')).toThrow(PayloadError);
  });

  it('rejects invalid semver values from --app-version or the manifest', () => {
    expect(() => parsePackageWindowsArgs(['--app-version', 'v0.0.21'], '0.0.21')).toThrow(ReleaseVersionError);
    expect(() => parsePackageWindowsArgs(['--app-version', '0.01.0'], '0.0.21')).toThrow(ReleaseVersionError);
    expect(() => parsePackageWindowsArgs(['--app-version', 'not-a-version'], '0.0.21')).toThrow(ReleaseVersionError);
    expect(() => parsePackageWindowsArgs([], 'not-a-version')).toThrow(ReleaseVersionError);
  });

  it('rejects unknown flags and unexpected positional arguments', () => {
    expect(() => parsePackageWindowsArgs(['--form-app', 'out/app'], '0.0.21')).toThrow(PayloadError);
    expect(() => parsePackageWindowsArgs(['--unknown'], '0.0.21')).toThrow(PayloadError);
    expect(() => parsePackageWindowsArgs(['unexpected-arg'], '0.0.21')).toThrow(PayloadError);
  });
});
