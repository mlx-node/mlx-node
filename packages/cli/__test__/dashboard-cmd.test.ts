/**
 * Tests for the thin `mlx dashboard` command: flag parsing, help, port
 * validation, and the browser-open gate.
 *
 * `@mlx-node/dashboard` is mocked so no real HTTP server starts and the native
 * addon is never touched. `process.exit` is mocked to throw so validation
 * failures are observable as rejections instead of tearing down the worker.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vite-plus/test';

vi.mock('@mlx-node/dashboard', () => ({
  startDashboardServer: vi.fn(async () => ({
    url: 'http://127.0.0.1:6590',
    port: 6590,
    close: vi.fn(async () => {}),
  })),
}));

import { startDashboardServer } from '@mlx-node/dashboard';

import { run as runDashboard, parseDashboardArgs } from '../src/commands/dashboard.js';

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
  // `run` installs SIGINT/SIGTERM handlers to stay alive; drop them so repeated
  // test runs don't accumulate listeners.
  process.removeAllListeners('SIGINT');
  process.removeAllListeners('SIGTERM');
});

describe('mlx dashboard flag parsing', () => {
  it('parses default flags to port 6590, host 127.0.0.1, open true', () => {
    const args = parseDashboardArgs([]);
    expect(args).toMatchObject({ help: false, port: 6590, host: '127.0.0.1', open: true });
    expect(args.db).toBeUndefined();
    expect(args.modelsDir).toBeUndefined();
  });

  it('respects --no-open', () => {
    expect(parseDashboardArgs(['--no-open']).open).toBe(false);
  });

  it('respects --port 0 (ephemeral port)', () => {
    expect(parseDashboardArgs(['--port', '0']).port).toBe(0);
  });

  it('passes through --db, --models-dir, and --host', () => {
    const args = parseDashboardArgs(['--db', '/tmp/dash.db', '--models-dir', '/tmp/models', '--host', 'localhost']);
    expect(args.db).toBe('/tmp/dash.db');
    expect(args.modelsDir).toBe('/tmp/models');
    expect(args.host).toBe('localhost');
  });

  it('throws on a non-numeric --port at the parse layer', () => {
    expect(() => parseDashboardArgs(['--port', 'abc'])).toThrow('Invalid --port');
  });
});

describe('mlx dashboard run()', () => {
  it('prints usage for --help without starting a server', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    await runDashboard(['--help']);

    const out = logSpy.mock.calls.map((c) => String(c[0])).join('\n');
    expect(out).toContain('mlx dashboard');
    expect(out).toContain('--no-open');
    expect(startDashboardServer).not.toHaveBeenCalled();
  });

  it('exits with a clear error for a non-numeric --port', async () => {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(console, 'log').mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation(((code?: number) => {
      throw new Error(`process.exit(${code})`);
    }) as never);

    await expect(runDashboard(['--port', 'abc'])).rejects.toThrow('process.exit(1)');

    const errors = errSpy.mock.calls.map((c) => String(c[0])).join('\n');
    expect(errors).toContain('Invalid --port');
    expect(exitSpy).toHaveBeenCalledWith(1);
    expect(startDashboardServer).not.toHaveBeenCalled();
  });

  it('does not open a browser when --no-open is set', async () => {
    vi.spyOn(console, 'log').mockImplementation(() => {});
    const openBrowser = vi.fn();

    await runDashboard(['--no-open'], { openBrowser });

    expect(startDashboardServer).toHaveBeenCalledTimes(1);
    expect(startDashboardServer).toHaveBeenCalledWith(expect.objectContaining({ port: 6590, host: '127.0.0.1' }));
    expect(openBrowser).not.toHaveBeenCalled();
  });

  it('opens the browser at the server url by default', async () => {
    vi.spyOn(console, 'log').mockImplementation(() => {});
    const openBrowser = vi.fn();

    await runDashboard([], { openBrowser });

    expect(openBrowser).toHaveBeenCalledWith('http://127.0.0.1:6590');
  });

  it('warns loudly when binding a non-loopback host', async () => {
    vi.spyOn(console, 'log').mockImplementation(() => {});
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const openBrowser = vi.fn();

    await runDashboard(['--host', '0.0.0.0', '--no-open'], { openBrowser });

    const warnings = warnSpy.mock.calls.map((c) => String(c[0])).join('\n');
    expect(warnings).toContain('non-loopback');
    expect(startDashboardServer).toHaveBeenCalledWith(expect.objectContaining({ host: '0.0.0.0' }));
  });
});
