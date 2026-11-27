import { execFileSync, spawnSync } from 'node:child_process';
import { chmod, mkdir, mkdtemp, readdir, readFile, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import {
  isPrivateWindowsSddl,
  publishDesktopEndpoint,
  readDesktopEndpoint,
  WINDOWS_ACL_INSPECT_SCRIPTS,
  type DesktopEndpoint,
} from '../../../packages/server/src/host/desktop-endpoint.js';

const windows = process.platform === 'win32';
const dirs: string[] = [];
afterEach(async () => {
  await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});
async function location() {
  const dir = await mkdtemp(join(tmpdir(), 'mlx-desktop-endpoint-'));
  dirs.push(dir);
  return join(dir, 'desktop', 'inference.json');
}
const endpoint: DesktopEndpoint = {
  models: [{ name: 'local', path: '/models/local', base: 'local', variant: 'BF16', format: 'mlx' }],
  version: 1,
  pid: process.pid,
  url: 'http://127.0.0.1:12345',
  token: 'private',
};

function icacls(args: string[]): void {
  const exe = join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'icacls.exe');
  execFileSync(exe, args, { windowsHide: true, stdio: 'ignore' });
}

it('publishes privately and removes only its own generation', async () => {
  const path = await location();
  const releaseOld = await publishDesktopEndpoint(endpoint, path);
  if (!windows) {
    expect((await stat(path)).mode & 0o777).toBe(0o600);
    expect((await stat(join(path, '..'))).mode & 0o777).toBe(0o700);
  }
  expect(JSON.parse(await readFile(path, 'utf8'))).toEqual(endpoint);
  const releaseNew = await publishDesktopEndpoint({ ...endpoint, token: 'new' }, path);
  await releaseOld();
  expect(JSON.parse(await readFile(path, 'utf8')).token).toBe('new');
  await releaseNew();
  await expect(stat(path)).rejects.toMatchObject({ code: 'ENOENT' });
});

async function writeRaw(path: string, text: string, mode = 0o600): Promise<void> {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  await writeFile(path, text, { mode });
  if (!windows) await chmod(path, mode);
}

it('reads back a published endpoint', async () => {
  const path = await location();
  await publishDesktopEndpoint(endpoint, path);
  expect(await readDesktopEndpoint(path)).toEqual(endpoint);
});

it('reads null for a missing file or text that is not an endpoint', async () => {
  const path = await location();
  expect(await readDesktopEndpoint(path)).toBeNull();
  await writeRaw(path, '{not json');
  expect(await readDesktopEndpoint(path)).toBeNull();
  await writeRaw(path, 'null');
  expect(await readDesktopEndpoint(path)).toBeNull();
  await writeRaw(path, JSON.stringify({ ...endpoint, models: [null] }));
  expect(await readDesktopEndpoint(path)).toBeNull();
});

it('reads null for a model without a base, a build or a known format', async () => {
  const path = await location();
  const model = endpoint.models[0]!;
  const bad: unknown[] = [
    { name: model.name, path: model.path, variant: 'BF16', format: 'mlx' },
    { ...model, base: '' },
    { ...model, variant: '' },
    { ...model, variant: 4 },
    { ...model, format: 'safetensors' },
    { name: model.name, path: model.path, base: 'local', variant: 'BF16' },
  ];
  for (const one of bad) {
    await writeRaw(path, JSON.stringify({ ...endpoint, models: [one] }));
    expect(await readDesktopEndpoint(path)).toBeNull();
  }
  await expect(publishDesktopEndpoint({ ...endpoint, models: [{ ...model, base: '' }] }, path)).rejects.toThrow();
});

it('reads a model whose images flag is true, false or missing, and refuses one that is not a boolean', async () => {
  const path = await location();
  const model = endpoint.models[0]!;
  for (const images of [true, false, undefined]) {
    const one = { ...endpoint, models: [{ ...model, images }] };
    await writeRaw(path, JSON.stringify(one));
    expect(await readDesktopEndpoint(path)).toEqual(one);
  }
  await writeRaw(path, JSON.stringify({ ...endpoint, models: [{ ...model, images: 'yes' }] }));
  expect(await readDesktopEndpoint(path)).toBeNull();
});

it.skipIf(windows)('reads null when group or other may read the file', async () => {
  const path = await location();
  await writeRaw(path, JSON.stringify(endpoint), 0o640);
  expect(await readDesktopEndpoint(path)).toBeNull();
  await chmod(path, 0o604);
  expect(await readDesktopEndpoint(path)).toBeNull();
  await chmod(path, 0o600);
  expect(await readDesktopEndpoint(path)).toEqual(endpoint);
});

it('reads null for a directory at the endpoint path', async () => {
  const path = await location();
  await mkdir(path, { recursive: true });
  expect(await readDesktopEndpoint(path)).toBeNull();
});

it('reads null through a symlink', async () => {
  const path = await location();
  const real = join(dirname(path), 'real.json');
  await writeRaw(real, JSON.stringify(endpoint));
  try {
    await symlink(real, path);
  } catch (error) {
    if (windows && (error as NodeJS.ErrnoException).code === 'EPERM') return;
    throw error;
  }
  expect(await readDesktopEndpoint(path)).toBeNull();
});

it('reads null for an origin that is not loopback http, or an empty token', async () => {
  const path = await location();
  for (const url of [
    'http://localhost:12345',
    'https://127.0.0.1:12345',
    'http://127.0.0.1:12345/v1',
    'http://10.0.0.1:12345',
  ]) {
    await writeRaw(path, JSON.stringify({ ...endpoint, url }));
    expect(await readDesktopEndpoint(path)).toBeNull();
  }
  await writeRaw(path, JSON.stringify({ ...endpoint, token: '' }));
  expect(await readDesktopEndpoint(path)).toBeNull();
});

it('reads null when the publishing process is gone', async () => {
  const path = await location();
  const child = spawnSync(process.execPath, ['-e', '']);
  expect(child.pid).toBeGreaterThan(0);
  await writeRaw(path, JSON.stringify({ ...endpoint, pid: child.pid }));
  expect(await readDesktopEndpoint(path)).toBeNull();
});

describe('isPrivateWindowsSddl', () => {
  const user = 'S-1-5-21-3365401605-447589653-2711680764-1001';
  const other = 'S-1-5-21-3365401605-447589653-2711680764-1005';

  it('accepts owner-only protected and inherited DACLs restricted to user, SYSTEM, and Administrators', () => {
    expect(isPrivateWindowsSddl(`O:${user}G:${user}D:PAI(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;${user})`, user)).toEqual({
      ok: true,
      protectedDacl: true,
    });
    expect(
      isPrivateWindowsSddl(`O:${user}D:AI(A;ID;FA;;;S-1-5-18)(A;ID;FA;;;S-1-5-32-544)(A;ID;FA;;;${user})`, user),
    ).toEqual({
      ok: true,
      protectedDacl: false,
    });
    // Elevated admin process creates objects owned by BA (or SY); unelevated processes of the same user still accept them.
    expect(isPrivateWindowsSddl(`O:BAG:BAD:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;${user})`, user)).toEqual({
      ok: true,
      protectedDacl: true,
    });
    expect(isPrivateWindowsSddl(`O:SYG:SYD:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;${user})`, user)).toEqual({
      ok: true,
      protectedDacl: true,
    });
    // Explicit deny ACEs do not grant access, and trailing SACLs are ignored.
    expect(isPrivateWindowsSddl(`O:${user}D:P(D;;FA;;;WD)(A;;FA;;;${user})S:(AU;SAFA;FA;;;WD)`, user)).toEqual({
      ok: true,
      protectedDacl: true,
    });
  });

  it('rejects foreign owners, other local accounts/groups, missing/empty DACLs, and unknown ACE types', () => {
    for (const bad of [
      `O:${other}D:P(A;;FA;;;${user})`,
      `O:${user}D:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;${user})(A;;FR;;;${other})`,
      `O:${user}D:AI(A;ID;FR;;;BU)(A;ID;FA;;;${user})`,
      `O:${user}D:P(A;;FR;;;WD)(A;;FA;;;${user})`,
      `O:${user}D:P(A;;FR;;;AU)(A;;FA;;;${user})`,
      `O:${user}G:${user}`,
      `O:${user}D:NO_ACCESS_CONTROL`,
      `O:${user}D:P`,
      `O:${user}D:P(XA;;FA;;;${user};(WIN://SYSAPPID))`,
      `O:${user}D:P(OA;;FA;;;${user})`,
      `not-sddl`,
    ]) {
      expect(isPrivateWindowsSddl(bad, user).ok, bad).toBe(false);
    }
  });
});

describe.runIf(windows)('Windows NTFS ACL hardening and verification', () => {
  function runConstrainedPowerShell(script: string, extraEnv: Record<string, string>) {
    const ps = join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe');
    const guarded = `$ExecutionContext.SessionState.LanguageMode = 'ConstrainedLanguage'\n${script}`;
    const encoded = Buffer.from(guarded, 'utf16le').toString('base64');
    return spawnSync(ps, ['-NoProfile', '-NonInteractive', '-EncodedCommand', encoded], {
      env: { ...process.env, ...extraEnv },
      windowsHide: true,
      encoding: 'utf8',
    });
  }

  it('hardens an open directory on publish, round-trips Windows model paths, and rejects file/dir ACL relaxation', async () => {
    const path = await location();
    const dir = dirname(path);
    await mkdir(dir, { recursive: true });
    // Grant Everyone and BUILTIN\Users read access before publish — publishDesktopEndpoint must wipe both.
    icacls([dir, '/grant', '*S-1-1-0:(OI)(CI)(R)', '/grant', '*S-1-5-32-545:(OI)(CI)(R)', '/q']);

    const winEndpoint: DesktopEndpoint = {
      ...endpoint,
      models: [
        {
          name: 'Hy4-flash-SPARK-4Bit-Fast',
          path: 'C:\\Users\\lynwe\\.mlx-node\\models\\hy4-flash-gguf-spark-4bit-fast\\Hy4-flash-SPARK-4Bit-Fast.gguf',
          base: 'Hy4-flash',
          variant: 'SPARK-4Bit-Fast',
          format: 'gguf',
          modelType: 'hy_v4_flash',
          kind: 'chat',
          images: false,
        },
      ],
    };
    await publishDesktopEndpoint(winEndpoint, path);
    // Repeated reads verify file and directory ACLs live without trusting stale stat caches.
    expect(await readDesktopEndpoint(path)).toEqual(winEndpoint);
    expect(await readDesktopEndpoint(path)).toEqual(winEndpoint);

    // Granting another local group read access on the file fails closed immediately.
    icacls([path, '/grant', '*S-1-5-32-545:(R)', '/q']);
    expect(await readDesktopEndpoint(path)).toBeNull();

    // Re-publishing re-hardens both directory and file so reads succeed again.
    await publishDesktopEndpoint(winEndpoint, path);
    expect(await readDesktopEndpoint(path)).toEqual(winEndpoint);

    // Granting another local group write/read access on the containing directory must also fail closed.
    icacls([dir, '/grant', '*S-1-5-32-545:(W)', '/q']);
    expect(await readDesktopEndpoint(path)).toBeNull();
  });

  it('runs Windows ACL inspection under PowerShell Constrained Language mode and rejects junctions', async () => {
    const path = await location();
    const dir = dirname(path);
    await publishDesktopEndpoint(endpoint, path);

    const pathRun = runConstrainedPowerShell(WINDOWS_ACL_INSPECT_SCRIPTS.path, {
      MLX_ENDPOINT_ACL_PATH: dir,
    });
    expect(pathRun.status, pathRun.stderr).toBe(0);
    expect(pathRun.stdout.trim()).toMatch(/^O:.+D:P/);

    const pairRun = runConstrainedPowerShell(WINDOWS_ACL_INSPECT_SCRIPTS.pair, {
      MLX_ENDPOINT_ACL_FILE: path,
      MLX_ENDPOINT_ACL_DIR: dir,
    });
    expect(pairRun.status, pairRun.stderr).toBe(0);
    expect(pairRun.stdout.trim().split(/\r?\n/).filter(Boolean)).toHaveLength(2);

    const junction = join(dirname(dir), 'junction-desktop');
    const cmd = join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'cmd.exe');
    const mk = spawnSync(cmd, ['/c', 'mklink', '/J', junction, dir], { windowsHide: true });
    expect(mk.status).toBe(0);
    try {
      const juncRun = runConstrainedPowerShell(WINDOWS_ACL_INSPECT_SCRIPTS.path, {
        MLX_ENDPOINT_ACL_PATH: junction,
      });
      expect(juncRun.status).not.toBe(0);
      expect(await readDesktopEndpoint(join(junction, 'inference.json'))).toBeNull();
      const junctionEndpoint: DesktopEndpoint = { ...endpoint, token: 'never-written-through-junction' };
      await expect(publishDesktopEndpoint(junctionEndpoint, join(junction, 'inference.json'))).rejects.toThrow();
      expect(await readdir(dir)).toEqual(['inference.json']);
      expect(await readFile(path, 'utf8')).not.toContain('never-written-through-junction');
    } finally {
      spawnSync(cmd, ['/c', 'rmdir', junction], { windowsHide: true });
    }
  });
});
