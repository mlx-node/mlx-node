/**
 * Copies bundled skills into `ROOT/plugins/<id>`, the folders handed to the agent as `pluginDirs`. A copy is written
 * into a fresh temp folder and switched in by one rename, so an agent never reads a half-written folder. "Installed"
 * means exactly the manifest's files with their hashes, plus `.installed.json`, and nothing else: no link, no extra.
 */

import { randomBytes } from 'node:crypto';
import { chmod, lstat, mkdir, mkdtemp, readdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import {
  bundleEntrySha,
  bundlePathSegments,
  isSafeSegment,
  loadManifest,
  readBundleFile,
  sha256Hex,
  SkillBundleError,
  type SkillsManifest,
} from './manifest.js';

export const INSTALLED_FILE = '.installed.json';
const TMP_PREFIX = '.tmp-';
const OLD_PREFIX = '.old-';

export interface InstalledRecord {
  version: 1;
  id: string;
  bundle: string;
  manifestSha256: string;
  installedAt: number;
}

export interface SkillsInstallerOptions {
  /** The package's `skills/` folder. */
  skillsDir: string;
  /** `ROOT/plugins`. */
  pluginsDir: string;
  now?: () => number;
}

/** What one `sync` did per wanted bundle: installed and verified, the shipped bundle is corrupt, or the copy failed. */
export type SyncResult = Record<string, 'ok' | 'skill-bundle-corrupt' | 'install-failed'>;

async function lstatOrNull(path: string): Promise<Awaited<ReturnType<typeof lstat>> | null> {
  try {
    return await lstat(path);
  } catch {
    return null;
  }
}

export class SkillsInstaller {
  private readonly skillsDir: string;
  private readonly pluginsDir: string;
  private readonly now: () => number;

  constructor(options: SkillsInstallerOptions) {
    this.skillsDir = options.skillsDir;
    this.pluginsDir = options.pluginsDir;
    this.now = options.now ?? Date.now;
  }

  pluginDir(id: string): string {
    if (!isSafeSegment(id)) throw new SkillBundleError();
    return join(this.pluginsDir, id);
  }

  /** Clears crash leftovers, installs every wanted bundle that is not verified, removes every other folder. */
  async sync(wanted: readonly string[]): Promise<SyncResult> {
    await mkdir(this.pluginsDir, { recursive: true, mode: 0o700 });
    await chmod(this.pluginsDir, 0o700);
    const names = await readdir(this.pluginsDir);
    for (const name of names) {
      if (name.startsWith(TMP_PREFIX) || name.startsWith(OLD_PREFIX)) {
        await rm(join(this.pluginsDir, name), { recursive: true, force: true });
      }
    }
    let manifest: SkillsManifest | null;
    try {
      manifest = loadManifest(this.skillsDir);
    } catch {
      manifest = null;
    }
    const result: SyncResult = {};
    for (const id of wanted) {
      if (manifest === null || !isSafeSegment(id)) {
        result[id] = 'skill-bundle-corrupt';
        continue;
      }
      try {
        if (!(await this.verified(id, manifest))) await this.install(id, manifest);
        result[id] = 'ok';
      } catch (error) {
        result[id] = error instanceof SkillBundleError ? 'skill-bundle-corrupt' : 'install-failed';
      }
    }
    // A failed sync removes nothing: a selection that rolls back finds its folders where they were.
    if (!wanted.every((id) => result[id] === 'ok')) return result;
    for (const name of await readdir(this.pluginsDir)) {
      if (name.startsWith('.') || wanted.includes(name)) continue;
      await this.remove(name);
    }
    return result;
  }

  /** Every bundled file is checked first; a mismatch writes nothing. */
  async install(id: string, manifest: SkillsManifest = loadManifest(this.skillsDir)): Promise<void> {
    const bundle = manifest.bundles[id];
    if (bundle === undefined) throw new SkillBundleError();
    const target = this.pluginDir(id);
    const files = bundle.files.map((file) => {
      const segments = bundlePathSegments(file.path);
      if (segments === null) throw new SkillBundleError();
      return { segments, data: readBundleFile(this.skillsDir, bundle, file) };
    });
    await mkdir(this.pluginsDir, { recursive: true, mode: 0o700 });
    const tmp = await mkdtemp(join(this.pluginsDir, `${TMP_PREFIX}${id}-`));
    try {
      await chmod(tmp, 0o700);
      for (const { segments, data } of files) {
        let dir = tmp;
        for (const part of segments.slice(0, -1)) {
          dir = join(dir, part);
          await mkdir(dir, { mode: 0o700 }).catch((error: NodeJS.ErrnoException) => {
            if (error.code !== 'EEXIST') throw error;
          });
        }
        await writeFile(join(dir, segments[segments.length - 1]!), data, { mode: 0o600, flag: 'wx' });
      }
      const record: InstalledRecord = {
        version: 1,
        id,
        bundle: bundle.root,
        manifestSha256: bundleEntrySha(bundle),
        installedAt: this.now(),
      };
      await writeFile(join(tmp, INSTALLED_FILE), `${JSON.stringify(record)}\n`, { mode: 0o600, flag: 'wx' });
      let old: string | null = null;
      if ((await lstatOrNull(target)) !== null) {
        old = join(this.pluginsDir, `${OLD_PREFIX}${id}-${randomBytes(6).toString('hex')}`);
        await rename(target, old);
      }
      await rename(tmp, target);
      if (old !== null) await rm(old, { recursive: true, force: true });
    } catch (error) {
      await rm(tmp, { recursive: true, force: true });
      throw error;
    }
  }

  async remove(id: string): Promise<void> {
    if (id === '' || id.startsWith('.') || id.includes('/')) return;
    const target = join(this.pluginsDir, id);
    if ((await lstatOrNull(target)) === null) return;
    const old = join(this.pluginsDir, `${OLD_PREFIX}${id}-${randomBytes(6).toString('hex')}`);
    await rename(target, old);
    await rm(old, { recursive: true, force: true });
  }

  /** A real directory holding exactly the manifest's files with their hashes, and a matching record. */
  async verified(id: string, manifest: SkillsManifest = loadManifest(this.skillsDir)): Promise<boolean> {
    const bundle = manifest.bundles[id];
    if (bundle === undefined || !isSafeSegment(id)) return false;
    const target = join(this.pluginsDir, id);
    const st = await lstatOrNull(target);
    if (st === null || !st.isDirectory()) return false;
    try {
      const recordStat = await lstatOrNull(join(target, INSTALLED_FILE));
      if (recordStat === null || !recordStat.isFile()) return false;
      const record = JSON.parse(await readFile(join(target, INSTALLED_FILE), 'utf8')) as Partial<InstalledRecord>;
      if (record.version !== 1 || record.id !== id || record.manifestSha256 !== bundleEntrySha(bundle)) return false;
      const expected = new Map(bundle.files.map((f) => [f.path, f]));
      const found = await this.walk(target, '');
      if (found === null || found.length !== expected.size) return false;
      for (const path of found) {
        const file = expected.get(path);
        if (file === undefined) return false;
        const data = await readFile(join(target, path));
        if (data.length !== file.bytes || sha256Hex(data) !== file.sha256) return false;
      }
      return true;
    } catch {
      return false;
    }
  }

  /** Every regular file below `dir` (relative, `/`-joined) except the record; null on any link or special file. */
  private async walk(dir: string, prefix: string): Promise<string[] | null> {
    const out: string[] = [];
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const rel = prefix === '' ? entry.name : `${prefix}/${entry.name}`;
      if (rel === INSTALLED_FILE) continue;
      if (entry.isSymbolicLink()) return null;
      if (entry.isDirectory()) {
        const inner = await this.walk(join(dir, entry.name), rel);
        if (inner === null) return null;
        out.push(...inner);
      } else if (entry.isFile()) {
        out.push(rel);
      } else {
        return null;
      }
    }
    return out;
  }
}
