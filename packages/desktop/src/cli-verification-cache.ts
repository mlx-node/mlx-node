import { createHash, randomUUID } from 'node:crypto';
import { constants, type Stats } from 'node:fs';
import { mkdir, open, rename, rm, stat, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

interface FileFingerprint {
  id: string;
  stamp: string;
  digest: string;
  mode: number;
}

interface Verification {
  version: 1;
  key: string;
  files: FileFingerprint[];
}

const hash = (text: string): string => createHash('sha256').update(text).digest('hex');
const isHash = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const stamp = (info: Stats): string =>
  hash(JSON.stringify([info.dev, info.ino, info.size, info.mtimeMs, info.ctimeMs, info.mode]));

/** Persist successful probes, using metadata to avoid rereading unchanged runtime binaries. */
export class CliVerificationCache {
  private previous?: Verification;
  private loaded = false;

  constructor(private readonly path: string) {}

  private async load(): Promise<void> {
    if (this.loaded) return;
    this.loaded = true;
    try {
      const file = await open(this.path, constants.O_RDONLY | constants.O_NONBLOCK);
      let text: string;
      try {
        if (!(await file.stat()).isFile()) return;
        const buffer = Buffer.alloc(16 * 1024 + 1);
        const { bytesRead } = await file.read(buffer, 0, buffer.length, 0);
        if (bytesRead === buffer.length) return;
        text = buffer.subarray(0, bytesRead).toString('utf8');
      } finally {
        await file.close();
      }
      const data = JSON.parse(text);
      if (
        data?.version === 1 &&
        isHash(data.key) &&
        Array.isArray(data.files) &&
        data.files.length <= 16 &&
        data.files.every(
          (item: FileFingerprint) =>
            item && isHash(item.id) && isHash(item.stamp) && isHash(item.digest) && Number.isInteger(item.mode),
        )
      ) {
        this.previous = data;
      }
    } catch {
      // Missing, unreadable or damaged caches are misses, never evidence of readiness.
    }
  }

  private async fingerprint(path: string): Promise<FileFingerprint> {
    const info = await stat(path);
    if (!info.isFile()) throw new Error(`The app runtime path is not a regular file: ${path}`);
    const id = hash(path);
    const metadata = stamp(info);
    const cached = this.previous?.files.find((file) => file.id === id && file.stamp === metadata);
    if (cached) return cached;

    const file = await open(path, constants.O_RDONLY | constants.O_NONBLOCK);
    const digest = createHash('sha256');
    try {
      if (stamp(await file.stat()) !== metadata)
        throw new Error('The app runtime changed while checking. Retry setup.');
      // Stream large binaries instead of allocating their entire contents in the main process.
      const buffer = Buffer.alloc(256 * 1024);
      for (;;) {
        const { bytesRead } = await file.read(buffer, 0, buffer.length, null);
        if (!bytesRead) break;
        digest.update(buffer.subarray(0, bytesRead));
      }
      if (stamp(await file.stat()) !== metadata || stamp(await stat(path)) !== metadata)
        throw new Error('The app runtime changed while checking. Retry setup.');
    } finally {
      await file.close();
    }
    return { id, stamp: metadata, digest: digest.digest('hex'), mode: info.mode };
  }

  async verify(paths: string[], probe: () => Promise<void>): Promise<void> {
    await this.load();
    const files: FileFingerprint[] = [];
    for (const path of paths) files.push(await this.fingerprint(path));
    // Timestamps/inodes only select the cheap path; identical bytes keep the same verdict.
    const key = hash(JSON.stringify(['delegate-help-v1', files.map(({ id, digest, mode }) => [id, digest, mode])]));
    if (this.previous?.key !== key) {
      await probe();
      for (const [index, path] of paths.entries()) {
        if (stamp(await stat(path)) !== files[index].stamp)
          throw new Error('The app runtime changed while checking. Retry setup.');
      }
    }
    const next: Verification = { version: 1, key, files };
    if (JSON.stringify(this.previous) === JSON.stringify(next)) return;
    this.previous = next;
    const temp = `${this.path}.${randomUUID()}.tmp`;
    try {
      await mkdir(dirname(this.path), { recursive: true, mode: 0o700 });
      await writeFile(temp, JSON.stringify(next), { mode: 0o600, flag: 'wx' });
      await rename(temp, this.path);
    } catch {
      // Keep the in-memory result if disk caching is unavailable. Failures remain retryable.
    } finally {
      await rm(temp, { force: true }).catch(() => {});
    }
  }
}
