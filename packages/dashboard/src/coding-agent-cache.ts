import { randomUUID } from 'node:crypto';
import { constants } from 'node:fs';
import { mkdir, open, rename, rm, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

export interface DetectionResult {
  installed: boolean;
  needsUpdate?: boolean;
  source?: { startLine: number; endLine: number };
  checkedAt: string;
}

/** Only hashes, verdicts and line references are persisted. Instruction text never enters this file. */
export class CodingAgentCache {
  private readonly entries = new Map<string, DetectionResult>();
  private loading?: Promise<void>;
  private writes = Promise.resolve();

  constructor(private readonly path: string) {}

  load(): Promise<void> {
    return (this.loading ??= (async () => {
      try {
        const file = await open(this.path, constants.O_RDONLY | constants.O_NONBLOCK);
        let text: string;
        try {
          if (!(await file.stat()).isFile()) return;
          const buffer = Buffer.alloc(32 * 1024 + 1);
          const { bytesRead } = await file.read(buffer, 0, buffer.length, 0);
          if (bytesRead === buffer.length) return;
          text = buffer.subarray(0, bytesRead).toString('utf8');
        } finally {
          await file.close();
        }
        const data = JSON.parse(text);
        if (data?.version !== 1 || !Array.isArray(data.entries)) return;
        for (const item of data.entries.slice(-64)) {
          if (
            typeof item?.key === 'string' &&
            /^[a-f0-9]{64}$/.test(item.key) &&
            typeof item.installed === 'boolean' &&
            typeof item.checkedAt === 'string' &&
            Number.isFinite(Date.parse(item.checkedAt))
          ) {
            this.entries.set(item.key, {
              installed: item.installed,
              needsUpdate: item.needsUpdate === true,
              ...(Number.isSafeInteger(item.source?.startLine) &&
              Number.isSafeInteger(item.source?.endLine) &&
              item.source.startLine >= 1 &&
              item.source.endLine >= item.source.startLine
                ? { source: { startLine: item.source.startLine, endLine: item.source.endLine } }
                : {}),
              checkedAt: item.checkedAt,
            });
          }
        }
      } catch {
        // An absent or damaged cache is a miss; it must never become an installation verdict.
      }
    })());
  }

  get(key: string): DetectionResult | undefined {
    return this.entries.get(key);
  }

  set(key: string, result: DetectionResult): Promise<void> {
    this.entries.delete(key);
    this.entries.set(key, result);
    while (this.entries.size > 64) this.entries.delete(this.entries.keys().next().value!);
    const data = JSON.stringify({ version: 1, entries: [...this.entries].map(([key, value]) => ({ key, ...value })) });
    this.writes = this.writes.then(async () => {
      const temp = `${this.path}.${randomUUID()}.tmp`;
      try {
        await mkdir(dirname(this.path), { recursive: true });
        await writeFile(temp, data, { mode: 0o600, flag: 'wx' });
        await rename(temp, this.path);
      } catch {
        // The in-memory result remains usable if disk caching is unavailable.
      } finally {
        await rm(temp, { force: true }).catch(() => {});
      }
    });
    return this.writes;
  }
}
