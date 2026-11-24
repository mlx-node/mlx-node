import { statfs } from 'node:fs/promises';
import { totalmem } from 'node:os';

import type { ApiPaths } from '../context.js';

/** What the Models page needs to know about this machine: its memory, and the free space of the models disk. */
export interface HostInfo {
  totalMemBytes: number;
  /** `null` when the models folder cannot be read. */
  diskFreeBytes: number | null;
}

export async function handleHost(ctx: ApiPaths): Promise<HostInfo> {
  let diskFreeBytes: number | null;
  try {
    const fs = await statfs(ctx.modelsDir);
    diskFreeBytes = Number(fs.bavail) * Number(fs.bsize);
  } catch {
    diskFreeBytes = null;
  }
  return { totalMemBytes: totalmem(), diskFreeBytes };
}
