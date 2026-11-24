/**
 * The two layouts of a GGUF model that Chat loads with no `config.json`, as the user has them: Hy4-flash (split set at
 * the root, LICENSE, README and a `partial` record from `mlx download`) and Qwen3.8-Flash-Next (split set in a
 * variant folder, an MTP GGUF in another, no record). Each folder also holds a link to itself.
 */

import { mkdirSync, symlinkSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { type CatalogBuild, catalogBuildSlug, MODEL_CATALOG } from '@mlx-node/agent/catalog';

import { DOWNLOAD_COMPLETE_MARKER } from '../../src/models.js';

/** A GGUF v3 header with no tensors and one key, `general.architecture`. */
export function ggufHeader(architecture: string): Buffer {
  const string = (text: string): Buffer => {
    const bytes = Buffer.from(text);
    const length = Buffer.alloc(8);
    length.writeBigUInt64LE(BigInt(bytes.length));
    return Buffer.concat([length, bytes]);
  };
  const header = Buffer.alloc(24);
  header.write('GGUF');
  header.writeUInt32LE(3, 4);
  header.writeBigUInt64LE(1n, 16);
  const stringType = Buffer.alloc(4);
  stringType.writeUInt32LE(8);
  return Buffer.concat([header, string('general.architecture'), stringType, string(architecture)]);
}

export function catalogBuildOf(label: string, id: string): CatalogBuild {
  const build = MODEL_CATALOG.find((entry) => entry.label === label)?.builds.find((candidate) => candidate.id === id);
  if (build === undefined) throw new Error(`no build ${label} ${id}`);
  return build;
}

export interface ConfiglessLayout {
  label: string;
  slug: string;
  /** The family Local models shows, from the header. */
  family: string;
  /** The header's `general.architecture`. */
  architecture: string;
  /** The quant label Local models shows, from the file name. */
  quant: string;
  /** The weight file that carries the header. */
  firstPart: string;
  /** Writes the layout into `dir`, the first part with `header`. */
  plant: (dir: string, header?: Buffer) => void;
}

const HY4_PARTS = [1, 2].map((part) => `Hy4-flash-Q4_K_M-0000${part}-of-00002.gguf`);

export const HY4: ConfiglessLayout = {
  label: 'Hy4-flash',
  slug: catalogBuildSlug(catalogBuildOf('Hy4-flash', 'Q4_K_M')),
  family: 'hy_v4_flash',
  architecture: 'hy_v4_flash',
  quant: 'Q4_K_M',
  firstPart: HY4_PARTS[0]!,
  plant: (dir, header = ggufHeader('hy_v4_flash')) => {
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, HY4_PARTS[0]!), header);
    writeFileSync(join(dir, HY4_PARTS[1]!), 'engram rows');
    writeFileSync(join(dir, 'LICENSE'), 'license');
    writeFileSync(join(dir, 'README.md'), '# Hy4-flash');
    writeFileSync(
      join(dir, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({
        repo: 'AngelSlim/Hy4-flash-GGUF',
        revision: '6'.repeat(40),
        files: [...HY4_PARTS, 'LICENSE', 'README.md'],
        scope: 'partial',
        completedAt: '2026-11-18T11:30:13.744Z',
      }),
    );
    symlinkSync(dir, join(dir, 'hy4-flash-gguf'));
  },
};

const FLASH_NEXT_PARTS = [1, 2, 3, 4].map(
  (part) => `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000${part}-of-00004.gguf`,
);

export const FLASH_NEXT: ConfiglessLayout = {
  label: 'Qwen3.8-Flash-Next',
  slug: catalogBuildSlug(catalogBuildOf('Qwen3.8-Flash-Next', 'UD-Q4_K_XL')),
  family: 'qwen4_exp',
  architecture: 'qwen4exp',
  quant: 'UD-Q4_K_XL',
  firstPart: FLASH_NEXT_PARTS[0]!,
  plant: (dir, header = ggufHeader('qwen4exp')) => {
    mkdirSync(join(dir, 'UD-Q4_K_XL'), { recursive: true });
    mkdirSync(join(dir, 'MTP-Q8_0'), { recursive: true });
    FLASH_NEXT_PARTS.forEach((part, index) => writeFileSync(join(dir, part), index === 0 ? header : 'experts'));
    writeFileSync(join(dir, 'MTP-Q8_0', 'mtp-Qwen3.8-Flash-Next-Q8_0.gguf'), ggufHeader('qwen4exp'));
    symlinkSync(dir, join(dir, 'qwen3.8-flash-next-gguf'));
  },
};
