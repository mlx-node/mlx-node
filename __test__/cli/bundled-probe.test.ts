import { describe, expect, it } from 'vitest';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  compareReleaseTags,
  findBundledAppBinary,
} from '../../packages/cli/src/commands/agent/grok-build.js';

describe('findBundledAppBinary', () => {
  it('finds Contents/Resources/native/mlx-agent from inside the bundled app', () => {
    const root = mkdtempSync(join(tmpdir(), 'bundled-'));
    try {
      const deep = join(root, 'Contents/Resources/app/node_modules/@mlx-node/cli/dist/commands/agent');
      mkdirSync(deep, { recursive: true });
      const bin = join(root, 'Contents/Resources/native/mlx-agent');
      mkdirSync(join(root, 'Contents/Resources/native'), { recursive: true });
      writeFileSync(bin, 'x');
      expect(findBundledAppBinary(join(deep, 'grok-build.js'))).toBe(bin);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('returns undefined outside a bundle', () => {
    expect(findBundledAppBinary('/tmp/somewhere/grok-build.js')).toBeUndefined();
  });
});

describe('compareReleaseTags', () => {
  it('orders semver triplets', () => {
    expect(compareReleaseTags('mlx-agent-v0.1.0', 'mlx-agent-v0.2.0')).toBeLessThan(0);
    expect(compareReleaseTags('mlx-agent-v0.2.0', 'mlx-agent-v0.2.0')).toBe(0);
    expect(compareReleaseTags('mlx-agent-v1.0.0', 'mlx-agent-v0.9.9')).toBeGreaterThan(0);
  });

  it('sorts malformed tags below valid ones', () => {
    expect(compareReleaseTags('garbage', 'mlx-agent-v0.0.1')).toBeLessThan(0);
  });
});
