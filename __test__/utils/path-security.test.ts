import { describe, expect, it, afterEach } from 'vite-plus/test';
import { resolve } from 'node:path';
import { validatePathContainment, resolveAndValidatePath, getAllowedRoot, PathTraversalError } from '@mlx-node/trl';

describe('Path Security', () => {
  describe('validatePathContainment', () => {
    it('accepts paths within the allowed root', () => {
      const root = '/home/user/data';
      const validPath = '/home/user/data/subdir/file.txt';

      expect(() => validatePathContainment(validPath, root)).not.toThrow();
    });

    it('accepts paths at the root level', () => {
      const root = '/home/user/data';
      const validPath = '/home/user/data/file.txt';

      expect(() => validatePathContainment(validPath, root)).not.toThrow();
    });

    it('accepts the root path itself', () => {
      const root = '/home/user/data';

      expect(() => validatePathContainment(root, root)).not.toThrow();
    });

    it('rejects paths that escape via parent traversal', () => {
      const root = '/home/user/data';
      const escapedPath = '/home/user/etc/passwd';

      expect(() => validatePathContainment(escapedPath, root)).toThrow(PathTraversalError);
    });

    it('rejects paths with explicit parent traversal sequences', () => {
      const root = '/home/user/data';
      const escapedPath = resolve(root, '../../../etc/passwd');

      expect(() => validatePathContainment(escapedPath, root)).toThrow(PathTraversalError);
    });

    it('rejects absolute paths outside the root', () => {
      const root = '/home/user/data';
      const outsidePath = '/etc/passwd';

      expect(() => validatePathContainment(outsidePath, root)).toThrow(PathTraversalError);
    });

    it('handles normalized paths correctly', () => {
      const root = '/home/user/data';
      const normalizedPath = '/home/user/data/./subdir/../file.txt';

      // After normalization this is /home/user/data/file.txt, which is valid
      expect(() => validatePathContainment(normalizedPath, root)).not.toThrow();
    });

    it('throws PathTraversalError with correct properties', () => {
      const root = '/home/user/data';
      const escapedPath = '/etc/passwd';

      try {
        validatePathContainment(escapedPath, root);
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(PathTraversalError);
        const pte = error as PathTraversalError;
        expect(pte.resolvedPath).toBe(escapedPath);
        expect(pte.allowedRoot).toBe(root);
        expect(pte.message).toContain('Path traversal detected');
      }
    });
  });

  describe('resolveAndValidatePath', () => {
    it('resolves relative paths within allowed root', () => {
      const root = '/home/user/data';
      const relativePath = 'subdir/file.txt';

      const result = resolveAndValidatePath(relativePath, root);
      expect(result).toBe('/home/user/data/subdir/file.txt');
    });

    it('rejects relative paths that escape the root', () => {
      const root = '/home/user/data';
      const escapingPath = '../../../etc/passwd';

      expect(() => resolveAndValidatePath(escapingPath, root)).toThrow(PathTraversalError);
    });

    it('handles absolute paths that are within root', () => {
      const root = '/home/user/data';
      const absoluteWithinRoot = '/home/user/data/file.txt';

      // When you call resolvePath with an absolute path, it returns that absolute path
      // Since the absolute path is within root, this should succeed
      const result = resolveAndValidatePath(absoluteWithinRoot, root);
      expect(result).toBe(absoluteWithinRoot);
    });

    it('rejects absolute paths outside root', () => {
      const root = '/home/user/data';
      const absoluteOutsideRoot = '/etc/passwd';

      expect(() => resolveAndValidatePath(absoluteOutsideRoot, root)).toThrow(PathTraversalError);
    });
  });

  describe('getAllowedRoot', () => {
    const originalEnv = process.env['MLX_NODE_DATA_ROOT'];

    afterEach(() => {
      if (originalEnv === undefined) {
        delete process.env['MLX_NODE_DATA_ROOT'];
      } else {
        process.env['MLX_NODE_DATA_ROOT'] = originalEnv;
      }
    });

    it('returns cwd when no options or env var', () => {
      delete process.env['MLX_NODE_DATA_ROOT'];
      const result = getAllowedRoot();
      expect(result).toBe(process.cwd());
    });

    it('uses options.allowedRoot when provided', () => {
      const customRoot = '/custom/data/root';
      const result = getAllowedRoot({ allowedRoot: customRoot });
      expect(result).toBe(customRoot);
    });

    it('uses MLX_NODE_DATA_ROOT env var when set', () => {
      process.env['MLX_NODE_DATA_ROOT'] = '/env/data/root';
      const result = getAllowedRoot();
      expect(result).toBe(resolve('/env/data/root'));
    });

    it('prefers options.allowedRoot over env var', () => {
      process.env['MLX_NODE_DATA_ROOT'] = '/env/data/root';
      const customRoot = '/custom/data/root';
      const result = getAllowedRoot({ allowedRoot: customRoot });
      expect(result).toBe(customRoot);
    });
  });
});

describe('Dataset path traversal protection', () => {
  it('loadLocalGsm8kDataset rejects path traversal attempts', async () => {
    const { loadLocalGsm8kDataset } = await import('@mlx-node/trl');

    // Attempt to escape to parent directories
    await expect(loadLocalGsm8kDataset('train', { basePath: '../../../etc' })).rejects.toThrow(PathTraversalError);
  });

  it('loadLocalGsm8kDataset rejects absolute paths outside cwd', async () => {
    const { loadLocalGsm8kDataset } = await import('@mlx-node/trl');

    // Attempt to use absolute path outside cwd
    await expect(loadLocalGsm8kDataset('train', { basePath: '/etc' })).rejects.toThrow(PathTraversalError);
  });

  it('loadLocalGsm8kDataset accepts valid paths within cwd', async () => {
    const { loadLocalGsm8kDataset } = await import('@mlx-node/trl');
    const validBasePath = resolve(process.cwd(), 'data/gsm8k');

    // This should not throw a PathTraversalError (may fail for other reasons like file not found)
    try {
      await loadLocalGsm8kDataset('train', { basePath: validBasePath, limit: 1 });
    } catch (error) {
      // Should not be a PathTraversalError
      expect(error).not.toBeInstanceOf(PathTraversalError);
    }
  });

  it('loadLocalGsm8kDataset allows custom allowedRoot', async () => {
    const { loadLocalGsm8kDataset } = await import('@mlx-node/trl');
    const customRoot = '/custom/root';

    // With a custom allowedRoot, paths relative to that root should work
    // This will throw file not found, but not PathTraversalError
    try {
      await loadLocalGsm8kDataset('train', { basePath: 'data', allowedRoot: customRoot, limit: 1 });
    } catch (error) {
      // Should not be PathTraversalError (file just doesn't exist)
      expect(error).not.toBeInstanceOf(PathTraversalError);
      expect((error as Error).message).toContain('Failed to read');
    }
  });
});
