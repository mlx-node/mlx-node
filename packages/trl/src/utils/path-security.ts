/**
 * Path security utilities to prevent directory traversal attacks.
 *
 * These utilities ensure that user-provided paths stay within allowed directories,
 * preventing malicious paths like `../../../etc/` from accessing arbitrary files.
 */

import { resolve as resolvePath, normalize, relative, isAbsolute } from 'node:path';

/**
 * Error thrown when a path traversal attempt is detected.
 */
export class PathTraversalError extends Error {
  constructor(
    public readonly resolvedPath: string,
    public readonly allowedRoot: string,
  ) {
    super(`Path traversal detected: "${resolvedPath}" is outside allowed directory "${allowedRoot}"`);
    this.name = 'PathTraversalError';
  }
}

/**
 * Validates that a resolved path is contained within an allowed root directory.
 * Prevents path traversal attacks via '../' sequences.
 *
 * @param resolvedPath - The fully resolved absolute path to validate
 * @param allowedRoot - The root directory that paths must be contained within
 * @throws PathTraversalError if path escapes the allowed root
 */
export function validatePathContainment(resolvedPath: string, allowedRoot: string): void {
  const normalizedPath = normalize(resolvedPath);
  const normalizedRoot = normalize(allowedRoot);

  // Get relative path from root to target
  const relativePath = relative(normalizedRoot, normalizedPath);

  // If relative path starts with '..' or is absolute, it's outside the root
  if (relativePath.startsWith('..') || isAbsolute(relativePath)) {
    throw new PathTraversalError(resolvedPath, allowedRoot);
  }
}

/**
 * Resolves a user-provided path and validates it stays within an allowed root.
 *
 * @param userPath - The user-provided path (may be relative or absolute)
 * @param allowedRoot - The root directory that the path must be contained within
 * @returns The resolved absolute path
 * @throws PathTraversalError if the resolved path escapes the allowed root
 */
export function resolveAndValidatePath(userPath: string, allowedRoot: string): string {
  const resolved = resolvePath(allowedRoot, userPath);
  validatePathContainment(resolved, allowedRoot);
  return resolved;
}

/**
 * Options for configuring path validation behavior.
 */
export interface PathValidationOptions {
  /**
   * The root directory that all paths must be contained within.
   * Defaults to process.cwd() if not specified.
   */
  allowedRoot?: string;
}

/**
 * Get the allowed root directory from options or environment.
 * Checks MLX_NODE_DATA_ROOT environment variable first, then falls back to cwd.
 *
 * @param options - Optional validation options
 * @returns The allowed root directory path
 */
export function getAllowedRoot(options?: PathValidationOptions): string {
  if (options?.allowedRoot) {
    return options.allowedRoot;
  }

  // Check environment variable for configurable data root
  const envRoot = process.env['MLX_NODE_DATA_ROOT'];
  if (envRoot) {
    return resolvePath(envRoot);
  }

  return process.cwd();
}
