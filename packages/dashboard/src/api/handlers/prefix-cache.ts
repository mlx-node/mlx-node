import { readdirSync, readFileSync, realpathSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

import {
  classifyProcess,
  liveFilePid,
  parseLiveFile,
  parseStateCache,
  parseStateNamespaces,
  pidAlive,
} from '../../live.js';
import {
  canLockNamespaces,
  defaultPrefixCacheRoot,
  isNamespaceName,
  PrefixCacheError,
  removePrefixNamespaceStep,
  scanPrefixCaches,
  type PrefixCacheDeleteResult,
  type PrefixCacheResponse,
  type PrefixScanRoot,
} from '../../prefix-cache.js';
import type { ApiPaths, ApiRequest } from '../context.js';
import { requireBody } from '../context.js';
import { ApiError } from '../errors.js';

const LIVE_FILES_MAX = 32;
const LIVE_FILE_BYTES_MAX = 256 * 1024;

export interface ListedPrefixRoot extends PrefixScanRoot {
  /** The real path the roots are merged by; the configured text when the folder is missing. */
  canonical: string;
  models: Map<string, string>;
}

function canonicalRoot(path: string): string {
  try {
    return realpathSync(path);
  } catch {
    return resolve(path);
  }
}

/**
 * The roots the page lists: the runtime's own, plus the cache dir each running engine's heartbeat names. An engine
 * whose heartbeat says nothing about its cache is attached nowhere.
 */
export function prefixCacheRoots(
  ctx: Pick<ApiPaths, 'prefixCacheRoot' | 'runtimeDir'>,
  now: number,
  isAlive: (pid: number) => boolean = pidAlive,
): ListedPrefixRoot[] {
  const base = ctx.prefixCacheRoot ?? defaultPrefixCacheRoot();
  const roots: ListedPrefixRoot[] = [
    { root: base, isDefault: true, engines: [], canonical: canonicalRoot(base), models: new Map() },
  ];
  const rootOf = (dir: string): ListedPrefixRoot => {
    const canonical = canonicalRoot(dir);
    const known = roots.find((root) => root.canonical === canonical);
    if (known !== undefined) return known;
    const added: ListedPrefixRoot = { root: dir, isDefault: false, engines: [], canonical, models: new Map() };
    roots.push(added);
    return added;
  };
  let names: string[];
  try {
    names = readdirSync(ctx.runtimeDir).sort();
  } catch {
    names = [];
  }
  let read = 0;
  for (const name of names) {
    const pid = liveFilePid(name);
    if (pid === null || !isAlive(pid)) continue;
    if (read >= LIVE_FILES_MAX) break;
    read += 1;
    const path = join(ctx.runtimeDir, name);
    try {
      if (statSync(path).size > LIVE_FILE_BYTES_MAX) continue;
      const raw = JSON.parse(readFileSync(path, 'utf8')) as unknown;
      const file = parseLiveFile(raw);
      if (file === null || classifyProcess(file, now, isAlive) === 'dead') continue;
      const state = parseStateCache((raw as { stateCache?: unknown }).stateCache);
      if (state === null) continue;
      const root = state.dir === null ? roots[0]! : rootOf(state.dir);
      root.engines.push({ pid: file.pid, host: file.host, limitBytes: state.limitBytes });
      for (const { model, namespace } of parseStateNamespaces((raw as { models?: unknown }).models)) {
        if (!root.models.has(namespace)) root.models.set(namespace, model);
      }
    } catch {
      continue;
    }
  }
  return roots;
}

export function handlePrefixCacheGet(ctx: ApiPaths): PrefixCacheResponse {
  const canLock = canLockNamespaces();
  const scan = scanPrefixCaches(prefixCacheRoots(ctx, Date.now()), { canLock });
  return { roots: scan.roots, counting: scan.counting, canLock };
}

const BODY_SHAPE = 'Body must be {"root": <folder>, "openedMs": <number>}';

function apiErrorOf(err: PrefixCacheError): ApiError {
  switch (err.kind) {
    case 'bad-name':
      return ApiError.badRequest(err.message);
    case 'not-found':
      return ApiError.notFound(err.message);
    case 'not-a-cache':
      return new ApiError('E_NOT_FOUND', err.message, err.kind);
    case 'held':
    case 'changed':
    case 'moved':
    case 'no-lock-check':
      return new ApiError('E_CONFLICT', err.message, err.kind);
  }
}

export function handlePrefixCacheDelete(ctx: ApiPaths, req: ApiRequest): PrefixCacheDeleteResult {
  const namespace = req.params.namespace ?? '';
  if (!isNamespaceName(namespace)) throw ApiError.badRequest('Not a model cache name: 32 lowercase hex digits.');
  const body = requireBody(req) as { root?: unknown; openedMs?: unknown } | null;
  const root = body?.root;
  const openedMs = body?.openedMs;
  if (typeof root !== 'string' || root.length === 0 || typeof openedMs !== 'number' || !Number.isFinite(openedMs)) {
    throw ApiError.badRequest(BODY_SHAPE);
  }
  const asked = canonicalRoot(root);
  const listed = prefixCacheRoots(ctx, Date.now()).find((candidate) => candidate.canonical === asked);
  if (listed === undefined) {
    throw ApiError.forbidden('This folder is no longer listed. Its engine stopped. Refresh the page.');
  }
  try {
    return removePrefixNamespaceStep(listed.canonical, namespace, openedMs);
  } catch (err) {
    if (err instanceof PrefixCacheError) throw apiErrorOf(err);
    throw err;
  }
}
