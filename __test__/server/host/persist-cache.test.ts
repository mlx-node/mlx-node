import { describe, expect, it } from 'vite-plus/test';

import {
  parsePersistEnv,
  PERSIST_PAGED_CACHE_DEFAULT,
  persistPagedCacheFlag,
  resolvePersistPagedCache,
} from '../../../packages/server/src/host/persist-cache.js';

describe('the paged KV disk cache switch', () => {
  it('reads the env as `parse_bool_env` in cold_tier.rs', () => {
    for (const on of ['1', 'true', ' ON ', 'Yes']) expect(parsePersistEnv(on)).toBe(true);
    for (const off of ['0', 'false', 'Off', ' no']) expect(parsePersistEnv(off)).toBe(false);
    for (const none of [undefined, '', 'maybe', '2']) expect(parsePersistEnv(none)).toBeUndefined();
  });

  it('resolves flag > MLX_PERSIST_PAGED_CACHE > the entry default (on)', () => {
    expect(PERSIST_PAGED_CACHE_DEFAULT).toEqual({ byDefault: true });
    expect(resolvePersistPagedCache(PERSIST_PAGED_CACHE_DEFAULT, {})).toBe(true);
    expect(resolvePersistPagedCache(PERSIST_PAGED_CACHE_DEFAULT, { MLX_PERSIST_PAGED_CACHE: '0' })).toBe(false);
    expect(resolvePersistPagedCache(PERSIST_PAGED_CACHE_DEFAULT, { MLX_PERSIST_PAGED_CACHE: 'junk' })).toBe(true);
    expect(resolvePersistPagedCache({ flag: false, byDefault: true }, { MLX_PERSIST_PAGED_CACHE: '1' })).toBe(false);
    expect(resolvePersistPagedCache({ flag: true, byDefault: true }, { MLX_PERSIST_PAGED_CACHE: '0' })).toBe(true);
    expect(resolvePersistPagedCache({ byDefault: false }, {})).toBe(false);
  });

  it('names the serve flag that carries a resolved switch', () => {
    expect(persistPagedCacheFlag(true)).toBe('--persist-cache');
    expect(persistPagedCacheFlag(false)).toBe('--no-persist-cache');
  });
});
