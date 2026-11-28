/**
 * Reads the live router location directly from a React Router navigator (e.g. `BrowserHistory` or `MemoryHistory`)
 * so guards can see URL updates synchronously before React commits the route transition.
 */

export interface LiveLocation {
  pathname: string;
  search: string;
  hash: string;
  key?: string;
}

export function stripBasename(pathname: string, basename: string): string | null {
  let base = basename;
  while (base.length > 1 && base.endsWith('/')) base = base.slice(0, -1);
  if (base === '' || base === '/') return pathname;
  if (!base.startsWith('/')) base = `/${base}`;
  if (!pathname.toLowerCase().startsWith(base.toLowerCase())) return null;
  const nextChar = pathname.charAt(base.length);
  if (nextChar !== '' && nextChar !== '/') return null;
  return pathname.slice(base.length) || '/';
}

export function readLiveLocation(navigator: unknown, basename: string): LiveLocation | null {
  const raw = (navigator as { location?: Partial<LiveLocation> } | null | undefined)?.location;
  if (raw === undefined || typeof raw.pathname !== 'string') return null;
  const pathname = stripBasename(raw.pathname, basename);
  if (pathname === null) return null;
  return {
    pathname,
    search: typeof raw.search === 'string' ? raw.search : '',
    hash: typeof raw.hash === 'string' ? raw.hash : '',
    ...(typeof raw.key === 'string' ? { key: raw.key } : {}),
  };
}
