import { useLayoutEffect, useState, useSyncExternalStore } from 'react';
import { useLocation } from 'react-router-dom';

import { routeById, routeForPath } from './routes';

export interface Crumb {
  label: string;
  to?: string;
}

interface Registered {
  pathname: string;
  crumbs: readonly Crumb[];
}

const FIELD = '\u0000';
const ITEM = '\u0001';

let current: Registered | null = null;
const listeners = new Set<() => void>();

function notify(): void {
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function snapshot(): Registered | null {
  return current;
}

function crumbsKey(crumbs: readonly Crumb[]): string {
  return crumbs.map((crumb) => `${crumb.label}${FIELD}${crumb.to ?? ''}`).join(ITEM);
}

function parseKey(key: string): Crumb[] {
  if (key.length === 0) return [];
  return key.split(ITEM).map((item) => {
    const at = item.indexOf(FIELD);
    const to = item.slice(at + FIELD.length);
    return to.length > 0 ? { label: item.slice(0, at), to } : { label: item.slice(0, at) };
  });
}

/** Called by `<PageHeader>`: the title bar shows these crumbs while the page that registered them is current. */
export function useCrumbs(crumbs: readonly Crumb[]): void {
  const location = useLocation();
  const [pathname] = useState(location.pathname);
  const key = crumbsKey(crumbs);
  useLayoutEffect(() => {
    const entry: Registered = { pathname, crumbs: parseKey(key) };
    current = entry;
    notify();
    return () => {
      if (current !== entry) return;
      current = null;
      notify();
    };
  }, [pathname, key]);
}

/** Crumbs of a pathname from the route table alone: what the bar shows from the click until the page registers its own. */
export function routeCrumbs(pathname: string): Crumb[] {
  const route = routeForPath(pathname);
  if (route === undefined) return [];
  const parent = route.parent === undefined ? undefined : routeById(route.parent);
  if (parent === undefined) return [{ label: route.label }];
  return [{ label: parent.label, to: parent.path }, { label: route.label }];
}

/** The crumbs the title bar shows for `pathname`. */
export function useCurrentCrumbs(pathname: string): readonly Crumb[] {
  const registered = useSyncExternalStore(subscribe, snapshot, snapshot);
  if (registered !== null && registered.pathname === pathname && registered.crumbs.length > 0) return registered.crumbs;
  return routeCrumbs(pathname);
}
