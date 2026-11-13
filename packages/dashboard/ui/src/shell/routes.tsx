import {
  Boxes,
  Bot,
  Cpu,
  Globe,
  HardDrive,
  LayoutDashboard,
  type LucideIcon,
  MessageCircle,
  MessagesSquare,
  TrendingUp,
} from 'lucide-react';
import type { ComponentType } from 'react';

import Cache from '../pages/cache';
import Chat from '../pages/chat';
import CodingAgents from '../pages/coding-agents';
import Engine from '../pages/engine';
import Metrics from '../pages/metrics';
import Models from '../pages/models';
import Overview from '../pages/overview';
import Providers from '../pages/providers';
import SessionDetail from '../pages/session-detail';
import Sessions from '../pages/sessions';

export type NavSection = 'top' | 'observe' | 'manage';

export interface AppRoute {
  id:
    | 'overview'
    | 'chat'
    | 'sessions'
    | 'session-detail'
    | 'metrics'
    | 'engine'
    | 'cache'
    | 'models'
    | 'coding-agents'
    | 'providers';
  path: string;
  label: string;
  /** Present on nav items only. */
  icon?: LucideIcon;
  section?: NavSection;
  /** Parent route id for detail pages; drives the breadcrumb and the active nav item. */
  parent?: AppRoute['id'];
  element: ComponentType;
  end?: boolean;
}

/** The one route table: routes, sidebar, breadcrumb, ⌘1…9 and the palette read it. Nav items keep this order. */
export const APP_ROUTES: readonly AppRoute[] = [
  { id: 'overview', path: '/', label: 'Overview', icon: LayoutDashboard, section: 'top', element: Overview, end: true },
  { id: 'chat', path: '/chat', label: 'Chat', icon: MessageCircle, section: 'top', element: Chat },
  { id: 'sessions', path: '/sessions', label: 'Sessions', icon: MessagesSquare, section: 'observe', element: Sessions },
  { id: 'session-detail', path: '/sessions/:id', label: 'Session', parent: 'sessions', element: SessionDetail },
  { id: 'metrics', path: '/metrics', label: 'Metrics', icon: TrendingUp, section: 'observe', element: Metrics },
  { id: 'engine', path: '/engine', label: 'Engine', icon: Cpu, section: 'observe', element: Engine },
  { id: 'cache', path: '/cache', label: 'Cache', icon: HardDrive, section: 'observe', element: Cache },
  { id: 'models', path: '/models', label: 'Models', icon: Boxes, section: 'manage', element: Models },
  {
    id: 'coding-agents',
    path: '/coding-agents',
    label: 'Coding Agents',
    icon: Bot,
    section: 'manage',
    element: CodingAgents,
  },
  { id: 'providers', path: '/providers', label: 'Web Search', icon: Globe, section: 'manage', element: Providers },
];

export const NAV_SECTIONS: ReadonlyArray<{ id: NavSection; label: string | null }> = [
  { id: 'top', label: null },
  { id: 'observe', label: 'Observe' },
  { id: 'manage', label: 'Manage' },
];

/** The sidebar items in table order; item n answers ⌘n. */
export const NAV_ROUTES: readonly AppRoute[] = APP_ROUTES.filter((route) => route.icon !== undefined);

function segments(path: string): string[] {
  return path.split('/').filter((part) => part.length > 0);
}

function matches(route: AppRoute, pathname: string): boolean {
  const want = segments(route.path);
  const have = segments(pathname);
  if (want.length !== have.length) return false;
  return want.every((part, index) => part.startsWith(':') || part === have[index]);
}

/** The route whose path matches `pathname` exactly (a `:param` matches one segment), or undefined. */
export function routeForPath(pathname: string): AppRoute | undefined {
  return APP_ROUTES.find((route) => matches(route, pathname));
}

export function routeById(id: AppRoute['id']): AppRoute | undefined {
  return APP_ROUTES.find((route) => route.id === id);
}
