import {
  Box,
  Download,
  type LucideIcon,
  MessageCircle,
  MessageCirclePlus,
  MessageSquare,
  Monitor,
  Moon,
  PaletteIcon,
  PanelLeft,
  RefreshCw,
  SquareTerminal,
  Sun,
} from 'lucide-react';
import { createElement, type ReactNode, useEffect, useSyncExternalStore } from 'react';

import type { CommandItem } from '../components/arc/command-palette/command-palette';
import { CHAT } from '../lib/chat-strings';
import { chatTitle } from '../lib/chat-turns';
import type { ChatHostState, ConversationRow } from '../lib/chat-types';
import { formatBytes, formatRelativeTime } from '../lib/format';
import { tildePath } from '../lib/paths';
import { isoMs, sessionTitle } from '../lib/sessions';
import type { LocalModel, SessionRow } from '../lib/types';
import type { ThemePreference } from '../theme/theme';
import { NAV_ROUTES } from './routes';

const glyph = (icon: LucideIcon): ReactNode => createElement(icon, { size: 16, strokeWidth: 1.75 });

const THEME_ICONS: Record<ThemePreference, LucideIcon> = { system: Monitor, light: Sun, dark: Moon };

export interface AppCommand extends CommandItem {
  run(): void | Promise<void>;
}

const registered = new Map<symbol, readonly AppCommand[]>();
let pageCommands: readonly AppCommand[] = [];
const listeners = new Set<() => void>();

function publish(): void {
  pageCommands = [...registered.values()].flat();
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function snapshot(): readonly AppCommand[] {
  return pageCommands;
}

/** A page registers its own commands while it is mounted. */
export function useCommands(commands: readonly AppCommand[]): void {
  useEffect(() => {
    const token = Symbol('page-commands');
    registered.set(token, commands);
    publish();
    return () => {
      registered.delete(token);
      publish();
    };
  }, [commands]);
}

/** Every command the mounted pages registered, in mount order. */
export function usePageCommands(): readonly AppCommand[] {
  return useSyncExternalStore(subscribe, snapshot, snapshot);
}

export interface GlobalCommandDeps {
  navigate: (to: string) => void;
  refresh(): void | Promise<void>;
  /** Absent while the window is too narrow for an expanded sidebar. */
  toggleSidebar?: () => void;
  theme: ThemePreference;
  setTheme: (preference: ThemePreference) => void;
  resetColors: () => void;
}

const THEMES: ReadonlyArray<{ value: ThemePreference; label: string }> = [
  { value: 'system', label: 'System' },
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
];

/** The shell's own groups: Go to and Actions. */
export function globalCommands(deps: GlobalCommandDeps): AppCommand[] {
  const goTo: AppCommand[] = NAV_ROUTES.map((route, index) => ({
    id: `go:${route.id}`,
    label: route.label,
    group: 'Go to',
    keywords: route.section === undefined || route.section === 'top' ? [] : [route.section],
    shortcut: `⌘ ${index + 1}`,
    icon: route.icon === undefined ? undefined : glyph(route.icon),
    run: () => deps.navigate(route.path),
  }));
  goTo.push({
    id: 'go:welcome',
    label: 'Choose a model',
    description: 'First-run setup',
    group: 'Go to',
    keywords: ['onboarding', 'welcome', 'download'],
    icon: glyph(Download),
    run: () => deps.navigate('/welcome'),
  });
  const actions: AppCommand[] = [
    {
      id: 'act:new-chat',
      label: CHAT.newChat,
      group: 'Actions',
      keywords: ['chat', 'conversation'],
      icon: glyph(MessageCirclePlus),
      shortcut: '⌘ ⇧ O',
      run: () => deps.navigate('/chat?mode=chat'),
    },
    {
      id: 'act:new-work',
      label: CHAT.newWork,
      group: 'Actions',
      keywords: ['work', 'agent', 'folder'],
      icon: glyph(SquareTerminal),
      run: () => deps.navigate('/chat?mode=work'),
    },
    {
      id: 'act:refresh',
      label: 'Refresh data',
      description: 'Read new traces now',
      group: 'Actions',
      keywords: ['reload', 'ingest'],
      icon: glyph(RefreshCw),
      run: () => deps.refresh(),
    },
  ];
  if (deps.toggleSidebar !== undefined) {
    const toggle = deps.toggleSidebar;
    actions.push({
      id: 'act:sidebar',
      label: 'Toggle sidebar',
      group: 'Actions',
      keywords: ['collapse', 'rail'],
      icon: glyph(PanelLeft),
      shortcut: '⌘ B',
      run: () => toggle(),
    });
  }
  for (const theme of THEMES) {
    actions.push({
      id: `act:theme:${theme.value}`,
      label: `Theme: ${theme.label}`,
      description: deps.theme === theme.value ? 'Current' : undefined,
      group: 'Actions',
      keywords: ['appearance', 'dark mode'],
      icon: glyph(THEME_ICONS[theme.value]),
      run: () => deps.setTheme(theme.value),
    });
  }
  actions.push({
    id: 'act:reset-colors',
    label: 'Reset model colors',
    description: 'Reassign chart colors from scratch',
    group: 'Actions',
    keywords: ['palette'],
    icon: glyph(PaletteIcon),
    run: () => deps.resetColors(),
  });
  return [...goTo, ...actions];
}

export function modelCommands(models: readonly LocalModel[], navigate: (to: string) => void): AppCommand[] {
  return models.map((model) => ({
    id: `model:${model.name}`,
    label: model.name,
    description: [model.modelType, model.quant ?? 'full precision', formatBytes(model.sizeBytes)].join(' · '),
    group: 'Models',
    icon: glyph(Box),
    keywords: [model.modelType, model.quant ?? ''].filter((word) => word.length > 0),
    run: () => navigate(`/models?focus=${encodeURIComponent(model.name)}`),
  }));
}

export function sessionCommands(
  sessions: readonly SessionRow[],
  navigate: (to: string) => void,
  home: string | null,
): AppCommand[] {
  return sessions.map((session) => {
    const shown = tildePath(session.cwd, home);
    const parts = shown.split('/').filter((part) => part.length > 0);
    const folder = parts[parts.length - 1] ?? shown;
    const when = formatRelativeTime(isoMs(session.lastActiveAt));
    const description = `${folder} · ${when}`;
    return {
      id: `session:${session.id}`,
      label: sessionTitle(session),
      description: session.live ? `${description} · live` : description,
      group: 'Sessions',
      icon: glyph(MessageSquare),
      keywords: [session.cwd, session.branch ?? '', session.model ?? ''].filter((word) => word.length > 0),
      run: () => navigate(`/sessions/${encodeURIComponent(session.id)}`),
    };
  });
}

/** The newest 20 conversations of both modes, by `updatedAt`, each with its mode, folder, age and live state. */
export function conversationCommands(
  rows: readonly ConversationRow[],
  active: ChatHostState['active'],
  navigate: (to: string) => void,
  home: string | null,
): AppCommand[] {
  return [...rows]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, 20)
    .map((row) => {
      const parts: string[] = [row.mode === 'work' ? CHAT.modeWork : CHAT.modeChat];
      if (row.mode === 'work' && row.cwd !== null) {
        const shown = tildePath(row.cwd, home);
        const segments = shown.split('/').filter((part) => part.length > 0);
        parts.push(segments[segments.length - 1] ?? shown);
      }
      parts.push(formatRelativeTime(row.updatedAt));
      const live = active.find((entry) => entry.conversationId === row.id);
      if (live?.waiting === true) parts.push(CHAT.waiting.toLowerCase());
      else if (live?.busy === true) parts.push(CHAT.running.toLowerCase());
      return {
        id: `conversation:${row.id}`,
        label: chatTitle(row.title, row.mode),
        description: parts.join(' · '),
        group: CHAT.commands.conversations,
        icon: glyph(row.mode === 'work' ? SquareTerminal : MessageCircle),
        keywords: [row.mode, row.cwd ?? '', row.modelId].filter((word) => word.length > 0),
        run: () => navigate(`/chat?c=${encodeURIComponent(row.id)}`),
      };
    });
}
