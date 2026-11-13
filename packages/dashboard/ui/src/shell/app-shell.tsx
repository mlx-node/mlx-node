import { AnimatePresence, animate, motion, useMotionValue, useTransform } from 'motion/react';
import {
  type CSSProperties,
  type ReactElement,
  type ReactNode,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';
import { useLocation, useNavigate, useNavigationType, useOutlet } from 'react-router-dom';

import { motionX } from '../components/arc-x/motion/motion';
import { useStaticMotion } from '../components/arc-x/motion/use-static-motion';
import { mutate } from '../lib/api';
import { errMessage } from '../lib/errors';
import { ensureModelColors } from '../lib/model-colors';
import { toast } from '../lib/toast';
import type { MetricsModels, ModelsResponse } from '../lib/types';
import { reloadMountedQueries, useJson } from '../lib/use-api';
import { storedMode } from '../pages/chat/drafts';
import { useTheme } from '../theme/theme';
import styles from './app-shell.module.css';
import { CommandMenu } from './command-menu';
import { globalCommands } from './commands';
import { ConnectionBanner } from './connection-banner';
import { routeCrumbs, useCurrentCrumbs } from './crumbs';
import { type Hotkey, focusPageSearch, useHotkeys } from './hotkeys';
import { NAV_ROUTES } from './routes';
import { ScrollRootContext, useScrollMemory } from './scroll-root';
import { resetColorsWithToast } from './settings-popover';
import { Sidebar } from './sidebar';
import { TitleBar } from './title-bar';
import { useShellStatus } from './use-shell-status';

export const SIDEBAR_STORAGE_KEY = 'mlx-node:sidebar:v1';
/** Below this window width the sidebar is always the rail. */
export const NARROW_BELOW_PX = 880;
const EXPANDED_WIDTH_PX = 208;
const RAIL_WIDTH_PX = 52;

type SidebarPreference = 'expanded' | 'rail';

function readSidebarPreference(): SidebarPreference {
  try {
    return globalThis.localStorage?.getItem(SIDEBAR_STORAGE_KEY) === 'rail' ? 'rail' : 'expanded';
  } catch {
    return 'expanded';
  }
}

function storeSidebarPreference(preference: SidebarPreference): void {
  try {
    globalThis.localStorage?.setItem(SIDEBAR_STORAGE_KEY, preference);
  } catch {
    // The choice still applies to this window.
  }
}

/**
 * Seeds the color registry in a fixed order: traced models in first-seen order, then every served id (a model with
 * none contributes its name), sorted. The second seed waits for the first to answer or fail.
 */
function useModelColorSeed(): void {
  const traced = useJson<MetricsModels>('/metrics/models');
  const local = useJson<ModelsResponse>('/models');
  const tracedSettled = traced.data !== undefined || traced.error !== undefined;
  useEffect(() => {
    if (traced.data !== undefined) ensureModelColors(traced.data.models.map((entry) => entry.model));
    if (!tracedSettled || local.data === undefined) return;
    const served: string[] = [];
    for (const model of local.data.models) {
      const ids = model.servedIds ?? [];
      if (ids.length === 0) served.push(model.name);
      else served.push(...ids);
    }
    served.sort();
    ensureModelColors(served);
  }, [traced.data, tracedSettled, local.data]);
}

function subscribeResize(listener: () => void): () => void {
  window.addEventListener('resize', listener);
  return () => window.removeEventListener('resize', listener);
}

function isNarrow(): boolean {
  return window.innerWidth < NARROW_BELOW_PX;
}

/** The route element, swapped with the `page` motion: the old page leaves, then the new one mounts and enters. */
function PageFrame({ pathname, onSwap }: { pathname: string; onSwap: () => void }): ReactElement {
  const outlet = useOutlet();
  const still = useStaticMotion();
  if (still) {
    return (
      <div key={pathname} className={styles.pageFrame}>
        {outlet}
      </div>
    );
  }
  const { in: enter, out } = motionX.page;
  return (
    <AnimatePresence mode="wait" initial={false} onExitComplete={onSwap}>
      <motion.div
        key={pathname}
        className={styles.pageFrame}
        initial={{ opacity: 0, y: enter.y }}
        animate={{ opacity: 1, y: 0 }}
        exit={{
          opacity: 0,
          y: out.y,
          filter: `blur(${out.blur}px)`,
          transition: { duration: out.duration, ease: [...out.ease] },
        }}
        transition={{ duration: enter.duration, ease: [...enter.ease] }}
      >
        {outlet}
      </motion.div>
    </AnimatePresence>
  );
}

/** The frame around every page but `/welcome`: title bar, sidebar, the one scroll root, the palette. */
export function AppShell(): ReactElement {
  const location = useLocation();
  const navigationType = useNavigationType();
  const navigate = useNavigate();
  const still = useStaticMotion();
  const theme = useTheme();
  const status = useShellStatus();
  useModelColorSeed();

  const narrow = useSyncExternalStore(subscribeResize, isNarrow, () => false);
  const [preference, setPreference] = useState<SidebarPreference>(readSidebarPreference);
  const rail = narrow || preference === 'rail';
  const toggleSidebar = useCallback(() => {
    setPreference((current) => {
      const next: SidebarPreference = current === 'rail' ? 'expanded' : 'rail';
      storeSidebarPreference(next);
      return next;
    });
  }, []);

  const width = useMotionValue(rail ? RAIL_WIDTH_PX : EXPANDED_WIDTH_PX);
  const sidebarWidth = useTransform(width, (value) => `${value}px`);
  useEffect(() => {
    const target = rail ? RAIL_WIDTH_PX : EXPANDED_WIDTH_PX;
    if (still) {
      width.jump(target);
      return;
    }
    const controls = animate(width, target, motionX.spring.glide);
    return () => controls.stop();
  }, [rail, still, width]);

  const [paletteOpen, setPaletteOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [main, setMain] = useState<HTMLElement | null>(null);

  const refresh = useCallback(async () => {
    try {
      await mutate('POST', '/ingest');
      reloadMountedQueries();
      toast.success('Data refreshed');
    } catch (err) {
      toast.error('Could not refresh data', { description: errMessage(err) });
    }
  }, []);

  const commands = useMemo(
    () =>
      globalCommands({
        navigate: (to) => void navigate(to),
        refresh,
        toggleSidebar: narrow ? undefined : toggleSidebar,
        theme: theme.preference,
        setTheme: theme.setPreference,
        resetColors: resetColorsWithToast,
      }),
    [navigate, refresh, narrow, toggleSidebar, theme.preference, theme.setPreference],
  );
  const navigateTo = useCallback((to: string) => void navigate(to), [navigate]);

  const hotkeys = useMemo<Hotkey[]>(
    () => [
      { key: 'k', meta: true, allowInInput: true, run: () => setPaletteOpen((open) => !open) },
      ...NAV_ROUTES.map((route, index) => ({
        key: String(index + 1),
        meta: true,
        allowInInput: true,
        run: () => void navigate(route.path),
      })),
      { key: 'o', meta: true, shift: true, allowInInput: true, run: () => void navigate(`/chat?mode=${storedMode()}`) },
      { key: 'b', meta: true, allowInInput: true, run: () => (narrow ? undefined : toggleSidebar()) },
      { key: ',', meta: true, allowInInput: true, run: () => setSettingsOpen(true) },
      { key: '/', run: () => focusPageSearch() },
    ],
    [navigate, narrow, toggleSidebar],
  );
  useHotkeys(hotkeys);

  const crumbs = useCurrentCrumbs(location.pathname);

  const scroll = useScrollMemory(main);
  const shownPath = useRef(location.pathname);
  const shownKey = useRef(location.key);
  const latest = useRef({ key: location.key, pop: navigationType === 'POP' });
  const [swaps, setSwaps] = useState(0);
  const [announcement, setAnnouncement] = useState('');

  const afterSwap = useCallback(() => {
    const { key, pop } = latest.current;
    scroll.settle(pop ? 'pop' : 'push', key);
    main?.focus({ preventScroll: true });
    setSwaps((count) => count + 1);
  }, [scroll, main]);

  const actions = useRef({ remember: scroll.remember, afterSwap });
  useLayoutEffect(() => {
    actions.current = { remember: scroll.remember, afterSwap };
  });

  const { key, pathname } = location;
  const pop = navigationType === 'POP';
  useLayoutEffect(() => {
    latest.current = { key, pop };
    if (shownPath.current === pathname) {
      shownKey.current = key;
      return;
    }
    actions.current.remember(shownKey.current);
    shownPath.current = pathname;
    shownKey.current = key;
    if (still) actions.current.afterSwap();
  }, [key, pathname, pop, still]);

  useEffect(() => {
    if (swaps === 0) return;
    const heading = main?.querySelector('h1')?.textContent?.trim();
    const fallback = routeCrumbs(shownPath.current).at(-1)?.label ?? '';
    setAnnouncement(heading !== undefined && heading.length > 0 ? heading : fallback);
  }, [swaps, main]);

  const shellStyle = { '--sidebar-w': sidebarWidth } as unknown as CSSProperties;
  return (
    <ScrollRootContext.Provider value={main}>
      <motion.div data-shell="" className={styles.shell} style={shellStyle} data-sidebar={rail ? 'rail' : 'expanded'}>
        <SkipLink target={main} />
        <TitleBar
          crumbs={crumbs}
          narrow={narrow}
          rail={rail}
          onToggleSidebar={toggleSidebar}
          onSearch={() => setPaletteOpen(true)}
        />
        <Sidebar status={status} rail={rail} settingsOpen={settingsOpen} onSettingsOpenChange={setSettingsOpen} />
        <main ref={setMain} data-scroll-root="" tabIndex={-1} className={styles.main} id="content">
          <ConnectionBanner connected={status.connected} />
          <PageFrame pathname={location.pathname} onSwap={afterSwap} />
        </main>
        <div className={styles.announcer} aria-live="polite" role="status">
          {announcement}
        </div>
      </motion.div>
      <CommandMenu open={paletteOpen} onOpenChange={setPaletteOpen} commands={commands} navigate={navigateTo} />
    </ScrollRootContext.Provider>
  );
}

function SkipLink({ target }: { target: HTMLElement | null }): ReactNode {
  return (
    <a
      href="#content"
      className={styles.skip}
      onClick={(event) => {
        event.preventDefault();
        target?.focus();
      }}
    >
      Skip to content
    </a>
  );
}
