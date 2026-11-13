import { ArrowDown } from 'lucide-react';
import { LayoutGroup, motion } from 'motion/react';
import { type ReactElement, type ReactNode, useState } from 'react';
import { NavLink } from 'react-router-dom';

import { LiveDot } from '../components/arc-x/live-dot/live-dot';
import { motionX } from '../components/arc-x/motion/motion';
import { useStaticMotion } from '../components/arc-x/motion/use-static-motion';
import { Tooltip } from '../components/arc/tooltip/tooltip';
import { CHAT, fill } from '../lib/chat-strings';
import { type AppRoute, NAV_ROUTES, NAV_SECTIONS } from './routes';
import styles from './sidebar.module.css';
import { StatusFooter } from './status-footer';
import type { ShellStatus } from './use-shell-status';

export interface SidebarProps {
  status: ShellStatus;
  rail: boolean;
  settingsOpen: boolean;
  onSettingsOpenChange: (open: boolean) => void;
}

interface Badge {
  node: ReactNode;
  /** Tooltip addition in the rail. */
  text: string;
  tone: 'accent' | 'success' | 'warning';
}

function badgeFor(route: AppRoute, status: ShellStatus): Badge | null {
  if (route.id === 'chat' && status.chat !== null) {
    const { waiting, running } = status.chat;
    if (waiting > 0) {
      const label = fill(CHAT.badgeWaiting, { n: waiting });
      return { node: <LiveDot tone="warning" pulse size={6} label={label} />, text: label, tone: 'warning' };
    }
    if (running > 0) {
      const label = fill(CHAT.badgeRunning, { n: running });
      return { node: <LiveDot tone="success" pulse size={6} label={label} />, text: label, tone: 'success' };
    }
  }
  if (route.id === 'models' && status.runningDownloads !== null && status.runningDownloads > 0) {
    const count = status.runningDownloads;
    return {
      node: (
        <span className={styles.count}>
          {count}
          <ArrowDown size={12} strokeWidth={1.75} aria-hidden />
        </span>
      ),
      text: `${count} ${count === 1 ? 'download' : 'downloads'} running`,
      tone: 'accent',
    };
  }
  if (route.id === 'engine' && status.primary !== null && status.primary.inFlight > 0) {
    const label = `${status.primary.inFlight} active`;
    return { node: <LiveDot tone="success" pulse size={6} label={label} />, text: label, tone: 'success' };
  }
  return null;
}

function NavItem({ route, badge, rail }: { route: AppRoute; badge: Badge | null; rail: boolean }): ReactElement {
  const still = useStaticMotion();
  const Icon = route.icon;
  const link = (
    <NavLink to={route.path} end={route.end} className={styles.item}>
      {({ isActive }) => (
        <>
          {isActive ? (
            still ? (
              <span className={styles.highlight} aria-hidden="true" />
            ) : (
              <motion.span
                // A mode switch remounts the link (the rail wraps it in a Tooltip): one id per mode keeps the
                // highlight from gliding there from the old row position.
                layoutId={rail ? 'nav-active-rail' : 'nav-active'}
                className={styles.highlight}
                transition={motionX.spring.glide}
                aria-hidden="true"
              />
            )
          ) : null}
          <span className={styles.icon} aria-hidden="true">
            {Icon === undefined ? null : <Icon size={16} strokeWidth={1.75} />}
            {badge !== null && rail ? <span className={styles.badgeDot} data-tone={badge.tone} /> : null}
          </span>
          <span className={styles.label}>{route.label}</span>
          {badge !== null && !rail ? <span className={styles.badge}>{badge.node}</span> : null}
        </>
      )}
    </NavLink>
  );
  if (!rail) return <li className={styles.row}>{link}</li>;
  const tip = badge === null ? route.label : `${route.label} · ${badge.text}`;
  return (
    <li className={styles.row}>
      <Tooltip content={tip} side="right">
        {link}
      </Tooltip>
    </li>
  );
}

/** Nine nav items in three sections, then the live status footer. */
export function Sidebar({ status, rail, settingsOpen, onSettingsOpenChange }: SidebarProps): ReactElement {
  // Labels fade only after a mode switch, never on the first paint.
  const [shownRail, setShownRail] = useState(rail);
  const [switched, setSwitched] = useState(false);
  if (shownRail !== rail) {
    setShownRail(rail);
    setSwitched(true);
  }
  return (
    <aside
      data-nav=""
      aria-label="Sidebar"
      className={styles.root}
      data-mode={rail ? 'rail' : 'expanded'}
      data-switched={switched ? '' : undefined}
    >
      <div className={styles.scroll}>
        <nav aria-label="Sections" className={styles.nav}>
          <LayoutGroup id="sidebar">
            {NAV_SECTIONS.map((section) => {
              const routes = NAV_ROUTES.filter((route) => route.section === section.id);
              if (routes.length === 0) return null;
              return (
                <div key={section.id} className={styles.section}>
                  {section.label === null ? null : (
                    <p className={styles.sectionLabel}>
                      <span className={styles.sectionText}>{section.label}</span>
                    </p>
                  )}
                  <ul className={styles.list}>
                    {routes.map((route) => (
                      <NavItem key={route.id} route={route} badge={badgeFor(route, status)} rail={rail} />
                    ))}
                  </ul>
                </div>
              );
            })}
          </LayoutGroup>
        </nav>
      </div>
      <StatusFooter
        status={status}
        rail={rail}
        settingsOpen={settingsOpen}
        onSettingsOpenChange={onSettingsOpenChange}
      />
    </aside>
  );
}
