import type { ReactElement } from 'react';
import { Link } from 'react-router-dom';

import { LiveDot, type LiveDotProps } from '../components/arc-x/live-dot/live-dot';
import { LiveLabel } from '../components/arc-x/live-label/live-label';
import { SwapText } from '../components/arc-x/swap-text/swap-text';
import { Tooltip } from '../components/arc/tooltip/tooltip';
import { SettingsPopover } from './settings-popover';
import styles from './sidebar.module.css';
import type { ShellStatus } from './use-shell-status';

export interface FooterLine {
  tone: LiveDotProps['tone'];
  pulse: boolean;
  /** The label shimmers: work in progress that will end. */
  working: boolean;
  line1: string;
  line2: string | null;
}

/** What the footer says for a shell status. */
export function footerLine(status: ShellStatus): FooterLine {
  if (!status.connected) return { tone: 'warning', pulse: false, working: false, line1: 'Reconnecting…', line2: null };
  switch (status.engine) {
    case 'running': {
      const primary = status.primary;
      let line2: string | null = null;
      if (primary !== null) {
        const models = `${primary.modelCount} ${primary.modelCount === 1 ? 'model' : 'models'}`;
        // The 208 px sidebar fits two of the three parts; the port gives way while turns run.
        line2 = primary.inFlight > 0 ? `${models} · ${primary.inFlight} active` : `:${primary.port} · ${models}`;
      }
      return { tone: 'success', pulse: true, working: false, line1: 'Inference running', line2 };
    }
    case 'loading':
      return { tone: 'accent', pulse: true, working: true, line1: 'Loading model…', line2: null };
    case 'starting':
      return { tone: 'accent', pulse: true, working: true, line1: 'Starting inference…', line2: null };
    case 'restarting':
      return { tone: 'warning', pulse: true, working: true, line1: 'Restarting inference…', line2: null };
    case 'unresponsive':
      return { tone: 'warning', pulse: false, working: false, line1: 'Inference not responding', line2: null };
    case 'failed':
      return {
        tone: 'danger',
        pulse: false,
        working: false,
        line1: 'Inference failed',
        line2: 'Open Engine for details',
      };
    case 'untrusted':
      return {
        tone: 'danger',
        pulse: false,
        working: false,
        line1: 'Output not trustworthy',
        line2: 'Open Engine for details',
      };
    case 'stopped':
      return {
        tone: 'neutral',
        pulse: false,
        working: false,
        line1: 'Inference stopped',
        line2: 'Start it from the menu bar',
      };
    case undefined:
      return { tone: 'neutral', pulse: false, working: false, line1: 'Control Panel ready', line2: null };
  }
}

export interface StatusFooterProps {
  status: ShellStatus;
  rail: boolean;
  settingsOpen: boolean;
  onSettingsOpenChange: (open: boolean) => void;
}

export function StatusFooter({ status, rail, settingsOpen, onSettingsOpenChange }: StatusFooterProps): ReactElement {
  const line = footerLine(status);
  const name = line.line2 === null ? line.line1 : `${line.line1}, ${line.line2}`;
  const dot = <LiveDot tone={line.tone} pulse={line.pulse} label={line.line1} />;
  const link = (
    <Link to="/engine" className={styles.status} aria-label={name} data-tone={line.tone}>
      <span className={styles.statusDot}>{dot}</span>
      <span className={styles.statusText}>
        <LiveLabel active={line.working} className={styles.line1}>
          {line.line1}
        </LiveLabel>
        {line.line2 !== null && !rail ? (
          <span className={styles.line2}>
            <SwapText text={line.line2} granularity="char" />
          </span>
        ) : null}
      </span>
    </Link>
  );
  return (
    <footer className={styles.footer} data-mode={rail ? 'rail' : 'expanded'}>
      {rail ? (
        <Tooltip content={name} side="right">
          {link}
        </Tooltip>
      ) : (
        link
      )}
      <SettingsPopover open={settingsOpen} onOpenChange={onSettingsOpenChange} rail={rail} />
    </footer>
  );
}
