import { Settings } from 'lucide-react';
import { type ReactElement, useEffect, useState } from 'react';

import { ButtonLink } from '../components/arc-x/button-link/button-link';
import { IconButton } from '../components/arc-x/icon-button/icon-button';
import { KeyValue } from '../components/arc-x/key-value/key-value';
import { PathText } from '../components/arc-x/path-text/path-text';
import { Rule } from '../components/arc-x/rule/rule';
import { Button } from '../components/arc/button/button';
import { Popover, PopoverContent, PopoverTrigger } from '../components/arc/popover/popover';
import SegmentedControl, { type Segment } from '../components/arc/segmented-control/segmented-control';
import { Switch } from '../components/arc/switch/switch';
import { Tooltip } from '../components/arc/tooltip/tooltip';
import { getJson, mutate } from '../lib/api';
import { resetModelColors } from '../lib/model-colors';
import { PROVIDERS } from '../lib/providers-strings';
import { toast } from '../lib/toast';
import type { HealthResponse } from '../lib/types';
import { type ThemePreference, useTheme } from '../theme/theme';
import styles from './sidebar.module.css';

const THEME_OPTIONS: Segment[] = [
  { value: 'system', label: 'System' },
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
];

function isThemePreference(value: string): value is ThemePreference {
  return value === 'system' || value === 'light' || value === 'dark';
}

export function resetColorsWithToast(): void {
  resetModelColors();
  toast.success('Model colors reset');
}

/** What fits the value column of the 288 px popover in 12 px mono, so the CSS ellipsis never cuts the tail too. */
const PATH_CHARS = 23;

/** The directories this window reads, fetched each time the popover opens; null until then or after a failure. */
function useDataPaths(open: boolean): HealthResponse | null {
  const [paths, setPaths] = useState<HealthResponse | null>(null);
  useEffect(() => {
    if (!open) return;
    let current = true;
    getJson<HealthResponse>('/health').then(
      (body) => {
        if (current) setPaths(body);
      },
      () => {
        if (current) setPaths(null);
      },
    );
    return () => {
      current = false;
    };
  }, [open]);
  return paths;
}

function pathItem(key: string, label: string, path: string) {
  return { key, label, value: <PathText path={path} maxChars={PATH_CHARS} />, mono: true, copy: path };
}

export interface SettingsPopoverProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  rail: boolean;
}

interface WorkSettings {
  claudeCode: boolean;
}

const CLAUDE_CODE = {
  title: 'Claude Code in Work',
  on: 'Work loads your Claude Code plugins, hooks, skills, agents and MCP servers, as in a terminal.',
  off: 'Work skips your Claude Code plugins, hooks, skills, agents, rules and MCP servers.',
  failed: 'Could not save the Claude Code setting',
};

/**
 * The app-wide Work settings, fetched each time the popover opens; null until then or after a failed read. A save
 * shows the new value at once and goes back on a failure.
 */
function useWorkSettings(open: boolean): { settings: WorkSettings | null; save: (next: WorkSettings) => void } {
  const [settings, setSettings] = useState<WorkSettings | null>(null);
  useEffect(() => {
    if (!open) return;
    let current = true;
    getJson<WorkSettings>('/chat/work/settings').then(
      (body) => {
        if (current) setSettings(body);
      },
      () => {
        if (current) setSettings(null);
      },
    );
    return () => {
      current = false;
    };
  }, [open]);
  const save = (next: WorkSettings): void => {
    const before = settings;
    setSettings(next);
    mutate<WorkSettings>('PUT', '/chat/work/settings', next).then(setSettings, () => {
      setSettings(before);
      toast.error(CLAUDE_CODE.failed);
    });
  };
  return { settings, save };
}

/** The gear in the sidebar footer: theme, model colors, Claude Code in Work, and where the data lives. */
export function SettingsPopover({ open, onOpenChange, rail }: SettingsPopoverProps): ReactElement {
  const theme = useTheme();
  const paths = useDataPaths(open);
  const work = useWorkSettings(open);
  return (
    <Popover open={open} onOpenChange={onOpenChange}>
      <Tooltip content="Settings" side={rail ? 'right' : 'top'}>
        <PopoverTrigger asChild>
          <IconButton label="Settings" tooltip={false} icon={<Settings size={16} strokeWidth={1.75} aria-hidden />} />
        </PopoverTrigger>
      </Tooltip>
      <PopoverContent side="top" align={rail ? 'start' : 'end'} aria-label="Settings" className={styles.settings}>
        <div className={styles.settingsBlock}>
          <span className={styles.settingsLabel}>Theme</span>
          <SegmentedControl
            label="Theme"
            className={styles.settingsTheme}
            options={THEME_OPTIONS}
            value={theme.preference}
            onValueChange={(value) => {
              if (isThemePreference(value)) theme.setPreference(value);
            }}
          />
        </div>
        <Rule tone="strong" />
        <div className={styles.settingsBlock}>
          <div className={styles.settingsRow}>
            <span className={styles.settingsTitle}>Model colors</span>
            <Button variant="ghost" size="sm" onClick={resetColorsWithToast}>
              Reset
            </Button>
          </div>
          <p className={styles.settingsCaption}>Colors follow each model across pages. Reset reassigns them.</p>
        </div>
        {work.settings !== null ? (
          <>
            <Rule tone="strong" />
            <div className={styles.settingsBlock}>
              <div className={styles.settingsRow}>
                <span className={styles.settingsTitle}>{CLAUDE_CODE.title}</span>
                <Switch
                  aria-label={CLAUDE_CODE.title}
                  checked={work.settings.claudeCode}
                  onCheckedChange={(claudeCode) => work.save({ claudeCode })}
                />
              </div>
              <p className={styles.settingsCaption}>{work.settings.claudeCode ? CLAUDE_CODE.on : CLAUDE_CODE.off}</p>
            </div>
          </>
        ) : null}
        <Rule tone="strong" />
        <div className={styles.settingsBlock}>
          <span className={styles.settingsLabel}>{PROVIDERS.settingsLabel}</span>
          <ButtonLink to="/providers" size="sm" variant="ghost" onClick={() => onOpenChange(false)}>
            {PROVIDERS.settingsLink}
          </ButtonLink>
        </div>
        {paths !== null ? (
          <>
            <Rule tone="strong" />
            <div className={styles.settingsBlock}>
              <span className={styles.settingsLabel}>Data</span>
              <KeyValue
                layout="rows"
                ariaLabel="Data"
                items={[
                  pathItem('models', 'Models', paths.modelsDir),
                  pathItem('traces', 'Traces', paths.tracesDir),
                  pathItem('runtime', 'Runtime', paths.runtimeDir),
                ]}
              />
            </div>
          </>
        ) : null}
      </PopoverContent>
    </Popover>
  );
}
