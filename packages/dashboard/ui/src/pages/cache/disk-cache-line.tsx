import type { ReactElement, ReactNode } from 'react';

import { Badge, type BadgeTone } from '../../components/arc/badge/badge';
import { pickProcess } from '../../lib/engine-view';
import { formatDuration, formatNumber } from '../../lib/format';
import { useRuntimeLive } from '../../lib/live';
import type { CacheScope, HostKind, RuntimeLive, RuntimeProcessLive } from '../../lib/types';

/** `WHOLE_PROMPT_CAPTURE_BLOCKS` of the server: a cap this high never ends a capture walk before the prompt does. */
export const WHOLE_PROMPT_CAPTURE_BLOCKS = 63_125;

export type DiskCacheSwitch = 'on' | 'off' | 'unknown' | 'no-engine';

export function diskCacheSwitch(process: Pick<RuntimeProcessLive, 'persistPagedCache'> | null): DiskCacheSwitch {
  if (process === null) return 'no-engine';
  if (process.persistPagedCache === null) return 'unknown';
  return process.persistPagedCache ? 'on' : 'off';
}

/**
 * The most one turn's capture walk may write: a limit, not a measured write. The time cap can end a walk early; the
 * next turn's walk picks up what it left. Null when the engine does not say (an older engine, or the cache is off).
 */
export function captureLimitText(capture: RuntimeProcessLive['coldCapture']): string | null {
  if (capture === null) return null;
  const most =
    capture.blocks >= WHOLE_PROMPT_CAPTURE_BLOCKS
      ? 'the whole prompt'
      : `${formatNumber(capture.blocks)} prefix ${capture.blocks === 1 ? 'block' : 'blocks'}`;
  return `Per turn: up to ${most}, ${formatDuration(capture.ms)} cap`;
}

const SWITCH: Record<DiskCacheSwitch, { word: string; tone: BadgeTone; note: string | null }> = {
  on: { word: 'On', tone: 'success', note: null },
  off: { word: 'Off', tone: 'warning', note: null },
  unknown: { word: 'Unknown', tone: 'neutral', note: 'engine started before this field' },
  'no-engine': { word: 'No engine', tone: 'neutral', note: 'inference is not running' },
};

const HOST_WORD: Record<HostKind, string> = {
  desktop: 'desktop',
  serve: 'mlx serve',
  launch: 'mlx launch',
  embedded: 'embedded',
};

/** Which engine the line reads, when more than one runs: their switches can differ. */
function engineNote(live: RuntimeLive | undefined, process: RuntimeProcessLive | null): string | null {
  if (live === undefined || process === null) return null;
  if (live.processes.filter((p) => p.status !== 'dead').length < 2) return null;
  return `${HOST_WORD[process.host]} engine, pid ${String(process.pid)}`;
}

function turnsText(scope: Pick<CacheScope, 'enabledTurns' | 'disabledTurns' | 'trendWindowDays'>): string {
  const days = `Last ${String(scope.trendWindowDays)} days, all cache folders`;
  const on = scope.enabledTurns;
  const off = scope.disabledTurns;
  if (on + off === 0) return `${days}: no turn recorded the cache state`;
  return `${days}: ${formatNumber(on)} ${on === 1 ? 'turn' : 'turns'} with it on, ${formatNumber(off)} off`;
}

/** A dot after an item, inside it: a wrapped line ends with the dot and never starts with one. */
function Dot(): ReactElement {
  return (
    <span aria-hidden="true" className="ml-2">
      ·
    </span>
  );
}

/**
 * The switch the running engine's host resolved (not proof of a write), the most one turn may capture, and how many
 * recorded turns ran with the tier on and off (in every cache folder, not only the one this page shows).
 */
export function DiskCacheLine({ scope }: { scope: CacheScope | undefined }): ReactElement {
  const live = useRuntimeLive();
  const process = pickProcess(live, null);
  const state = live === undefined ? null : diskCacheSwitch(process);
  const shown = state === null ? null : SWITCH[state];
  const limit = state === 'on' && process !== null ? captureLimitText(process.coldCapture) : null;
  const notes = [engineNote(live, process), shown?.note ?? null].filter((n): n is string => n !== null);
  const parts: string[] = [];
  if (limit !== null) parts.push(limit);
  if (scope !== undefined) parts.push(turnsText(scope));
  const badge: ReactNode =
    shown === null ? (
      <span className="text-muted">—</span>
    ) : (
      <Badge size="sm" tone={shown.tone} dot>
        {shown.word}
      </Badge>
    );
  return (
    <p className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 text-body" data-disk-cache={state ?? 'waiting'}>
      <span className="text-secondary">Disk cache:</span>
      <span className="inline-flex items-center text-secondary">
        {badge}
        {notes.length === 0 ? null : <span className="ml-2">{`(${notes.join('; ')})`}</span>}
        {parts.length === 0 ? null : <Dot />}
      </span>
      {parts.map((part, k) => (
        <span key={part} className="text-secondary tabular-nums">
          {part}
          {k < parts.length - 1 ? <Dot /> : null}
        </span>
      ))}
    </p>
  );
}
