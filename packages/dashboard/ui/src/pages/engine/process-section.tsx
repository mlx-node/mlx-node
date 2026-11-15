import type { ReactElement } from 'react';

import { KeyValue } from '../../components/arc-x/key-value/key-value';
import { LiveLabel } from '../../components/arc-x/live-label/live-label';
import { Panel } from '../../components/arc-x/panel/panel';
import { Placeholder } from '../../components/arc-x/placeholder/placeholder';
import { Alert } from '../../components/arc/alert/alert';
import { Badge } from '../../components/arc/badge/badge';
import { nativeErrorText } from '../../lib/engine-view';
import { formatClock, formatDuration, formatNumber, formatRelativeTime } from '../../lib/format';
import type { EngineStatus, RuntimeLive, RuntimeProcessLive, SupervisorLive } from '../../lib/types';

const BADGE: Record<EngineStatus, { tone: 'success' | 'info' | 'warning' | 'danger' | 'neutral'; text: string }> = {
  running: { tone: 'success', text: 'Running' },
  loading: { tone: 'info', text: 'Loading a model' },
  starting: { tone: 'info', text: 'Starting' },
  restarting: { tone: 'warning', text: 'Restarting' },
  unresponsive: { tone: 'warning', text: 'Not responding' },
  failed: { tone: 'danger', text: 'Failed' },
  untrusted: { tone: 'danger', text: 'Not trustworthy' },
  stopped: { tone: 'neutral', text: 'Stopped' },
};

export function StatusBadge({ status }: { status: EngineStatus | undefined }): ReactElement {
  if (status === undefined) return <Placeholder width={72} height={20} radius="pill" inline />;
  const badge = BADGE[status];
  const active = status === 'loading' || status === 'starting' || status === 'restarting';
  return (
    <Badge tone={badge.tone} dot size="sm">
      <LiveLabel active={active}>{badge.text}</LiveLabel>
    </Badge>
  );
}

/** The alert under the header for a status that needs one; null otherwise. */
export function StatusAlert({
  live,
  process,
}: {
  live: RuntimeLive | undefined;
  process: RuntimeProcessLive | null;
}): ReactElement | null {
  if (live === undefined) return null;
  const supervisor = live.supervisor;
  switch (live.status) {
    case 'restarting':
      return (
        <Alert tone="warning" title="Inference is restarting">
          {`Failures in a row: ${formatNumber(supervisor?.streak ?? 0)}.`}
        </Alert>
      );
    case 'unresponsive': {
      const age = process?.ageMs ?? live.processes.find((item) => item.status === 'unresponsive')?.ageMs ?? null;
      return (
        <Alert tone="warning" title="Inference is not responding">
          {`No heartbeat for ${formatDuration(age)}. The process is alive; a long model load can block it.`}
        </Alert>
      );
    }
    case 'failed': {
      const exit = supervisor?.lastExit ?? null;
      const exitText =
        exit === null
          ? 'Last exit: —.'
          : `Last exit: ${exit.verdict}, code ${exit.code === null ? '—' : String(exit.code)}, signal ${exit.signal ?? '—'}, ${formatClock(exit.atMs)}.`;
      const shortLived = supervisor?.shortLivedFailures ?? 0;
      const failuresText = shortLived > 0 ? `Short-lived failures in a row: ${formatNumber(shortLived)}. ` : '';
      return (
        <Alert tone="danger" title="Inference failed">
          {`It stopped and was not restarted. ${failuresText}${exitText}`}
        </Alert>
      );
    }
    case 'untrusted': {
      const errors = supervisor === null ? '—' : nativeErrorText(supervisor);
      const plural =
        supervisor !== null && supervisor.nativeErrorCount === 1 && !supervisor.nativeErrorsCapped ? 'error' : 'errors';
      return (
        <Alert tone="danger" title="Output not trustworthy">
          {`The process reported ${errors} native ${plural}, last in ${supervisor?.lastNativeErrorContext ?? 'unknown'} at ${formatClock(supervisor?.lastNativeErrorAt ?? null)}. Restart it from the menu bar.`}
        </Alert>
      );
    }
    default:
      return null;
  }
}

function lastExitText(supervisor: SupervisorLive): string | null {
  const exit = supervisor.lastExit;
  if (exit === null) return null;
  const parts = [exit.verdict, `code ${exit.code === null ? '—' : String(exit.code)}`];
  if (exit.signal !== null) parts.push(exit.signal);
  return parts.join(' · ');
}

export function SupervisorPanel({ supervisor }: { supervisor: SupervisorLive }): ReactElement {
  return (
    <Panel title="Supervisor" description="The desktop app starts and watches this process.">
      <div className="grid gap-3">
        <KeyValue
          layout="grid"
          ariaLabel="Supervisor"
          items={[
            {
              key: 'state',
              label: 'State',
              value: supervisor.state === 'lying' ? 'not trustworthy' : supervisor.state,
            },
            { key: 'generation', label: 'Generation', value: formatNumber(supervisor.generation), numeric: true },
            { key: 'streak', label: 'Failures in a row', value: formatNumber(supervisor.streak), numeric: true },
            {
              key: 'short-lived',
              label: 'Short-lived failures',
              value: formatNumber(supervisor.shortLivedFailures),
              numeric: true,
            },
            {
              key: 'exit',
              label: 'Last exit',
              value: lastExitText(supervisor),
              hint: supervisor.lastExit === null ? undefined : formatRelativeTime(supervisor.lastExit.atMs),
            },
            { key: 'health', label: 'Health check', value: supervisor.healthStatus },
            {
              key: 'native',
              label: 'Native errors',
              value: nativeErrorText(supervisor),
              hint:
                supervisor.lastNativeErrorContext === null
                  ? undefined
                  : `last in ${supervisor.lastNativeErrorContext} · ${formatRelativeTime(supervisor.lastNativeErrorAt)}`,
              numeric: true,
            },
            { key: 'updated', label: 'Updated', value: formatRelativeTime(supervisor.writtenAt) },
          ]}
        />
        <p className="text-caption text-secondary">Start, stop and restart are in the menu bar.</p>
      </div>
    </Panel>
  );
}
