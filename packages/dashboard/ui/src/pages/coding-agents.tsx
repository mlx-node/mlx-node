import { Download, RefreshCw } from 'lucide-react';
import {
  Fragment,
  type ReactElement,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';

import type { CodingAgentId, CodingAgentsState } from '../../../src/coding-agents.js';
import { AgentMark } from '../components/arc-x/agent-mark/agent-mark';
import { ButtonLink } from '../components/arc-x/button-link/button-link';
import { CodePanel } from '../components/arc-x/code-panel/code-panel';
import { Disclosure } from '../components/arc-x/disclosure/disclosure';
import { KeyValue } from '../components/arc-x/key-value/key-value';
import { LiveLabel } from '../components/arc-x/live-label/live-label';
import { ModelBadge } from '../components/arc-x/model-badge/model-badge';
import { MorphLoader } from '../components/arc-x/morph-loader/morph-loader';
import { Panel } from '../components/arc-x/panel/panel';
import { PathText } from '../components/arc-x/path-text/path-text';
import { Placeholder } from '../components/arc-x/placeholder/placeholder';
import { Rule } from '../components/arc-x/rule/rule';
import { Alert } from '../components/arc/alert/alert';
import { Button } from '../components/arc/button/button';
import { mutate } from '../lib/api';
import { getConnectionGeneration, subscribeConnection } from '../lib/connection';
import { useJson } from '../lib/use-api';
import { type AppCommand, useCommands } from '../shell/commands';
import { Page, PageHeader } from '../shell/page';

type Busy = null | 'refresh' | CodingAgentId;
type Agent = CodingAgentsState['agents'][number];
type Request = { action: 'refresh' } | { action: 'install'; agent: CodingAgentId };

const REFRESH_MS = 30_000;

function actionLabel(agent: Agent): string {
  if (agent.status === 'error') return 'Check again';
  if (agent.status === 'needs-update') return 'Update';
  return 'Install';
}

function workingLabel(agent: Agent): string {
  if (agent.status === 'error') return 'Checking…';
  if (agent.status === 'needs-update') return 'Updating…';
  return 'Installing…';
}

function requestFor(agent: Agent): Request {
  return agent.status === 'error' ? { action: 'refresh' } : { action: 'install', agent: agent.id };
}

interface AgentRowProps {
  agent: Agent;
  available: boolean;
  busy: Busy;
  onRun: (body: Request, busy: Busy) => void;
}

function AgentRow({ agent, available, busy, onRun }: AgentRowProps): ReactElement {
  const working = busy === agent.id;
  const installed = agent.status === 'installed';
  const showButton = !working && !installed;
  const cell = useRef<HTMLDivElement>(null);
  const hadFocus = useRef(false);

  // A replaced button must not drop focus to <body>; when it comes back it takes focus again.
  useLayoutEffect(() => {
    const node = cell.current;
    if (node === null || !hadFocus.current) return;
    const button = node.querySelector('button');
    if (button === null) {
      node.focus();
      return;
    }
    if (document.activeElement === node) button.focus();
    hadFocus.current = false;
  }, [showButton]);

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-3 p-4">
      <AgentMark id={agent.id} size={28} className="shrink-0" />
      <div className="grid min-w-[min(100%,14rem)] flex-1 grid-cols-1 gap-0.5">
        <h2 className="m-0 text-strong font-medium text-foreground">{agent.name}</h2>
        <p className="m-0 text-body text-secondary [overflow-wrap:anywhere]">
          Instructions in{' '}
          <span className="inline-flex max-w-full align-bottom">
            <PathText path={agent.path} />.
          </span>
        </p>
        {agent.detail !== null && agent.detail !== '' && !installed ? (
          <p
            className={agent.status === 'error' ? 'm-0 text-body text-danger' : 'm-0 text-body text-secondary'}
            role={agent.status === 'error' ? 'alert' : undefined}
          >
            {agent.detail}
          </p>
        ) : null}
      </div>
      <div ref={cell} tabIndex={-1} aria-live="polite" className="flex min-w-[96px] justify-end">
        {showButton ? (
          <Button
            type="button"
            variant="secondary"
            style={{ minWidth: 96 }}
            disabled={!available || busy !== null}
            onClick={() => {
              hadFocus.current = true;
              onRun(requestFor(agent), agent.id);
            }}
          >
            {actionLabel(agent)}
          </Button>
        ) : (
          <span key="status" className="flex min-h-8 items-center gap-1.5 text-body font-medium text-secondary">
            <MorphLoader size={14} status={working ? 'loading' : 'success'} />
            {working ? <LiveLabel active>{workingLabel(agent)}</LiveLabel> : <span>Installed</span>}
          </span>
        )}
      </div>
    </div>
  );
}

function LoadingRows(): ReactElement {
  return (
    <div role="status" className="grid">
      <span className="sr-only">Loading coding agents…</span>
      {[0, 1, 2].map((index) => (
        <div key={index} className="flex items-center gap-4 p-4" aria-hidden="true">
          <Placeholder width={28} height={28} radius="pill" />
          <div className="grid flex-1 gap-1.5">
            <Placeholder width="30%" />
            <Placeholder width="65%" />
          </div>
          <Placeholder width={96} height={32} radius="control" />
        </div>
      ))}
    </div>
  );
}

export default function CodingAgents(): ReactElement {
  const state = useJson<CodingAgentsState>('/coding-agents');
  const { reload } = state;
  const connection = useSyncExternalStore(subscribeConnection, getConnectionGeneration);
  const [busy, setBusy] = useState<Busy>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const active = useRef(true);
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
    };
  }, []);
  const data = state.data;

  const run = useCallback(
    async (body: Request, key: Busy) => {
      setBusy(key);
      setProblem(null);
      try {
        await mutate('POST', '/coding-agents', body);
      } catch (error) {
        if (active.current) setProblem(error instanceof Error ? error.message : 'Could not update coding agents.');
      } finally {
        if (active.current) {
          setBusy(null);
          reload();
        }
      }
    },
    [reload],
  );

  useEffect(() => {
    reload();
  }, [connection, reload]);

  useEffect(() => {
    const refresh = (): void => {
      if (!document.hidden) reload();
    };
    const timer = setInterval(refresh, REFRESH_MS);
    window.addEventListener('focus', refresh);
    return () => {
      clearInterval(timer);
      window.removeEventListener('focus', refresh);
    };
  }, [reload]);

  const commands = useMemo((): AppCommand[] => {
    const list: AppCommand[] = [
      {
        id: 'coding-agents:refresh',
        label: 'Check coding agent status',
        group: 'Coding Agents',
        run: () => void run({ action: 'refresh' }, 'refresh'),
      },
    ];
    if (data?.available === true) {
      for (const agent of data.agents) {
        if (agent.status === 'not-installed' || agent.status === 'needs-update') {
          list.push({
            id: `coding-agents:install:${agent.id}`,
            label: `Install instructions for ${agent.name}`,
            group: 'Coding Agents',
            run: () => void run({ action: 'install', agent: agent.id }, agent.id),
          });
        }
      }
    }
    return list;
  }, [data, run]);
  useCommands(commands);

  const failure = problem ?? state.error?.message ?? null;

  return (
    <Page width="reading" busy={state.refreshing}>
      <PageHeader
        title="Coding Agents"
        description="Let coding agents hand GitHub work to a local model."
        actions={
          <Button
            type="button"
            variant="ghost"
            size="sm"
            loading={busy === 'refresh'}
            disabled={busy !== null}
            onClick={() => void run({ action: 'refresh' }, 'refresh')}
          >
            {busy === 'refresh' ? null : <RefreshCw size={16} aria-hidden="true" />}
            Check status
          </Button>
        }
      />

      {data !== undefined && !data.available ? (
        <Alert
          tone="warning"
          title={data.model !== null ? 'Command setup needs attention' : 'A local model is required'}
        >
          <span className="grid justify-items-start gap-2">
            <span>{data.unavailableReason}</span>
            {data.model !== null ? (
              <Button
                type="button"
                variant="secondary"
                size="sm"
                disabled={busy !== null}
                onClick={() => void run({ action: 'refresh' }, 'refresh')}
              >
                Retry setup
              </Button>
            ) : (
              <ButtonLink to="/welcome" variant="secondary" size="sm">
                <Download size={16} aria-hidden="true" />
                Install a model
              </ButtonLink>
            )}
          </span>
        </Alert>
      ) : null}

      {failure !== null ? (
        <Alert tone="danger" title="Could not update coding agents">
          {failure}
        </Alert>
      ) : null}

      {data !== undefined ? (
        <Panel title="Setup">
          <KeyValue
            layout="rows"
            ariaLabel="Setup"
            items={[
              {
                key: 'model',
                label: 'Local model',
                value: data.model !== null ? <ModelBadge name={data.model} size="sm" /> : 'None installed',
              },
              data.command !== null
                ? {
                    key: 'command',
                    label: 'Command',
                    value: <PathText path={data.command} />,
                    mono: true,
                    copy: data.command,
                  }
                : { key: 'command', label: 'Command', value: 'Not set up' },
            ]}
          />
        </Panel>
      ) : null}

      <Panel ariaLabel="Coding agents" padding="none">
        {data === undefined ? (
          state.loading ? (
            <LoadingRows />
          ) : (
            <p className="m-0 p-4 text-body text-secondary">Coding agent status is not available.</p>
          )
        ) : (
          <div className="grid">
            {data.agents.map((agent, index) => (
              <Fragment key={agent.id}>
                {index > 0 ? <Rule /> : null}
                <AgentRow
                  agent={agent}
                  available={data.available}
                  busy={busy}
                  onRun={(body, key) => void run(body, key)}
                />
              </Fragment>
            ))}
          </div>
        )}
      </Panel>

      {data?.available === true ? (
        <Panel title="How it works">
          <div className="grid gap-3">
            <ol className="m-0 grid list-decimal gap-2 ps-5 text-body text-foreground">
              <li>
                Install adds a block between <code className="font-mono">portato managed instructions</code> markers.
                Update replaces only that block.
              </li>
              <li>Start a new agent session after installation.</li>
              <li>Your coding agent approves each delegated task before the local worker runs it.</li>
            </ol>
            <p className="m-0 text-body text-secondary">
              The app includes the mlx command. GitHub work requires an authenticated GitHub CLI.
            </p>
            {data.block !== null ? (
              <Disclosure title="What gets added">
                <CodePanel code={data.block} label="Instruction block" maxLines={16} />
              </Disclosure>
            ) : null}
          </div>
        </Panel>
      ) : null}
    </Page>
  );
}
