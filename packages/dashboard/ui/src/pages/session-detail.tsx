import { GitBranch, RefreshCw } from 'lucide-react';
import { type ReactElement, useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom';

import { ButtonLink } from '../components/arc-x/button-link/button-link';
import { IconButton } from '../components/arc-x/icon-button/icon-button';
import { KeyValue, type KeyValueItem } from '../components/arc-x/key-value/key-value';
import { ModelBadge } from '../components/arc-x/model-badge/model-badge';
import { Panel } from '../components/arc-x/panel/panel';
import { PathText } from '../components/arc-x/path-text/path-text';
import { Alert } from '../components/arc/alert/alert';
import { Badge } from '../components/arc/badge/badge';
import { ExpandableText } from '../components/arc/expandable-text/expandable-text';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/arc/tabs/tabs';
import {
  isAppConversation,
  locked,
  openInChatUrl,
  SessionActions,
  sessionUrl,
  useSessionActions,
} from '../components/session-dialogs';
import { Transcript } from '../components/transcript';
import { CHAT } from '../lib/chat-strings';
import { formatNumber, formatRelativeTime } from '../lib/format';
import {
  AGENT_MESSAGES,
  AGENT_MESSAGES_TIP,
  copyResumeCommand,
  groupTurns,
  isoMs,
  LIVE_FRESH_MS,
  LIVE_HINT,
  openTurnGroup,
  sessionTitle,
  sessionTitleTip,
  TRANSCRIPT_ROWS_TIP,
  useFreshness,
} from '../lib/sessions';
import type { SessionDetail as SessionDetailData, SessionDetailResponse } from '../lib/types';
import { useJson } from '../lib/use-api';
import { type AppCommand, useCommands } from '../shell/commands';
import { Page, PageHeader } from '../shell/page';
import { Activity } from './session-detail/activity';
import { INSPECTOR_ID } from './session-detail/turn-inspector';

type Tab = 'activity' | 'transcript';

const CRUMBS = [{ label: 'Sessions', to: '/sessions' }] as const;

function isSubagentKind(kind: string | null): boolean {
  return kind !== null && kind.startsWith('subagent');
}

function isTab(value: string | null): value is Tab {
  return value === 'activity' || value === 'transcript';
}

/** What fits the narrowest facts-grid cell in 12 px mono beside the copy button, so the CSS ellipsis never cuts the tail too. */
const DIRECTORY_CHARS = 17;

/** The agent writes `git_root_dir` with a trailing slash and `cwd` without one. */
function withoutTrailingSlash(path: string): string {
  let end = path.length;
  while (end > 1 && path[end - 1] === '/') end -= 1;
  return path.slice(0, end);
}

function ParentLine({ session }: { session: SessionDetailData }): ReactElement | null {
  const parent = session.parent;
  if (parent === null) return null;
  return (
    <p className="m-0 text-body text-secondary">
      {isSubagentKind(session.kind) ? 'Subagent of' : 'Forked from'}{' '}
      {parent.title === null ? (
        <span title={parent.id}>a session that no longer exists</span>
      ) : (
        <Link
          className="link"
          to={sessionUrl(parent.id)}
          title={sessionTitleTip({ id: parent.id, title: parent.title })}
        >
          {sessionTitle({ id: parent.id, title: parent.title })}
        </Link>
      )}
    </p>
  );
}

function sessionModels(data: SessionDetailResponse): string[] {
  const models: string[] = [];
  const add = (model: string | null): void => {
    if (model !== null && model !== '' && !models.includes(model)) models.push(model);
  };
  add(data.session.model);
  for (const turn of data.turns) add(turn.model);
  return models;
}

function facts(data: SessionDetailResponse): KeyValueItem[] {
  const session = data.session;
  const models = sessionModels(data);
  const items: KeyValueItem[] = [
    {
      key: 'cwd',
      label: 'Directory',
      value: <PathText path={session.cwd} maxChars={DIRECTORY_CHARS} />,
      mono: true,
      copy: session.cwd,
    },
    {
      key: 'branch',
      label: 'Branch',
      value:
        session.branch === null ? null : (
          <span className="inline-flex min-w-0 items-center gap-1">
            <GitBranch size={12} aria-hidden="true" className="shrink-0" />
            <span className="truncate">{session.branch}</span>
          </span>
        ),
      mono: true,
    },
    {
      key: 'models',
      label: 'Models',
      value:
        models.length === 0 ? null : (
          <span className="flex min-w-0 flex-wrap gap-1">
            {models.map((model) => (
              <ModelBadge key={model} name={model} size="sm" maxChars={24} />
            ))}
          </span>
        ),
    },
    {
      key: 'created',
      label: 'Created',
      value: <span title={session.createdAt}>{formatRelativeTime(isoMs(session.createdAt))}</span>,
    },
    {
      key: 'last',
      label: 'Last active',
      value: <span title={session.lastActiveAt}>{formatRelativeTime(isoMs(session.lastActiveAt))}</span>,
    },
    {
      key: 'messages',
      label: AGENT_MESSAGES,
      labelTitle: AGENT_MESSAGES_TIP,
      value: formatNumber(session.messageCount),
      numeric: true,
    },
  ];
  if (session.title.trim() === '')
    items.push({ key: 'id', label: 'Session ID', value: session.id, mono: true, copy: session.id });
  if (session.headCommit !== null) {
    items.push({
      key: 'commit',
      label: 'Commit',
      value: session.headCommit.slice(0, 7),
      mono: true,
      copy: session.headCommit,
    });
  }
  if (session.gitRoot !== null && withoutTrailingSlash(session.gitRoot) !== withoutTrailingSlash(session.cwd)) {
    items.push({
      key: 'root',
      label: 'Git root',
      value: <PathText path={session.gitRoot} maxChars={DIRECTORY_CHARS} />,
      mono: true,
    });
  }
  if (session.remotes.length > 0) {
    const more = session.remotes.length - 1;
    items.push({
      key: 'remotes',
      label: 'Remotes',
      value: (
        <span title={session.remotes.join('\n')}>{`${session.remotes[0]!}${more > 0 ? ` +${String(more)}` : ''}`}</span>
      ),
      mono: true,
    });
  }
  if (isAppConversation(session.origin))
    items.push({ key: 'mode', label: CHAT.mode, value: session.origin === 'work' ? CHAT.modeWork : CHAT.modeChat });
  else if (session.agentName !== null) items.push({ key: 'agent', label: 'Agent', value: session.agentName });
  if (session.reasoningEffort !== null) {
    items.push({ key: 'reasoning', label: 'Reasoning', value: session.reasoningEffort });
  }
  return items;
}

export default function SessionDetail(): ReactElement {
  const { id = '' } = useParams();
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  // Live refetching only while the last body said the session runs.
  const [liveQuery, setLiveQuery] = useState(false);
  const detail = useJson<SessionDetailResponse>(`/sessions/${encodeURIComponent(id)}`, { live: liveQuery });
  const data = detail.data;
  const bodyLive = data?.session.live === true;
  if (data !== undefined && bodyLive !== liveQuery) setLiveQuery(bodyLive);
  const { fresh, arrivedAt } = useFreshness(data, LIVE_FRESH_MS);
  const live = bodyLive && fresh;
  const reloadDetail = detail.reload;
  useEffect(() => {
    window.addEventListener('focus', reloadDetail);
    return () => window.removeEventListener('focus', reloadDetail);
  }, [reloadDetail]);

  const turns = data?.turns ?? [];
  const groups = useMemo(() => groupTurns(data?.transcript ?? [], data?.turns ?? []), [data]);
  // Chosen from the first body of each session only: a live refetch that brings the first turn must not switch tabs.
  const [defaultTab, setDefaultTab] = useState<{ id: string; tab: Tab } | null>(null);
  if (data !== undefined && defaultTab?.id !== id) {
    setDefaultTab({ id, tab: data.turns.length > 0 ? 'activity' : 'transcript' });
  }
  const rawTab = params.get('tab');
  const tab: Tab = isTab(rawTab)
    ? rawTab
    : defaultTab?.id === id
      ? defaultTab.tab
      : turns.length > 0
        ? 'activity'
        : 'transcript';
  const rawTurn = Number(params.get('turn'));
  const hasTurnParam = params.get('turn') !== null && Number.isInteger(rawTurn) && rawTurn >= 1;
  const openGroup = useMemo(() => openTurnGroup(groups), [groups]);
  const count = turns.length + (openGroup !== null ? 1 : 0);
  const defaultTurn = turns.length === 0 || (bodyLive && openGroup !== null) ? count : turns.length;
  const selected = hasTurnParam && rawTurn <= count ? rawTurn : defaultTurn;
  const focusTurn = hasTurnParam ? rawTurn : null;
  const [scrollToInspector, setScrollToInspector] = useState(0);

  const update = (changes: Record<string, string | null>): void => {
    setParams(
      (current) => {
        const next = new URLSearchParams(current);
        for (const [key, value] of Object.entries(changes)) {
          if (value === null) next.delete(key);
          else next.set(key, value);
        }
        return next;
      },
      { replace: true },
    );
  };

  useEffect(() => {
    if (scrollToInspector === 0) return;
    document.getElementById(INSPECTOR_ID)?.scrollIntoView?.({ block: 'start' });
  }, [scrollToInspector]);

  const session = data?.session;
  const reload = detail.reload;
  const onDeleted = (): void => void navigate('/sessions', { replace: true });
  const handles = useSessionActions(
    session ?? { id, title: '', live: false, subagentCount: 0, origin: 'agent', cwd: '' },
    {
      onRenamed: reload,
      onDeleted,
      onConflict: reload,
    },
  );

  const parent = session?.parent ?? null;
  const resumeCommand = data?.resumeCommand;
  const commands = useMemo((): AppCommand[] => {
    if (session === undefined || resumeCommand === undefined) return [];
    const list: AppCommand[] = [];
    if (resumeCommand !== null)
      list.push({
        id: 'session:copy-resume',
        label: 'Copy resume command',
        group: 'Session',
        run: () => copyResumeCommand(resumeCommand),
      });
    if (isAppConversation(session.origin))
      list.push({
        id: 'session:open-in-chat',
        label: CHAT.commands.openInChat,
        group: 'Session',
        run: () => void navigate(openInChatUrl(session.id)),
      });
    if (!locked(session)) {
      list.push(
        { id: 'session:rename', label: 'Rename session…', group: 'Session', run: () => handles.rename() },
        { id: 'session:delete', label: 'Delete session…', group: 'Session', run: () => handles.remove() },
      );
    }
    list.push(
      tab === 'activity'
        ? {
            id: 'session:show-transcript',
            label: 'Show transcript',
            group: 'Session',
            run: () => update({ tab: 'transcript' }),
          }
        : {
            id: 'session:show-activity',
            label: 'Show activity',
            group: 'Session',
            run: () => update({ tab: 'activity' }),
          },
    );
    if (parent !== null && parent.title !== null) {
      list.push({
        id: 'session:open-parent',
        label: 'Open parent session',
        group: 'Session',
        run: () => void navigate(sessionUrl(parent.id)),
      });
    }
    return list;
  }, [session, resumeCommand, tab, parent]);
  useCommands(commands);

  if (detail.error !== undefined && data === undefined) {
    return (
      <Page>
        <PageHeader title="Session" crumbs={CRUMBS} />
        <Alert tone="danger" title="Could not open this session">
          <span className="grid justify-items-start gap-2">
            <span>{detail.error.message}</span>
            <ButtonLink to="/sessions" variant="secondary" size="sm">
              Back to sessions
            </ButtonLink>
          </span>
        </Alert>
      </Page>
    );
  }

  if (data === undefined || session === undefined) {
    return (
      <Page busy>
        <PageHeader title="Session" crumbs={CRUMBS} titlePlaceholder="24rem" />
        <Panel ariaLabel="Session details" busy>
          <KeyValue
            loading
            ariaLabel="Session details"
            items={['Directory', 'Branch', 'Models', 'Created', 'Last active', AGENT_MESSAGES].map((label) => ({
              key: label,
              label,
              value: null,
            }))}
          />
        </Panel>
      </Page>
    );
  }

  const transcriptLink = (turn: number): string => {
    const next = new URLSearchParams(params);
    next.set('tab', 'transcript');
    next.set('turn', String(turn));
    return `?${next.toString()}`;
  };

  return (
    <Page busy={detail.refreshing}>
      <PageHeader
        title={sessionTitle(session)}
        crumbs={CRUMBS}
        meta={
          <>
            {session.live ? (
              <Badge tone="success" size="sm" dot title={LIVE_HINT}>
                Live
              </Badge>
            ) : null}
            {session.kind !== null ? (
              <Badge size="sm" outline>
                {session.kind}
              </Badge>
            ) : null}
            {isAppConversation(session.origin) ? (
              <Badge size="sm" outline>
                {session.origin === 'work' ? CHAT.modeWork : CHAT.modeChat}
              </Badge>
            ) : null}
          </>
        }
        actions={
          <>
            {isAppConversation(session.origin) ? (
              <ButtonLink to={openInChatUrl(session.id)} variant="secondary">
                {CHAT.commands.openInChat}
              </ButtonLink>
            ) : null}
            <SessionActions
              session={session}
              resumeCommand={data.resumeCommand}
              onRenamed={reload}
              onDeleted={onDeleted}
              onConflict={reload}
            />
            <IconButton
              label="Refresh"
              variant="secondary"
              icon={<RefreshCw size={16} aria-hidden="true" />}
              loading={detail.refreshing}
              onClick={reload}
            />
          </>
        }
      />

      <Panel ariaLabel="Session details">
        <div className="grid min-w-0 gap-3">
          <ParentLine session={session} />
          <KeyValue ariaLabel="Session details" items={facts(data)} />
          {session.lastTurnSummary !== null && session.lastTurnSummary !== '' ? (
            <div className="grid min-w-0 gap-1">
              <span className="text-caption text-secondary">Last turn</span>
              <ExpandableText lines={2} moreLabel="Show more" lessLabel="Show less" className="text-body">
                {session.lastTurnSummary}
              </ExpandableText>
            </div>
          ) : null}
        </div>
      </Panel>

      <Tabs value={tab} onValueChange={(value) => update({ tab: isTab(value) ? value : null })}>
        <TabsList aria-label="Session views">
          <TabsTrigger value="activity">Activity</TabsTrigger>
          <TabsTrigger value="transcript" title={TRANSCRIPT_ROWS_TIP}>
            Transcript
            <span className="ms-1.5 text-muted tabular-nums">{formatNumber(data.transcript.length)}</span>
          </TabsTrigger>
        </TabsList>
        <TabsContent value="activity">
          {tab === 'activity' ? (
            <Activity
              data={data}
              groups={groups}
              selected={selected}
              onSelect={(turn) => update({ turn: String(turn) })}
              transcriptLink={transcriptLink}
              liveQuery={liveQuery}
              live={live}
            />
          ) : null}
        </TabsContent>
        <TabsContent value="transcript">
          {tab === 'transcript' ? (
            <Transcript
              entries={data.transcript}
              turns={data.turns}
              focusTurn={focusTurn}
              model={session.model}
              sessionLive={session.live}
              live={live}
              asOfMs={arrivedAt ?? undefined}
              transcriptError={data.transcriptError}
              onInspectTurn={(turn) => {
                update({ tab: 'activity', turn: String(turn) });
                setScrollToInspector((count) => count + 1);
              }}
            />
          ) : null}
        </TabsContent>
      </Tabs>
      {handles.dialogs}
    </Page>
  );
}
