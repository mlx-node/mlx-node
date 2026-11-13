import { RefreshCw } from 'lucide-react';
import { type ReactElement, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';

import { IconButton } from '../components/arc-x/icon-button/icon-button';
import { Placeholder } from '../components/arc-x/placeholder/placeholder';
import { Alert } from '../components/arc/alert/alert';
import { Button } from '../components/arc/button/button';
import { DashboardApiError, mutate } from '../lib/api';
import { PROVIDERS } from '../lib/providers-strings';
import type {
  ProviderCheckReply,
  ProviderId,
  ProviderKeyReply,
  ProviderSelectionInput,
  ProvidersResponse,
} from '../lib/providers-types';
import {
  checkKeyFormat,
  fill,
  type Progress,
  progressOf,
  providerById,
  providerName,
  vanished,
} from '../lib/providers-view';
import { toast } from '../lib/toast';
import { useJson } from '../lib/use-api';
import { type AppCommand, useCommands } from '../shell/commands';
import { Page, PageHeader, PageSection } from '../shell/page';
import { NowPanel } from './providers/now-panel';
import { ProviderCard } from './providers/provider-card';
import { type Selecting, UsePanel } from './providers/use-panel';

type Busy =
  | { kind: 'key' | 'check' | 'remove'; id: ProviderId }
  | { kind: 'select'; input: ProviderSelectionInput }
  | null;

function reasonOf(error: unknown): string | undefined {
  return error instanceof DashboardApiError ? error.reason : undefined;
}

function codeOf(error: unknown): string | undefined {
  return error instanceof DashboardApiError ? error.code : undefined;
}

/** A notice sentence with its "Cache page" words as a link to the Cache page. */
function withCacheLink(text: string): ReactNode {
  const at = text.indexOf(PROVIDERS.cacheLink);
  if (at === -1) return text;
  return (
    <>
      {text.slice(0, at)}
      <Link className="link" to="/cache">
        {PROVIDERS.cacheLink}
      </Link>
      {text.slice(at + PROVIDERS.cacheLink.length)}
    </>
  );
}

function LoadingPanels(): ReactElement {
  return (
    <div role="status" aria-busy="true" className="grid gap-4">
      <span className="sr-only">Loading Web Search settings…</span>
      <Placeholder height={96} radius="panel" />
      <Placeholder height={176} radius="panel" />
      <Placeholder height={208} radius="panel" />
    </div>
  );
}

export default function Providers(): ReactElement {
  const state = useJson<ProvidersResponse>('/providers');
  const { reload } = state;
  const data = state.data;
  const [busy, setBusy] = useState<Busy>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [installFailed, setInstallFailed] = useState<{ corrupt: boolean; input: ProviderSelectionInput } | null>(null);
  const [gone, setGone] = useState<ProviderId | null>(null);
  const [status, setStatus] = useState('');
  const [checkingAll, setCheckingAll] = useState(false);
  const active = useRef(true);
  const removedByPage = useRef<ProviderId | null>(null);
  const seen = useRef<ProvidersResponse | undefined>(undefined);

  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
    };
  }, []);

  useEffect(() => {
    const onFocus = (): void => reload();
    window.addEventListener('focus', onFocus);
    return () => window.removeEventListener('focus', onFocus);
  }, [reload]);

  // Any body that shows a provider in use as gone, which this page did not remove, means its key left the Keychain;
  // a key gone at app start comes from the server, which saw the selection before it cleared it.
  useEffect(() => {
    const before = seen.current;
    seen.current = data;
    if (data === undefined || before === data) return;
    const id = data.vanished ?? (before === undefined ? null : vanished(before, data, removedByPage.current));
    if (id !== null) setGone(id);
  }, [data]);

  const apply = useCallback(
    (next: ProvidersResponse): void => {
      const before = seen.current;
      state.patch(() => next);
      if (before === undefined) return;
      const change = progressOf(before, next);
      if (change !== null) {
        setProgress(change);
        setStatus(change.status);
      }
    },
    [state],
  );

  const settle = useCallback((): void => {
    if (!active.current) return;
    setBusy(null);
  }, []);

  const saveKey = useCallback(
    async (id: ProviderId, raw: string): Promise<string | null> => {
      const key = checkKeyFormat(raw);
      if (key === null) return PROVIDERS.badFormat;
      const hadKey = data !== undefined && providerById(data, id)?.configured === true;
      setBusy({ kind: 'key', id });
      try {
        const reply = await mutate<ProviderKeyReply>('PUT', `/providers/${id}/key`, { key });
        if (!active.current) return null;
        apply(reply.providers);
        if (reply.saved) return null;
        const rejected = fill(PROVIDERS.rejectedOnSave, { provider: providerName(id) });
        return hadKey ? `${rejected} ${PROVIDERS.earlierKept}` : rejected;
      } catch (error) {
        if (active.current) reload();
        if (codeOf(error) === 'E_BAD_REQUEST') return PROVIDERS.badFormat;
        if (reasonOf(error) === 'keychain-unavailable' || codeOf(error) === 'E_UNAVAILABLE') {
          return PROVIDERS.keychainRefused;
        }
        return PROVIDERS.failed;
      } finally {
        settle();
      }
    },
    [apply, data, reload, settle],
  );

  const checkKey = useCallback(
    async (id: ProviderId): Promise<void> => {
      setBusy({ kind: 'check', id });
      try {
        const reply = await mutate<ProviderCheckReply>('POST', `/providers/${id}/validate`);
        if (active.current) apply(reply.providers);
      } catch (error) {
        if (!active.current) return;
        reload();
        if (reasonOf(error) !== 'keychain-unavailable') toast.error(PROVIDERS.failed);
      } finally {
        settle();
      }
    },
    [apply, reload, settle],
  );

  const removeKey = useCallback(
    async (id: ProviderId): Promise<void> => {
      const wasSearch = data?.search === id;
      const wasFirecrawl = id === 'firecrawl' && data?.firecrawl === true;
      setBusy({ kind: 'remove', id });
      removedByPage.current = id;
      try {
        const reply = await mutate<ProvidersResponse>('DELETE', `/providers/${id}/key`);
        if (!active.current) return;
        apply(reply);
        if (wasSearch) toast.success(fill(PROVIDERS.removedToast, { provider: providerName(id) }));
        else if (wasFirecrawl) toast.success(PROVIDERS.removedToastFirecrawl);
      } catch (error) {
        removedByPage.current = null;
        if (active.current) reload();
        throw new Error(reasonOf(error) === 'keychain-unavailable' ? PROVIDERS.keychainTitle : PROVIDERS.failed);
      } finally {
        settle();
      }
    },
    [apply, data, reload, settle],
  );

  const select = useCallback(
    async (input: ProviderSelectionInput): Promise<void> => {
      setBusy({ kind: 'select', input });
      setInstallFailed(null);
      try {
        const reply = await mutate<ProvidersResponse>('PUT', '/providers/selection', input);
        if (active.current) apply(reply);
      } catch (error) {
        if (!active.current) return;
        reload();
        if (codeOf(error) === 'E_CONFLICT') {
          const id: ProviderId | null = input.search ?? (input.firecrawl === true ? 'firecrawl' : null);
          toast.error(id === null ? PROVIDERS.failed : fill(PROVIDERS.needKey, { provider: providerName(id) }));
        } else if (reasonOf(error) !== 'keychain-unavailable') {
          setInstallFailed({ corrupt: reasonOf(error) === 'skill-bundle-corrupt', input });
        }
      } finally {
        settle();
      }
    },
    [apply, reload, settle],
  );

  const locked = data?.keychain === 'unavailable';
  const idle = busy === null;
  const anyConfigured = data?.providers.some((p) => p.configured) === true;
  const canCheckAll = idle && !checkingAll && !locked && anyConfigured;

  const checkAll = useCallback(async (): Promise<void> => {
    if (data === undefined || !canCheckAll) return;
    setCheckingAll(true);
    try {
      for (const p of data.providers) {
        if (!p.configured) continue;
        await checkKey(p.id);
        if (!active.current) return;
      }
    } finally {
      if (active.current) setCheckingAll(false);
    }
  }, [canCheckAll, checkKey, data]);

  const latestCheckAll = useRef(checkAll);
  useEffect(() => {
    latestCheckAll.current = checkAll;
  });
  const commands = useMemo(
    (): AppCommand[] => [
      {
        id: 'providers:check-all',
        label: PROVIDERS.checkAll,
        group: PROVIDERS.commandGroup,
        run: () => void latestCheckAll.current(),
      },
    ],
    [],
  );
  useCommands(commands);

  const selecting: Selecting = busy?.kind === 'select' ? busy.input : null;

  return (
    <Page width="reading" busy={state.refreshing}>
      <PageHeader
        title={PROVIDERS.nav}
        description={PROVIDERS.description}
        actions={
          <IconButton
            label={PROVIDERS.checkAll}
            icon={<RefreshCw size={16} aria-hidden="true" />}
            loading={checkingAll}
            disabled={!checkingAll && !canCheckAll}
            onClick={() => void checkAll()}
          />
        }
      />
      <div role="status" aria-live="polite" className="sr-only">
        {status}
      </div>

      {data === undefined ? (
        state.error !== undefined ? (
          <Alert tone="warning" title={PROVIDERS.loadError}>
            <span className="grid justify-items-start gap-2">
              <Button type="button" variant="secondary" size="sm" onClick={reload}>
                {PROVIDERS.tryAgain}
              </Button>
            </span>
          </Alert>
        ) : (
          <LoadingPanels />
        )
      ) : (
        <>
          {locked ? (
            <Alert tone="danger" title={PROVIDERS.keychainTitle}>
              <span className="grid justify-items-start gap-2">
                <span>{PROVIDERS.keychainText}</span>
                <Button type="button" variant="secondary" size="sm" onClick={reload}>
                  {PROVIDERS.tryAgain}
                </Button>
              </span>
            </Alert>
          ) : null}
          {gone !== null ? (
            <Alert
              tone="warning"
              title={fill(PROVIDERS.vanishedTitle, { provider: providerName(gone) })}
              onDismiss={() => setGone(null)}
            >
              {PROVIDERS.vanishedText}
            </Alert>
          ) : null}

          <NowPanel data={data} />

          <UsePanel
            data={data}
            selecting={selecting}
            locked={locked || !idle}
            lockedReason={locked ? PROVIDERS.keychainTitle : undefined}
            progress={progress}
            onSelect={(input) => void select(input)}
          />

          {installFailed !== null ? (
            <Alert tone="danger" title={PROVIDERS.installFailedTitle}>
              <span className="grid justify-items-start gap-2">
                <span>{installFailed.corrupt ? PROVIDERS.installCorrupt : PROVIDERS.installOther}</span>
                {installFailed.corrupt ? null : (
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    disabled={!idle}
                    onClick={() => void select(installFailed.input)}
                  >
                    {PROVIDERS.tryAgain}
                  </Button>
                )}
              </span>
            </Alert>
          ) : null}

          <Alert tone="info" title={PROVIDERS.leavesTitle}>
            {PROVIDERS.leavesText}
          </Alert>
          {data.search === 'brave' ? (
            <Alert tone="info" title={PROVIDERS.historyTitle}>
              {withCacheLink(PROVIDERS.historyBrave)}
            </Alert>
          ) : null}
          {data.search === 'exa' ? (
            <Alert tone="info" title={PROVIDERS.historyTitle}>
              {withCacheLink(PROVIDERS.historyExa)}
            </Alert>
          ) : null}

          <PageSection title={PROVIDERS.keys}>
            {data.providers.map((p) => {
              const own = busy !== null && busy.kind !== 'select' && busy.id === p.id;
              return (
                <ProviderCard
                  key={p.id}
                  provider={p}
                  checking={own && busy.kind !== 'remove'}
                  locked={locked || (!idle && !own)}
                  lockedReason={locked ? PROVIDERS.keychainTitle : undefined}
                  onSave={(key) => saveKey(p.id, key)}
                  onCheck={() => void checkKey(p.id)}
                  onRemove={() => removeKey(p.id)}
                  onRemoved={() => {
                    removedByPage.current = null;
                  }}
                />
              );
            })}
          </PageSection>
        </>
      )}
    </Page>
  );
}
