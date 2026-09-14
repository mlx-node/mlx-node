import { prettyModelName } from '@/components/model-logos';
import { Button } from '@/components/ui/button';
import { mutate } from '@/lib/api';
import { getConnectionGeneration, subscribeConnection } from '@/lib/connection';
import { useJson } from '@/lib/use-api';
import { Check, ChevronDown, Download, LoaderCircle, RefreshCw, Terminal } from 'lucide-react';
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { Link } from 'react-router-dom';

import type { CodingAgentId, CodingAgentsState } from '../../../src/coding-agents.js';

function AgentLogo({ id }: { id: CodingAgentId }) {
  if (id === 'claude')
    return (
      <svg viewBox="0 0 40 40" className="size-9 text-[#d97757]" fill="none" aria-hidden>
        {Array.from({ length: 12 }, (_, i) => (
          <path
            key={i}
            d="M20 4v32"
            stroke="currentColor"
            strokeWidth="2.4"
            strokeLinecap="round"
            transform={`rotate(${i * 15} 20 20)`}
          />
        ))}
      </svg>
    );
  if (id === 'codex')
    return (
      <span className="flex size-9 items-center justify-center rounded-xl bg-gradient-to-br from-violet-300 to-indigo-500 text-white">
        <Terminal className="size-6" aria-hidden />
      </span>
    );
  return (
    <svg viewBox="0 0 40 40" className="size-9" fill="none" stroke="currentColor" strokeWidth="2.4" aria-hidden>
      <path d="M30 9a14 14 0 1 0 2 19M8 34 34 5M19 22l14-13-8 17" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function CodingAgents() {
  const state = useJson<CodingAgentsState>('/coding-agents');
  const { reload } = state;
  const connection = useSyncExternalStore(subscribeConnection, getConnectionGeneration);
  const [sending, setSending] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const refreshing = useRef(false);
  const active = useRef(true);
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
    };
  }, []);
  const data = state.data;
  const busy = sending || !!data?.agents.some((row) => ['waiting', 'checking', 'installing'].includes(row.status));

  const run = useCallback(
    async (action: 'detect' | 'install', agent?: CodingAgentId, force = false) => {
      setSending(true);
      setProblem(null);
      try {
        await mutate('POST', '/coding-agents', { action, ...(agent ? { agent } : {}), ...(force ? { force } : {}) });
      } catch (error) {
        if (active.current) setProblem(error instanceof Error ? error.message : 'Could not update coding agents.');
      } finally {
        if (active.current) {
          setSending(false);
          reload();
        }
      }
    },
    [reload],
  );

  const refreshMetadata = useCallback(async () => {
    if (refreshing.current) return;
    refreshing.current = true;
    try {
      await mutate('POST', '/coding-agents', { action: 'refresh' });
      if (active.current) {
        setProblem(null);
        reload();
      }
    } catch (error) {
      if (active.current) setProblem(error instanceof Error ? error.message : 'Could not refresh coding agents.');
    } finally {
      refreshing.current = false;
    }
  }, [reload]);

  useEffect(() => {
    void refreshMetadata();
  }, [connection, refreshMetadata]);

  useEffect(() => {
    // Job polls read only cached state. Idle/focus refreshes inspect metadata, never run the model.
    const refresh = () => {
      if (!document.hidden) void refreshMetadata();
    };
    const timer = setInterval(busy ? reload : refresh, busy ? 1000 : 30_000);
    if (!busy) window.addEventListener('focus', refresh);
    return () => {
      clearInterval(timer);
      window.removeEventListener('focus', refresh);
    };
  }, [busy, reload, refreshMetadata]);

  return (
    <section className="mx-auto max-w-4xl space-y-5" aria-labelledby="coding-agents-title">
      <div className="flex items-center justify-between gap-4 pb-2">
        <h1 id="coding-agents-title" className="text-3xl font-semibold tracking-tight">
          Coding Agents
        </h1>
        <Button variant="ghost" size="sm" disabled={!data?.available || busy} onClick={() => void run('detect')}>
          <RefreshCw className={`size-4 ${busy ? 'animate-spin' : ''}`} aria-hidden /> Check status
        </Button>
      </div>

      <div className="bg-muted/70 rounded-3xl px-6 py-5">
        <p className="text-muted-foreground text-base leading-relaxed">
          Let your coding agents hand GitHub work to a local model. Review comments, investigate failed checks, and
          gather context while using fewer cloud tokens. Your coding agent approves each delegated task before the local
          worker runs it.
        </p>
      </div>

      {data && !data.available && (
        <div
          className="bg-muted/70 flex flex-wrap items-center justify-between gap-4 rounded-3xl px-6 py-5"
          role="status"
        >
          <div>
            <p className="font-medium">{data.model ? 'Command setup needs attention' : 'A local model is required'}</p>
            <p className="text-muted-foreground mt-1 text-sm">{data.unavailableReason}</p>
          </div>
          {data.model ? (
            <Button disabled={busy} onClick={() => void refreshMetadata()}>
              Retry setup
            </Button>
          ) : (
            <Button asChild>
              <Link to="/welcome">
                <Download className="size-4" aria-hidden /> Install a model
              </Link>
            </Button>
          )}
        </div>
      )}

      {(problem || state.error) && (
        <p role="alert" className="text-destructive text-sm">
          {problem || state.error?.message}
        </p>
      )}

      {!data ? (
        state.loading && (
          <div role="status" className="bg-muted/70 flex items-center gap-3 rounded-3xl p-6 text-sm">
            <LoaderCircle className="size-4 animate-spin" aria-hidden /> Loading coding agents…
          </div>
        )
      ) : (
        <div className="bg-muted/70 rounded-3xl px-5 sm:px-6">
          {data.agents.map((agent) => {
            const working = agent.status === 'checking' || agent.status === 'installing';
            const installed = agent.status === 'installed' && data.available;
            const label =
              agent.status === 'installing'
                ? 'Installing…'
                : agent.status === 'checking'
                  ? 'Checking…'
                  : agent.status === 'waiting'
                    ? 'Waiting…'
                    : agent.status === 'error'
                      ? 'Check again'
                      : agent.status === 'needs-update'
                        ? 'Update…'
                        : agent.status === 'unchecked' && data.available
                          ? 'Check status'
                          : 'Install…';
            return (
              <div key={agent.id} className="flex items-center gap-4 border-b py-6 last:border-b-0 sm:gap-5">
                <div className="flex w-10 shrink-0 justify-center">
                  <AgentLogo id={agent.id} />
                </div>
                <div className="min-w-0 flex-1">
                  <h2 className="text-xl font-medium tracking-tight">{agent.name}</h2>
                  <p className="text-muted-foreground mt-1 break-words text-sm leading-relaxed">
                    Instructions in {agent.path}.
                  </p>
                  {agent.detail && agent.status !== 'installed' && (
                    <p
                      className={`mt-1 text-sm leading-relaxed ${agent.status === 'error' ? 'text-destructive' : 'text-muted-foreground'}`}
                      role={agent.status === 'error' ? 'alert' : undefined}
                    >
                      {agent.detail}
                    </p>
                  )}
                </div>
                <div className="shrink-0" aria-live="polite">
                  {installed ? (
                    <details className="relative">
                      <summary className="flex cursor-pointer list-none items-center gap-2 rounded-lg px-2 py-2 text-base font-medium [&::-webkit-details-marker]:hidden">
                        <span className="bg-foreground text-background flex size-5 items-center justify-center rounded-full">
                          <Check className="size-3.5" strokeWidth={3} aria-hidden />
                        </span>{' '}
                        Installed <ChevronDown className="size-4" aria-hidden />
                      </summary>
                      <div className="bg-popover absolute right-0 z-10 mt-1 w-40 rounded-xl border p-1 shadow-lg">
                        <button
                          className="hover:bg-accent w-full rounded-lg px-3 py-2 text-left text-sm disabled:opacity-50"
                          disabled={busy}
                          onClick={() => void run('detect', agent.id, true)}
                        >
                          Recheck with model
                        </button>
                      </div>
                    </details>
                  ) : (
                    <Button
                      variant="secondary"
                      className="min-w-28 rounded-xl bg-black/[0.06] text-base shadow-none hover:bg-black/10 dark:bg-white/10"
                      disabled={!data.available || busy}
                      onClick={() =>
                        void run(
                          ['not-installed', 'needs-update'].includes(agent.status) ? 'install' : 'detect',
                          agent.id,
                        )
                      }
                    >
                      {working && <LoaderCircle className="size-4 animate-spin" aria-hidden />}
                      {label}
                    </Button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {data?.model && (
        <p className="text-muted-foreground px-1 text-sm">
          Check status uses {prettyModelName(data.model)} on this Mac for new or changed files. It may need to load the
          model. Opening this page uses cached results. The app includes the mlx command. Start a new agent session
          after installation. GitHub work requires an authenticated GitHub CLI.
        </p>
      )}
    </section>
  );
}
