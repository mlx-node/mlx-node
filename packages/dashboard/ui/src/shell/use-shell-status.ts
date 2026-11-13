import { useEffect, useRef, useSyncExternalStore } from 'react';
import { useLocation } from 'react-router-dom';

import { getTransportUp, subscribeConnection } from '../lib/connection';
import { useRuntimeLive } from '../lib/live';
import type { DownloadsResponse, EngineStatus } from '../lib/types';
import { useJson } from '../lib/use-api';
import { chatActivity, useChatHost } from '../pages/chat/use-chat-host';

export interface ShellStatus {
  /** undefined = no live snapshot yet. */
  engine: EngineStatus | undefined;
  /** First running process. */
  primary: { port: number; modelCount: number; inFlight: number } | null;
  /** Downloads in state running or committing; null before the first answer. */
  runningDownloads: number | null;
  /** RPC port up. */
  connected: boolean;
  /** Chat and Work turns that run, and conversations that wait for the user; null before the first host state. */
  chat: { running: number; waiting: number } | null;
}

/** Live status from the pushed live channel; downloads reload on navigation and window focus. No timer. */
export function useShellStatus(): ShellStatus {
  const live = useRuntimeLive();
  const downloads = useJson<DownloadsResponse>('/downloads');
  const { pathname } = useLocation();
  const { reload } = downloads;

  const lastPath = useRef(pathname);
  useEffect(() => {
    if (lastPath.current === pathname) return;
    lastPath.current = pathname;
    reload();
  }, [pathname, reload]);

  useEffect(() => {
    window.addEventListener('focus', reload);
    return () => window.removeEventListener('focus', reload);
  }, [reload]);

  const running = live?.processes.find((process) => process.status === 'running');
  const primary =
    running === undefined
      ? null
      : {
          port: running.port,
          modelCount: running.models.length,
          inFlight: running.models.reduce((sum, model) => sum + model.inFlight, 0),
        };
  const jobs = downloads.data?.jobs;
  const runningDownloads =
    jobs === undefined ? null : jobs.filter((job) => job.state === 'running' || job.state === 'committing').length;
  const connected = useSyncExternalStore(subscribeConnection, getTransportUp, getTransportUp);
  const chat = chatActivity(useChatHost());
  return { engine: live?.status, primary, runningDownloads, connected, chat };
}
