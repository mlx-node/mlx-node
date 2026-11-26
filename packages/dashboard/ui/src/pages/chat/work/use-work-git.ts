import { useCallback, useEffect, useRef, useState } from 'react';

import { DashboardApiError, getJson, mutate } from '../../../lib/api';
import type { WorkGit, WorkSwitchReply } from '../../../lib/chat-types';

/** Reads on window focus or show at most this often. */
const SHOW_MS = 1_000;

export type WorkSwitchResult =
  | { kind: 'switched' }
  | { kind: 'refused'; text: string }
  | { kind: 'timeout' }
  /** The app refused before git ran: `reason` is the route's (`dirty`, `folder-busy`, …), `message` its sentence. */
  | { kind: 'error'; reason: string | undefined; message: string };

export interface WorkGitState {
  /** The read of the folder shown now; null before it answers, or when it failed for another reason. */
  git: WorkGit | null;
  /** The folder rule's sentence when the server refused the folder. */
  problem: string | null;
  /** The branch a switch is under way to. */
  switching: string | null;
  refresh: () => void;
  switchTo: (branch: string) => Promise<WorkSwitchResult | null>;
}

interface Answer {
  folder: string;
  git: WorkGit | null;
  problem: string | null;
}

/**
 * The work tree of the New Work screen's folder (null: no read). Read when the folder changes, when `working` (the
 * working Work conversations) changes, on `refresh`, and when the window regains focus or shows. An answer for a folder
 * that is no longer shown, or older than the newest read, is dropped.
 */
export function useWorkGit(folder: string | null, working: string): WorkGitState {
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [switching, setSwitching] = useState<string | null>(null);
  const folderRef = useRef(folder);
  folderRef.current = folder;
  const seq = useRef(0);
  const shownAt = useRef(0);

  const refresh = useCallback(() => {
    const cwd = folderRef.current;
    if (cwd === null) return;
    const n = ++seq.current;
    const fresh = (): boolean => n === seq.current && folderRef.current === cwd;
    getJson<WorkGit>(`/chat/work/git?cwd=${encodeURIComponent(cwd)}`).then(
      (git) => {
        if (fresh()) setAnswer({ folder: cwd, git, problem: null });
      },
      (error: unknown) => {
        if (!fresh()) return;
        const refused = error instanceof DashboardApiError && error.reason === 'cwd-invalid';
        setAnswer({ folder: cwd, git: null, problem: refused ? error.message : null });
      },
    );
  }, []);

  useEffect(() => {
    if (folder !== null) refresh();
  }, [folder, working, refresh]);

  useEffect(() => {
    if (folder === null) return;
    const onShow = (): void => {
      if (document.visibilityState !== 'visible') return;
      const now = Date.now();
      if (now - shownAt.current < SHOW_MS) return;
      shownAt.current = now;
      refresh();
    };
    window.addEventListener('focus', onShow);
    document.addEventListener('visibilitychange', onShow);
    return () => {
      window.removeEventListener('focus', onShow);
      document.removeEventListener('visibilitychange', onShow);
    };
  }, [folder, refresh]);

  const switchTo = useCallback(
    async (branch: string): Promise<WorkSwitchResult | null> => {
      const cwd = folderRef.current;
      if (cwd === null) return null;
      seq.current += 1;
      setSwitching(branch);
      try {
        const reply = await mutate<WorkSwitchReply>('POST', '/chat/work/git/switch', { cwd, branch });
        if (folderRef.current === cwd) setAnswer({ folder: cwd, git: reply.git, problem: null });
        if (reply.outcome === 'switched') return { kind: 'switched' };
        return reply.outcome === 'timeout' ? { kind: 'timeout' } : { kind: 'refused', text: reply.text };
      } catch (error) {
        refresh();
        return error instanceof DashboardApiError
          ? { kind: 'error', reason: error.reason, message: error.message }
          : { kind: 'error', reason: undefined, message: '' };
      } finally {
        setSwitching(null);
      }
    },
    [refresh],
  );

  const shown = answer !== null && answer.folder === folder ? answer : null;
  return { git: shown?.git ?? null, problem: shown?.problem ?? null, switching, refresh, switchTo };
}
