/**
 * The Unload action of the Chat page. After a success the unloaded names are hidden from the resident list at once,
 * until a host state no longer lists them or the engine is busy again, so no place says that the model is in memory
 * while the next heartbeat is on its way.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';

import type { UnloadControl } from '../../components/unload-button';
import { DashboardApiError, mutate } from '../../lib/api';
import { CHAT, fill } from '../../lib/chat-strings';
import type { ChatHostState } from '../../lib/chat-types';
import { toast } from '../../lib/toast';

export const UNLOAD_PATH = '/chat/engine/unload';

const NONE: ReadonlySet<string> = new Set();

/** An answer or a load holds the engine: Unload is off. */
export function engineBusy(host: Pick<ChatHostState, 'loading' | 'active'>): boolean {
  return host.loading.length > 0 || host.active.some((entry) => entry.busy);
}

export interface Unload {
  /** The models the engine holds now, as the page shows them. */
  resident: ChatHostState['resident'];
  /** For a model in `resident`: the Unload action. */
  control: (modelId: string) => UnloadControl;
}

export function useUnload(host: ChatHostState | null): Unload {
  const [working, setWorking] = useState(false);
  const [hidden, setHidden] = useState<ReadonlySet<string>>(NONE);
  const busy = host === null || engineBusy(host);

  useEffect(() => {
    if (hidden.size === 0 || host === null) return;
    if (busy || !host.resident.some((entry) => hidden.has(entry.modelId))) setHidden(NONE);
  }, [host, hidden, busy]);

  const unload = useCallback(async (modelId: string): Promise<void> => {
    setWorking(true);
    try {
      const answer = await mutate<{ unloaded: string[] }>('POST', UNLOAD_PATH);
      setHidden(new Set([...answer.unloaded, modelId]));
      toast.success(fill(CHAT.engine.unloaded, { model: modelId }));
    } catch (err) {
      const reason = err instanceof DashboardApiError ? err.reason : undefined;
      toast.error(reason === 'engine-busy' ? CHAT.engine.unloadBusy : CHAT.engine.unloadFailed);
    } finally {
      setWorking(false);
    }
  }, []);

  const resident = useMemo(
    () => (host === null ? [] : host.resident.filter((entry) => !hidden.has(entry.modelId))),
    [host, hidden],
  );
  const state: UnloadControl['state'] = working ? 'working' : busy ? 'off' : 'idle';
  const control = useCallback(
    (modelId: string): UnloadControl => ({ state, onUnload: () => void unload(modelId) }),
    [state, unload],
  );
  return { resident, control };
}
