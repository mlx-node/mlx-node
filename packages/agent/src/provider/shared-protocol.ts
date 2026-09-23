/** Private, local-only transport. Tools and agent execution never cross this boundary. */
import { homedir } from 'node:os';
import { join } from 'node:path';

import type { Api, AssistantMessage, Context, Model, SimpleStreamOptions } from '@earendil-works/pi-ai';
import type { PerformanceMetrics } from '@mlx-node/lm';

import type { DiscoveredModelLike } from '../types.js';
import type { SharedEvent } from './shared-events.js';
import type { TurnRecorder } from './stream-adapter.js';

export const SHARED_PROTOCOL = 'mlx-delegate-inference-v2';
export const SHARED_IDLE_MS = 5 * 60_000;
export const SHARED_REQUEST_LIMIT = 20;
export const SHARED_SESSION_LIMIT = 4;

export interface SharedProfile {
  discovered: DiscoveredModelLike;
  persistPagedCache: boolean;
  preserveEmbeddedGemmaDraft: boolean;
}

export interface SharedRequest {
  clientId: string;
  profile: SharedProfile;
  model: Model<Api>;
  context: Context;
  options: Pick<SimpleStreamOptions, 'sessionId' | 'maxTokens' | 'temperature' | 'reasoning' | 'thinkingBudgets'>;
  rootSessionId?: string;
  rootSessionFile?: string;
  thinkingBudget?: number;
}

/** Stable within one caller, isolated from other callers resuming the same Pi session. */
export function sharedCacheOwners(request: Pick<SharedRequest, 'clientId' | 'options' | 'rootSessionId'>): {
  owner: string;
  root: string;
} {
  const session = request.options.sessionId || request.rootSessionId || 'default';
  return {
    owner: JSON.stringify([request.clientId, session]),
    root: JSON.stringify([request.clientId, request.rootSessionId || session]),
  };
}

export type SharedFrame =
  | { event: SharedEvent }
  | { performance: PerformanceMetrics; message: AssistantMessage }
  | { record: Parameters<TurnRecorder>[0] }
  | { model: Pick<Model<Api>, 'contextWindow' | 'maxTokens' | 'input'> }
  | { error: string };

export interface SharedEndpoint {
  protocol: string;
  pid: number;
  port: number;
  token: string;
}

/** Election is scoped to this private directory; listening ports are OS-assigned. */
export function sharedLocation(home = homedir()): { directory: string } {
  return { directory: join(home, '.mlx-node', 'delegate') };
}
