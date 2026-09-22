/** Private, local-only transport. Tools and agent execution never cross this boundary. */
import { createHash } from 'node:crypto';
import { homedir } from 'node:os';
import { join } from 'node:path';

import type { Api, AssistantMessage, Context, Model, SimpleStreamOptions } from '@earendil-works/pi-ai';
import type { PerformanceMetrics } from '@mlx-node/lm';

import type { DiscoveredModelLike } from '../types.js';
import type { SharedEvent } from './shared-events.js';
import type { TurnRecorder } from './stream-adapter.js';

export const SHARED_PROTOCOL = 'mlx-delegate-inference-v1';
export const SHARED_IDLE_MS = 5 * 60_000;
export const SHARED_REQUEST_LIMIT = 20;

export interface SharedProfile {
  discovered: DiscoveredModelLike;
  persistPagedCache: boolean;
  preserveEmbeddedGemmaDraft: boolean;
}

export interface SharedRequest {
  profile: SharedProfile;
  model: Model<Api>;
  context: Context;
  options: Pick<SimpleStreamOptions, 'sessionId' | 'maxTokens' | 'temperature' | 'reasoning' | 'thinkingBudgets'>;
  rootSessionId?: string;
  rootSessionFile?: string;
  thinkingBudget?: number;
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

/** One port per OS user, independent of checkout, model, caller, or agent config. */
export function sharedLocation(home = homedir()): { directory: string; port: number } {
  const hash = createHash('sha256').update(home).digest().readUInt32BE(0);
  return { directory: join(home, '.mlx-node', 'delegate'), port: 19000 + (hash % 10000) };
}
