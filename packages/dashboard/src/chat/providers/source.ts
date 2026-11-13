/**
 * What the agent hosts need from the provider settings. Every method but the two waits reads memory only: no create or
 * load path waits on the Keychain past the key cache's timeout.
 */

import type { ChatWebEnv } from '../agent/profiles.js';
import type { ChatHostState } from '../types.js';
import type { ProviderId } from './types.js';

export interface ProviderSource {
  /** The `mlx_web` server env of a new or loaded Chat session. */
  chatWeb(): ChatWebEnv;
  /** Key variables of the selected providers for the Work agent env (name → key). */
  workKeys(): Record<string, string>;
  /** Installed and verified provider skill folders for Work sessions. */
  workPluginDirs(): string[];
  /** Redaction needles: every current and retired provider key. */
  secretNeedles(): string[];
  /** The same keys with their provider, for the printed-key check of Work tool output. */
  keyNeedles(): Array<{ provider: ProviderId; key: string }>;
  /** A web tool result said the provider rejected `key` or has no credit; ignored when `key` is no longer current. */
  markKeyCheck(provider: ProviderId, check: 'rejected' | 'no-credit', key: string): void;
  /** Resolves when the key cache is filled and no provider change (and its skill sync) is in progress. */
  settled(): Promise<void>;
  /** True when the Chat skill is installed and verified (the first call runs the start-up skill sync). */
  chatSkillReady(): Promise<boolean>;
  summary(): ChatHostState['providers'];
}
