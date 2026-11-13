/**
 * The message contract between a dashboard renderer and the runtime that answers
 * it, multiplexed over one {@link RpcPort}.
 *
 * Every field crosses a structured clone, so nothing here may be a function, a
 * class instance or a `URLSearchParams`. {@link ApiResponse} already is a plain
 * envelope — that is what makes this boundary cheap — and `ApiCall.path` already
 * carries the query string as text for exactly the same reason.
 *
 * Requests and events share ONE id space and are told apart by `kind`. A request
 * id is answered by exactly one `response`; a subscribe id is instead the
 * subscription handle every later `event` is tagged with, so both directions
 * multiplex without a second channel.
 */

import type { ApiResponse } from '../api/errors.js';
import type { ChatHostEvent, ConversationEvent } from '../chat/types.js';
import type { DownloadEvent } from '../download.js';
import type { LiveEvent, LiveTopic } from '../live.js';
import type { ApiCall } from '../runtime.js';

export type { LiveEvent, LiveTopic } from '../live.js';

/** What a `chat` stream watches: the host state, or one conversation. */
export type ChatScope = { scope: 'host' } | { scope: 'conversation'; conversationId: string };

/**
 * The `window.postMessage` tag the preload uses to hand the SPA its port.
 *
 * A port cannot be `contextBridge`d, so the documented Electron hop is
 * `window.postMessage({ type: TAG, generation }, '*', [port])` from the preload
 * into the page. The tag is here rather than in the shell because BOTH ends must
 * agree on it and this is the module that already defines what crosses the port.
 */
export const DASHBOARD_PORT_MESSAGE = 'mlx:dashboard-port';

/**
 * Page → preload: the RPC transport missed its post-cancellation bound and the
 * supervised CONTROL PANEL runtime must be replaced, not merely handed another
 * port to the same wedged process. The envelope also carries the broker
 * generation so MAIN can reject a delayed report after replacement.
 */
export const DASHBOARD_UNRESPONSIVE_MESSAGE = 'mlx:control-panel-unresponsive';

export type RpcRequest =
  | { kind: 'call'; id: number; call: ApiCall }
  /**
   * Withdraw a call whose renderer-side deadline expired. Not answered on its
   * own: the original call stays pending until the runtime returns either an
   * acknowledged cancellation or the real result of work already started.
   */
  | { kind: 'cancel'; id: number }
  /** Open a download-progress subscription; `id` tags every event that follows. */
  | { kind: 'subscribe'; id: number; jobId: string }
  /** Open a live engine-state subscription; `id` tags every `live` reply that follows. */
  | { kind: 'watch'; id: number; topic: LiveTopic }
  /** Open a Chat stream; `id` tags every `chat` reply that follows. Closed by `unsubscribe`. */
  | ({ kind: 'chat'; id: number } & ChatScope)
  /** Close the subscription (any kind) opened under `id`. Never answered. */
  | { kind: 'unsubscribe'; id: number };

export type RpcReply =
  | { kind: 'response'; id: number; response: ApiResponse }
  | { kind: 'event'; id: number; event: DownloadEvent }
  | { kind: 'live'; id: number; event: LiveEvent }
  | { kind: 'chat'; id: number; event: ChatHostEvent | ConversationEvent };

/**
 * Whether an inbound payload is a well-formed request. A port is only as trusted
 * as whoever holds the other end, and a malformed message must be dropped rather
 * than crash the host's message loop.
 */
export function isRpcRequest(value: unknown): value is RpcRequest {
  if (typeof value !== 'object' || value === null) return false;
  const msg = value as {
    kind?: unknown;
    id?: unknown;
    call?: unknown;
    jobId?: unknown;
    topic?: unknown;
    scope?: unknown;
    conversationId?: unknown;
  };
  if (typeof msg.id !== 'number') return false;
  switch (msg.kind) {
    case 'call':
      return typeof msg.call === 'object' && msg.call !== null;
    case 'cancel':
      return true;
    case 'subscribe':
      return typeof msg.jobId === 'string';
    case 'watch':
      return msg.topic === 'live';
    case 'chat':
      return (
        msg.scope === 'host' ||
        (msg.scope === 'conversation' && typeof msg.conversationId === 'string' && msg.conversationId !== '')
      );
    case 'unsubscribe':
      return true;
    default:
      return false;
  }
}

/** Whether an inbound payload is a well-formed reply. Same reasoning as {@link isRpcRequest}. */
export function isRpcReply(value: unknown): value is RpcReply {
  if (typeof value !== 'object' || value === null) return false;
  const msg = value as { kind?: unknown; id?: unknown; response?: unknown; event?: unknown };
  if (typeof msg.id !== 'number') return false;
  switch (msg.kind) {
    case 'response':
      return typeof msg.response === 'object' && msg.response !== null;
    case 'event':
    case 'live':
    case 'chat':
      return typeof msg.event === 'object' && msg.event !== null;
    default:
      return false;
  }
}
