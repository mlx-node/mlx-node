/**
 * The error text law of the Chat and provider routes: an `ApiError` passes through; anything else becomes one fixed
 * `ApiError` of its route. The raw error is dropped: never logged, never attached as a cause, never put in a reply,
 * because keyring, `fetch`, `fs` and agent errors can quote a key, a path or agent text.
 */

import { SessionWriteError } from '../agent-sessions/writes.js';
import { ApiError, type ApiErrorCode } from '../api/errors.js';
import { AgentHostError, FAILURE_MESSAGES, InteractionError } from './agent/agent-host.js';
import type { ChatFailureCode } from './types.js';

export type GuardedRoute = 'chat' | 'provider-key' | 'provider-validate' | 'provider-selection';

const ROUTE_FAILURES: Readonly<Record<GuardedRoute, string>> = {
  chat: 'The agent could not be reached.',
  'provider-key': 'The Keychain cannot be reached.',
  'provider-validate': 'The provider could not be reached.',
  'provider-selection': 'Skills could not be installed.',
};

const CODE_OF_FAILURE: Readonly<Record<ChatFailureCode, ApiErrorCode>> = {
  'inference-stopped': 'E_UNAVAILABLE',
  'inference-failed': 'E_UNAVAILABLE',
  'no-models': 'E_UNAVAILABLE',
  'agent-missing': 'E_UNAVAILABLE',
  'agent-too-old': 'E_UNAVAILABLE',
  'agent-incompatible': 'E_UNAVAILABLE',
  'agent-protocol-error': 'E_UNAVAILABLE',
  'agent-crashed': 'E_UNAVAILABLE',
  'agent-failed': 'E_UNAVAILABLE',
  'agent-slow': 'E_UNAVAILABLE',
  'keychain-unavailable': 'E_UNAVAILABLE',
  'skill-bundle-corrupt': 'E_UNAVAILABLE',
  'model-load-failed': 'E_UNAVAILABLE',
  'profile-mismatch': 'E_CONFLICT',
  'model-unavailable': 'E_CONFLICT',
  'cwd-mismatch': 'E_CONFLICT',
  'conversation-busy': 'E_CONFLICT',
  'engine-busy': 'E_CONFLICT',
  'cwd-invalid': 'E_BAD_REQUEST',
  'slash-text': 'E_BAD_REQUEST',
};

/** The typed route failure of a code: its API code, its fixed sentence, and the code as `reason`. */
export function chatRouteError(code: ChatFailureCode): ApiError {
  return new ApiError(CODE_OF_FAILURE[code], FAILURE_MESSAGES[code], code);
}

const SESSION_WRITE_MESSAGES: Readonly<Record<SessionWriteError['kind'], string>> = {
  'bad-request': 'That request is not valid.',
  'not-found': 'No such conversation.',
  conflict: 'The conversation is in use. Try again in a moment.',
};

/** Any thrown value as the fixed `ApiError` of `route`. */
export function routeError(route: GuardedRoute, error: unknown): ApiError {
  if (error instanceof ApiError) return error;
  if (error instanceof AgentHostError) return chatRouteError(error.code);
  if (error instanceof InteractionError) {
    return error.kind === 'not-found'
      ? new ApiError('E_NOT_FOUND', 'No such request.')
      : new ApiError('E_BAD_REQUEST', 'That option was not offered.');
  }
  if (error instanceof SessionWriteError) {
    const code: ApiErrorCode =
      error.kind === 'bad-request' ? 'E_BAD_REQUEST' : error.kind === 'not-found' ? 'E_NOT_FOUND' : 'E_CONFLICT';
    return new ApiError(code, SESSION_WRITE_MESSAGES[error.kind]);
  }
  return new ApiError('E_UNAVAILABLE', ROUTE_FAILURES[route]);
}

/** Runs a handler body under the error text law of `route`. */
export async function guarded<T>(route: GuardedRoute, run: () => T | Promise<T>): Promise<T> {
  try {
    return await run();
  } catch (error) {
    throw routeError(route, error);
  }
}
