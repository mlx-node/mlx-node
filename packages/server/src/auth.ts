/**
 * Optional bearer-token gate.
 *
 * Enforced at the single choke point in `createHandler`'s returned closure —
 * after the CORS/OPTIONS early-return, before `routeRequest` — so there is
 * exactly one place a route can be added without inheriting the gate.
 *
 * Scope, deliberately narrow: this is a shared-secret check for a server
 * bound to loopback and supervised by a local app. It is NOT a
 * multi-tenant auth system — there is one token, no rotation, no scopes,
 * no per-caller identity.
 */

import { timingSafeEqual } from 'node:crypto';
import type { IncomingMessage, ServerResponse } from 'node:http';

/** `Bearer ` prefix length, used to slice the credential out of `authorization`. */
const BEARER_PREFIX = 'bearer ';

/**
 * Extract the presented credential, or `null` when none is usable.
 *
 * `x-api-key` is checked FIRST because Anthropic clients (including Claude
 * Code, the primary consumer of `/v1/messages`) send it. Checking it first
 * also means a stale `authorization` header injected by an intermediary
 * cannot shadow the caller's real key.
 *
 * Array-valued headers are rejected outright. Node collapses repeated
 * non-allowlisted headers such as `x-api-key` into a `string[]`; joining or
 * picking one arbitrarily would let a caller smuggle a second value past a
 * front proxy that only inspected the first.
 */
export function extractPresentedToken(req: IncomingMessage): string | null {
  const apiKey = req.headers['x-api-key'];
  if (apiKey !== undefined) {
    return typeof apiKey === 'string' ? apiKey : null;
  }

  const authorization = req.headers.authorization;
  if (typeof authorization !== 'string') return null;
  // Scheme is case-insensitive per RFC 7235; the credential is not.
  if (authorization.length <= BEARER_PREFIX.length) return null;
  if (authorization.slice(0, BEARER_PREFIX.length).toLowerCase() !== BEARER_PREFIX) return null;
  return authorization.slice(BEARER_PREFIX.length);
}

/**
 * Constant-time-ish credential comparison.
 *
 * The length check in front of `timingSafeEqual` is unavoidable — the
 * primitive throws on mismatched lengths. It therefore LEAKS the length of
 * the configured token to an attacker who can time responses. That is an
 * accepted trade: the token is a locally-generated high-entropy secret, and
 * knowing its length does not meaningfully reduce the search space. The
 * byte-by-byte content comparison, which is the part that would otherwise
 * allow incremental guessing, stays constant-time.
 */
export function tokensMatch(presented: string, expected: string): boolean {
  // The empty string is not a credential, in either position. `timingSafeEqual`
  // on two zero-length buffers returns TRUE, so a server misconfigured with an
  // empty token would authenticate every caller who sent an empty `x-api-key`.
  // `resolveAuthToken` rejects an empty token before it can reach a server built
  // through `createServer`; this is the second latch, for a caller that
  // constructs `createHandler` directly. Refusing is the fail-closed direction —
  // an empty token 401s everything rather than admitting everything.
  if (presented === '' || expected === '') return false;
  const presentedBytes = Buffer.from(presented, 'utf8');
  const expectedBytes = Buffer.from(expected, 'utf8');
  if (presentedBytes.length !== expectedBytes.length) return false;
  return timingSafeEqual(presentedBytes, expectedBytes);
}

/**
 * `true` when the caller offered SOME credential, valid or not.
 *
 * Used only by the `/health` carve-out: an anonymous poll degrades to a
 * liveness-only body, but a caller who presented a wrong (or malformed)
 * credential gets a 401 so a mistyped token surfaces instead of masquerading
 * as a healthy 200.
 */
export function hasCredential(req: IncomingMessage): boolean {
  return req.headers['x-api-key'] !== undefined || req.headers.authorization !== undefined;
}

/** `true` when the request carries a credential matching `expected`. */
export function isAuthorized(req: IncomingMessage, expected: string): boolean {
  const presented = extractPresentedToken(req);
  if (presented === null) return false;
  return tokensMatch(presented, expected);
}

/**
 * 401 with `WWW-Authenticate: Bearer`. The body deliberately says nothing
 * about whether a credential was absent, malformed, or simply wrong.
 */
export function sendUnauthorized(res: ServerResponse): void {
  res.writeHead(401, {
    'WWW-Authenticate': 'Bearer realm="mlx-node"',
    'Content-Type': 'application/json',
  });
  res.end(
    JSON.stringify({
      error: {
        type: 'authentication_error',
        message: 'Missing or invalid API key',
        code: null,
        param: null,
      },
    }),
  );
}
