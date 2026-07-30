/** Bind-address and port helpers shared by every inference host front-end. */

import { createServer as netCreateServer } from 'node:net';

/**
 * Wrap a bare IPv6 literal in `[...]` so it is safe inside a URL authority.
 *
 * Any host carrying a `:` that is not already bracketed is an IPv6 literal
 * (`::1`, `2001:db8::1`, or a scoped `fe80::1%en0`); an unbracketed one makes
 * the trailing `:<port>` ambiguous and `new URL()` reject it. IPv4/hostnames
 * (no `:`) and already-bracketed literals pass through unchanged. A scoped
 * literal is bracketed too: `new URL` still rejects the `%`, but the printed
 * string stays well-formed rather than crashing the caller.
 *
 * `@mlx-node/dashboard` carries its own copy for its own bind logic; this one
 * is the inference host's, so the two can diverge without coupling.
 */
export function bracketHost(host: string): string {
  return host.includes(':') && !host.startsWith('[') ? `[${host}]` : host;
}

/**
 * Convert the one bracketed bind form the host security policy accepts into
 * the bare literal Node's `server.listen()` requires.
 *
 * Brackets belong to URL authorities, not socket bind addresses: passing
 * `[::1]` to Node performs a DNS lookup and fails with `ENOTFOUND`. This is
 * deliberately exact rather than a general bracket stripper. In particular,
 * `[::]` and bracketed routable IPv6 text must not be turned into working
 * wildcard/network binds by a helper intended only to repair loopback.
 */
export function normalizeLoopbackBindHost(host: string): string {
  return host === '[::1]' ? '::1' : host;
}

/**
 * Build the URL a client should connect to for a `(host, port)` bind.
 *
 * A wildcard bind has no connectable literal host, so advertise the matching
 * loopback instead: `::` binds IPv6 → `[::1]`, `0.0.0.0`/`''` bind IPv4 →
 * `127.0.0.1`. A concrete host is advertised as-is with any IPv6 literal
 * bracketed.
 */
export function hostUrl(host: string, port: number): string {
  const wildcard = host === '0.0.0.0' || host === '::' || host === '';
  const displayHost = wildcard ? (host === '::' ? '[::1]' : '127.0.0.1') : bracketHost(host);
  return `http://${displayHost}:${port}`;
}

/**
 * Does this BIND address keep the socket unreachable from another machine?
 *
 * Distinct from the desktop supervisor's `isLoopbackHttpUrl`, which classifies
 * a URL a client would connect TO. This one classifies the string handed to
 * `server.listen(port, host)`, where the failure modes are different: the
 * wildcards (`0.0.0.0`, `::`, `''`) are not routable addresses at all yet bind
 * every interface, and `hostUrl` deliberately ADVERTISES them as loopback — so
 * a check written against the advertised URL would call a wildcard bind safe.
 *
 * Loopback: the whole `127.0.0.0/8` block, `::1` (bracketed or not), and
 * `localhost`. Anything else — a LAN address, a wildcard, a hostname that
 * resolves off-box — is treated as reachable, because a predicate that has to
 * resolve DNS to answer would either block startup or guess.
 */
export function isLoopbackBindHost(host: string): boolean {
  if (host === 'localhost' || host === '::1' || host === '[::1]') return true;
  // `::ffff:127.0.0.1` is the IPv4-mapped form Node accepts on a dual-stack
  // socket; it binds the same loopback interface.
  if (/^::ffff:127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/i.test(host)) return true;
  return /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(host);
}

/**
 * Ask the kernel for a free port by binding to port 0, reading the assigned
 * port, and closing immediately.
 *
 * Racy in theory (another process could grab the port between the close and
 * the real listen), but acceptable for a developer-local host. Callers that
 * cannot tolerate the race should pass `port: 0` straight through to
 * `createServer` and read the bound port back off the socket instead — the
 * inference host does exactly that when no port is requested at all; this
 * helper exists for the case where the port must be known BEFORE the server
 * starts (e.g. baking `ANTHROPIC_BASE_URL` into a child's env).
 */
export async function pickFreePort(): Promise<number> {
  return await new Promise<number>((resolve, reject) => {
    const probe = netCreateServer();
    probe.unref();
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const addr = probe.address();
      if (addr && typeof addr === 'object') {
        const port = addr.port;
        probe.close(() => resolve(port));
      } else {
        probe.close(() => reject(new Error('could not determine free port')));
      }
    });
  });
}
