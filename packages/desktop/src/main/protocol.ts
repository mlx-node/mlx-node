/**
 * The `app://` scheme that serves the Control Panel SPA out of the bundle.
 *
 * The app has no TCP port for its own UI — that was the point of choosing a custom
 * protocol over a localhost server. Nothing to firewall, nothing another process
 * can connect to, no port to collide on.
 *
 * All routing decisions live in `www.ts` so they are testable without Electron;
 * this file is the I/O.
 */

import { pathToFileURL } from 'node:url';

import { net, protocol } from 'electron';

import { CSP, decideWww } from './www.js';

/**
 * Must be called BEFORE `app.whenReady()` — Chromium reads the privileged-scheme
 * table once during startup, and a later registration is silently ignored.
 *
 * `standard: true` is the load-bearing flag. It gives `app://` real origin
 * semantics, so an absolute `/assets/index-<hash>.js` resolves against the host
 * instead of being treated as opaque — which is what lets `BrowserRouter`
 * (pushState) work with no `HashRouter` and no vite `base` change.
 *
 * `bypassCSP` stays false: the header `www.ts` supplies is the policy, and a
 * scheme that bypassed CSP would quietly make it decorative.
 */
export function registerAppScheme(): void {
  protocol.registerSchemesAsPrivileged([
    {
      scheme: 'app',
      privileges: {
        standard: true,
        secure: true,
        supportFetchAPI: true,
        stream: true,
        corsEnabled: true,
        bypassCSP: false,
      },
    },
  ]);
}

/** Serve the SPA from `wwwRoot`. Call after `app.whenReady()`. */
export function installAppProtocol(wwwRoot: string): void {
  protocol.handle('app', async (request) => {
    const { pathname } = new URL(request.url);
    const decision = decideWww(wwwRoot, pathname);

    if (decision.kind === 'notFound') {
      return new Response(`Not found: ${pathname}`, {
        status: 404,
        headers: { 'Content-Type': 'text/plain; charset=utf-8' },
      });
    }

    // `net.fetch` over file:// streams and honours range requests, which matters
    // because the bundle is read from inside the .app on every navigation.
    const res = await net.fetch(pathToFileURL(decision.file).toString());
    const headers = new Headers(res.headers);
    headers.set('Content-Type', decision.contentType);
    headers.set('Content-Security-Policy', CSP);
    headers.set('Cache-Control', decision.immutable ? 'public, max-age=31536000, immutable' : 'no-cache');
    return new Response(res.body, { status: res.status, headers });
  });
}
