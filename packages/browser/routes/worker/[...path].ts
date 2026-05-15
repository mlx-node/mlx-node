import { defineHandler } from 'void';

const WORKER_CACHE_CONTROL = 'public, max-age=31536000, immutable';
const ERROR_CACHE_CONTROL = 'no-store';

type AssetBinding = {
  fetch(request: Request): Promise<Response>;
};

function workerHeaders(source: Headers, cacheControl = WORKER_CACHE_CONTROL) {
  const headers = new Headers(source);
  headers.set('Cache-Control', cacheControl);
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
  if (!headers.has('Content-Type')) {
    headers.set('Content-Type', 'text/javascript; charset=utf-8');
  }
  return headers;
}

function assetBinding(env: Record<string, unknown>) {
  const binding = env.ASSETS as AssetBinding | undefined;
  return typeof binding?.fetch === 'function' ? binding : null;
}

function safeAssetPath(value: string | undefined) {
  const path = value?.replace(/^\/+/g, '') ?? '';
  if (!path || path.includes('..') || path.includes('\\')) return null;
  return path;
}

export const GET = defineHandler(async (c) => {
  const path = safeAssetPath(c.req.param('path'));
  if (!path) {
    return new Response('Not found', {
      status: 404,
      headers: workerHeaders(new Headers({ 'Content-Type': 'text/plain; charset=utf-8' }), ERROR_CACHE_CONTROL),
    });
  }

  const assets = assetBinding(c.env);
  if (!assets) {
    return new Response('Static asset binding is not configured', {
      status: 503,
      headers: workerHeaders(new Headers({ 'Content-Type': 'text/plain; charset=utf-8' }), ERROR_CACHE_CONTROL),
    });
  }

  const assetUrl = new URL(c.req.url);
  assetUrl.pathname = `/assets/${path}`;
  const asset = await assets.fetch(
    new Request(assetUrl, {
      headers: c.req.raw.headers,
    }),
  );

  if (asset.status === 404) {
    return new Response('Not found', {
      status: 404,
      headers: workerHeaders(new Headers({ 'Content-Type': 'text/plain; charset=utf-8' }), ERROR_CACHE_CONTROL),
    });
  }

  const cacheControl =
    asset.status === 304 || (asset.status >= 200 && asset.status < 400)
      ? WORKER_CACHE_CONTROL
      : ERROR_CACHE_CONTROL;

  return new Response(asset.body, {
    status: asset.status,
    statusText: asset.statusText,
    headers: workerHeaders(asset.headers, cacheControl),
  });
});
