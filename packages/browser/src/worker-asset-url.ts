export function workerAssetUrl(input: URL | string) {
  const url = input instanceof URL ? input : new URL(input, globalThis.location?.href ?? 'http://localhost/');

  if (!url.pathname.startsWith('/assets/')) {
    return url;
  }

  const assetPath = url.pathname.slice('/assets/'.length);
  if (!assetPath || assetPath.includes('..') || assetPath.includes('\\')) {
    return url;
  }

  const workerUrl = new URL(`/worker/${assetPath}`, url.origin);
  workerUrl.search = url.search;
  return workerUrl;
}
