const DEFAULT_MODEL_STORAGE_PREFIX = 'models/qwen3.5-0.8b-mlx-bf16';

type EnvLike = Record<string, unknown>;

type AuthContext = {
  env: EnvLike;
  req: {
    header(name: string): string | undefined;
  };
};

export type UploadedPart = {
  partNumber: number;
  etag: string;
};

export function configuredModelPrefix(env: EnvLike) {
  const value = typeof env.MODEL_STORAGE_PREFIX === 'string' ? env.MODEL_STORAGE_PREFIX : DEFAULT_MODEL_STORAGE_PREFIX;
  return value.replace(/^\/+|\/+$/g, '') || DEFAULT_MODEL_STORAGE_PREFIX;
}

export function sanitizeModelPath(value: unknown) {
  const raw = typeof value === 'string' ? value : '';
  if (!raw || raw.includes('\0') || raw.startsWith('/')) return null;

  const parts = raw.split('/').filter(Boolean);
  if (parts.length === 0 || parts.some((part) => part === '.' || part === '..')) return null;
  return parts.join('/');
}

export function keyForModelPath(env: EnvLike, modelPath: string) {
  return `${configuredModelPrefix(env)}/${modelPath}`;
}

export function sanitizeStorageKey(env: EnvLike, value: unknown) {
  const raw = typeof value === 'string' ? value.replace(/^\/+/, '') : '';
  if (!raw || raw.includes('\0')) return null;

  const parts = raw.split('/').filter(Boolean);
  if (parts.some((part) => part === '.' || part === '..')) return null;

  const key = parts.join('/');
  const prefix = `${configuredModelPrefix(env)}/`;
  return key.startsWith(prefix) ? key : null;
}

export function inferModelContentType(pathname: string) {
  if (pathname.endsWith('.json')) return 'application/json; charset=utf-8';
  if (pathname.endsWith('.txt')) return 'text/plain; charset=utf-8';
  if (pathname.endsWith('.safetensors')) return 'application/octet-stream';
  if (pathname.endsWith('.model')) return 'application/octet-stream';
  return 'application/octet-stream';
}

export function baseModelHeaders(contentType?: string) {
  const headers = new Headers();
  headers.set('Accept-Ranges', 'bytes');
  headers.set('Access-Control-Allow-Origin', '*');
  headers.set('Access-Control-Allow-Headers', 'Range, Content-Type, Authorization');
  headers.set('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS, PUT, POST');
  headers.set('Cache-Control', 'public, max-age=31536000, immutable');
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
  if (contentType) headers.set('Content-Type', contentType);
  return headers;
}

export function uploadHeaders() {
  const headers = new Headers();
  headers.set('Access-Control-Allow-Origin', '*');
  headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  headers.set('Access-Control-Allow-Methods', 'OPTIONS, PUT, POST');
  headers.set('Cache-Control', 'no-store');
  return headers;
}

export function jsonResponse(body: unknown, init: ResponseInit = {}) {
  const headers = new Headers(init.headers);
  headers.set('Content-Type', 'application/json; charset=utf-8');
  for (const [key, value] of uploadHeaders()) {
    if (!headers.has(key)) headers.set(key, value);
  }
  return new Response(JSON.stringify(body), { ...init, headers });
}

export function requireUploadAuth(c: AuthContext) {
  const token = typeof c.env.MODEL_UPLOAD_TOKEN === 'string' ? c.env.MODEL_UPLOAD_TOKEN : '';
  if (!token) {
    return jsonResponse({ error: 'MODEL_UPLOAD_TOKEN is not configured' }, { status: 503 });
  }

  const auth = c.req.header('authorization') ?? '';
  if (auth !== `Bearer ${token}`) {
    return jsonResponse({ error: 'unauthorized' }, { status: 401 });
  }

  return null;
}

export function parseRangeHeader(value: string | null, size: number) {
  if (!value) return null;
  const match = /^bytes=(\d*)-(\d*)$/.exec(value.trim());
  if (!match) return { unsatisfiable: true as const };

  const [, startText, endText] = match;
  if (!startText && !endText) return { unsatisfiable: true as const };

  if (!startText) {
    const suffixLength = Number.parseInt(endText!, 10);
    if (!Number.isSafeInteger(suffixLength) || suffixLength <= 0) return { unsatisfiable: true as const };
    const length = Math.min(suffixLength, size);
    const offset = size - length;
    return { offset, length, end: size - 1 };
  }

  const offset = Number.parseInt(startText, 10);
  if (!Number.isSafeInteger(offset) || offset < 0 || offset >= size) {
    return { unsatisfiable: true as const };
  }

  const requestedEnd = endText ? Number.parseInt(endText, 10) : size - 1;
  if (!Number.isSafeInteger(requestedEnd) || requestedEnd < offset) return { unsatisfiable: true as const };
  const end = Math.min(requestedEnd, size - 1);
  return { offset, length: end - offset + 1, end };
}
