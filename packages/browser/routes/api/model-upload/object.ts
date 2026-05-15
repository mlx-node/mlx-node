import { defineHandler } from 'void';
import { storage } from 'void/storage';

import {
  inferModelContentType,
  jsonResponse,
  requireUploadAuth,
  sanitizeStorageKey,
  uploadHeaders,
} from '../../_model-storage';

export const OPTIONS = defineHandler(() => new Response(null, { status: 204, headers: uploadHeaders() }));

export const PUT = defineHandler(async (c) => {
  const authFailure = requireUploadAuth(c);
  if (authFailure) return authFailure;

  const url = new URL(c.req.url);
  const key = sanitizeStorageKey(c.env, url.searchParams.get('key'));
  if (!key) return jsonResponse({ error: 'invalid key' }, { status: 400 });

  await storage.put(key, c.req.raw.body, {
    httpMetadata: {
      contentType: c.req.header('content-type') ?? inferModelContentType(key),
    },
  });

  return jsonResponse({ ok: true, key });
});
