import { defineHandler } from 'void';
import { storage } from 'void/storage';

import { jsonResponse, requireUploadAuth, sanitizeStorageKey, uploadHeaders } from '../../_model-storage';

type AbortBody = {
  key?: unknown;
  uploadId?: unknown;
};

export const OPTIONS = defineHandler(() => new Response(null, { status: 204, headers: uploadHeaders() }));

export const POST = defineHandler(async (c) => {
  const authFailure = requireUploadAuth(c);
  if (authFailure) return authFailure;

  const body = (await c.req.json().catch(() => ({}))) as AbortBody;
  const key = sanitizeStorageKey(c.env, body.key);
  const uploadId = typeof body.uploadId === 'string' ? body.uploadId : '';
  if (!key || !uploadId) return jsonResponse({ error: 'invalid multipart abort' }, { status: 400 });

  await storage.resumeMultipartUpload(key, uploadId).abort();
  return jsonResponse({ ok: true, key });
});
