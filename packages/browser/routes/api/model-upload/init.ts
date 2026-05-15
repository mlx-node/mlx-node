import { defineHandler } from 'void';
import { storage } from 'void/storage';

import {
  inferModelContentType,
  jsonResponse,
  requireUploadAuth,
  sanitizeStorageKey,
  uploadHeaders,
} from '../../_model-storage';

type InitBody = {
  key?: unknown;
  contentType?: unknown;
};

export const OPTIONS = defineHandler(() => new Response(null, { status: 204, headers: uploadHeaders() }));

export const POST = defineHandler(async (c) => {
  const authFailure = requireUploadAuth(c);
  if (authFailure) return authFailure;

  const body = (await c.req.json().catch(() => ({}))) as InitBody;
  const key = sanitizeStorageKey(c.env, body.key);
  if (!key) return jsonResponse({ error: 'invalid key' }, { status: 400 });

  const contentType = typeof body.contentType === 'string' ? body.contentType : inferModelContentType(key);
  const upload = await storage.createMultipartUpload(key, {
    httpMetadata: { contentType },
  });

  return jsonResponse({ key: upload.key, uploadId: upload.uploadId });
});
