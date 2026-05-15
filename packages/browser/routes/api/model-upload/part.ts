import { defineHandler } from 'void';
import { storage } from 'void/storage';

import { jsonResponse, requireUploadAuth, sanitizeStorageKey, uploadHeaders } from '../../_model-storage';

export const OPTIONS = defineHandler(() => new Response(null, { status: 204, headers: uploadHeaders() }));

export const PUT = defineHandler(async (c) => {
  const authFailure = requireUploadAuth(c);
  if (authFailure) return authFailure;

  const url = new URL(c.req.url);
  const key = sanitizeStorageKey(c.env, url.searchParams.get('key'));
  const uploadId = url.searchParams.get('uploadId') ?? '';
  const partNumber = Number.parseInt(url.searchParams.get('partNumber') ?? '', 10);
  if (!key || !uploadId || !Number.isSafeInteger(partNumber) || partNumber < 1) {
    return jsonResponse({ error: 'invalid multipart part request' }, { status: 400 });
  }
  if (!c.req.raw.body) return jsonResponse({ error: 'missing request body' }, { status: 400 });

  const upload = storage.resumeMultipartUpload(key, uploadId);
  const part = await upload.uploadPart(partNumber, c.req.raw.body);
  return jsonResponse(part);
});
