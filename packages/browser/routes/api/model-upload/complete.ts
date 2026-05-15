import { defineHandler } from 'void';
import { storage } from 'void/storage';

import {
  type UploadedPart,
  jsonResponse,
  requireUploadAuth,
  sanitizeStorageKey,
  uploadHeaders,
} from '../../_model-storage';

type CompleteBody = {
  key?: unknown;
  uploadId?: unknown;
  parts?: unknown;
};

function sanitizeParts(value: unknown): UploadedPart[] | null {
  if (!Array.isArray(value)) return null;
  const parts = value.map((part) => ({
    partNumber:
      typeof part === 'object' && part !== null ? Number((part as { partNumber?: unknown }).partNumber) : Number.NaN,
    etag: typeof part === 'object' && part !== null ? (part as { etag?: unknown }).etag : undefined,
  }));
  if (
    parts.some(
      (part) =>
        !Number.isSafeInteger(part.partNumber) ||
        part.partNumber < 1 ||
        typeof part.etag !== 'string' ||
        part.etag.length === 0,
    )
  ) {
    return null;
  }
  return parts as UploadedPart[];
}

export const OPTIONS = defineHandler(() => new Response(null, { status: 204, headers: uploadHeaders() }));

export const POST = defineHandler(async (c) => {
  const authFailure = requireUploadAuth(c);
  if (authFailure) return authFailure;

  const body = (await c.req.json().catch(() => ({}))) as CompleteBody;
  const key = sanitizeStorageKey(c.env, body.key);
  const uploadId = typeof body.uploadId === 'string' ? body.uploadId : '';
  const parts = sanitizeParts(body.parts);
  if (!key || !uploadId || !parts) return jsonResponse({ error: 'invalid multipart completion' }, { status: 400 });

  const object = await storage.resumeMultipartUpload(key, uploadId).complete(parts);
  return jsonResponse({ ok: true, key: object.key, size: object.size, etag: object.httpEtag });
});
