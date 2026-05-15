import { defineHandler } from 'void';
import { storage } from 'void/storage';

import {
  baseModelHeaders,
  inferModelContentType,
  keyForModelPath,
  parseRangeHeader,
  sanitizeModelPath,
} from '../../_model-storage';

export const OPTIONS = defineHandler(() => {
  return new Response(null, { status: 204, headers: baseModelHeaders() });
});

export const GET = defineHandler(async (c) => {
  const modelPath = sanitizeModelPath(c.req.param('path'));
  if (!modelPath) return new Response('Not found', { status: 404, headers: baseModelHeaders('text/plain') });

  const key = keyForModelPath(c.env, modelPath);
  const head = await storage.head(key);
  if (!head) return new Response('Not found', { status: 404, headers: baseModelHeaders('text/plain') });

  const contentType = head.httpMetadata?.contentType ?? inferModelContentType(modelPath);
  const range = parseRangeHeader(c.req.header('range') ?? null, head.size);
  if (range && 'unsatisfiable' in range) {
    const headers = baseModelHeaders(contentType);
    headers.set('Content-Range', `bytes */${head.size}`);
    return new Response('Range Not Satisfiable', { status: 416, headers });
  }

  const object = range
    ? await storage.get(key, { range: { offset: range.offset, length: range.length } })
    : await storage.get(key);
  if (!object) return new Response('Not found', { status: 404, headers: baseModelHeaders('text/plain') });

  const headers = baseModelHeaders(contentType);
  object.writeHttpMetadata(headers);
  headers.set('ETag', object.httpEtag);
  if (range) {
    headers.set('Content-Length', `${range.length}`);
    headers.set('Content-Range', `bytes ${range.offset}-${range.end}/${head.size}`);
    return new Response(object.body, { status: 206, headers });
  }

  headers.set('Content-Length', `${object.size}`);
  return new Response(object.body, { headers });
});
