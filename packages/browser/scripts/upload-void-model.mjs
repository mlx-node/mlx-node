#!/usr/bin/env node
import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dirname = path.dirname(fileURLToPath(import.meta.url));
const defaultModelDir = path.resolve(
  dirname,
  '../../../../mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16',
);
const defaultBaseUrl = 'https://mlx-node-browser.void.app';
const defaultPrefix = 'models/qwen3.5-0.8b-mlx-bf16';
const defaultPartSize = 32 * 1024 * 1024;
const maxSinglePutSize = 32 * 1024 * 1024;

function usage() {
  console.error(`Usage:
  MODEL_UPLOAD_TOKEN=<token> yarn workspace @mlx-node/browser upload:void-model [model-dir] [base-url]

Environment:
  MODEL_UPLOAD_TOKEN       required bearer token configured in Void
  MODEL_STORAGE_PREFIX     storage prefix; default ${defaultPrefix}
  MODEL_UPLOAD_PART_MB     multipart part size; default 32
  MODEL_UPLOAD_MODEL_URL   model read base; default <base-url>/api/model
  MODEL_UPLOAD_API_URL     upload API base; default <base-url>/api/model-upload`);
}

function normalizeBaseUrl(value) {
  const url = new URL(value);
  if (!url.pathname.endsWith('/')) url.pathname += '/';
  return url;
}

function relativeUrl(base, relPath) {
  return new URL(
    relPath
      .split('/')
      .map((part) => encodeURIComponent(part))
      .join('/'),
    base,
  );
}

function contentTypeFor(file) {
  if (file.endsWith('.json')) return 'application/json; charset=utf-8';
  if (file.endsWith('.txt')) return 'text/plain; charset=utf-8';
  if (file.endsWith('.safetensors')) return 'application/octet-stream';
  if (file.endsWith('.model')) return 'application/octet-stream';
  return 'application/octet-stream';
}

async function listFiles(root) {
  const out = [];
  async function walk(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const absolute = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await walk(absolute);
      } else if (entry.isFile()) {
        out.push(absolute);
      }
    }
  }
  await walk(root);
  return out.sort();
}

async function request(url, init) {
  const resp = await fetch(url, init);
  if (resp.ok) return resp;

  const text = await resp.text().catch(() => '');
  throw new Error(`${init.method ?? 'GET'} ${url} failed: ${resp.status} ${resp.statusText}${text ? `\n${text}` : ''}`);
}

async function remoteHasSameSize(modelBaseUrl, relPath, size) {
  const resp = await fetch(relativeUrl(modelBaseUrl, relPath), { method: 'HEAD', cache: 'no-store' }).catch(() => null);
  if (!resp?.ok) return false;
  return Number(resp.headers.get('content-length')) === size;
}

async function uploadObject(uploadBaseUrl, token, key, filePath, size, contentType) {
  const url = new URL('object', uploadBaseUrl);
  url.searchParams.set('key', key);
  await request(url, {
    method: 'PUT',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': contentType,
      'Content-Length': `${size}`,
    },
    body: createReadStream(filePath),
    duplex: 'half',
  });
}

async function uploadMultipart(uploadBaseUrl, token, key, filePath, size, contentType, partSize) {
  const initResp = await request(new URL('init', uploadBaseUrl), {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ key, contentType, size }),
  });
  const { uploadId } = await initResp.json();
  const parts = [];

  try {
    const file = await fs.open(filePath, 'r');
    try {
      const buffer = Buffer.allocUnsafe(partSize);
      let offset = 0;
      let partNumber = 1;
      while (offset < size) {
        const bytesToRead = Math.min(partSize, size - offset);
        const { bytesRead } = await file.read(buffer, 0, bytesToRead, offset);
        if (bytesRead <= 0) break;

        const partUrl = new URL('part', uploadBaseUrl);
        partUrl.searchParams.set('key', key);
        partUrl.searchParams.set('uploadId', uploadId);
        partUrl.searchParams.set('partNumber', `${partNumber}`);

        const body = buffer.subarray(0, bytesRead);
        const partResp = await request(partUrl, {
          method: 'PUT',
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/octet-stream',
            'Content-Length': `${bytesRead}`,
          },
          body,
        });
        const part = await partResp.json();
        parts.push(part);

        offset += bytesRead;
        process.stdout.write(
          `\r  multipart ${partNumber} (${Math.min(100, Math.floor((offset / size) * 100))}%)`,
        );
        partNumber += 1;
      }
      process.stdout.write('\n');
    } finally {
      await file.close();
    }

    await request(new URL('complete', uploadBaseUrl), {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ key, uploadId, parts }),
    });
  } catch (error) {
    await request(new URL('abort', uploadBaseUrl), {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ key, uploadId }),
    }).catch(() => {});
    throw error;
  }
}

async function main() {
  const token = process.env.MODEL_UPLOAD_TOKEN;
  if (!token) {
    usage();
    process.exit(1);
  }

  const modelDir = path.resolve(process.argv[2] ?? defaultModelDir);
  const baseUrl = normalizeBaseUrl(process.argv[3] ?? process.env.MODEL_UPLOAD_BASE_URL ?? defaultBaseUrl);
  const uploadBaseUrl = normalizeBaseUrl(process.env.MODEL_UPLOAD_API_URL ?? new URL('/api/model-upload/', baseUrl));
  const modelBaseUrl = normalizeBaseUrl(process.env.MODEL_UPLOAD_MODEL_URL ?? new URL('/api/model/', baseUrl));
  const prefix = (process.env.MODEL_STORAGE_PREFIX ?? defaultPrefix).replace(/^\/+|\/+$/g, '');
  const partSizeMb = Number.parseFloat(process.env.MODEL_UPLOAD_PART_MB ?? '');
  const partSize =
    Number.isFinite(partSizeMb) && partSizeMb >= 8 ? Math.round(partSizeMb * 1024 * 1024) : defaultPartSize;

  const stat = await fs.stat(modelDir);
  if (!stat.isDirectory()) throw new Error(`Model path is not a directory: ${modelDir}`);

  const files = await listFiles(modelDir);
  console.log(`Uploading ${files.length} files from ${modelDir}`);
  console.log(`Read URL: ${modelBaseUrl.href}`);
  console.log(`Storage prefix: ${prefix}`);

  for (const filePath of files) {
    const relPath = path.relative(modelDir, filePath).split(path.sep).join('/');
    const key = `${prefix}/${relPath}`;
    const { size } = await fs.stat(filePath);
    const contentType = contentTypeFor(relPath);

    if (await remoteHasSameSize(modelBaseUrl, relPath, size)) {
      console.log(`skip ${relPath} (${size} bytes already present)`);
      continue;
    }

    console.log(`upload ${relPath} (${size} bytes)`);
    if (size <= maxSinglePutSize) {
      await uploadObject(uploadBaseUrl, token, key, filePath, size, contentType);
    } else {
      await uploadMultipart(uploadBaseUrl, token, key, filePath, size, contentType, partSize);
    }
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack || error.message : error);
  process.exit(1);
});
