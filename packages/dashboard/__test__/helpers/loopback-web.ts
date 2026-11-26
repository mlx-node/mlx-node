/**
 * The real `mlx_web` with every provider origin on one loopback fake, and a throwaway Exa key set through the providers
 * service (its key check goes to the fake too): the address ledger alone then decides each `web_fetch`. The live smoke
 * runs its address-ledger rows on it; nothing reaches a real provider.
 */

import { randomBytes } from 'node:crypto';
import { mkdirSync, realpathSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import type { ChatPaths } from '../../src/chat/paths.js';
import { MemorySecretStore } from '../../src/chat/providers/secrets.js';
import { ProvidersService, type ProvidersServiceOptions } from '../../src/chat/providers/service.js';
import { CHECK_OK, type FakeAnswer, type FakeRequest, startFakeProvider } from './fake-providers.js';

const WEB_DIR = realpathSync(fileURLToPath(new URL('../../src/chat/web/', import.meta.url)));

/** An `mlx_web` entry: `server.ts` from source, as `server-ts-bootstrap.ts` runs it, with every origin at `baseUrl`. */
export function loopbackWebServerText(baseUrl: string): string {
  const tools = pathToFileURL(join(WEB_DIR, 'tools.ts')).href;
  const server = pathToFileURL(join(WEB_DIR, 'server.ts')).href;
  return `import { existsSync } from 'node:fs';
import { registerHooks } from 'node:module';
import { fileURLToPath } from 'node:url';

registerHooks({
  resolve(spec, ctx, next) {
    if (spec.startsWith('.') && spec.endsWith('.js') && ctx.parentURL !== undefined) {
      const asTs = new URL(spec.slice(0, -3) + '.ts', ctx.parentURL);
      if (existsSync(fileURLToPath(asTs))) return { url: asTs.href, format: 'module-typescript', shortCircuit: true };
    }
    return next(spec, ctx);
  },
});

const base = ${JSON.stringify(baseUrl)};
const { PRODUCTION_DEPS } = await import(${JSON.stringify(tools)});
PRODUCTION_DEPS.baseUrls = { exa: base, deepseek: base, brave: base, firecrawl: base };
await import(${JSON.stringify(server)});
`;
}

export interface LoopbackWeb {
  /** Exa selected with the throwaway key: `chatWeb()` names it for search and fetch. */
  providers: ProvidersService;
  webServer: { command: string; script: string };
  /** Every address a `web_fetch` sent to the provider, in order. */
  fetched(): string[];
  /** Deletes the key (the selection goes with it) and stops the fake. */
  close(): Promise<void>;
}

function answer(request: FakeRequest): FakeAnswer {
  if (request.path === '/monitors') return { json: CHECK_OK.exa };
  if (request.path === '/contents') {
    return { json: { results: [{ title: 'Loopback page', text: 'This page comes from the loopback provider.' }] } };
  }
  return { json: { results: [] } };
}

function fetchedUrls(request: FakeRequest): string[] {
  if (request.path !== '/contents') return [];
  try {
    const urls = (JSON.parse(request.body) as { urls?: unknown }).urls;
    return Array.isArray(urls) ? urls.filter((url): url is string => typeof url === 'string') : [];
  } catch {
    return [];
  }
}

/** Writes the entry to `<paths.root>/tmp/loopback-web-server.mjs`; throws when the key is not taken. */
export async function startLoopbackWeb(
  paths: ChatPaths,
  workPath: ProvidersServiceOptions['workPath'],
): Promise<LoopbackWeb> {
  const fake = await startFakeProvider(answer);
  const providers = new ProvidersService({
    paths,
    secrets: new MemorySecretStore(),
    baseUrls: { exa: fake.baseUrl, deepseek: fake.baseUrl, brave: fake.baseUrl, firecrawl: fake.baseUrl },
    workPath,
  });
  const reply = await providers.putKey('exa', `loopback-${randomBytes(12).toString('hex')}`);
  if (!reply.saved || providers.chatWeb().fetchProvider !== 'exa') {
    await fake.close();
    throw new Error('the loopback Exa key was not taken');
  }
  const script = join(paths.root, 'tmp', 'loopback-web-server.mjs');
  mkdirSync(dirname(script), { recursive: true });
  writeFileSync(script, loopbackWebServerText(fake.baseUrl));
  return {
    providers,
    webServer: { command: process.execPath, script },
    fetched: () => fake.requests.flatMap(fetchedUrls),
    close: async () => {
      await providers.deleteKey('exa');
      await fake.close();
    },
  };
}
