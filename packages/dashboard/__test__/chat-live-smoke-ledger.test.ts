/**
 * The setup of the live smoke's address-ledger rows (`scripts/chat-live-smoke.ts`), without the agent: the server the
 * agent would start (`chatSessionParams`), on the ledger the host writes from the user text, answers each `web_fetch`
 * as the agent would get it.
 */

import { spawn } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import { AddressLedger, writeLedgerFile } from '../src/chat/agent/address-ledger.js';
import { type ChatWebEnv, chatSessionParams } from '../src/chat/agent/profiles.js';
import { chatPaths } from '../src/chat/paths.js';
import { MemorySecretStore } from '../src/chat/providers/secrets.js';
import { ProvidersService } from '../src/chat/providers/service.js';
import { webServerScript } from '../src/chat/service.js';
import { formatErrorText, parseWebResultText } from '../src/chat/web/result-text.js';
import { type LoopbackWeb, startLoopbackWeb } from './helpers/loopback-web.js';

const LISTED = 'https://example.com/marker-0123abcd';
const UNLISTED = 'https://doc.rust-lang.org/';

interface McpServer {
  command: string;
  args: string[];
  env: Array<{ name: string; value: string }>;
}

/** Each `web_fetch` through the server the agent would start for this web env; the text of each answer. */
function fetchThrough(
  webServer: { command: string; script: string },
  web: ChatWebEnv,
  ledgerFile: string,
  urls: string[],
): Promise<string[]> {
  const params = chatSessionParams({
    cwd: tmpdir(),
    modelId: 'm',
    webServer,
    web,
    prompt: '',
    pluginDir: '',
    ledgerFile,
  });
  const server = (params.mcpServers as McpServer[])[0]!;
  return new Promise((resolve, reject) => {
    const child = spawn(server.command, server.args, {
      env: {
        PATH: process.env.PATH ?? '/usr/bin:/bin',
        ...Object.fromEntries(server.env.map((e) => [e.name, e.value])),
      },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8').on('data', (d: string) => (stdout += d));
    child.stderr.setEncoding('utf8').on('data', (d: string) => (stderr += d));
    child.on('error', reject);
    child.on('close', () => {
      try {
        expect(stderr).toBe('');
        const replies = stdout
          .split('\n')
          .filter((line) => line !== '')
          .map((line) => JSON.parse(line) as { id: number; result: { content: Array<{ text: string }> } });
        replies.sort((a, b) => a.id - b.id);
        resolve(replies.map((reply) => reply.result.content[0]!.text));
      } catch (error) {
        reject(error);
      }
    });
    urls.forEach((url, i) => {
      const call = {
        jsonrpc: '2.0',
        id: i + 1,
        method: 'tools/call',
        params: { name: 'web_fetch', arguments: { url } },
      };
      child.stdin.write(`${JSON.stringify(call)}\n`);
    });
    child.stdin.end();
  });
}

function ledgerFromUserText(root: string): string {
  const ledger = new AddressLedger();
  ledger.addUserText(`Read the page ${LISTED} with web_fetch and tell me what it says.`);
  const path = join(root, 'ledger.json');
  writeLedgerFile(path, ledger.members());
  return path;
}

let open: LoopbackWeb | null = null;
afterEach(async () => {
  await open?.close();
  open = null;
});

describe('live smoke: the address-ledger setup', () => {
  it('without a provider (what the smoke had), both reads are not-configured: the ledger is never asked', async () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-smoke-ledger-'));
    const providers = new ProvidersService({
      paths: chatPaths(join(root, 'chat')),
      secrets: new MemorySecretStore(),
      fetch: () => Promise.reject(new Error('no network')),
      workPath: () => '',
    });
    const texts = await fetchThrough(
      { command: process.execPath, script: webServerScript() },
      providers.chatWeb(),
      ledgerFromUserText(root),
      [LISTED, UNLISTED],
    );
    expect(texts).toEqual([formatErrorText('not-configured', 'none'), formatErrorText('not-configured', 'none')]);
  });

  it('on the loopback provider, the user address reaches it and an invented one is blocked before it', async () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-smoke-ledger-'));
    open = await startLoopbackWeb(chatPaths(join(root, 'chat')), () => '');
    const web = open.providers.chatWeb();
    expect([web.fetchProvider, web.fetchKey.startsWith('loopback-')]).toEqual(['exa', true]);
    const [listed, unlisted] = await fetchThrough(open.webServer, web, ledgerFromUserText(root), [LISTED, UNLISTED]);
    expect(parseWebResultText(listed!)).toMatchObject({ kind: 'page', provider: 'exa', url: LISTED });
    expect(unlisted).toBe(formatErrorText('blocked', 'exa'));
    expect(open.fetched()).toEqual([LISTED]);
    expect(`${listed}${unlisted}`).not.toContain(web.fetchKey);
  });

  it('closing deletes the throwaway key and the selection', async () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-smoke-ledger-'));
    const web = await startLoopbackWeb(chatPaths(join(root, 'chat')), () => '');
    await web.close();
    expect(web.providers.chatWeb()).toEqual({
      searchProvider: 'none',
      searchKey: '',
      fetchProvider: 'none',
      fetchKey: '',
    });
  });
});
