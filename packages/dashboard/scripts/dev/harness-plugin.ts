/**
 * Dev-server plugin: serves the real dashboard runtime to the browser tab over Vite's HMR websocket. The tab's
 * `bridge.ts` turns that socket into the MessagePort `main.tsx` expects from the desktop preload, so the SPA runs
 * unchanged. Dev only: nothing under `ui/` imports this, and the production build never sees it.
 */

import { randomBytes } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import type { Plugin } from 'vite-plus';

import { failure } from '../../src/api/errors.js';
import { MemorySecretStore } from '../../src/chat/providers/secrets.js';
import { PROVIDER_IDS } from '../../src/chat/providers/settings.js';
import type { ProviderId } from '../../src/chat/providers/types.js';
import { isLoopbackHttpOrigin } from '../../src/chat/web/http.js';
import { type RpcRuntime, serveRuntimeOverPort } from '../../src/rpc/host.js';
import type { RpcPort, RpcPortEvents } from '../../src/rpc/port.js';
import type { ChatRuntimeOptions } from '../../src/runtime.js';

export const DEV_OPEN = 'mlx:dev-open';
export const DEV_RPC = 'mlx:dev-rpc';
export const DEV_RPC_REPLY = 'mlx:dev-rpc-reply';
export const DEV_CLOSE = 'mlx:dev-close';
/** The bridge asks for a Work folder (the desktop's native dialog), and the plugin answers with the harness folder. */
export const DEV_PICK_FOLDER = 'mlx:dev-pick-folder';
export const DEV_FOLDER = 'mlx:dev-folder';
const CLIENT_DISCONNECT = 'vite:client:disconnect';

/** Absolute path of the browser half, served by Vite under `/@fs/`. */
export const BRIDGE_FILE = join(import.meta.dirname, 'bridge.ts');

/** The component gallery: an HTML page outside `ui/`, served at {@link GALLERY_PATH}. */
export const GALLERY_PATH = '/__gallery';
const GALLERY_DIR = join(import.meta.dirname, 'gallery');

/** `gallery.html` with its entry script pointed at the absolute `/@fs/` path of `gallery.tsx`. */
export function galleryHtml(): string {
  const html = readFileSync(join(GALLERY_DIR, 'gallery.html'), 'utf8');
  return html.replaceAll('src="./gallery.tsx"', `src="/@fs${join(GALLERY_DIR, 'gallery.tsx')}"`);
}

function isGalleryUrl(url: string | undefined): boolean {
  if (url === undefined) return false;
  const query = url.indexOf('?');
  const path = query === -1 ? url : url.slice(0, query);
  return path === GALLERY_PATH || path === `${GALLERY_PATH}/`;
}

/**
 * Routes that would reach the network. They answer E_UNAVAILABLE unless `MLX_DEV_NETWORK=1`. The provider routes are
 * not here: they reach only the fake providers on 127.0.0.1, and a key must be checked to be saved.
 */
const GUARDED: ReadonlyArray<{ method: string; path: string }> = [
  { method: 'POST', path: '/api/downloads' },
  { method: 'GET', path: '/api/catalog/updates' },
];

/** One connected browser tab, as the plugin sees the HMR socket's client. */
export interface HarnessClient {
  send(event: string, payload?: unknown): void;
}

/** The slice of Vite's `server.ws` the plugin uses. */
export interface HarnessChannel {
  on(event: string, listener: (data: unknown, client: HarnessClient) => void): void;
  off(event: string, listener: (data: unknown, client: HarnessClient) => void): void;
}

export interface HarnessOptions {
  /** Let the guarded routes through (a real download, the update check). Default: `MLX_DEV_NETWORK === '1'`. */
  allowNetwork?: boolean;
  /** The Chat options the served runtime was built with; checked by {@link assertHarnessChat}. */
  chat: ChatRuntimeOptions;
  /** What the folder dialog answers: a throwaway folder in the fixture home. Null answers "nothing chosen". */
  workFolder?: string | null;
}

/** The scripted stand-in for `mlx-agent`; Chat in the harness runs only on it. */
export const FAKE_AGENT_FILE = resolve(import.meta.dirname, '../../__test__/helpers/fake-agent.mjs');
/**
 * The fake agent's script in the harness, picked by a word of the prompt. Chat: "search", "read <address>", "shell",
 * "error", "long", "slow", "agent-stop" (the agent ends the turn as cancelled on its own), "untitled" (the agent writes no
 * title); any other prompt gets a Markdown answer. Work: "run" (a command with an always option),
 * "write" (a file edit), "edit" (a read, then an edit with the edits-for-this-conversation option), "look" (a listing,
 * a search and a read), "risky" (a flagged command), "plan" (a plan to approve), "key" (a command that prints the
 * selected provider key), "auto" (three known commands, an edit, then a command that is not on Auto's list),
 * "everything" (a printf, an edit outside the folder, an MCP tool and a helper's request with no tool call); any other
 * Work prompt lists the folder.
 */
export const HARNESS_CHAT_SCENARIO = resolve(
  import.meta.dirname,
  '../../__test__/fixtures/chat-scenarios/chat-harness.json',
);

/** A 60-character model name in the harness's endpoint and the agent's model list. */
export const HARNESS_LONG_MODEL = 'qwen3.6-35b-a3b-agentworld-instruct-ud-q4-k-xl-mlx-dwq-8bits';

/**
 * Chat for the harness: the fake agent, keys in memory, the fake providers (`providerBaseUrls`, all loopback), an app
 * folder inside the fixture home, and an endpoint that names two models and a throwaway token. `engineUrl` is the
 * harness's fake engine on loopback (`startHarnessEngine`): the app selects the model there before each prompt.
 */
export function harnessChatOptions(
  home: string,
  providerBaseUrls: Record<ProviderId, string>,
  engineUrl = 'http://127.0.0.1:9',
): ChatRuntimeOptions {
  const token = randomBytes(24).toString('hex');
  return {
    root: join(home, 'chat'),
    homeDir: home,
    baseEnv: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: home, LANG: 'en_US.UTF-8' },
    agentCommand: { command: process.execPath, args: [FAKE_AGENT_FILE, HARNESS_CHAT_SCENARIO] },
    secrets: new MemorySecretStore(),
    providerBaseUrls,
    readEndpoint: () =>
      Promise.resolve({
        version: 1,
        pid: process.pid,
        url: engineUrl,
        token,
        models: [
          { name: 'qwen3.5-4b', path: join(home, 'models', 'qwen3.5-4b') },
          { name: 'qwen3.5-9b', path: join(home, 'models', 'qwen3.5-9b') },
          { name: HARNESS_LONG_MODEL, path: join(home, 'models', HARNESS_LONG_MODEL) },
        ],
      }),
  };
}

/**
 * The start refusals: Chat create and prompt spawn processes and the provider routes call provider APIs, which no route
 * guard would stop, so the harness runs only on the fake agent, with keys in memory and the providers on loopback.
 */
export function assertHarnessChat(chat: ChatRuntimeOptions | undefined): void {
  if (chat === undefined) throw new Error('dev harness: Chat options are required');
  const command = chat.agentCommand;
  if (command === undefined || command.args.length === 0 || resolve(command.args[0]!) !== FAKE_AGENT_FILE) {
    throw new Error('dev harness: Chat must run on the fake agent');
  }
  if (!(chat.secrets instanceof MemorySecretStore)) throw new Error('dev harness: keys must be kept in memory');
  const home = chat.baseEnv?.HOME;
  if (chat.homeDir === undefined || home === undefined || resolve(home) !== resolve(chat.homeDir)) {
    throw new Error('dev harness: Work must run in the fixture home');
  }
  const urls = chat.providerBaseUrls;
  if (urls === undefined || !PROVIDER_IDS.every((id) => urls[id] !== undefined && isLoopbackHttpOrigin(urls[id]))) {
    throw new Error('dev harness: every provider base URL must be a loopback http origin');
  }
}

/** `runtime` with the network routes answered locally unless `allowNetwork`. */
export function guardRuntime(runtime: RpcRuntime, allowNetwork: boolean): RpcRuntime {
  const guarded: RpcRuntime = {
    call(call, signal) {
      if (!allowNetwork) {
        const query = call.path.indexOf('?');
        const path = query === -1 ? call.path : call.path.slice(0, query);
        const method = call.method.toUpperCase();
        if (GUARDED.some((route) => route.method === method && route.path === path)) {
          return Promise.resolve(failure('E_UNAVAILABLE', 'Disabled in the dev harness'));
        }
      }
      return runtime.call(call, signal);
    },
    subscribe: (jobId, listener) => runtime.subscribe(jobId, listener),
    watchLive: (listener) => runtime.watchLive(listener),
    ...(runtime.watchChat === undefined ? {} : { watchChat: runtime.watchChat.bind(runtime) }),
  };
  return guarded;
}

interface Channel {
  client: HarnessClient;
  events: RpcPortEvents | null;
  dispose: () => void;
  closed: boolean;
}

function channelOf(data: unknown): string | null {
  if (typeof data !== 'object' || data === null) return null;
  const channel = (data as { channel?: unknown }).channel;
  return typeof channel === 'string' && channel.length > 0 ? channel : null;
}

/**
 * Serve `runtime` to every tab that opens a channel on `ws`. Returns the detach, which also closes every channel.
 * Messages cross as JSON: RPC requests, replies and download / live events are plain data.
 */
export function attachHarness(ws: HarnessChannel, runtime: RpcRuntime, workFolder: string | null = null): () => void {
  const channels = new Map<string, Channel>();

  const close = (id: string): void => {
    const channel = channels.get(id);
    if (channel === undefined) return;
    channels.delete(id);
    channel.closed = true;
    channel.events?.onClose();
    channel.dispose();
  };

  const onOpen = (data: unknown, client: HarnessClient): void => {
    const id = channelOf(data);
    if (id === null || channels.has(id)) return;
    const channel: Channel = { client, events: null, dispose: () => {}, closed: false };
    const port: RpcPort = {
      postMessage(message: unknown): void {
        if (!channel.closed) client.send(DEV_RPC_REPLY, { channel: id, message });
      },
      listen(events: RpcPortEvents): () => void {
        channel.events = events;
        return () => {
          if (channel.events === events) channel.events = null;
        };
      },
      close(): void {
        channel.closed = true;
        channels.delete(id);
      },
    };
    channels.set(id, channel);
    channel.dispose = serveRuntimeOverPort(runtime, port);
  };

  const onRpc = (data: unknown): void => {
    const id = channelOf(data);
    if (id === null) return;
    channels.get(id)?.events?.onMessage((data as { message?: unknown }).message);
  };

  const onClose = (data: unknown): void => {
    const id = channelOf(data);
    if (id !== null) close(id);
  };

  const onPickFolder = (data: unknown, client: HarnessClient): void => {
    const requestId = typeof data === 'object' && data !== null ? (data as { requestId?: unknown }).requestId : null;
    if (typeof requestId !== 'string' || requestId === '' || requestId.length > 64) return;
    client.send(DEV_FOLDER, { requestId, path: workFolder });
  };

  const onDisconnect = (_data: unknown, client: HarnessClient): void => {
    for (const [id, channel] of channels) if (channel.client === client) close(id);
  };

  ws.on(DEV_OPEN, onOpen);
  ws.on(DEV_RPC, onRpc);
  ws.on(DEV_CLOSE, onClose);
  ws.on(DEV_PICK_FOLDER, onPickFolder);
  ws.on(CLIENT_DISCONNECT, onDisconnect);
  return () => {
    ws.off(DEV_OPEN, onOpen);
    ws.off(DEV_RPC, onRpc);
    ws.off(DEV_CLOSE, onClose);
    ws.off(DEV_PICK_FOLDER, onPickFolder);
    ws.off(CLIENT_DISCONNECT, onDisconnect);
    for (const id of channels.keys()) close(id);
  };
}

/** The Vite plugin: injects the bridge after the entry script and serves `runtime` over the HMR socket. */
export function devHarness(runtime: RpcRuntime, options: HarnessOptions): Plugin {
  assertHarnessChat(options.chat);
  const allowNetwork = options.allowNetwork ?? process.env.MLX_DEV_NETWORK === '1';
  const served = guardRuntime(runtime, allowNetwork);
  return {
    name: 'mlx-node-dev-harness',
    apply: 'serve',
    transformIndexHtml: {
      order: 'post',
      handler: (_html, context) =>
        isGalleryUrl(context.originalUrl ?? context.path)
          ? []
          : [{ tag: 'script', attrs: { type: 'module', src: `/@fs${BRIDGE_FILE}` }, injectTo: 'body' }],
    },
    configureServer(server) {
      const detach = attachHarness(server.ws as unknown as HarnessChannel, served, options.workFolder ?? null);
      server.httpServer?.once('close', detach);
      server.middlewares.use((request, response, next) => {
        if (request.method !== 'GET' || !isGalleryUrl(request.url)) {
          next();
          return;
        }
        server.transformIndexHtml(GALLERY_PATH, galleryHtml(), request.originalUrl).then(
          (html) => {
            response.statusCode = 200;
            response.setHeader('Content-Type', 'text/html; charset=utf-8');
            response.end(html);
          },
          (err: unknown) => next(err),
        );
      });
    },
  };
}
