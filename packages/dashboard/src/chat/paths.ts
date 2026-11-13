/**
 * The app data folder of Chat and Work (`~/.mlx-node/chat`). Every file the app writes here is metadata or a fixed
 * text; chat text is written only by the agent, into its own home `chat-agent/`.
 */

import { randomUUID } from 'node:crypto';
import { chmod, mkdir, readdir, rename, unlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import { mlxNodeHome } from '../paths.js';

export function defaultChatRoot(): string {
  return join(mlxNodeHome(), 'chat');
}

export interface ChatPaths {
  root: string;
  providers: string;
  conversations: string;
  /** `HOME` of the Chat agent: empty, never written by the app. */
  chatHome: string;
  /** cwd of the Chat agent process and of every Chat session: empty. */
  chatWorkspace: string;
  /** `GROK_HOME` of the Chat agent. */
  chatAgent: string;
  chatConfig: string;
  chatRequirements: string;
  chatSessions: string;
  plugins: string;
  /** The Work guard plugin folder (`workGuardManifest`), outside `plugins/` so a skill sync never removes it. */
  workGuard: string;
  /** `TMPDIR` of the Chat agent. */
  tmp: string;
  /** Work: the permission history files, one per conversation (metadata only). */
  permissionLog: string;
}

export function chatPaths(root: string = defaultChatRoot()): ChatPaths {
  const chatAgent = join(root, 'chat-agent');
  return {
    root,
    providers: join(root, 'providers.json'),
    conversations: join(root, 'conversations.json'),
    chatHome: join(root, 'chat-home'),
    chatWorkspace: join(root, 'chat-workspace'),
    chatAgent,
    chatConfig: join(chatAgent, 'config.toml'),
    chatRequirements: join(chatAgent, 'requirements.toml'),
    chatSessions: join(chatAgent, 'sessions'),
    plugins: join(root, 'plugins'),
    workGuard: join(root, 'work-guard'),
    tmp: join(root, 'tmp'),
    permissionLog: join(root, 'permission'),
  };
}

export function pluginDir(paths: ChatPaths, id: string): string {
  return join(paths.plugins, id);
}

/** Creates the app folders with mode 0700 (and repairs the mode of existing ones). */
export async function ensureChatDirs(paths: ChatPaths): Promise<void> {
  for (const dir of [
    paths.root,
    paths.chatHome,
    paths.chatWorkspace,
    paths.chatAgent,
    paths.chatSessions,
    paths.plugins,
    paths.tmp,
  ]) {
    await mkdir(dir, { recursive: true, mode: 0o700 });
    await chmod(dir, 0o700);
  }
}

/** Writes the Work guard plugin (`plugin.json` only) before every Work agent start; a failure stops the start. */
export async function writeWorkGuard(paths: ChatPaths, manifest: string): Promise<void> {
  for (const dir of [paths.root, paths.workGuard]) {
    await mkdir(dir, { recursive: true, mode: 0o700 });
    await chmod(dir, 0o700);
  }
  await writeFileAtomic(join(paths.workGuard, 'plugin.json'), manifest);
}

/** Temp file and rename in the same folder, mode 0600. */
export async function writeFileAtomic(path: string, text: string): Promise<void> {
  const temp = `${path}.${randomUUID()}.tmp`;
  try {
    await writeFile(temp, text, { mode: 0o600, flag: 'wx' });
    await rename(temp, path);
  } finally {
    await unlink(temp).catch(() => {});
  }
}

async function unlinkQuiet(path: string): Promise<void> {
  try {
    await unlink(path);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code !== 'ENOENT' && code !== 'ENOTDIR') throw error;
  }
}

/** The agent's full-text index of chat text, and its WAL files. */
export async function removeChatSearchIndex(paths: ChatPaths): Promise<void> {
  for (const suffix of ['', '-wal', '-shm'])
    await unlinkQuiet(join(paths.chatSessions, `session_search.sqlite${suffix}`));
}

/** Every `sessions/<cwd>/prompt_history.jsonl` of the Chat agent home: the agent keeps each typed prompt there. */
export async function removeChatPromptHistory(paths: ChatPaths): Promise<void> {
  let names: string[];
  try {
    names = await readdir(paths.chatSessions);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return;
    throw error;
  }
  for (const name of names) await unlinkQuiet(join(paths.chatSessions, name, 'prompt_history.jsonl'));
}
