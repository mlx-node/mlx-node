import { createHash } from 'node:crypto';
import { constants } from 'node:fs';
import { mkdir, open, realpath, type FileHandle } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';

import {
  DELEGATION_PROMPT,
  INSTALL_CHECK_GENERATION,
  delegationCommand,
  delegationPrompt,
  expandHome,
  localCompletion,
  parseLocalJson,
  preferredLocalModel,
} from '@mlx-node/agent/delegate';
import type { LocalInferenceConnection } from '@mlx-node/agent/delegate';

import { ApiError } from './api/errors.js';
import { CodingAgentCache, type DetectionResult } from './coding-agent-cache.js';
import {
  DETECTION_INPUT_PREFIX,
  DETECTION_SYSTEM,
  detectionMessage,
  detectionResult,
  COMMAND_SELECTION_SYSTEM,
  commandSelectionSource,
  commandSelectionResult,
} from './coding-agent-detection.js';

export type CodingAgentId = 'claude' | 'codex' | 'grok';
export type InstallStatus =
  | 'unchecked'
  | 'waiting'
  | 'checking'
  | 'installed'
  | 'not-installed'
  | 'needs-update'
  | 'installing'
  | 'error';
export interface CodingAgentRow {
  id: CodingAgentId;
  name: string;
  path: string;
  status: InstallStatus;
  detail: string | null;
  checkedAt: string | null;
}
export interface CodingAgentsState {
  command: string | null;
  model: string | null;
  available: boolean;
  unavailableReason: string | null;
  agents: CodingAgentRow[];
}
export interface CodingAgentsOptions {
  prepareCommand?(): Promise<string>;
  listModels(): Promise<string[]>;
  connect(): Promise<LocalInferenceConnection>;
  complete?: typeof localCompletion;
  home?: string;
  env?: NodeJS.ProcessEnv;
}

const MAX_FILE_BYTES = 48 * 1024;

function fingerprint(text: string): string {
  return createHash('sha256').update(text).digest('hex');
}

function detectionKey(model: string, path: string, text: string, command: string): string {
  return fingerprint(
    JSON.stringify([
      DETECTION_SYSTEM,
      DETECTION_INPUT_PREFIX,
      DELEGATION_PROMPT,
      INSTALL_CHECK_GENERATION,
      model,
      path,
      fingerprint(text),
      command,
    ]),
  );
}

async function readInstructions(file: FileHandle): Promise<string> {
  if (!(await file.stat()).isFile()) throw new Error('The instruction path must point to a regular file.');
  const buffer = Buffer.alloc(MAX_FILE_BYTES + 1);
  let length = 0;
  while (length < buffer.length) {
    const { bytesRead } = await file.read(buffer, length, buffer.length - length, null);
    if (!bytesRead) break;
    length += bytesRead;
  }
  if (length > MAX_FILE_BYTES)
    throw new Error('This instruction file is too large to check completely. Shorten it and try again.');
  return buffer.subarray(0, length).toString('utf8');
}

export class CodingAgentsService {
  private readonly home: string;
  private readonly env: NodeJS.ProcessEnv;
  private readonly complete: typeof localCompletion;
  private readonly rows = new Map<CodingAgentId, CodingAgentRow>();
  private readonly observed = new Map<CodingAgentId, string>();
  private readonly cache: CodingAgentCache;
  private readonly abort = new AbortController();
  private readonly work = new Set<Promise<void>>();
  private chain: Promise<void> = Promise.resolve();
  private model: string | null = null;
  private command: string | null = null;
  private unavailableReason: string | null = null;
  private initialized = false;
  private refreshing?: Promise<CodingAgentsState>;

  constructor(private readonly options: CodingAgentsOptions) {
    this.home = options.home ?? homedir();
    this.env = options.env ?? process.env;
    this.complete = options.complete ?? localCompletion;
    this.cache = new CodingAgentCache(join(this.home, '.mlx-node', 'coding-agents.json'));
    for (const [id, name] of [
      ['claude', 'Claude Code'],
      ['codex', 'Codex'],
      ['grok', 'Grok'],
    ] as const) {
      this.rows.set(id, { id, name, path: '', status: 'unchecked', detail: null, checkedAt: null });
    }
  }

  private async paths(): Promise<void> {
    const claude = join(expandHome(this.env.CLAUDE_CONFIG_DIR || join(this.home, '.claude'), this.home), 'CLAUDE.md');
    const codexDir = expandHome(this.env.CODEX_HOME || join(this.home, '.codex'), this.home);
    let codex = join(codexDir, 'AGENTS.md');
    try {
      if ((await this.read(join(codexDir, 'AGENTS.override.md'))).trim()) codex = join(codexDir, 'AGENTS.override.md');
    } catch (error) {
      // An unreadable override must produce a per-agent error, never an install into the shadowed base file.
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') codex = join(codexDir, 'AGENTS.override.md');
    }
    const grok = join(expandHome(this.env.GROK_HOME || join(this.home, '.grok'), this.home), 'AGENTS.md');
    for (const [id, path] of [
      ['claude', claude],
      ['codex', codex],
      ['grok', grok],
    ] as const) {
      const row = this.rows.get(id)!;
      if (row.path !== path) {
        row.path = path;
        if (!this.busy(row)) Object.assign(row, { status: 'unchecked', detail: null, checkedAt: null });
        this.observed.delete(id);
      }
    }
  }

  async state(): Promise<CodingAgentsState> {
    if (!this.initialized) return this.refresh();
    return this.snapshot();
  }

  /** Explicit metadata refresh. Ordinary job polling only reads the in-memory snapshot. */
  refresh(): Promise<CodingAgentsState> {
    if (this.refreshing) return this.refreshing;
    const task = this.refreshMetadata().finally(() => {
      this.refreshing = undefined;
    });
    this.refreshing = task;
    return task;
  }

  private async refreshMetadata(): Promise<CodingAgentsState> {
    await this.paths();
    await this.cache.load();
    const models = await this.options.listModels();
    const preferred = await preferredLocalModel(this.home, this.env);
    const selected = preferred ?? this.env.ANTHROPIC_MODEL ?? [...models].sort()[0];
    const model = selected && models.includes(selected) ? selected : null;
    let command: string | null = null;
    let commandError: string | null = null;
    if (model) {
      try {
        if (!this.options.prepareCommand) throw new Error('Open the updated mlx-node app to set up its command.');
        command = await this.options.prepareCommand();
        if (!command.startsWith('/')) throw new Error('The app command must use an absolute path.');
        delegationCommand(command);
      } catch (error) {
        command = null;
        commandError =
          error instanceof Error ? error.message : 'The app command is unavailable. Restart mlx-node and retry.';
      }
    }
    if (this.model !== model || this.command !== command) {
      this.model = model;
      this.command = command;
      this.observed.clear();
      for (const row of this.rows.values())
        if (!this.busy(row)) Object.assign(row, { status: 'unchecked', detail: null, checkedAt: null });
    }
    this.unavailableReason = model
      ? commandError
      : models.length === 0
        ? 'Install a local model first to check and set up coding agents.'
        : 'Your default local model is no longer installed. Download it again or choose an installed model in mlx agent.';
    for (const row of this.rows.values()) {
      if (!model || !command || this.busy(row)) continue;
      try {
        const text = await this.read(row.path);
        const key = detectionKey(model, row.path, text, command);
        if (this.observed.get(row.id) !== key) {
          Object.assign(row, { status: 'unchecked', detail: null, checkedAt: null });
          this.observed.set(row.id, key);
        }
        const cached = this.cache.get(key);
        if (!text.trim()) this.applyResult(row, { installed: false, checkedAt: new Date().toISOString() });
        else if (cached) this.applyResult(row, cached);
      } catch (error) {
        Object.assign(row, { status: 'error', detail: (error as Error).message, checkedAt: null });
        this.observed.delete(row.id);
      }
    }
    this.initialized = true;
    return this.snapshot();
  }

  private snapshot(): CodingAgentsState {
    return {
      model: this.model,
      command: this.command,
      available: this.model !== null && this.command !== null,
      unavailableReason: this.unavailableReason,
      agents: [...this.rows.values()].map((row) => ({ ...row })),
    };
  }

  private busy(row: CodingAgentRow): boolean {
    return row.status === 'waiting' || row.status === 'checking' || row.status === 'installing';
  }

  async start(action: 'detect' | 'install', id?: string, force = false): Promise<CodingAgentsState> {
    if (this.abort.signal.aborted) throw new ApiError('E_UNAVAILABLE', 'The control panel is closing.');
    const state = await this.refresh();
    if (!state.available) throw new ApiError('E_UNAVAILABLE', state.unavailableReason!);
    const targets = id === undefined ? [...this.rows.values()] : [this.rows.get(id as CodingAgentId)];
    if (targets.some((row) => !row)) throw ApiError.badRequest('Unknown coding agent.');
    for (const target of targets) {
      const row = target!;
      if (this.busy(row)) continue;
      if (action === 'detect' && !force && ['installed', 'not-installed', 'needs-update'].includes(row.status))
        continue;
      row.status = 'waiting';
      row.detail = null;
      const task = this.chain.then(async () => {
        try {
          this.abort.signal.throwIfAborted();
          row.status = action === 'install' ? 'installing' : 'checking';
          await this.perform(row, action, force);
        } catch (error) {
          row.status = 'error';
          row.detail = error instanceof Error ? error.message : 'Could not check this coding agent.';
        }
      });
      this.chain = task;
      this.work.add(task);
      void task.finally(() => this.work.delete(task));
    }
    return { ...state, agents: [...this.rows.values()].map((row) => ({ ...row })) };
  }

  private async read(path: string): Promise<string> {
    try {
      const file = await open(path, constants.O_RDONLY | constants.O_NONBLOCK);
      try {
        return await readInstructions(file);
      } finally {
        await file.close();
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return '';
      throw error;
    }
  }

  private async classify(
    connection: LocalInferenceConnection,
    text: string,
    command: string,
  ): Promise<Omit<DetectionResult, 'checkedAt'>> {
    const answer = parseLocalJson(
      await this.complete(
        connection,
        DETECTION_SYSTEM,
        [{ role: 'user', content: detectionMessage(text, command) }],
        this.abort.signal,
        INSTALL_CHECK_GENERATION.max_output_tokens,
      ),
    );
    return detectionResult(answer, text);
  }

  private async perform(row: CodingAgentRow, action: 'detect' | 'install', force: boolean): Promise<void> {
    this.abort.signal.throwIfAborted();
    const state = await this.refresh();
    if (!state.available || !state.model || !state.command) throw new Error(state.unavailableReason!);
    const path = row.path;
    const before = await this.read(path);
    this.abort.signal.throwIfAborted();
    let connection: LocalInferenceConnection | undefined;
    const classify = async (text: string): Promise<Omit<DetectionResult, 'checkedAt'>> => {
      if (!text.trim()) return { installed: false };
      connection ??= { ...(await this.options.connect()), model: state.model! };
      this.abort.signal.throwIfAborted();
      return this.classify(connection, text, state.command!);
    };
    const key = detectionKey(state.model, path, before, state.command);
    const cached = force ? undefined : this.cache.get(key);
    let result = cached ?? (await classify(before));
    let after = before;
    if (action === 'install' && !result.installed) {
      const prompt = delegationPrompt(state.command);
      const verify = async (): Promise<Omit<DetectionResult, 'checkedAt'>> => {
        if (Buffer.byteLength(after) > MAX_FILE_BYTES)
          throw new Error('There is not enough room to add and verify the prompt. Shorten the instruction file first.');
        return classify(after);
      };
      // Older cached verdicts do not have a source reference. Recheck only when
      // an update needs that reference; ordinary cached status checks stay free.
      if (result.needsUpdate && !result.source) result = await classify(before);
      const replacements: { start: number; end: number }[] = [];
      while (result.needsUpdate) {
        if (!result.source || replacements.length >= 64)
          throw new Error('The local model could not finish updating the commands. No changes were made.');
        connection ??= { ...(await this.options.connect()), model: state.model };
        const replacement = `${delegationCommand(state.command)} delegate github --caller-approved`;
        const answer = parseLocalJson(
          await this.complete(
            connection,
            COMMAND_SELECTION_SYSTEM,
            [
              {
                role: 'user',
                content: JSON.stringify({
                  currentCommand: replacement,
                  source: commandSelectionSource(after, result.source),
                }),
              },
            ],
            this.abort.signal,
            INSTALL_CHECK_GENERATION.max_output_tokens,
          ),
        );
        const span = commandSelectionResult(answer, after, result.source);
        // Never re-edit generated text. A contradictory verdict must leave the
        // original file untouched instead of causing a retry loop or duplication.
        if (
          after.slice(span.start, span.end) === replacement ||
          replacements.some((old) => span.start < old.end && span.end > old.start)
        )
          throw new Error('The local model could not verify the updated command. No changes were made.');
        after = after.slice(0, span.start) + replacement + after.slice(span.end);
        const delta = replacement.length - (span.end - span.start);
        for (const old of replacements) {
          if (old.start >= span.end) {
            old.start += delta;
            old.end += delta;
          }
        }
        replacements.push({ start: span.start, end: span.start + replacement.length });
        // Reclassify the complete candidate, including any remaining old routes.
        result = await verify();
      }
      if (!result.installed && replacements.length === 0) {
        after = `${before}${before && !before.endsWith('\n') ? '\n' : ''}${before ? '\n' : ''}${prompt}\n`;
        result = await verify();
      }
      // Every obsolete directive must be gone before touching user files.
      if (!result.installed)
        throw new Error('The local model could not verify the delegation prompt. No changes were made.');
      const fresh = await this.refresh();
      if (fresh.model !== state.model || fresh.command !== state.command || row.path !== path)
        throw new Error('The model or instruction location changed. Check again.');
      this.abort.signal.throwIfAborted();
      await mkdir(dirname(path), { recursive: true });
      // Follow existing symlinks and preserve the file inode, permissions and unrelated content.
      let target = path;
      try {
        target = await realpath(path);
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      }
      const file = await open(target, constants.O_RDWR | constants.O_CREAT | constants.O_NONBLOCK, 0o600);
      try {
        const current = await readInstructions(file);
        if (current !== before) throw new Error('The instruction file changed while checking it. Check again.');
        // Preserve the inode/symlink target while supporting an in-place upgrade.
        const bytes = Buffer.from(after);
        let written = 0;
        while (written < bytes.length) {
          const chunk = await file.write(bytes, written, bytes.length - written, written);
          if (chunk.bytesWritten === 0) throw new Error('Could not write the instruction file.');
          written += chunk.bytesWritten;
        }
        await file.truncate(bytes.length);
        await file.sync();
      } finally {
        await file.close();
      }
      if ((await this.read(path)) !== after)
        throw new Error('The instruction file changed during installation. Check again.');
      result = { installed: true };
    } else if ((await this.read(path)) !== before) {
      throw new Error('The instruction file changed while checking it. Check again.');
    }
    const current = await this.refresh();
    if (current.model !== state.model || current.command !== state.command || row.path !== path)
      throw new Error('The model or instruction location changed. Check again.');
    this.abort.signal.throwIfAborted();
    const detection = { ...result, checkedAt: new Date().toISOString() };
    const afterKey = detectionKey(state.model, path, after, state.command);
    await this.cache.set(afterKey, detection);
    this.applyResult(row, detection);
    this.observed.set(row.id, afterKey);
    // Custom config directories can point multiple agents at the same instruction file.
    for (const other of this.rows.values()) {
      if (other.id !== row.id && other.path === path && !this.busy(other)) {
        Object.assign(other, { status: row.status, detail: row.detail, checkedAt: row.checkedAt });
        this.observed.set(other.id, afterKey);
      }
    }
  }

  private applyResult(row: CodingAgentRow, result: DetectionResult): void {
    Object.assign(row, {
      status: result.installed ? 'installed' : result.needsUpdate ? 'needs-update' : 'not-installed',
      checkedAt: result.checkedAt,
      detail: result.needsUpdate
        ? 'Update these instructions to use the command included with the app.'
        : result.installed
          ? 'GitHub delegation is configured. Start a new agent session to load the instructions.'
          : null,
    });
  }

  async close(): Promise<void> {
    this.abort.abort();
    await Promise.allSettled(this.work);
  }
}
