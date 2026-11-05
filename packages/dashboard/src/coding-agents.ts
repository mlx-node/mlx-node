import { constants } from 'node:fs';
import { mkdir, open, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';

import { delegationCommand, delegationPrompt, expandHome, preferredLocalModel } from '@mlx-node/agent/delegate';

import { ApiError } from './api/errors.js';

export type CodingAgentId = 'claude' | 'codex' | 'grok';
export type InstallStatus = 'unchecked' | 'installed' | 'not-installed' | 'needs-update' | 'error';
export interface CodingAgentRow {
  id: CodingAgentId;
  name: string;
  path: string;
  status: InstallStatus;
  detail: string | null;
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
  home?: string;
  env?: NodeJS.ProcessEnv;
}

export const MANAGED_BEGIN = '<!-- portato managed instructions: begin -->';
export const MANAGED_END = '<!-- portato managed instructions: end -->';

export function managedBlock(command: string): string {
  return `${MANAGED_BEGIN}\n${delegationPrompt(command)}\n${MANAGED_END}`;
}

type Block = { start: number; end: number } | null | 'invalid';

/** One begin marker followed by one end marker. Anything else cannot be replaced safely. */
export function findManagedBlock(text: string): Block {
  const start = text.indexOf(MANAGED_BEGIN);
  const endMarker = text.indexOf(MANAGED_END);
  if (start === -1 && endMarker === -1) return null;
  if (
    start === -1 ||
    endMarker < start ||
    text.indexOf(MANAGED_BEGIN, start + 1) !== -1 ||
    text.indexOf(MANAGED_END, endMarker + 1) !== -1
  )
    return 'invalid';
  return { start, end: endMarker + MANAGED_END.length };
}

const INVALID_BLOCK =
  'The portato managed instructions markers in this file are incomplete or repeated. Fix them and check again.';

export class CodingAgentsService {
  private readonly home: string;
  private readonly env: NodeJS.ProcessEnv;
  private readonly rows = new Map<CodingAgentId, CodingAgentRow>();
  private chain: Promise<unknown> = Promise.resolve();
  private model: string | null = null;
  private command: string | null = null;
  private unavailableReason: string | null = null;
  private closed = false;

  constructor(private readonly options: CodingAgentsOptions) {
    this.home = options.home ?? homedir();
    this.env = options.env ?? process.env;
    for (const [id, name] of [
      ['claude', 'Claude Code'],
      ['codex', 'Codex'],
      ['grok', 'Grok'],
    ] as const) {
      this.rows.set(id, { id, name, path: '', status: 'unchecked', detail: null });
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
    this.rows.get('claude')!.path = claude;
    this.rows.get('codex')!.path = codex;
    this.rows.get('grok')!.path = grok;
  }

  state(): Promise<CodingAgentsState> {
    return this.refresh();
  }

  /** Reads every instruction file again. String matching is cheap, so there is no cache. */
  refresh(): Promise<CodingAgentsState> {
    return this.serial(() => this.scan());
  }

  install(id: string): Promise<CodingAgentsState> {
    return this.serial(async () => {
      const state = await this.scan();
      if (!state.available) throw new ApiError('E_UNAVAILABLE', state.unavailableReason!);
      const row = this.rows.get(id as CodingAgentId);
      if (!row) throw ApiError.badRequest('Unknown coding agent.');
      try {
        const before = await this.read(row.path);
        const block = findManagedBlock(before);
        if (block === 'invalid') throw new Error(INVALID_BLOCK);
        const managed = managedBlock(this.command!);
        const after = block
          ? before.slice(0, block.start) + managed + before.slice(block.end)
          : `${before}${before && !before.endsWith('\n') ? '\n' : ''}${before ? '\n' : ''}${managed}\n`;
        if (after !== before) {
          await mkdir(dirname(row.path), { recursive: true });
          // writeFile follows symlinks and keeps the inode and mode of an existing file.
          await writeFile(row.path, after, { mode: 0o600 });
        }
      } catch (error) {
        Object.assign(row, { status: 'error', detail: (error as Error).message });
        return this.snapshot();
      }
      return this.scan();
    });
  }

  private serial<T>(task: () => Promise<T>): Promise<T> {
    if (this.closed) return Promise.reject(new ApiError('E_UNAVAILABLE', 'The control panel is closing.'));
    const run = this.chain.then(task);
    this.chain = run.catch(() => {});
    return run;
  }

  private async scan(): Promise<CodingAgentsState> {
    await this.paths();
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
    this.model = model;
    this.command = command;
    this.unavailableReason = model
      ? commandError
      : models.length === 0
        ? 'Install a local model first to set up coding agents.'
        : 'Your default local model is no longer installed. Download it again or choose an installed model in mlx agent.';
    for (const row of this.rows.values()) {
      if (!command) {
        Object.assign(row, { status: 'unchecked', detail: null });
        continue;
      }
      try {
        const text = await this.read(row.path);
        const block = findManagedBlock(text);
        if (block === 'invalid') Object.assign(row, { status: 'error', detail: INVALID_BLOCK });
        else if (!block) Object.assign(row, { status: 'not-installed', detail: null });
        else if (text.slice(block.start, block.end) === managedBlock(command))
          Object.assign(row, {
            status: 'installed',
            detail: 'GitHub delegation is configured. Start a new agent session to load the instructions.',
          });
        else
          Object.assign(row, {
            status: 'needs-update',
            detail: 'Update these instructions to use the command included with the app.',
          });
      } catch (error) {
        Object.assign(row, { status: 'error', detail: (error as Error).message });
      }
    }
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

  private async read(path: string): Promise<string> {
    try {
      // O_NONBLOCK keeps a FIFO at the instruction path from hanging the control panel.
      const file = await open(path, constants.O_RDONLY | constants.O_NONBLOCK);
      try {
        if (!(await file.stat()).isFile()) throw new Error('The instruction path must point to a regular file.');
        return await file.readFile('utf8');
      } finally {
        await file.close();
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return '';
      throw error;
    }
  }

  async close(): Promise<void> {
    this.closed = true;
    await this.chain;
  }
}
