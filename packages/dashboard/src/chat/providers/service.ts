/**
 * The web search providers: the selection (exactly one search provider or none, Firecrawl independent, in Firecrawl
 * Cloud or on this Mac), the keys, their checks, and the skill folders that follow the selection. Keys live in the
 * Keychain; this service keeps a cache filled once and refreshed only by the key routes, so no create or load path
 * waits on the Keychain past {@link KEYCHAIN_CALL_MS}. A replaced or removed key stays in memory as a redaction needle
 * until the process exits. Changes run one at a time; after each one the skills are synced and the agent hosts are
 * told to restart when idle. The check of Firecrawl on this Mac is a live fact in memory: no read waits for it, and
 * its result never restarts a host (the mlx_web server checks the server again before each page read).
 */

import { join } from 'node:path';

import { ApiError } from '../../api/errors.js';
import { PROVIDER_KEY_ENV } from '../agent/env.js';
import type { ChatWebEnv } from '../agent/profiles.js';
import type { ChatPaths } from '../paths.js';
import { findOnPath } from '../skills/cli-check.js';
import { SkillsInstaller, type SyncResult } from '../skills/install.js';
import { CHAT_SKILL_BUNDLE, defaultSkillsDir, loadManifest, type SkillsManifest } from '../skills/manifest.js';
import type { ChatHostState } from '../types.js';
import { PROVIDER_ORIGINS } from '../web/http.js';
import { checkLocal, type LocalCheckResult, parseLocalAddress } from '../web/local.js';
import { checkKeyFormat, type SecretStore } from './secrets.js';
import {
  defaultSettings,
  PROVIDER_IDS,
  type ProviderSettings,
  readSettings,
  repairSettings,
  writeSettings,
} from './settings.js';
import type { ProviderSource } from './source.js';
import type {
  KeyCheck,
  LocalFirecrawl,
  ProviderCheckReply,
  ProviderFacts,
  ProviderId,
  ProviderKeyReply,
  ProviderLocalCheckReply,
  ProviderLocalReply,
  ProviderSelectionInput,
  ProvidersResponse,
  ProviderState,
  SearchProviderId,
} from './types.js';
import { validateKey } from './validate.js';

/** Every Keychain call races this. */
export const KEYCHAIN_CALL_MS = 8_000;
/** A providers read asks the shell again at most this often: each ask runs the user's shell start files. */
export const WORK_PATH_ASK_GAP_MS = 15_000;
/** A providers read starts a check of Firecrawl on this Mac at most this often. */
export const LOCAL_CHECK_GAP_MS = 15_000;

const PROVIDER_NAMES: Readonly<Record<ProviderId, string>> = {
  exa: 'Exa',
  deepseek: 'DeepSeek',
  brave: 'Brave',
  firecrawl: 'Firecrawl',
};

const KEYCHAIN_MESSAGE = 'The Keychain cannot be reached.';
const SETTINGS_UNSAVED_MESSAGE = 'The key changed, but the settings could not be saved.';

function keychainError(): ApiError {
  return new ApiError('E_UNAVAILABLE', KEYCHAIN_MESSAGE, 'keychain-unavailable');
}

function localNotSetUp(): ApiError {
  return new ApiError('E_CONFLICT', 'Set up Firecrawl on this Mac first.', 'local-not-set-up');
}

export interface ProvidersServiceOptions {
  paths: ChatPaths;
  secrets: SecretStore;
  skillsDir?: string;
  /** Provider origins for key checks. Production leaves it unset (the fixed provider hosts). */
  baseUrls?: Partial<Record<ProviderId, string>>;
  fetch?: typeof fetch;
  /**
   * The `PATH` a Work command gets: where the `firecrawl` CLI is looked for. Asked only while Firecrawl is on: once
   * at start, then by `GET /api/providers` and by a route that turns Firecrawl on, while the CLI is not found.
   */
  workPath: () => string | undefined | Promise<string | undefined>;
  now?: () => number;
  keychainCallMs?: number;
  validateTimeoutMs?: number;
}

/** What the Chat service hears from a change, and what it tells back. */
export interface ProviderHooks {
  /** The selection, a key or a skill folder changed: both agent hosts restart when idle. */
  changed(): void;
  /** A key became known that was not a redaction needle yet: what the pages were sent may hold it. */
  keysChanged(): void;
  /** A host still runs on the old selection with a turn in flight. */
  applyPending(): boolean;
}

type Raced<T> = { kind: 'value'; value: T } | { kind: 'timeout' } | { kind: 'error' };

/** A check of Firecrawl on this Mac and its place in the order the checks started. */
interface LocalChecked {
  result: LocalCheckResult;
  seq: number;
}

export class ProvidersService implements ProviderSource {
  private readonly opts: ProvidersServiceOptions;
  private readonly now: () => number;
  private readonly installer: SkillsInstaller;
  private readonly skillsDir: string;
  private settings: ProviderSettings = defaultSettings();
  /** A provider is in the map once the Keychain answered for it: its key, or null for none. */
  private readonly keys = new Map<ProviderId, string | null>();
  /** Bumped on every cache write of a provider's key: a check or a late read that started before it is dropped. */
  private readonly generations = new Map<ProviderId, number>();
  /** Replaced and removed keys, with their provider: redaction needles until the Control Panel exits. */
  private readonly retired = new Map<string, ProviderId>();
  private readonly facts = new Map<ProviderId, ProviderFacts>();
  private readonly installed = new Map<string, boolean>();
  private keychainDown = false;
  /** The selected provider whose key was not in the Keychain at start; told to the page once. */
  private vanished: ProviderId | null = null;
  private cliFound = false;
  /** The last answer of `workPath`; before the first, the CLI counts as not found. */
  private workPath: string | undefined;
  private pathAsked: Promise<void> | null = null;
  private everAsked = false;
  private pathAskedAt = Number.NEGATIVE_INFINITY;
  /** Providers whose Keychain write timed out and is still pending: a newer write could land before it. */
  private readonly writing = new Set<ProviderId>();
  /** The last check of the saved address that ended; a check that started earlier never replaces it. */
  private localCheck: { url: string; result: LocalCheckResult; at: number; seq: number } | null = null;
  private localRunning: { url: string; done: Promise<LocalChecked> } | null = null;
  private localSeq = 0;
  private localStartedAt = Number.NEGATIVE_INFINITY;
  private chain: Promise<void> = Promise.resolve();
  private started: Promise<void> | null = null;
  private hooks: ProviderHooks = { changed: () => {}, keysChanged: () => {}, applyPending: () => false };

  constructor(options: ProvidersServiceOptions) {
    this.opts = options;
    this.now = options.now ?? Date.now;
    this.skillsDir = options.skillsDir ?? defaultSkillsDir();
    this.installer = new SkillsInstaller({
      skillsDir: this.skillsDir,
      pluginsDir: options.paths.plugins,
      now: this.now,
    });
  }

  attach(hooks: ProviderHooks): void {
    this.hooks = hooks;
  }

  // ── what the hosts read (memory only) ─────────────────────────────────────────────────────────────────────────

  chatWeb(): ChatWebEnv {
    const search = this.selectedSearch();
    const fetchProvider = this.fetchProvider();
    const local = this.localInUse() ? this.settings.localUrl : null;
    return {
      searchProvider: search ?? 'none',
      searchKey: search === null ? '' : (this.keys.get(search) ?? ''),
      fetchProvider: fetchProvider ?? 'none',
      fetchKey: fetchProvider === null || local !== null ? '' : (this.keys.get(fetchProvider) ?? ''),
      fetchUrl: local ?? '',
    };
  }

  workKeys(): Record<string, string> {
    const out: Record<string, string> = {};
    const add = (id: ProviderId): void => {
      const key = this.keys.get(id);
      if (typeof key === 'string') for (const name of PROVIDER_KEY_ENV[id]) out[name] = key;
    };
    const search = this.selectedSearch();
    if (search !== null) add(search);
    if (this.firecrawlInWork() && !this.localInUse()) add('firecrawl');
    return out;
  }

  workFirecrawlUrl(): string | null {
    return this.firecrawlInWork() && this.localInUse() ? this.settings.localUrl : null;
  }

  workPluginDirs(): string[] {
    const dirs: string[] = [];
    const search = this.selectedSearch();
    if (search !== null && this.installed.get(search) === true) dirs.push(join(this.opts.paths.plugins, search));
    if (this.firecrawlInWork()) dirs.push(join(this.opts.paths.plugins, 'firecrawl'));
    return dirs;
  }

  secretNeedles(): string[] {
    const needles = new Set(this.retired.keys());
    for (const key of this.keys.values()) if (typeof key === 'string') needles.add(key);
    return [...needles];
  }

  keyNeedles(): Array<{ provider: ProviderId; key: string }> {
    const out: Array<{ provider: ProviderId; key: string }> = [];
    for (const [id, key] of this.keys) if (typeof key === 'string') out.push({ provider: id, key });
    for (const [key, provider] of this.retired) if (!out.some((n) => n.key === key)) out.push({ provider, key });
    return out;
  }

  markKeyCheck(provider: ProviderId, check: 'rejected' | 'no-credit', key: string): void {
    void this.exclusive(async () => {
      if (key === '' || this.keys.get(provider) !== key) return;
      this.settings.checks[provider] = { status: check, at: this.now() };
      await writeSettings(this.opts.paths.providers, this.settings);
    }).catch(() => {});
  }

  /** Never waits for the shell: a Work host that starts before the answer is restarted by it. */
  async settled(): Promise<void> {
    await this.start();
    await this.chain;
    // Firecrawl may have come on since start (a late Keychain answer).
    if (!this.everAsked && this.firecrawlOn()) void this.askPath();
    await this.refreshCli();
  }

  async chatSkillReady(): Promise<boolean> {
    await this.start();
    await this.chain;
    if (this.installed.get(CHAT_SKILL_BUNDLE) !== true) await this.exclusive(() => this.sync()).catch(() => {});
    return this.installed.get(CHAT_SKILL_BUNDLE) === true;
  }

  summary(): ChatHostState['providers'] {
    const search = this.selectedSearch();
    return {
      search,
      searchCheck: search === null ? null : (this.settings.checks[search]?.status ?? null),
      firecrawl: this.firecrawlOn(),
      firecrawlLocal: this.localInUse(),
      fetch: this.fetchProvider(),
      applyPending: this.hooks.applyPending(),
    };
  }

  // ── routes ────────────────────────────────────────────────────────────────────────────────────────────────────

  /**
   * `GET /api/providers`: re-verifies the installed skill folders. While the Keychain is marked down, every provider is
   * read again (each read bounded); keys the cache did not have yet are taken, and the mark clears when all answer.
   * With an address saved it answers with the last check of Firecrawl on this Mac and starts the next one in the
   * background, at most once per {@link LOCAL_CHECK_GAP_MS}: the reply never waits for it.
   */
  async list(): Promise<ProvidersResponse> {
    await this.start();
    this.checkLocalInBackground();
    const reread = this.keychainDown ? await this.rereadAll() : null;
    const due = this.pathAsked !== null || this.now() - this.pathAskedAt >= WORK_PATH_ASK_GAP_MS;
    if (this.firecrawlOn() && !this.cliFound && due) await this.askPath();
    return this.exclusive(async () => {
      if (reread !== null) await this.applyReread(reread);
      await this.verifyInstalled();
      await this.refreshCli();
      const reply = this.snapshot();
      this.vanished = null;
      return reply;
    });
  }

  /** `PUT /api/providers/:id/key`: checked first; a rejected key is not saved and any earlier key stays. */
  async putKey(id: ProviderId, raw: string): Promise<ProviderKeyReply> {
    const key = checkKeyFormat(raw);
    if (key === null) throw ApiError.badRequest('That does not look like an API key.');
    await this.start();
    const result = await validateKey(id, key, this.validateDeps(id));
    if (result.check === 'rejected') {
      return { saved: false, check: 'rejected', providers: await this.exclusive(async () => this.snapshot()) };
    }
    const reply = await this.exclusive(async (): Promise<ProviderKeyReply> => {
      if (this.writing.has(id)) throw keychainError();
      await this.keychainWrite(
        id,
        () => this.opts.secrets.set(id, key),
        () => this.replaceKey(id, key),
      );
      this.replaceKey(id, key);
      this.recordCheck(id, result.check, result.facts);
      const prior = {
        search: this.settings.search,
        firecrawl: this.settings.firecrawl,
        firecrawlLocal: this.settings.firecrawlLocal,
      };
      // A Firecrawl key turns on Firecrawl Cloud only from Off: while Firecrawl on this Mac is in use, page addresses
      // keep going there until the user picks the cloud.
      if (id === 'firecrawl' && !this.settings.firecrawl) {
        this.settings.firecrawl = true;
        this.settings.firecrawlLocal = false;
      } else if (id !== 'firecrawl' && this.settings.search === null) this.settings.search = id;
      const enabled = this.settings.search !== prior.search || this.settings.firecrawl !== prior.firecrawl;
      await this.afterKeyChange(async () => {
        const synced = await this.sync().catch(() => null);
        // As on the select route, a provider is not turned on when its skill folder does not install.
        if (!enabled || synced?.[id] === 'ok') return;
        Object.assign(this.settings, prior);
        await this.sync().catch(() => {});
      });
      return { saved: true, check: result.check, providers: this.snapshot() };
    });
    return id === 'firecrawl' ? { ...reply, providers: await this.withWorkPath(reply.providers) } : reply;
  }

  /** `DELETE /api/providers/:id/key`: the selection that used the key is cleared and its skills are removed. */
  async deleteKey(id: ProviderId): Promise<ProvidersResponse> {
    await this.start();
    return this.exclusive(async () => {
      if (this.writing.has(id)) throw keychainError();
      await this.keychainWrite(
        id,
        () => this.opts.secrets.delete(id),
        () => this.replaceKey(id, null),
      );
      this.replaceKey(id, null);
      this.recordCheck(id, null, null);
      if (this.settings.search === id) this.settings.search = null;
      // Firecrawl on this Mac needs no key: only the cloud turns off.
      if (id === 'firecrawl' && !this.settings.firecrawlLocal) this.settings.firecrawl = false;
      await this.afterKeyChange(async () => {
        await this.sync().catch(() => {});
      });
      return this.snapshot();
    });
  }

  /**
   * `POST /api/providers/:id/validate`: re-reads the key from the Keychain, then checks it. A key route that changed
   * the key meanwhile wins: the result is dropped and the current state answers.
   */
  async validate(id: ProviderId): Promise<ProviderCheckReply> {
    await this.start();
    const generation = this.generation(id);
    const pending = (async () => this.opts.secrets.get(id))();
    const read = await this.race(() => pending);
    if (read.kind !== 'value') {
      this.keychainDown = true;
      if (read.kind === 'timeout') {
        pending.then(
          (value) => void this.exclusive(async () => this.applyRead(id, value, generation)).catch(() => {}),
          () => {},
        );
      }
      throw keychainError();
    }
    const key = read.value;
    const result = key === null ? null : await validateKey(id, key, this.validateDeps(id));
    return this.exclusive(async () => {
      if (this.generation(id) !== generation) {
        const current = this.configured(id) ? (this.settings.checks[id]?.status ?? null) : null;
        return { check: current, providers: this.snapshot() };
      }
      this.keychainDown = this.unknownKeys();
      const changed = this.keys.get(id) !== key;
      this.replaceKey(id, key);
      this.recordCheck(id, result?.check ?? null, result?.facts ?? null);
      repairSettings(this.settings, (p) => this.knownConfigured(p));
      try {
        await this.persistAndSync();
      } finally {
        if (changed) this.hooks.changed();
      }
      return { check: result?.check ?? null, providers: this.snapshot() };
    });
  }

  /** `PUT /api/providers/selection`: nothing changes unless every wanted skill folder installs. */
  async select(input: ProviderSelectionInput): Promise<ProvidersResponse> {
    await this.start();
    const reply = await this.exclusive(async () => {
      const previous = { ...this.settings, checks: { ...this.settings.checks } };
      const next = { ...previous, checks: { ...previous.checks } };
      if (input.search !== undefined) {
        if (input.search !== null && !this.configured(input.search)) throw this.needsKey(input.search);
        next.search = input.search;
      }
      if (input.firecrawl !== undefined) next.firecrawl = input.firecrawl;
      if (input.firecrawlLocal !== undefined) {
        if (input.firecrawlLocal && next.localUrl === null) throw localNotSetUp();
        next.firecrawlLocal = input.firecrawlLocal;
      }
      const crawlerPicked = input.firecrawl !== undefined || input.firecrawlLocal !== undefined;
      if (crawlerPicked && next.firecrawl && !next.firecrawlLocal && !this.configured('firecrawl')) {
        throw this.needsKey('firecrawl');
      }
      const changed =
        next.search !== previous.search ||
        next.firecrawl !== previous.firecrawl ||
        next.firecrawlLocal !== previous.firecrawlLocal;
      this.settings = next;
      let failed: 'skill-bundle-corrupt' | 'install-failed' | null;
      try {
        const result = await this.sync();
        const wanted = this.wantedProviders().map((p) => result[p]);
        failed = wanted.includes('skill-bundle-corrupt')
          ? 'skill-bundle-corrupt'
          : wanted.every((r) => r === 'ok')
            ? null
            : 'install-failed';
      } catch {
        failed = 'install-failed';
      }
      if (failed !== null) {
        this.settings = previous;
        await this.sync().catch(() => {});
        if (failed === 'skill-bundle-corrupt') {
          throw new ApiError(
            'E_UNAVAILABLE',
            "The app's skill files are damaged. Reinstall the app.",
            'skill-bundle-corrupt',
          );
        }
        throw new ApiError('E_UNAVAILABLE', 'Skills could not be installed.');
      }
      try {
        await writeSettings(this.opts.paths.providers, this.settings);
      } catch (error) {
        this.settings = previous;
        await this.sync().catch(() => {});
        throw error;
      }
      if (changed) this.hooks.changed();
      return this.snapshot();
    });
    return input.firecrawl === true || input.firecrawlLocal !== undefined ? this.withWorkPath(reply) : reply;
  }

  /**
   * `PUT /api/providers/local`: the address is checked first (see `checkLocal`); one whose check is not `ok` is not
   * saved, and the earlier address and the selection stay. A saved address turns on Firecrawl on this Mac only from
   * Off: while Firecrawl Cloud is in use, it stays in use until the user picks this Mac.
   */
  async putLocal(raw: string): Promise<ProviderLocalReply> {
    const url = parseLocalAddress(raw);
    if (url === null) throw ApiError.badRequest('Use an address on this Mac.');
    await this.start();
    const checked = await this.runLocalCheck(url);
    const { check } = checked.result;
    if (check !== 'ok') {
      return { saved: false, check, providers: await this.exclusive(async () => this.snapshot()) };
    }
    const reply = await this.exclusive(async (): Promise<ProviderLocalReply> => {
      const previous = { ...this.settings, checks: { ...this.settings.checks } };
      this.settings.localUrl = url;
      this.keepLocalCheck(url, checked);
      if (!this.settings.firecrawl) {
        this.settings.firecrawl = true;
        this.settings.firecrawlLocal = true;
      }
      try {
        const synced = await this.sync().catch(() => null);
        // As on the select route, Firecrawl is not turned on when its skill folder does not install.
        if (this.settings.firecrawl !== previous.firecrawl && synced?.firecrawl !== 'ok') {
          this.settings.firecrawl = previous.firecrawl;
          this.settings.firecrawlLocal = previous.firecrawlLocal;
          await this.sync().catch(() => {});
        }
        await this.saveOrRestore(previous);
      } finally {
        const s = this.settings;
        if (s.localUrl !== previous.localUrl || s.firecrawl !== previous.firecrawl) this.hooks.changed();
      }
      return { saved: true, check, providers: this.snapshot() };
    });
    return { ...reply, providers: await this.withWorkPath(reply.providers) };
  }

  /** `POST /api/providers/local/check`: checks the saved address now, past the gap of a providers read. */
  async checkLocalNow(): Promise<ProviderLocalCheckReply> {
    await this.start();
    const url = this.settings.localUrl;
    const checked = url === null ? null : await this.runLocalCheck(url);
    return { check: checked?.result.check ?? null, providers: await this.exclusive(async () => this.snapshot()) };
  }

  /**
   * `DELETE /api/providers/local`: the address is forgotten. When it was in use the crawler turns off; it never moves
   * to the cloud.
   */
  async deleteLocal(): Promise<ProvidersResponse> {
    await this.start();
    return this.exclusive(async () => {
      const previous = { ...this.settings, checks: { ...this.settings.checks } };
      const inUse = this.settings.firecrawl && this.settings.firecrawlLocal;
      this.settings.localUrl = null;
      this.settings.firecrawlLocal = false;
      if (inUse) this.settings.firecrawl = false;
      this.localCheck = null;
      try {
        await this.sync().catch(() => {});
        await this.saveOrRestore(previous);
      } finally {
        if (inUse) this.hooks.changed();
      }
      return this.snapshot();
    });
  }

  // ── internals ─────────────────────────────────────────────────────────────────────────────────────────────────

  /** First use: settings, the key cache (every read bounded), the invariants, the skill folders. */
  private start(): Promise<void> {
    this.started ??= this.exclusive(async () => {
      this.settings = await readSettings(this.opts.paths.providers);
      await Promise.all(PROVIDER_IDS.map((id) => this.fill(id)));
      const wanted = this.wantedProviders();
      const repaired = repairSettings(this.settings, (p) => this.knownConfigured(p));
      this.vanished = wanted.find((p) => !this.wantedProviders().includes(p)) ?? null;
      if (repaired) await writeSettings(this.opts.paths.providers, this.settings).catch(() => {});
      await this.sync().catch(() => {});
      await this.refreshCli();
      // Not awaited: its answer queues behind this step.
      if (this.firecrawlOn()) void this.askPath();
      this.checkLocalInBackground();
    });
    return this.started;
  }

  /** Starts a check of the saved address unless one runs or one started less than the gap ago. Never awaited. */
  private checkLocalInBackground(): void {
    const url = this.settings.localUrl;
    if (url === null || this.localRunning !== null || this.now() - this.localStartedAt < LOCAL_CHECK_GAP_MS) return;
    void this.runLocalCheck(url);
  }

  /** A check of `url`; a caller while one of the same address runs joins it. Kept only while `url` is saved. */
  private runLocalCheck(url: string): Promise<LocalChecked> {
    if (this.localRunning?.url === url) return this.localRunning.done;
    this.localStartedAt = this.now();
    const seq = ++this.localSeq;
    const done = checkLocal(url, { fetch: this.fetcher(), readCloud: true }).then((result) => {
      const checked = { result, seq };
      if (this.settings.localUrl === url) this.keepLocalCheck(url, checked);
      return checked;
    });
    const running = { url, done };
    this.localRunning = running;
    void done.finally(() => {
      if (this.localRunning === running) this.localRunning = null;
    });
    return done;
  }

  private keepLocalCheck(url: string, checked: LocalChecked): void {
    if (this.localCheck !== null && this.localCheck.url === url && this.localCheck.seq > checked.seq) return;
    this.localCheck = { url, result: checked.result, at: this.now(), seq: checked.seq };
  }

  /** Saves the settings; on a failure the earlier ones and their folders come back and the error is thrown. */
  private async saveOrRestore(previous: ProviderSettings): Promise<void> {
    try {
      await writeSettings(this.opts.paths.providers, this.settings);
    } catch (error) {
      this.settings = previous;
      await this.sync().catch(() => {});
      throw error;
    }
  }

  /** One question at a time: a caller while one is open joins it. A throw keeps the earlier answer. */
  private askPath(): Promise<void> {
    this.everAsked = true;
    if (this.pathAsked === null) this.pathAskedAt = this.now();
    this.pathAsked ??= (async () => this.opts.workPath())()
      .then((path) =>
        this.exclusive(async () => {
          this.workPath = path;
          await this.refreshCli();
        }),
      )
      .catch(() => {})
      .finally(() => {
        this.pathAsked = null;
      });
    return this.pathAsked;
  }

  /** A route that turned Firecrawl on: the reply waits for the Work PATH, as `list()` does. */
  private async withWorkPath(reply: ProvidersResponse): Promise<ProvidersResponse> {
    if (!this.firecrawlOn() || this.cliFound) return reply;
    await this.askPath();
    return this.exclusive(async () => this.snapshot());
  }

  private async fill(id: ProviderId): Promise<void> {
    const pending = (async () => this.opts.secrets.get(id))();
    const read = await this.race(() => pending);
    if (read.kind === 'value') {
      this.keys.set(id, read.value);
      return;
    }
    this.keychainDown = true;
    if (read.kind === 'timeout') {
      // The native call is left to finish; its answer fills the cache when it comes.
      pending.then(
        (value) => {
          if (this.keys.has(id)) return;
          this.replaceKey(id, value);
          this.keychainDown = this.unknownKeys();
          this.hooks.changed();
        },
        () => {},
      );
    }
  }

  private race<T>(call: () => Promise<T>): Promise<Raced<T>> {
    return new Promise((resolve) => {
      const timer = setTimeout(() => resolve({ kind: 'timeout' }), this.opts.keychainCallMs ?? KEYCHAIN_CALL_MS);
      let pending: Promise<T>;
      try {
        pending = call();
      } catch {
        clearTimeout(timer);
        resolve({ kind: 'error' });
        return;
      }
      pending.then(
        (value) => {
          clearTimeout(timer);
          resolve({ kind: 'value', value });
        },
        () => {
          clearTimeout(timer);
          resolve({ kind: 'error' });
        },
      );
    });
  }

  /**
   * A Keychain write. A failure is `keychain-unavailable`. After a timeout the provider takes no other write until
   * this one settles; a late success then completes the change under the lock.
   */
  private async keychainWrite(id: ProviderId, call: () => Promise<void>, late: () => void): Promise<void> {
    this.writing.add(id);
    const pending = (async () => call())();
    const result = await this.race(() => pending);
    if (result.kind !== 'timeout') this.writing.delete(id);
    if (result.kind === 'value') {
      this.keychainDown = false;
      return;
    }
    this.keychainDown = true;
    if (result.kind === 'timeout') {
      pending.then(
        () => void this.exclusive(async () => this.lateWrite(id, late)).catch(() => {}),
        () => this.writing.delete(id),
      );
    }
    throw keychainError();
  }

  private async lateWrite(id: ProviderId, late: () => void): Promise<void> {
    this.writing.delete(id);
    try {
      late();
      this.keychainDown = this.unknownKeys();
      if (repairSettings(this.settings, (p) => this.knownConfigured(p))) {
        await writeSettings(this.opts.paths.providers, this.settings).catch(() => {});
      }
      await this.sync().catch(() => {});
    } finally {
      this.hooks.changed();
    }
  }

  /** True while the Keychain has not answered for some provider. */
  private unknownKeys(): boolean {
    return PROVIDER_IDS.some((p) => !this.keys.has(p));
  }

  private generation(id: ProviderId): number {
    return this.generations.get(id) ?? 0;
  }

  /** A late Keychain read of one provider, taken only when no key route changed the key since it started. */
  private async applyRead(id: ProviderId, value: string | null, generation: number): Promise<void> {
    if (this.generation(id) !== generation) return;
    const changed = this.keys.get(id) !== value;
    this.replaceKey(id, value);
    this.keychainDown = this.unknownKeys();
    if (repairSettings(this.settings, (p) => this.knownConfigured(p))) {
      await writeSettings(this.opts.paths.providers, this.settings).catch(() => {});
    }
    if (changed) this.hooks.changed();
  }

  private async rereadAll(): Promise<Array<{ id: ProviderId; generation: number; read: Raced<string | null> }>> {
    return Promise.all(
      PROVIDER_IDS.map(async (id) => {
        const generation = this.generation(id);
        return { id, generation, read: await this.race(() => this.opts.secrets.get(id)) };
      }),
    );
  }

  /** Keys come only from the key routes once known; a re-read fills only the providers the cache never had. */
  private async applyReread(
    reads: ReadonlyArray<{ id: ProviderId; generation: number; read: Raced<string | null> }>,
  ): Promise<void> {
    let added = false;
    for (const { id, generation, read } of reads) {
      if (read.kind !== 'value' || this.keys.has(id) || this.generation(id) !== generation) continue;
      this.replaceKey(id, read.value);
      added = true;
    }
    this.keychainDown = reads.some((r) => r.read.kind !== 'value') || this.unknownKeys();
    if (!added) return;
    if (repairSettings(this.settings, (p) => this.knownConfigured(p))) {
      await writeSettings(this.opts.paths.providers, this.settings).catch(() => {});
    }
    this.hooks.changed();
  }

  private replaceKey(id: ProviderId, key: string | null): void {
    const old = this.keys.get(id);
    const grew = typeof key === 'string' && !this.secretNeedles().includes(key);
    if (typeof old === 'string' && old !== key) this.retired.set(old, id);
    this.keys.set(id, key);
    this.generations.set(id, this.generation(id) + 1);
    if (grew) this.hooks.keysChanged();
  }

  private recordCheck(id: ProviderId, check: KeyCheck | null, facts: ProviderFacts | null): void {
    if (check === null) delete this.settings.checks[id];
    else this.settings.checks[id] = { status: check, at: this.now() };
    if (facts === null) this.facts.delete(id);
    else this.facts.set(id, facts);
  }

  private async persistAndSync(): Promise<void> {
    await writeSettings(this.opts.paths.providers, this.settings);
    await this.sync();
  }

  /**
   * The key already changed in the Keychain and the cache: the skills follow, then the settings are saved. A skill
   * failure shows in the installed state; only a failed save is an error. The hosts hear it either way.
   */
  private async afterKeyChange(syncSkills: () => Promise<void>): Promise<void> {
    try {
      await syncSkills();
      try {
        await writeSettings(this.opts.paths.providers, this.settings);
      } catch {
        throw new ApiError('E_INTERNAL', SETTINGS_UNSAVED_MESSAGE);
      }
    } finally {
      this.hooks.changed();
    }
  }

  private wantedProviders(): ProviderId[] {
    const wanted: ProviderId[] = [];
    if (this.settings.search !== null) wanted.push(this.settings.search);
    if (this.settings.firecrawl) wanted.push('firecrawl');
    return wanted;
  }

  /** Installs the wanted bundles (the Chat skill, the selected providers) and removes every other folder. */
  private async sync(): Promise<SyncResult> {
    const wanted = [CHAT_SKILL_BUNDLE, ...this.wantedProviders()];
    let result: SyncResult;
    try {
      result = await this.installer.sync(wanted);
    } catch (error) {
      for (const id of wanted) this.installed.set(id, false);
      throw error;
    }
    for (const id of [CHAT_SKILL_BUNDLE, ...PROVIDER_IDS]) this.installed.set(id, result[id] === 'ok');
    return result;
  }

  /** Only wanted bundles count: a folder a failed sync left behind is not installed. */
  private async verifyInstalled(): Promise<void> {
    const manifest = this.manifest();
    const wanted = [CHAT_SKILL_BUNDLE, ...this.wantedProviders()];
    for (const id of [CHAT_SKILL_BUNDLE, ...PROVIDER_IDS]) {
      const ok =
        manifest !== null && wanted.includes(id) && (await this.installer.verified(id, manifest).catch(() => false));
      this.installed.set(id, ok);
    }
  }

  /** A change decides whether Work gets the Firecrawl key and skills: the hosts restart when idle. */
  private async refreshCli(): Promise<void> {
    const found = (await findOnPath('firecrawl', this.workPath)) !== null;
    if (found === this.cliFound) return;
    this.cliFound = found;
    if (this.firecrawlOn()) this.hooks.changed();
  }

  private manifest(): SkillsManifest | null {
    try {
      return loadManifest(this.skillsDir);
    } catch {
      return null;
    }
  }

  private exclusive<T>(run: () => Promise<T>): Promise<T> {
    const next = this.chain.then(run, run);
    this.chain = next.then(
      () => {},
      () => {},
    );
    return next;
  }

  private validateDeps(id: ProviderId): { baseUrl: string; fetch: typeof fetch; timeoutMs?: number } {
    return {
      baseUrl: this.opts.baseUrls?.[id] ?? PROVIDER_ORIGINS[id],
      fetch: this.fetcher(),
      timeoutMs: this.opts.validateTimeoutMs,
    };
  }

  private fetcher(): typeof fetch {
    return this.opts.fetch ?? ((input, init) => fetch(input, init));
  }

  private configured(id: ProviderId): boolean {
    return typeof this.keys.get(id) === 'string';
  }

  /** null when the Keychain has not answered for this provider. */
  private knownConfigured(id: ProviderId): boolean | null {
    return this.keys.has(id) ? this.configured(id) : null;
  }

  private selectedSearch(): SearchProviderId | null {
    const search = this.settings.search;
    return search !== null && this.configured(search) ? search : null;
  }

  /** Firecrawl reads pages: in the cloud with a key, or on this Mac with a saved address. */
  private firecrawlOn(): boolean {
    if (!this.settings.firecrawl) return false;
    return this.settings.firecrawlLocal ? this.settings.localUrl !== null : this.configured('firecrawl');
  }

  private localInUse(): boolean {
    return this.firecrawlOn() && this.settings.firecrawlLocal;
  }

  private localView(): LocalFirecrawl {
    const url = this.settings.localUrl;
    const last = url !== null && this.localCheck?.url === url ? this.localCheck : null;
    return {
      url,
      check: last?.result.check ?? null,
      checkedAt: last?.at ?? null,
      version: last?.result.version ?? null,
      cloud: last?.result.cloud ?? null,
    };
  }

  /**
   * Work gets the Firecrawl skills only when the `firecrawl` CLI is on its PATH, and with them the key (cloud) or the
   * address (this Mac).
   */
  private firecrawlInWork(): boolean {
    return this.firecrawlOn() && this.installed.get('firecrawl') === true && this.cliFound;
  }

  private fetchProvider(): ProviderId | null {
    return this.firecrawlOn() ? 'firecrawl' : this.selectedSearch();
  }

  private needsKey(id: ProviderId): ApiError {
    return new ApiError('E_CONFLICT', `Add a key for ${PROVIDER_NAMES[id]} first.`);
  }

  private snapshot(): ProvidersResponse {
    const manifest = this.manifest();
    const search = this.selectedSearch();
    const firecrawl = this.firecrawlOn();
    const providers = PROVIDER_IDS.map((id): ProviderState => {
      const bundle = manifest?.bundles[id];
      const check = this.settings.checks[id];
      const configured = this.configured(id);
      const selected = id === 'firecrawl' ? firecrawl : search === id;
      const installed = this.installed.get(id) === true;
      return {
        id,
        role: id === 'firecrawl' ? 'crawl' : 'search',
        configured,
        check: configured ? (check?.status ?? null) : null,
        checkedAt: configured ? (check?.at ?? null) : null,
        selected,
        keyEnv: [...PROVIDER_KEY_ENV[id]],
        skill: {
          installed,
          names: bundle === undefined ? [] : [...bundle.skills],
          upstream: bundle?.upstream === null || bundle === undefined ? null : { ...bundle.upstream },
        },
        work:
          id === 'firecrawl'
            ? { ready: this.firecrawlInWork(), missing: this.cliFound ? null : 'firecrawl-cli' }
            : { ready: selected && installed, missing: null },
        facts: configured ? (this.facts.get(id) ?? null) : null,
      };
    });
    return {
      search,
      firecrawl,
      firecrawlLocal: this.localInUse(),
      fetch: this.fetchProvider(),
      providers,
      local: this.localView(),
      chatSkill: { installed: this.installed.get(CHAT_SKILL_BUNDLE) === true },
      keychain: this.keychainDown ? 'unavailable' : 'ok',
      applyPending: this.hooks.applyPending(),
      vanished: this.vanished,
    };
  }
}
