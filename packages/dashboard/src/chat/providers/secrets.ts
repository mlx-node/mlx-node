/**
 * Where provider keys live: the macOS Keychain, service `mlx-node`, one account per provider. Nothing else in the app
 * stores a key. Callers go through the key cache of the providers service, which bounds every call by a timeout.
 */

import type { ProviderId } from './types.js';

export const KEYCHAIN_SERVICE = 'mlx-node';

export const KEYCHAIN_ACCOUNTS: Readonly<Record<ProviderId, string>> = {
  exa: 'search-exa-api-key',
  deepseek: 'search-deepseek-api-key',
  brave: 'search-brave-api-key',
  firecrawl: 'crawl-firecrawl-api-key',
};

export interface SecretStore {
  get(id: ProviderId): Promise<string | null>;
  set(id: ProviderId, key: string): Promise<void>;
  delete(id: ProviderId): Promise<void>;
}

interface KeyringEntry {
  getPassword(): Promise<string | undefined | null>;
  setPassword(password: string): Promise<void>;
  deletePassword(): Promise<boolean>;
}

type KeyringModule = { AsyncEntry: new (service: string, account: string) => KeyringEntry };

/** The Keychain through `@napi-rs/keyring`, loaded on first use: a load failure is a failed call. */
export class KeyringSecretStore implements SecretStore {
  private module: Promise<KeyringModule> | null = null;

  private async entry(id: ProviderId): Promise<KeyringEntry> {
    this.module ??= import('@napi-rs/keyring') as Promise<KeyringModule>;
    const { AsyncEntry } = await this.module;
    return new AsyncEntry(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNTS[id]);
  }

  async get(id: ProviderId): Promise<string | null> {
    const value = await (await this.entry(id)).getPassword();
    return typeof value === 'string' && value !== '' ? value : null;
  }

  async set(id: ProviderId, key: string): Promise<void> {
    await (await this.entry(id)).setPassword(key);
  }

  async delete(id: ProviderId): Promise<void> {
    await (await this.entry(id)).deletePassword();
  }
}

/** Keys in memory: tests and the dev harness. */
export class MemorySecretStore implements SecretStore {
  private readonly keys = new Map<ProviderId, string>();

  get(id: ProviderId): Promise<string | null> {
    return Promise.resolve(this.keys.get(id) ?? null);
  }

  set(id: ProviderId, key: string): Promise<void> {
    this.keys.set(id, key);
    return Promise.resolve();
  }

  delete(id: ProviderId): Promise<void> {
    this.keys.delete(id);
    return Promise.resolve();
  }
}

export const MIN_KEY_CHARS = 8;
export const MAX_KEY_CHARS = 512;

/** The key trimmed when it is 8 to 512 printable ASCII characters without spaces, else null. No prefix check. */
export function checkKeyFormat(raw: string): string | null {
  const key = raw.trim();
  if (key.length < MIN_KEY_CHARS || key.length > MAX_KEY_CHARS) return null;
  for (let i = 0; i < key.length; i++) {
    const c = key.charCodeAt(i);
    if (c <= 0x20 || c >= 0x7f) return null;
  }
  return key;
}
