import { whoAmI } from '@huggingface/hub';
import { input } from '@inquirer/prompts';
import { AsyncEntry } from '@napi-rs/keyring';

const KEYRING_SERVICE = 'mlx-node';
const KEYRING_ACCOUNT = 'huggingface-token';

const keyringEntry = new AsyncEntry(KEYRING_SERVICE, KEYRING_ACCOUNT);

/**
 * Keychain access routed through this object so tests can spy on `read`/`write`
 * without mocking `@napi-rs/keyring` at the module level. The exported helpers
 * call `keychain.read()` / `keychain.write()` so spies take effect.
 */
export const keychain = {
  async read(): Promise<string | undefined> {
    try {
      return (await keyringEntry.getPassword()) ?? undefined;
    } catch {
      // keyring v2 rejects on store errors (locked keychain, no Secret
      // Service on headless Linux) where v1 returned null; keep the
      // env-var/anonymous fallback working in that case.
      return undefined;
    }
  },
  async write(token: string): Promise<void> {
    await keyringEntry.setPassword(token);
  },
};

/**
 * Resolve the HuggingFace token for any download.
 *
 * Order: keychain (set via `mlx download --set-token`) first, then the
 * `HUGGINGFACE_TOKEN` environment variable, else `undefined` (anonymous
 * access). This is the single source of truth shared by both the model and
 * dataset downloaders so a token set once applies everywhere.
 */
export async function resolveHuggingFaceToken(): Promise<string | undefined> {
  return (await keychain.read()) ?? process.env.HUGGINGFACE_TOKEN ?? undefined;
}

/** Injectable IO so tests can drive `setToken` without an interactive prompt. */
export interface TokenPromptIO {
  /** Returns the token the user entered (already validated). */
  readToken: () => Promise<string>;
  log: (line: string) => void;
}

const defaultIO: TokenPromptIO = {
  readToken: () =>
    input({
      message: 'Enter your HuggingFace token:',
      required: true,
      theme: {
        validationFailureMode: 'clear',
      },
      validate: async (value) => {
        if (!value) {
          return 'Token is required';
        }
        if (!value.startsWith('hf_')) {
          return 'HuggingFace token must start with "hf_"';
        }
        try {
          const { auth } = await whoAmI({ accessToken: value });
          if (!auth) {
            return 'Invalid token';
          }
          return true;
        } catch {
          return 'Invalid token';
        }
      },
    }),
  log: (line) => console.log(line),
};

/**
 * Prompt for a HuggingFace token and persist it to the OS keychain.
 *
 * Shared by `mlx download --set-token` (download-wide) and the
 * `mlx download model --set-token` flag. Pass a custom `io` in tests.
 */
export async function setToken(io: TokenPromptIO = defaultIO): Promise<void> {
  const token = await io.readToken();
  if (token) {
    await keychain.write(token);
    io.log('HuggingFace token saved to the OS keychain.');
  }
}
