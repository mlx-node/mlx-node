/**
 * `createMlxProviderExtension` — the pi inline extension that registers
 * the in-process `mlx` provider.
 *
 * Task 8's `runAgent` passes the returned extension into pi's `main()`
 * via `extensionFactories`; pi calls the factory during extension load,
 * and `registerProvider` makes every discovered local model resolvable
 * as `mlx/<dir-name>` with no /login (the literal apiKey marks the
 * models available).
 *
 * Import discipline (load-bearing): pi is import-order sensitive to its
 * config env vars, so this module — which the CLI imports BEFORE those
 * env vars are set — must not runtime-import `@earendil-works/pi-coding-agent`
 * at module top level. Only type-only pi imports appear here; the
 * `ExtensionAPI` value arrives as the factory argument.
 */

import type { ExtensionAPI, InlineExtension } from '@earendil-works/pi-coding-agent';

import { MlxModelHost } from './model-host.js';
import type { MlxModelInfo } from './models.js';
import { makeMlxStreamSimple } from './stream-adapter.js';

/**
 * Build the `mlx-provider` inline extension serving `models`. The host
 * (one per process — it owns the single GPU-resident model) is created
 * eagerly so repeated factory invocations can never spawn a second
 * host, but stays lazy about weights: nothing loads until the first
 * `streamSimple` call.
 */
export function createMlxProviderExtension(models: MlxModelInfo[], host?: MlxModelHost): InlineExtension {
  const resolvedHost = host ?? new MlxModelHost(models.map((m) => m.discovered));
  const streamSimple = makeMlxStreamSimple(resolvedHost);
  return {
    name: 'mlx-provider',
    factory: (pi: ExtensionAPI) => {
      pi.registerProvider('mlx', {
        api: 'mlx',
        baseUrl: 'mlx://local',
        apiKey: 'mlx-local',
        streamSimple,
        models: models.map((m) => m.piModel),
      });
    },
  };
}
