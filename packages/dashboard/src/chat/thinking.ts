/**
 * Thinking levels on the host side. The levels of a model are the ones the agent lists for it (its echo of the
 * engine's `/v1/models` row); before an agent lists them, the engine's family table (`levelsFor` of
 * `@mlx-node/server/host/thinking-levels`). The host keeps no table of its own.
 */

import type { EffortOption } from './acp/messages.js';
import { isThinkingLevel } from './thinking-levels.js';
import { THINKING_LEVELS, type ModelThinking } from './types.js';

/** The `configId` of the level in `session/set_config_option` and in `configOptions`. */
export const EFFORT_CONFIG_ID = 'reasoning_effort';

/**
 * A model's levels from the agent's options: the known levels whose option id is the level itself (the engine's rows
 * name no other id), least thinking first; the default is the option marked so, else the first, as the agent picks
 * it. Null when none reads.
 */
export function modelThinking(options: readonly EffortOption[] | null): ModelThinking | null {
  if (options === null) return null;
  const levels = THINKING_LEVELS.filter((level) => options.some((o) => o.value === level && o.id === level));
  if (levels.length === 0) return null;
  const marked = options.find((o) => o.default && levels.some((level) => level === o.value))?.value;
  return { levels, default: isThinkingLevel(marked) ? marked : levels[0]! };
}
