export type { DiscoveredModelLike } from './types.js';

export { type CatalogEntry, MODEL_CATALOG, visibleCatalog } from './catalog.js';
export { createPermissionGateExtension } from './extensions/permission-gate.js';
export {
  createSubagentExtension,
  discoverSubagents,
  normalizeSubagentMode,
  type InProcessSubagentSession,
  type SubagentConfig,
  type SubagentExtensionOptions,
  type SubagentMode,
  type SubagentSessionCreateOptions,
} from './extensions/subagent.js';
export { createTerminalTitleExtension } from './extensions/terminal-title.js';
export { buildChatConfig } from './provider/chat-config.js';
export { contextToChatMessages, toolsToDefinitions } from './provider/convert-messages.js';
export { TurnEmitter } from './provider/events.js';
export { createMlxProviderExtension } from './provider/index.js';
export { MlxModelHost, type MlxModelHostOptions } from './provider/model-host.js';
export { discoverMlxModels, type MlxModelInfo } from './provider/models.js';
export { runAgent, type RunAgentMain, type RunAgentOptions } from './run-agent.js';
export { makeMlxStreamSimple, type StreamSimpleHost } from './provider/stream-adapter.js';
