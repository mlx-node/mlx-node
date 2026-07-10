export type { DiscoveredModelLike } from './types.js';

export { buildChatConfig } from './provider/chat-config.js';
export { contextToChatMessages, toolsToDefinitions } from './provider/convert-messages.js';
export { TurnEmitter } from './provider/events.js';
export { createMlxProviderExtension } from './provider/index.js';
export { MlxModelHost, type MlxModelHostOptions } from './provider/model-host.js';
export { discoverMlxModels, type MlxModelInfo } from './provider/models.js';
export { makeMlxStreamSimple, type StreamSimpleHost } from './provider/stream-adapter.js';
