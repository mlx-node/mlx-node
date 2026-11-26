import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { agentConfigDir } from '../src/paths.js';

describe('agentConfigDir', () => {
  // Mutations: reading `GROK_HOME` again, or ahead of `MLX_AGENT_HOME` (mlx-agent runs in grok-build's `~/.grok`) → red.
  it("reads MLX_AGENT_HOME and never grok-build's GROK_HOME", () => {
    expect(agentConfigDir({ GROK_HOME: '/x/.grok' }, '/h')).toBe(join('/h', '.mlx-agent'));
    expect(agentConfigDir({ MLX_AGENT_HOME: '/m', GROK_HOME: '/x/.grok' }, '/h')).toBe('/m');
    expect(agentConfigDir({ MLX_AGENT_HOME: '~/agent' }, '/h')).toBe(join('/h', 'agent'));
    expect(agentConfigDir({}, '/h')).toBe(join('/h', '.mlx-agent'));
  });
});
