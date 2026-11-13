import { execFileSync } from 'node:child_process';

import { describe, expect, it } from 'vite-plus/test';

import { chatAgentEnv, workAgentEnv, workOverlay } from '../src/chat/agent/env.js';
import {
  AGENT_ARGS,
  CHAT_PROMPT_HEAD,
  chatPrompt,
  chatSessionParams,
  initializeParams,
  loadSessionParams,
  NO_WEB,
  isWorkGuardHook,
  promptParams,
  skillBody,
  WORK_GUARD_PREFIX,
  WORK_RULES,
  workGuardHookName,
  workGuardManifest,
  workGuardName,
  workSessionParams,
} from '../src/chat/agent/profiles.js';
import { chatPaths } from '../src/chat/paths.js';

const paths = chatPaths('/r');
const endpoint = { url: 'http://127.0.0.1:4321', token: 'tok-INFERENCE' };

describe('chatAgentEnv', () => {
  it('is exactly the Chat table and inherits nothing', () => {
    const before = process.env.SOME_PARENT_VAR;
    process.env.SOME_PARENT_VAR = 'leak';
    try {
      expect(chatAgentEnv({ paths, endpoint, username: 'dev' })).toEqual({
        PATH: '/usr/bin:/bin:/usr/sbin:/sbin',
        HOME: '/r/chat-home',
        USER: 'dev',
        LOGNAME: 'dev',
        SHELL: '/bin/sh',
        LANG: 'en_US.UTF-8',
        TERM: 'dumb',
        TMPDIR: '/r/tmp',
        GROK_HOME: '/r/chat-agent',
        GROK_MODELS_BASE_URL: 'http://127.0.0.1:4321/v1',
        GROK_CLI_CHAT_PROXY_BASE_URL: 'http://127.0.0.1:4321/v1',
        GROK_XAI_API_BASE_URL: 'http://127.0.0.1:4321/v1',
        XAI_API_KEY: 'tok-INFERENCE',
        GROK_DISABLE_AUTOUPDATER: '1',
        GROK_IMAGE_GEN: '0',
        GROK_IMAGE_EDIT: '0',
        GROK_VIDEO_GEN: '0',
        DISABLE_TELEMETRY: '1',
        GROK_TELEMETRY_ENABLED: 'false',
        GROK_TELEMETRY_TRACE_UPLOAD: 'false',
        GROK_CLAUDE_HOOKS_ENABLED: 'false',
        GROK_CLAUDE_MCPS_ENABLED: 'false',
        GROK_CURSOR_HOOKS_ENABLED: 'false',
        GROK_CURSOR_MCPS_ENABLED: 'false',
        GROK_CODEX_HOOKS_ENABLED: 'false',
        GROK_CODEX_MCPS_ENABLED: 'false',
        GROK_CLAUDE_SKILLS_ENABLED: 'false',
        GROK_CURSOR_SKILLS_ENABLED: 'false',
        GROK_TURN_SUMMARY: '0',
        GROK_SESSION_SEARCH: '0',
        GROK_CONFIG: '{"models":{"max_completion_tokens":8192}}',
      });
    } finally {
      if (before === undefined) delete process.env.SOME_PARENT_VAR;
      else process.env.SOME_PARENT_VAR = before;
    }
  });
});

describe('workAgentEnv', () => {
  const base = {
    PATH: '/usr/bin:/opt/homebrew/bin',
    HOME: '/Users/dev',
    ELECTRON_RUN_AS_NODE: '1',
    MLX_DESKTOP_CLI: '{"executable":"x"}',
    NAPI_RS_NATIVE_LIBRARY_PATH: '/x.node',
    NODE_OPTIONS: '--inspect',
    EXA_API_KEY: 'old-exa',
    BRAVE_API_KEY: 'old-brave',
    DEEPSEEK_API_KEY: 'old-deepseek',
    FIRECRAWL_API_KEY: 'old-firecrawl',
    BRAVE_SEARCH_API_KEY: 'old-brave-search',
    GROK_IMAGE_GEN: '1',
    GROK_CONFIG: '{"mine":true}',
  };

  it('keeps the parent env, removes the app internals and every unselected key, sets the rest', () => {
    const env = workAgentEnv({ base, endpoint, agentHome: null, providerKeys: { EXA_API_KEY: 'new-exa' } });
    expect(env.PATH).toBe('/usr/bin:/opt/homebrew/bin');
    expect(env.HOME).toBe('/Users/dev');
    for (const name of ['ELECTRON_RUN_AS_NODE', 'MLX_DESKTOP_CLI', 'NAPI_RS_NATIVE_LIBRARY_PATH', 'NODE_OPTIONS']) {
      expect(env).not.toHaveProperty(name);
    }
    for (const name of ['BRAVE_API_KEY', 'BRAVE_SEARCH_API_KEY', 'DEEPSEEK_API_KEY', 'FIRECRAWL_API_KEY']) {
      expect(env).not.toHaveProperty(name);
    }
    expect(env.EXA_API_KEY).toBe('new-exa');
    expect(env.XAI_API_KEY).toBe('tok-INFERENCE');
    expect(env.GROK_MODELS_BASE_URL).toBe('http://127.0.0.1:4321/v1');
    expect(env.GROK_DISABLE_AUTOUPDATER).toBe('1');
    expect(env.GROK_IMAGE_GEN).toBe('1');
    expect(env.GROK_IMAGE_EDIT).toBe('0');
    expect(env.GROK_RESUME_COMMAND).toBe('mlx agent');
    expect(env.GROK_TURN_SUMMARY).toBe('0');
    expect(env.GROK_CONFIG).toBe('{"mine":true,"models":{"max_completion_tokens":16384}}');
    expect(env).not.toHaveProperty('GROK_HOME');
    expect(env.FIRECRAWL_NO_SEARCH_FEEDBACK).toBe('1');
    expect(env.FIRECRAWL_DISABLE_SEARCH_FEEDBACK).toBe('1');
    expect(env.FIRECRAWL_NO_ENDPOINT_FEEDBACK).toBe('1');
  });

  it('sets CDPATH empty, also over a CDPATH of the app, so the agent does not fill it from the login shell', () => {
    for (const own of [{}, { CDPATH: '/Users/dev:.' }]) {
      const env = workAgentEnv({ base: { ...base, ...own }, endpoint, agentHome: null, providerKeys: {} });
      expect(Object.hasOwn(env, 'CDPATH')).toBe(true);
      expect(env.CDPATH).toBe('');
    }
  });

  it('caps output at 16,384 tokens when the user set no config, and sets GROK_HOME only for a custom home', () => {
    const env = workAgentEnv({ base: { PATH: '/bin' }, endpoint, agentHome: '/custom/agent', providerKeys: {} });
    expect(env.GROK_CONFIG).toBe('{"models":{"max_completion_tokens":16384}}');
    expect(env.GROK_HOME).toBe('/custom/agent');
  });

  it("keeps the user's own overlay and puts the cap inside it; a value that is not an object is replaced", () => {
    const cap = '{"models":{"max_completion_tokens":16384}}';
    const rows: Array<[string | undefined, string]> = [
      [undefined, cap],
      ['', cap],
      ['not json', cap],
      ['[1]', cap],
      ['null', cap],
      ['{"models":"x"}', cap],
      ['{"features":{"a":true}}', '{"features":{"a":true},"models":{"max_completion_tokens":16384}}'],
      [
        '{"models":{"default_reasoning_effort":"low","max_completion_tokens":99999}}',
        '{"models":{"default_reasoning_effort":"low","max_completion_tokens":16384}}',
      ],
      ['{"models":{"max_completion_tokens":4096}}', '{"models":{"max_completion_tokens":4096}}'],
      ['{"models":{"max_completion_tokens":0}}', cap],
      ['{"models":{"max_completion_tokens":"8"}}', cap],
    ];
    for (const [base, want] of rows) {
      expect([base, workOverlay(base)]).toEqual([base, want]);
      const env = workAgentEnv({
        base: base === undefined ? {} : { GROK_CONFIG: base },
        endpoint,
        agentHome: null,
        providerKeys: {},
      });
      expect([base, env.GROK_CONFIG]).toEqual([base, want]);
    }
  });

  it('never sets a provider key variable name it does not know', () => {
    const env = workAgentEnv({ base: {}, endpoint, agentHome: null, providerKeys: { PATH: 'evil', EXA_API_KEY: 'k' } });
    expect(env.PATH).toBeUndefined();
    expect(env.EXA_API_KEY).toBe('k');
  });
});

describe('session contract', () => {
  it('initialize pins clientType "generic" and advertises no fs or terminal', () => {
    expect(initializeParams('9.9.9')).toEqual({
      protocolVersion: 1,
      clientCapabilities: { fs: { readTextFile: false, writeTextFile: false }, terminal: false },
      _meta: {
        clientType: 'generic',
        clientIdentifier: 'mlx-node-desktop',
        clientVersion: '9.9.9',
        startupHints: { nonInteractive: false },
      },
    });
    expect(AGENT_ARGS).toEqual(['--disable-web-search', 'agent', '--no-leader', 'stdio']);
  });

  it('the Chat session has a profile object, two tools, one plugin dir, and keys only as env values', () => {
    const params = chatSessionParams({
      cwd: '/r/chat-workspace',
      modelId: 'm',
      webServer: { command: '/app/exe', script: '/app/server.js' },
      web: { searchProvider: 'exa', searchKey: 'KEY-1', fetchProvider: 'exa', fetchKey: 'KEY-1' },
      prompt: 'P',
      pluginDir: '/r/plugins/chat-web',
      extraCaCerts: '/ca.pem',
    });
    expect(params).toEqual({
      cwd: '/r/chat-workspace',
      mcpServers: [
        {
          name: 'mlx_web',
          command: '/app/exe',
          args: ['/app/server.js'],
          env: [
            { name: 'MLX_WEB_SEARCH_PROVIDER', value: 'exa' },
            { name: 'MLX_WEB_SEARCH_KEY', value: 'KEY-1' },
            { name: 'MLX_WEB_FETCH_PROVIDER', value: 'exa' },
            { name: 'MLX_WEB_FETCH_KEY', value: 'KEY-1' },
            { name: 'NODE_NO_WARNINGS', value: '1' },
            { name: 'ELECTRON_RUN_AS_NODE', value: '1' },
            { name: 'NODE_EXTRA_CA_CERTS', value: '/ca.pem' },
          ],
        },
      ],
      _meta: {
        modelId: 'm',
        yoloMode: false,
        autoMode: false,
        askUserQuestion: false,
        agentProfile: {
          name: 'mlx-chat',
          description: 'mlx-node Chat mode',
          promptMode: 'full',
          promptBody: 'P',
          tools: ['search_tool', 'use_tool'],
          discoverSkills: true,
        },
        pluginDirs: ['/r/plugins/chat-web'],
      },
    });
    const meta = params._meta as Record<string, unknown>;
    expect(typeof meta.agentProfile).toBe('object');
  });

  it('with no provider the web server is still attached and answers "none"', () => {
    const params = chatSessionParams({
      cwd: '/w',
      modelId: 'm',
      webServer: { command: 'n', script: 's' },
      web: NO_WEB,
      prompt: 'P',
      pluginDir: '/p',
    });
    const servers = params.mcpServers as Array<{ env: Array<{ name: string; value: string }> }>;
    expect(servers).toHaveLength(1);
    expect(servers[0]!.env.slice(0, 4).map((e) => e.value)).toEqual(['none', '', 'none', '']);
  });

  it('the Work session sends the rules, the guard and only the given plugin dirs, and no MCP server', () => {
    expect(
      workSessionParams({ cwd: '/proj', modelId: 'm', guardDir: '/r/work-guard', pluginDirs: ['/r/plugins/exa'] }),
    ).toEqual({
      cwd: '/proj',
      mcpServers: [],
      _meta: {
        modelId: 'm',
        rules: WORK_RULES,
        yoloMode: false,
        autoMode: false,
        askUserQuestion: false,
        pluginDirs: ['/r/work-guard', '/r/plugins/exa'],
      },
    });
  });

  it('the guard plugin is one PreToolUse hook on Bash and monitor whose command prints an "ask" decision', () => {
    const name = workGuardName();
    const manifest = JSON.parse(workGuardManifest(name)) as {
      name: string;
      hooks: { hooks: Record<string, Array<{ matcher: string; hooks: Array<{ type: string; command: string }> }>> };
    };
    expect(manifest.name).toBe(name);
    expect(Object.keys(manifest.hooks.hooks)).toEqual(['PreToolUse']);
    const groups = manifest.hooks.hooks.PreToolUse!;
    expect(groups.map((g) => [g.matcher, g.hooks.map((h) => h.type)])).toEqual([['Bash|monitor', ['command']]]);
    const printed = execFileSync('/bin/sh', ['-c', groups[0]!.hooks[0]!.command], { encoding: 'utf8', env: {} });
    expect(JSON.parse(printed)).toEqual({
      hookSpecificOutput: { hookEventName: 'PreToolUse', permissionDecision: 'ask' },
    });
    expect(workGuardHookName(name)).toBe(`plugin/${name}/plugin:pre_tool_use[0].hooks[0]`);
  });

  it('the guard name is new at every start, so a folder cannot list it as disabled; only its own hook name matches', () => {
    const names = [workGuardName(), workGuardName()];
    expect(names[0]).not.toBe(names[1]);
    for (const name of names) {
      expect(name.startsWith(WORK_GUARD_PREFIX)).toBe(true);
      expect(name.length).toBe(WORK_GUARD_PREFIX.length + 16);
      expect(isWorkGuardHook(workGuardHookName(name))).toBe(true);
    }
    for (const other of [
      'plugin/mlx-work-guard/plugin:pre_tool_use[0].hooks[0]',
      `plugin/${WORK_GUARD_PREFIX}0123456789abcdeg/plugin:pre_tool_use[0].hooks[0]`,
      `plugin/${WORK_GUARD_PREFIX}0123456789abcdef0/plugin:pre_tool_use[0].hooks[0]`,
      `plugin/${WORK_GUARD_PREFIX}0123456789abcdef/plugin:pre_tool_use[1].hooks[0]`,
      `plugin/x/${WORK_GUARD_PREFIX}0123456789abcdef/plugin:pre_tool_use[0].hooks[0]`,
    ]) {
      expect(isWorkGuardHook(other)).toBe(false);
    }
  });

  it('a load resends everything of a new session plus noReplay', () => {
    const fresh = workSessionParams({ cwd: '/proj', modelId: 'm', guardDir: '/r/work-guard', pluginDirs: [] });
    expect(loadSessionParams(fresh, 'sid')).toEqual({
      sessionId: 'sid',
      cwd: '/proj',
      mcpServers: [],
      _meta: { ...(fresh._meta as object), noReplay: true },
    });
  });

  it('a prompt is one text block with exactly the keys type and text', () => {
    const params = promptParams('sid', 'hello');
    expect(params).toEqual({ sessionId: 'sid', prompt: [{ type: 'text', text: 'hello' }] });
    const block = (params.prompt as Array<Record<string, unknown>>)[0]!;
    expect(Object.keys(block).sort()).toEqual(['text', 'type']);
  });

  it('the Chat prompt is the head, a blank line and the skill body without frontmatter', () => {
    const skill = '---\nname: web-tools\ndescription: x\n---\n\n# Web tools\n\nbody\n';
    expect(skillBody(skill)).toBe('# Web tools\n\nbody\n');
    expect(chatPrompt(skill)).toBe(`${CHAT_PROMPT_HEAD}\n\n# Web tools\n\nbody\n`);
    expect(
      CHAT_PROMPT_HEAD.startsWith(
        "You are the Chat assistant of mlx-node. You run on a local model on the user's Mac.",
      ),
    ).toBe(true);
  });
});
