import { statSync } from 'node:fs';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it, vi } from 'vite-plus/test';

import type { GrokAgentLaunchOptions } from '../../packages/cli/src/commands/agent/grok-build.js';
import {
  type AgentRunDeps,
  chooseDefaultModel,
  configureAgentTracing,
  expandAgentDir,
  readPersistedDefaultModel,
  run,
  scanAgentArgs,
  withDefaultModel,
} from '../../packages/cli/src/commands/agent/index.js';

describe('scanAgentArgs', () => {
  describe('--models-dir extraction', () => {
    it('extracts a leading --models-dir pair and removes it from passthrough', () => {
      const scan = scanAgentArgs(['--models-dir', '/models', '-p', 'hi']);
      expect(scan.modelsDir).toBe('/models');
      expect(scan.passthrough).toEqual(['-p', 'hi']);
      expect(scan.modelsDirMissingValue).toBe(false);
    });

    it('extracts a trailing --models-dir pair, preserving preceding args in order', () => {
      const scan = scanAgentArgs(['-p', 'hi', '--effort', 'high', '--models-dir', '/models']);
      expect(scan.modelsDir).toBe('/models');
      expect(scan.passthrough).toEqual(['-p', 'hi', '--effort', 'high']);
    });

    it('supports the --models-dir=<dir> form', () => {
      const scan = scanAgentArgs(['--models-dir=/models', '-c']);
      expect(scan.modelsDir).toBe('/models');
      expect(scan.passthrough).toEqual(['-c']);
    });

    it('flags a --models-dir without a value', () => {
      const scan = scanAgentArgs(['--models-dir']);
      expect(scan.modelsDirMissingValue).toBe(true);
      expect(scan.modelsDir).toBeUndefined();
      expect(scan.passthrough).toEqual([]);
    });

    it('flags an empty --models-dir= value', () => {
      const scan = scanAgentArgs(['--models-dir=']);
      expect(scan.modelsDirMissingValue).toBe(true);
      expect(scan.modelsDir).toBeUndefined();
    });

    it('flags an empty space-form value without eating later args', () => {
      const scan = scanAgentArgs(['--models-dir', '', '-p', 'hi']);
      expect(scan.modelsDirMissingValue).toBe(true);
      expect(scan.modelsDir).toBeUndefined();
      expect(scan.passthrough).toEqual(['-p', 'hi']);
    });

    it('never consumes an option-looking token as the space-form value', () => {
      for (const nextFlag of ['--tools', '--help', '--effort', '-c']) {
        const scan = scanAgentArgs(['--models-dir', nextFlag]);
        expect(scan.modelsDirMissingValue).toBe(true);
        expect(scan.modelsDir).toBeUndefined();
        // The flag stays in passthrough — it was never a value.
        expect(scan.passthrough).toEqual([nextFlag]);
      }
    });

    it('still accepts a dash-leading dir via the = form', () => {
      const scan = scanAgentArgs(['--models-dir=-odd-dir', '-p', 'hi']);
      expect(scan.modelsDir).toBe('-odd-dir');
      expect(scan.modelsDirMissingValue).toBe(false);
      expect(scan.passthrough).toEqual(['-p', 'hi']);
    });
  });

  describe('trace extraction', () => {
    it('extracts --trace without forwarding it to the binary', () => {
      const scan = scanAgentArgs(['--trace', '-p', 'hi']);
      expect(scan.trace).toBe(true);
      expect(scan.traceDir).toBeUndefined();
      expect(scan.traceDirMissingValue).toBe(false);
      expect(scan.passthrough).toEqual(['-p', 'hi']);
    });

    it('extracts both --trace-dir forms and makes them imply tracing', () => {
      const spaced = scanAgentArgs(['--trace-dir', '/tmp/mlx-trace', '-p', 'hi']);
      expect(spaced.trace).toBe(true);
      expect(spaced.traceDir).toBe('/tmp/mlx-trace');
      expect(spaced.traceDirMissingValue).toBe(false);
      expect(spaced.passthrough).toEqual(['-p', 'hi']);

      const inline = scanAgentArgs(['--trace-dir=-odd-dir', '-c']);
      expect(inline.trace).toBe(true);
      expect(inline.traceDir).toBe('-odd-dir');
      expect(inline.traceDirMissingValue).toBe(false);
      expect(inline.passthrough).toEqual(['-c']);
    });

    it('reports a missing --trace-dir value without swallowing the next option', () => {
      for (const argv of [['--trace-dir'], ['--trace-dir='], ['--trace-dir', ''], ['--trace-dir', '--help']]) {
        const scan = scanAgentArgs(argv);
        expect(scan.trace).toBe(true);
        expect(scan.traceDir).toBeUndefined();
        expect(scan.traceDirMissingValue).toBe(true);
      }
      expect(scanAgentArgs(['--trace-dir', '--help']).passthrough).toEqual(['--help']);
    });

    it('does not hijack trace flags that occupy a binary option value slot', () => {
      const flagValue = scanAgentArgs(['--system-prompt-override', '--trace', '-p', 'hi']);
      expect(flagValue.trace).toBe(false);
      expect(flagValue.passthrough).toEqual(['--system-prompt-override', '--trace', '-p', 'hi']);

      const dirFlagValue = scanAgentArgs(['--system-prompt-override', '--trace-dir', '/still-passthrough']);
      expect(dirFlagValue.trace).toBe(false);
      expect(dirFlagValue.traceDir).toBeUndefined();
      expect(dirFlagValue.passthrough).toEqual(['--system-prompt-override', '--trace-dir', '/still-passthrough']);
    });
  });

  describe('--no-persist-cache extraction', () => {
    it('lifts the mlx-owned flag out of passthrough', () => {
      const scan = scanAgentArgs(['--no-persist-cache', '-p', 'hi']);
      expect(scan.persistPagedCache).toBe(false);
      expect(scan.passthrough).toEqual(['-p', 'hi']);
    });

    it('does not hijack the flag when it sits in a value-consumer slot', () => {
      const scan = scanAgentArgs(['--system-prompt-override', '--no-persist-cache']);
      expect(scan.persistPagedCache).toBe(true);
      expect(scan.passthrough).toEqual(['--system-prompt-override', '--no-persist-cache']);
    });
  });

  describe('update intercept', () => {
    it('detects a leading update positional', () => {
      expect(scanAgentArgs(['update']).update).toBe(true);
      expect(scanAgentArgs(['update', '--all']).update).toBe(true);
    });

    it('does not trip on update in a non-leading position', () => {
      const scan = scanAgentArgs(['-p', 'update']);
      expect(scan.update).toBe(false);
      expect(scan.passthrough).toEqual(['-p', 'update']);
    });

    it('detects update behind a stripped --models-dir pair (the binary would see it at args[0])', () => {
      const scan = scanAgentArgs(['--models-dir', '/x', 'update']);
      expect(scan.update).toBe(true);
      expect(scan.passthrough).toEqual(['update']);
    });
  });

  describe('help detection', () => {
    it('detects -h and --help', () => {
      expect(scanAgentArgs(['-h']).help).toBe(true);
      expect(scanAgentArgs(['--help']).help).toBe(true);
      expect(scanAgentArgs(['--effort', 'high', '--help']).help).toBe(true);
    });

    it('leaves per-subcommand help to the binary (they print their own help inside)', () => {
      for (const command of ['models', 'sessions', 'export', 'doctor', 'mcp', 'config', 'agent']) {
        const scan = scanAgentArgs([command, '--help']);
        expect(scan.help).toBe(false);
        expect(scan.passthrough).toEqual([command, '--help']);
      }
    });

    it('suppresses mlx help for a binary subcommand behind --models-dir too', () => {
      const scan = scanAgentArgs(['--models-dir', '/x', 'models', '--help']);
      expect(scan.help).toBe(false);
      expect(scan.passthrough).toEqual(['models', '--help']);
    });

    it('does not detect help when absent', () => {
      expect(scanAgentArgs(['-p', 'hello']).help).toBe(false);
    });
  });

  describe('passthrough preservation', () => {
    it('passes a leading subcommand through untouched', () => {
      const scan = scanAgentArgs(['sessions', 'list']);
      expect(scan.update).toBe(false);
      expect(scan.help).toBe(false);
      expect(scan.passthrough).toEqual(['sessions', 'list']);
    });

    it('passes -c, --resume and unknown flags through untouched, in order', () => {
      const argv = ['-c', '--resume', 'abc123', '--totally-unknown-flag', 'value', '-p', 'prompt text'];
      const scan = scanAgentArgs(argv);
      expect(scan.passthrough).toEqual(argv);
      expect(scan.modelsDir).toBeUndefined();
      expect(scan.help).toBe(false);
      expect(scan.update).toBe(false);
    });

    it('stops mlx flag extraction at the -- terminator', () => {
      const scan = scanAgentArgs(['-p', 'hi', '--', '--models-dir', '/x', '--help']);
      expect(scan.modelsDir).toBeUndefined();
      expect(scan.help).toBe(false);
      expect(scan.passthrough).toEqual(['-p', 'hi', '--', '--models-dir', '/x', '--help']);
    });

    it('returns empty passthrough for empty argv', () => {
      const scan = scanAgentArgs([]);
      expect(scan.passthrough).toEqual([]);
      expect(scan.help).toBe(false);
      expect(scan.update).toBe(false);
    });
  });

  describe('value-aware walk shares VALUE_CONSUMING_ARGS (WB-5, sibling of R3-2)', () => {
    // The scan must NOT hijack a token that sits in a grok value-consumer's
    // value slot. Mutation guard: reverting scanAgentArgs to the raw
    // exact-token scan strips the systemPrompt value + interprets a value
    // `--help`, failing these.
    it('does not strip a --models-dir that is the VALUE of --system-prompt-override', () => {
      const scan = scanAgentArgs(['--system-prompt-override', '--models-dir', 'x']);
      expect(scan.modelsDir).toBeUndefined();
      expect(scan.modelsDirMissingValue).toBe(false);
      expect(scan.passthrough).toEqual(['--system-prompt-override', '--models-dir', 'x']);
    });

    it('does not route to help for a --help that is the VALUE of --system-prompt-override', () => {
      const scan = scanAgentArgs(['--system-prompt-override', '--help', '-p', 'hi']);
      expect(scan.help).toBe(false);
      expect(scan.passthrough).toEqual(['--system-prompt-override', '--help', '-p', 'hi']);
    });

    it('does not treat update as the blocked positional when it is a consumed VALUE', () => {
      const scan = scanAgentArgs(['--system-prompt-override', 'update']);
      expect(scan.update).toBe(false);
      expect(scan.passthrough).toEqual(['--system-prompt-override', 'update']);
    });

    it('still recognizes a REAL --models-dir / --help / leading update in option-name position', () => {
      const withDir = scanAgentArgs(['--models-dir', 'x']);
      expect(withDir.modelsDir).toBe('x');
      expect(withDir.passthrough).toEqual([]);
      expect(scanAgentArgs(['--help']).help).toBe(true);
      expect(scanAgentArgs(['update']).update).toBe(true);
    });
  });

  describe('grok one-shot detection (version flags / leading subcommands)', () => {
    // The binary answers these before any model resolution and exits, so they
    // must bypass the inference server, discovery, and the first-run wizard.
    it('detects --version, -v and -V in option-name position, forwarding them verbatim', () => {
      for (const flag of ['--version', '-v', '-V']) {
        const scan = scanAgentArgs([flag]);
        expect(scan.grokOneShot).toBe(true);
        expect(scan.passthrough).toEqual([flag]);
      }
      expect(scanAgentArgs(['-p', '--version']).grokOneShot).toBe(true);
    });

    it('detects every leading binary subcommand except agent as a one-shot', () => {
      for (const command of [
        'models',
        'sessions',
        'export',
        'doctor',
        'mcp',
        'plugin',
        'memory',
        'config',
        'leader',
        'login',
        'logout',
        'setup',
        'share',
        'usage',
        'wrap',
        'trace',
      ]) {
        const scan = scanAgentArgs([command]);
        expect(scan.grokOneShot).toBe(true);
        expect(scan.passthrough).toEqual([command]);
      }
    });

    it('marks update a one-shot too — run() intercepts it before the handoff', () => {
      const scan = scanAgentArgs(['update']);
      expect(scan.update).toBe(true);
      expect(scan.grokOneShot).toBe(true);
    });

    it('does not treat the leading agent (ACP) subcommand as a one-shot', () => {
      const scan = scanAgentArgs(['agent']);
      expect(scan.grokOneShot).toBe(false);
      expect(scan.passthrough).toEqual(['agent']);
    });

    it('treats agent -h/--help as a one-shot — the binary prints subcommand help and exits', () => {
      for (const argv of [['agent', '--help'], ['agent', '-h']]) {
        const scan = scanAgentArgs(argv);
        expect(scan.grokOneShot).toBe(true);
        // Not mlx-level help: the binary prints the agent subcommand's own help.
        expect(scan.help).toBe(false);
        expect(scan.passthrough).toEqual(argv);
      }
    });

    it('detects a subcommand behind a stripped --models-dir pair', () => {
      const scan = scanAgentArgs(['--models-dir', '/x', 'models']);
      expect(scan.grokOneShot).toBe(true);
      expect(scan.passthrough).toEqual(['models']);
    });

    it('does not trip on subcommand words or version flags in non-leading / consumed positions', () => {
      expect(scanAgentArgs(['-p', 'hi', 'export']).grokOneShot).toBe(false);
      const consumed = scanAgentArgs(['--system-prompt-override', '--version']);
      expect(consumed.grokOneShot).toBe(false);
      expect(consumed.passthrough).toEqual(['--system-prompt-override', '--version']);
      const promptValue = scanAgentArgs(['-p', 'models']);
      expect(promptValue.grokOneShot).toBe(false);
      expect(promptValue.passthrough).toEqual(['-p', 'models']);
    });
  });
});

describe('configureAgentTracing', () => {
  async function withTempHome(fn: (home: string) => Promise<void> | void): Promise<void> {
    const home = await mkdtemp(join(tmpdir(), 'mlx-agent-trace-'));
    try {
      await fn(home);
    } finally {
      await rm(home, { recursive: true, force: true });
    }
  }

  it('does nothing when the CLI does not request diagnostics', () => {
    const env: NodeJS.ProcessEnv = {};
    const announcements: string[] = [];
    expect(
      configureAgentTracing({ trace: false }, { env, announce: (line) => announcements.push(line) }),
    ).toBeUndefined();
    expect(env.MLX_NODE_LOG).toBeUndefined();
    expect(env.MLX_NODE_LOG_FILE).toBeUndefined();
    expect(announcements).toEqual([]);
  });

  it('creates a private deterministic per-run file for --trace and announces its exact path once', async () => {
    await withTempHome((home) => {
      const env: NodeJS.ProcessEnv = {};
      const announcements: string[] = [];
      const logFile = configureAgentTracing(
        { trace: true },
        {
          env,
          homeDir: home,
          now: new Date('2026-07-14T01:02:03.456Z'),
          pid: 4242,
          announce: (line) => announcements.push(line),
        },
      );
      const expectedDir = join(home, '.mlx-node', 'logs', 'agent', '2026-07-14T01-02-03-456Z-pid-4242');
      const expectedFile = join(expectedDir, 'inference.log');
      expect(logFile).toBe(expectedFile);
      expect(env.MLX_NODE_LOG).toBe('mlx_core::inference=info,mlx_core::decode=info');
      expect(env.MLX_NODE_LOG_FILE).toBe(expectedFile);
      expect(statSync(expectedDir).mode & 0o777).toBe(0o700);
      expect(statSync(expectedFile).mode & 0o777).toBe(0o600);
      expect(announcements).toEqual([`mlx agent: inference log ${expectedFile}`]);
    });
  });

  it('uses an explicit --trace-dir and enables tracing even without a separate --trace', async () => {
    await withTempHome((home) => {
      const traceDir = join(home, 'chosen');
      const env: NodeJS.ProcessEnv = {};
      const logFile = configureAgentTracing({ trace: false, traceDir }, { env, announce: () => {} });
      expect(logFile).toBe(join(traceDir, 'inference.log'));
      expect(env.MLX_NODE_LOG).toBe('mlx_core::inference=info,mlx_core::decode=info');
      expect(env.MLX_NODE_LOG_FILE).toBe(logFile);
      expect(statSync(traceDir).mode & 0o777).toBe(0o700);
      expect(statSync(logFile!).mode & 0o777).toBe(0o600);
    });
  });

  it('preserves an explicit MLX_NODE_LOG filter', async () => {
    await withTempHome((home) => {
      const env: NodeJS.ProcessEnv = { MLX_NODE_LOG: 'mlx_core::inference=info' };
      const logFile = configureAgentTracing(
        { trace: true },
        {
          env,
          homeDir: home,
          now: new Date('2026-07-14T02:00:00.000Z'),
          pid: 7,
          announce: () => {},
        },
      );
      expect(logFile).toBe(join(home, '.mlx-node', 'logs', 'agent', '2026-07-14T02-00-00-000Z-pid-7', 'inference.log'));
      expect(env.MLX_NODE_LOG).toBe('mlx_core::inference=info');
      expect(env.MLX_NODE_LOG_FILE).toBe(logFile);
    });
  });

  it('preserves an explicit MLX_NODE_LOG_FILE over --trace-dir', async () => {
    await withTempHome((home) => {
      const explicit = join(home, 'explicit', 'native.log');
      const explicitEnvValue = `  ${explicit}  `;
      const env: NodeJS.ProcessEnv = { MLX_NODE_LOG_FILE: explicitEnvValue };
      const announcements: string[] = [];
      const logFile = configureAgentTracing(
        { trace: true, traceDir: join(home, 'ignored') },
        { env, announce: (line) => announcements.push(line) },
      );
      expect(logFile).toBe(explicit);
      expect(env.MLX_NODE_LOG_FILE).toBe(explicitEnvValue);
      expect(statSync(explicit).mode & 0o777).toBe(0o600);
      expect(announcements).toEqual([`mlx agent: inference log ${explicit}`]);
    });
  });
});

describe('withDefaultModel', () => {
  it('prepends -m <id> to a fresh run', () => {
    expect(withDefaultModel(['-p', 'hi'], 'qwen3.5-0.8b-mlx-bf16')).toEqual([
      '-m',
      'qwen3.5-0.8b-mlx-bf16',
      '-p',
      'hi',
    ]);
    expect(withDefaultModel([], 'some-model')).toEqual(['-m', 'some-model']);
  });

  it('suppresses injection for every model/session carrier', () => {
    for (const argv of [
      ['-m', 'other-model'],
      ['--model', 'other-model'],
      ['-c'],
      ['--continue'],
      ['-r', 'abc'],
      ['--resume', 'abc'],
      ['-s', 'abc'],
      ['--session-id', 'abc'],
      ['--fork-session'],
      ['--load', 'path/to/session'],
    ]) {
      expect(withDefaultModel([...argv, '-p', 'hi'], 'default-model')).toEqual([...argv, '-p', 'hi']);
    }
  });

  it('does not treat prompt text as a flag', () => {
    const argv = ['-p', 'please run --continue for me'];
    expect(withDefaultModel(argv, 'm')).toEqual(['-m', 'm', '-p', 'please run --continue for me']);
  });

  it('stops the carrier scan at the -- terminator', () => {
    expect(withDefaultModel(['--', '--model', 'x'], 'm')).toEqual(['-m', 'm', '--', '--model', 'x']);
  });

  describe('value-aware scan: a sentinel consumed as a VALUE must not suppress injection', () => {
    it('injects a local model when --model is the VALUE of --system-prompt-override (the leak this fix closes)', () => {
      // grok sets the prompt override to "--model" and leaves parsed.model
      // UNSET, so without a local injection the run lands on no explicit model.
      // A raw membership scan that saw the `--model` token would wrongly
      // forward this unchanged. The value-aware scan skips `--model` (it is
      // --system-prompt-override's value) → injects.
      // Mutation guard: reverting to `passthrough.some(CARRIER.has)` makes this
      // expect the unchanged argv and the test fails.
      expect(withDefaultModel(['--system-prompt-override', '--model', '-p', 'hi'], 'd')).toEqual([
        '-m',
        'd',
        '--system-prompt-override',
        '--model',
        '-p',
        'hi',
      ]);
    });

    it('injects a local model when a carrier (-c) is consumed as the VALUE of --client-identifier', () => {
      // --client-identifier consumes `-c` as its value (grok: args[++i]); the
      // benign reverse direction — the leftover run is a plain fresh run →
      // concrete -m.
      expect(withDefaultModel(['--client-identifier', '-c'], 'd')).toEqual(['-m', 'd', '--client-identifier', '-c']);
    });

    it('still classifies a REAL option name after its consumer sentinel skips its own value', () => {
      // A real `--session-id foo` (consumer + carrier) still suppresses
      // injection, and a real explicit `--model x` keeps its concrete
      // selection untouched. The sentinel classifies AND skips its value in
      // one pass.
      expect(withDefaultModel(['--session-id', 'foo'], 'd')).toEqual(['--session-id', 'foo']);
      expect(withDefaultModel(['--session-id', 'foo', '-c'], 'd')).toEqual(['--session-id', 'foo', '-c']);
      const explicit = ['--model', 'x', '-p', 'hi'];
      expect(withDefaultModel(explicit, 'd')).toEqual(explicit);
    });
  });
});

/**
 * Shared agent-dir expansion (re-exported from `@mlx-node/agent/paths`):
 * lone `~` and leading `~/` expand, `file://` URLs resolve, `~user` and
 * everything else pass verbatim.
 */
describe('expandAgentDir', () => {
  it('expands a lone ~ and a leading ~/ against the home dir', () => {
    expect(expandAgentDir('~', '/home/u')).toBe('/home/u');
    expect(expandAgentDir('~/tilde-agent', '/home/u')).toBe(join('/home/u', 'tilde-agent'));
    expect(expandAgentDir('~/a b/agent', '/home/u')).toBe(join('/home/u', 'a b/agent'));
  });

  it('does NOT expand ~user and passes other values verbatim', () => {
    expect(expandAgentDir('~user/agent', '/home/u')).toBe('~user/agent');
    expect(expandAgentDir('/abs/agent', '/home/u')).toBe('/abs/agent');
    expect(expandAgentDir('relative/agent', '/home/u')).toBe('relative/agent');
    // No trim: normalizePath default options leave whitespace alone, so a
    // padded value stays a literal (weird) path — parity over polish.
    expect(expandAgentDir(' ~/padded', '/home/u')).toBe(' ~/padded');
  });

  it('resolves file:// URLs', () => {
    expect(expandAgentDir('file:///abs/agent', '/home/u')).toBe('/abs/agent');
  });
});

/**
 * End-to-end argv ROUTING through `run()`: leading grok subcommands are
 * one-shots that run inside the binary, so they must reach `launch` verbatim
 * with needsServer:false — no `-m` injection ahead of them and no first-run
 * wizard. Everything else takes the discover → (wizard) → inject → launch
 * path.
 */
describe('run() argv routing', () => {
  function fakeModel(name: string): { name: string } {
    return { name };
  }

  /**
   * Injected fakes for run(): `discoverBatches[i]` is the result of the
   * i-th discovery call (last batch repeats). Records every call.
   */
  function makeDeps(discoverBatches: { name: string }[][] = [[fakeModel('fake-model')]], launchExit = 0) {
    const calls = {
      discover: [] as string[],
      wizard: [] as string[],
      launch: [] as GrokAgentLaunchOptions[],
    };
    const deps: AgentRunDeps = {
      resolveModelsDir: (explicit) => explicit ?? '/fake/models',
      discoverModels: (modelsDir) => {
        calls.discover.push(modelsDir);
        return Promise.resolve(discoverBatches[Math.min(calls.discover.length - 1, discoverBatches.length - 1)]!);
      },
      launch: (opts) => {
        calls.launch.push(opts);
        return Promise.resolve(launchExit);
      },
      wizard: (modelsDir) => {
        calls.wizard.push(modelsDir);
        return Promise.resolve();
      },
      // Hermetic default: never read the developer's real ~/.mlx-agent.
      readPersistedDefault: () => undefined,
    };
    return { deps, calls };
  }

  it('forwards each grok one-shot verbatim with no server, no discovery, no wizard', async () => {
    for (const argv of [
      ['--version'],
      ['-v'],
      ['-V'],
      ['models'],
      ['sessions'],
      ['export'],
      ['doctor'],
      ['config', '--list'],
      ['mcp'],
    ]) {
      const { deps, calls } = makeDeps([[]]);
      await run(argv, deps);
      expect(calls.launch).toHaveLength(1);
      expect(calls.launch[0]!.argv).toEqual(argv);
      expect(calls.launch[0]!.needsServer).toBe(false);
      expect(calls.launch[0]!.modelsDir).toBeUndefined();
      expect(calls.discover).toHaveLength(0);
      expect(calls.wizard).toHaveLength(0);
    }
  });

  it('routes the agent (ACP) subcommand through the normal model path', async () => {
    const { deps, calls } = makeDeps();
    await run(['agent'], deps);
    expect(calls.discover).toHaveLength(1);
    expect(calls.wizard).toHaveLength(0);
    expect(calls.launch).toHaveLength(1);
    expect(calls.launch[0]!.argv).toEqual(['-m', 'fake-model', 'agent']);
    expect(calls.launch[0]!.needsServer).toBe(true);
    expect(calls.launch[0]!.modelsDir).toBe('/fake/models');
  });

  it('routes a one-shot subcommand behind a stripped --models-dir pair', async () => {
    const { deps, calls } = makeDeps();
    await run(['--models-dir', '/x', 'models'], deps);
    expect(calls.launch).toHaveLength(1);
    expect(calls.launch[0]!.argv).toEqual(['models']);
    expect(calls.launch[0]!.needsServer).toBe(false);
    expect(calls.discover).toHaveLength(0);
  });

  it('exits 1 on a valueless --models-dir instead of consuming the next flag', async () => {
    for (const argv of [
      ['models', '--models-dir', '--local'],
      ['--models-dir', '--help'],
      ['--models-dir', '--no-persist-cache', '-p', 'hi'],
      ['--models-dir'],
      ['--models-dir', ''],
    ]) {
      const { deps, calls } = makeDeps();
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      const prevExitCode = process.exitCode;
      try {
        await run(argv, deps);
        expect(process.exitCode).toBe(1);
        expect(errorSpy.mock.calls.flat().join('\n')).toContain('Missing value for --models-dir');
      } finally {
        process.exitCode = prevExitCode;
        errorSpy.mockRestore();
      }
      // Nothing ran: no binary handoff (help or otherwise), no discovery, no wizard.
      expect(calls.launch).toHaveLength(0);
      expect(calls.discover).toHaveLength(0);
      expect(calls.wizard).toHaveLength(0);
    }
  });

  it('exits 1 on a valueless --trace-dir before importing or handing off to the agent', async () => {
    for (const argv of [['--trace-dir'], ['--trace-dir='], ['--trace-dir', ''], ['--trace-dir', '--help']]) {
      const { deps, calls } = makeDeps();
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      const prevExitCode = process.exitCode;
      try {
        await run(argv, deps);
        expect(process.exitCode).toBe(1);
        expect(errorSpy.mock.calls.flat().join('\n')).toContain('Missing value for --trace-dir');
      } finally {
        process.exitCode = prevExitCode;
        errorSpy.mockRestore();
      }
      expect(calls.launch).toHaveLength(0);
      expect(calls.discover).toHaveLength(0);
      expect(calls.wizard).toHaveLength(0);
    }
  });

  it('documents the mlx flags and the grok environment in the help preamble', async () => {
    const { deps, calls } = makeDeps();
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const prevExitCode = process.exitCode;
    try {
      await run(['--help'], deps);
      const output = logSpy.mock.calls.flat().join('\n');
      expect(output).toContain('--trace');
      expect(output).toContain('--trace-dir <dir>');
      expect(output).toContain('--no-persist-cache');
      expect(output).toContain('MLX_NODE_LOG');
      expect(output).toContain('MLX_NODE_LOG_FILE');
      expect(output).toContain('MLX_AGENT_BIN');
      expect(output).toContain('MLX_CLI_ENTRY');
      expect(output).toContain('~/.mlx-agent');
      expect(output).toContain('/v1/messages');
      expect(output).toContain('mlx agent update');
    } finally {
      process.exitCode = prevExitCode;
      logSpy.mockRestore();
    }
    expect(calls.launch).toHaveLength(1);
    expect(calls.launch[0]!.argv).toEqual(['--help']);
    expect(calls.launch[0]!.needsServer).toBe(false);
  });

  it('configures Rust tracing and strips mlx trace flags before the launch', async () => {
    const traceRoot = await mkdtemp(join(tmpdir(), 'mlx-agent-run-trace-'));
    const logDir = join(traceRoot, 'logs');
    const previousFilter = process.env.MLX_NODE_LOG;
    const previousFile = process.env.MLX_NODE_LOG_FILE;
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const prevExitCode = process.exitCode;
    try {
      delete process.env.MLX_NODE_LOG;
      delete process.env.MLX_NODE_LOG_FILE;
      const { deps, calls } = makeDeps();
      const baseResolve = deps.resolveModelsDir!;
      deps.resolveModelsDir = (explicit) => {
        // This dependency is resolved after the tracing setup, so observing
        // both values here guards the required ordering.
        expect(process.env.MLX_NODE_LOG).toBe('mlx_core::inference=info,mlx_core::decode=info');
        expect(process.env.MLX_NODE_LOG_FILE).toBe(join(logDir, 'inference.log'));
        return baseResolve(explicit);
      };

      await run(['--trace-dir', logDir, '-p', 'hi'], deps);

      expect(calls.launch).toHaveLength(1);
      expect(calls.launch[0]!.argv).toEqual(['-m', 'fake-model', '-p', 'hi']);
      expect(errorSpy.mock.calls.flat().join('\n')).toContain(
        `mlx agent: inference log ${join(logDir, 'inference.log')}`,
      );
    } finally {
      process.exitCode = prevExitCode;
      if (previousFilter === undefined) delete process.env.MLX_NODE_LOG;
      else process.env.MLX_NODE_LOG = previousFilter;
      if (previousFile === undefined) delete process.env.MLX_NODE_LOG_FILE;
      else process.env.MLX_NODE_LOG_FILE = previousFile;
      errorSpy.mockRestore();
      await rm(traceRoot, { recursive: true, force: true });
    }
  });

  it('routes update to the mlx-owned updater before anything else runs', async () => {
    const { deps, calls } = makeDeps();
    const applyUpdate = vi.fn(async () => ({
      downloaded: false,
      status: {
        current: 'mlx-agent-v0.2.0',
        latest: 'mlx-agent-v0.2.0',
        updateAvailable: false,
        managedPath: '/m/mlx-agent',
        pinned: 'managed' as const,
        newestInstalled: null,
      },
    }));
    const prevExitCode = process.exitCode;
    try {
      await run(['update'], { ...deps, applyUpdate });
      expect(applyUpdate).toHaveBeenCalledOnce();
      expect(process.exitCode).toBe(0);
    } finally {
      process.exitCode = prevExitCode;
    }
    expect(calls.launch).toHaveLength(0);
    expect(calls.discover).toHaveLength(0);
    expect(calls.wizard).toHaveLength(0);
  });

  it('still routes update to the updater even when --version rides along', async () => {
    const { deps, calls } = makeDeps();
    const applyUpdate = vi.fn(async () => ({
      downloaded: true,
      status: {
        current: 'mlx-agent-v0.3.0',
        latest: 'mlx-agent-v0.3.0',
        updateAvailable: false,
        managedPath: '/m/mlx-agent',
        pinned: 'managed' as const,
        newestInstalled: null,
      },
    }));
    const prevExitCode = process.exitCode;
    try {
      await run(['update', '--version'], { ...deps, applyUpdate });
      expect(applyUpdate).toHaveBeenCalledOnce();
      expect(process.exitCode).toBe(0);
    } finally {
      process.exitCode = prevExitCode;
    }
    expect(calls.launch).toHaveLength(0);
  });

  it('injects -m <id> on a fresh agent run', async () => {
    const { deps, calls } = makeDeps();
    await run(['-p', 'hi'], deps);
    expect(calls.discover).toHaveLength(1);
    expect(calls.wizard).toHaveLength(0);
    expect(calls.launch).toHaveLength(1);
    expect(calls.launch[0]!.argv).toEqual(['-m', 'fake-model', '-p', 'hi']);
    expect(calls.launch[0]!.modelsDir).toBe('/fake/models');
    expect(calls.launch[0]!.needsServer).toBe(true);
  });

  it('keeps the server path when --help is the VALUE of --system-prompt-override', async () => {
    // The launcher's one-shot decision must come from this routing, not a
    // raw argv rescan: a `--help` in a value slot is data for the binary.
    const { deps, calls } = makeDeps();
    await run(['--system-prompt-override', '--help', '-p', 'hi'], deps);
    expect(calls.discover).toHaveLength(1);
    expect(calls.launch).toHaveLength(1);
    expect(calls.launch[0]!.argv).toEqual(['-m', 'fake-model', '--system-prompt-override', '--help', '-p', 'hi']);
    expect(calls.launch[0]!.needsServer).toBe(true);
  });

  it('forwards --models-dir to the launch options, stripped from argv', async () => {
    const { deps, calls } = makeDeps();
    await run(['--models-dir', '/x', '-p', 'hi'], deps);
    expect(calls.discover).toEqual(['/x']);
    expect(calls.launch[0]!.modelsDir).toBe('/x');
    expect(calls.launch[0]!.argv).toEqual(['-m', 'fake-model', '-p', 'hi']);
  });

  it('keeps an explicit --model untouched (no injection)', async () => {
    const { deps, calls } = makeDeps();
    await run(['--model', 'other', '-p', 'hi'], deps);
    expect(calls.launch).toHaveLength(1);
    expect(calls.launch[0]!.argv).toEqual(['--model', 'other', '-p', 'hi']);
  });

  it('suppresses -m injection for session-carrying runs (the binary restores the saved model)', async () => {
    for (const argv of [
      ['-c'],
      ['--continue'],
      ['-r', 'abc'],
      ['--resume', 'abc'],
      ['-s', 'abc'],
      ['--session-id', 'abc'],
      ['--fork-session'],
      ['--load', 'sess'],
    ]) {
      const { deps, calls } = makeDeps();
      await run(argv, deps);
      expect(calls.launch).toHaveLength(1);
      expect(calls.launch[0]!.argv).toEqual(argv);
    }
  });

  it('still runs the wizard for a fresh agent run with no models, then injects the downloaded one', async () => {
    const { deps, calls } = makeDeps([[], [fakeModel('downloaded-model')]]);
    await run(['-p', 'hi'], deps);
    expect(calls.wizard).toHaveLength(1);
    expect(calls.discover).toHaveLength(2);
    expect(calls.launch).toHaveLength(1);
    expect(calls.launch[0]!.argv).toEqual(['-m', 'downloaded-model', '-p', 'hi']);
  });

  it('keeps non-subcommand flags on the wizard path with zero models', async () => {
    const { deps, calls } = makeDeps([[], [fakeModel('downloaded-model')]]);
    await run(['--effort', 'high'], deps);
    expect(calls.wizard).toHaveLength(1);
    expect(calls.launch).toHaveLength(1);
    expect(calls.launch[0]!.argv).toEqual(['-m', 'downloaded-model', '--effort', 'high']);
  });

  it('exits 1 when the wizard throws instead of launching', async () => {
    const { deps, calls } = makeDeps([[]]);
    deps.wizard = () => Promise.reject(new Error('Install a local model first'));
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const prevExitCode = process.exitCode;
    try {
      await run(['-p', 'hi'], deps);
      expect(process.exitCode).toBe(1);
      expect(errorSpy.mock.calls.flat().join('\n')).toContain('Install a local model first');
    } finally {
      process.exitCode = prevExitCode;
      errorSpy.mockRestore();
    }
    expect(calls.launch).toHaveLength(0);
  });

  it('exits 1 when no usable model remains after the wizard', async () => {
    const { deps, calls } = makeDeps([[], []]);
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const prevExitCode = process.exitCode;
    try {
      await run(['-p', 'hi'], deps);
      expect(process.exitCode).toBe(1);
      expect(errorSpy.mock.calls.flat().join('\n')).toContain('No usable model found');
    } finally {
      process.exitCode = prevExitCode;
      errorSpy.mockRestore();
    }
    expect(calls.launch).toHaveLength(0);
  });

  it('maps the launch return value to process.exitCode', async () => {
    const prevExitCode = process.exitCode;
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    try {
      const fresh = makeDeps([[fakeModel('m')]], 17);
      await run(['-p', 'hi'], fresh.deps);
      expect(process.exitCode).toBe(17);

      const oneShot = makeDeps([[]], 3);
      await run(['models'], oneShot.deps);
      expect(process.exitCode).toBe(3);

      const help = makeDeps([[]], 2);
      await run(['--help'], help.deps);
      expect(process.exitCode).toBe(2);
    } finally {
      process.exitCode = prevExitCode;
      logSpy.mockRestore();
    }
  });

  /**
   * Fresh-run injection vs grok's persisted `/model` default
   * (`<agentDir>/config.toml` → `default = "…"`, written by the binary's
   * model picker): a still-discovered persisted pick wins over the first
   * discovered model; a vanished pick falls back to it.
   */
  describe('persisted default model (config.toml)', () => {
    async function withTempAgentDir(
      configToml: string | undefined,
      fn: (agentDir: string) => Promise<void> | void,
    ): Promise<void> {
      const agentDir = await mkdtemp(join(tmpdir(), 'mlx-agent-config-'));
      try {
        if (configToml !== undefined) {
          await writeFile(join(agentDir, 'config.toml'), configToml);
        }
        await fn(agentDir);
      } finally {
        await rm(agentDir, { recursive: true, force: true });
      }
    }

    const twoModels = () => [[fakeModel('model-a'), fakeModel('model-b')]];

    it('prepends a persisted default that is still discovered', async () => {
      await withTempAgentDir('[models]\ndefault = "model-b"\n', async (agentDir) => {
        const { deps, calls } = makeDeps(twoModels());
        deps.readPersistedDefault = () => readPersistedDefaultModel(agentDir);
        await run(['-p', 'hi'], deps);
        expect(calls.launch[0]!.argv).toEqual(['-m', 'model-b', '-p', 'hi']);
      });
    });

    it('falls back to the first discovered model when the persisted default is gone', async () => {
      await withTempAgentDir('[models]\ndefault = "deleted-model"\n', async (agentDir) => {
        const { deps, calls } = makeDeps(twoModels());
        deps.readPersistedDefault = () => readPersistedDefaultModel(agentDir);
        await run(['-p', 'hi'], deps);
        expect(calls.launch[0]!.argv).toEqual(['-m', 'model-a', '-p', 'hi']);
      });
    });

    it('reads default = "…" from any section (minimal line scan, no TOML parser)', async () => {
      // Documented contract: the reader regex-scans for the first `default`
      // key anywhere in the file — grok only writes it under [models].
      const cases: Array<[string | undefined, { modelId: string } | undefined]> = [
        [undefined, undefined],
        ['[models]\nother = "x"\n', undefined],
        ['default = 42\n', undefined],
        ['default = "x"\n', { modelId: 'x' }],
        ['[server]\ndefault = "top"\n[models]\ndefault = "bottom"\n', { modelId: 'top' }],
      ];
      for (const [configToml, expected] of cases) {
        await withTempAgentDir(configToml, (agentDir) => {
          expect(readPersistedDefaultModel(agentDir)).toEqual(expected);
        });
      }
    });

    it('resolves the agent dir from GROK_HOME when no dir is passed', async () => {
      await withTempAgentDir('default = "model-b"\n', (agentDir) => {
        const prev = process.env.GROK_HOME;
        process.env.GROK_HOME = agentDir;
        try {
          expect(readPersistedDefaultModel()).toEqual({ modelId: 'model-b' });
        } finally {
          if (prev === undefined) {
            delete process.env.GROK_HOME;
          } else {
            process.env.GROK_HOME = prev;
          }
        }
      });
    });

    it('never throws on an unreadable agent dir (failure = no default)', () => {
      expect(readPersistedDefaultModel(join(tmpdir(), 'definitely-missing-mlx-agent-dir'))).toBeUndefined();
    });

    it('chooseDefaultModel is pure over the policy branches', () => {
      const models = ['model-a', 'model-b'];
      expect(chooseDefaultModel(models, undefined)).toEqual({ modelId: 'model-a' });
      expect(chooseDefaultModel(models, { modelId: 'model-b' })).toEqual({ modelId: 'model-b' });
      expect(chooseDefaultModel(models, { modelId: 'gone' })).toEqual({ modelId: 'model-a' });
    });
  });
});
