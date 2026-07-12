import { readFileSync } from 'node:fs';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { basename, join } from 'node:path';

import type { MlxModelInfo } from '@mlx-node/agent';
import { describe, expect, it, vi } from 'vite-plus/test';

import {
  type AgentRunDeps,
  chooseDefaultModel,
  expandPiAgentDir,
  readPersistedDefaultModel,
  run,
  scanAgentArgs,
  withDefaultModel,
  writePersistedDefaultModel,
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
      const scan = scanAgentArgs(['-p', 'hi', '--mode', 'json', '--models-dir', '/models']);
      expect(scan.modelsDir).toBe('/models');
      expect(scan.passthrough).toEqual(['-p', 'hi', '--mode', 'json']);
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
      for (const nextFlag of ['--local', '--help', '--no-session', '-c']) {
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

    it('detects update behind a stripped --models-dir pair (pi would see it at args[0])', () => {
      const scan = scanAgentArgs(['--models-dir', '/x', 'update']);
      expect(scan.update).toBe(true);
      expect(scan.passthrough).toEqual(['update']);
    });
  });

  describe('help detection', () => {
    it('detects -h and --help', () => {
      expect(scanAgentArgs(['-h']).help).toBe(true);
      expect(scanAgentArgs(['--help']).help).toBe(true);
      expect(scanAgentArgs(['--mode', 'json', '--help']).help).toBe(true);
    });

    it('leaves per-command help to pi (install/remove/uninstall/list/config pass through)', () => {
      for (const command of ['install', 'remove', 'uninstall', 'list', 'config']) {
        const scan = scanAgentArgs([command, '--help']);
        expect(scan.help).toBe(false);
        expect(scan.passthrough).toEqual([command, '--help']);
      }
    });

    it('suppresses mlx help for a pass-through command behind --models-dir too', () => {
      const scan = scanAgentArgs(['--models-dir', '/x', 'install', '--help']);
      expect(scan.help).toBe(false);
      expect(scan.passthrough).toEqual(['install', '--help']);
    });

    it('does not detect help when absent', () => {
      expect(scanAgentArgs(['-p', 'hello']).help).toBe(false);
    });
  });

  describe('passthrough preservation', () => {
    it('passes install through untouched', () => {
      const scan = scanAgentArgs(['install', 'npm:some-extension']);
      expect(scan.update).toBe(false);
      expect(scan.help).toBe(false);
      expect(scan.passthrough).toEqual(['install', 'npm:some-extension']);
    });

    it('passes -c, --resume and unknown flags through untouched, in order', () => {
      const argv = ['-c', '--resume', 'abc123', '--totally-unknown-flag', 'value', '-p', 'prompt text'];
      const scan = scanAgentArgs(argv);
      expect(scan.passthrough).toEqual(argv);
      expect(scan.modelsDir).toBeUndefined();
      expect(scan.help).toBe(false);
      expect(scan.update).toBe(false);
    });

    it('returns empty passthrough for empty argv', () => {
      const scan = scanAgentArgs([]);
      expect(scan.passthrough).toEqual([]);
      expect(scan.help).toBe(false);
      expect(scan.update).toBe(false);
    });
  });

  describe('value-aware walk shares VALUE_CONSUMING_ARGS (WB-5, sibling of R3-2)', () => {
    // The scan must NOT hijack a token that sits in a pi value-consumer's value
    // slot. Mutation guard: reverting scanAgentArgs to the raw exact-token scan
    // strips the systemPrompt value + interprets a value `--help`, failing these.
    it('does not strip a --models-dir that is the VALUE of --system-prompt', () => {
      const scan = scanAgentArgs(['--system-prompt', '--models-dir', 'x']);
      expect(scan.modelsDir).toBeUndefined();
      expect(scan.modelsDirMissingValue).toBe(false);
      expect(scan.passthrough).toEqual(['--system-prompt', '--models-dir', 'x']);
    });

    it('does not route to help for a --help that is the VALUE of --system-prompt', () => {
      const scan = scanAgentArgs(['--system-prompt', '--help', '-p', 'hi']);
      expect(scan.help).toBe(false);
      expect(scan.passthrough).toEqual(['--system-prompt', '--help', '-p', 'hi']);
    });

    it('does not treat update as the blocked positional when it is a consumed VALUE', () => {
      const scan = scanAgentArgs(['--system-prompt', 'update']);
      expect(scan.update).toBe(false);
      expect(scan.passthrough).toEqual(['--system-prompt', 'update']);
    });

    it('still recognizes a REAL --models-dir / --help / leading update in option-name position', () => {
      const withDir = scanAgentArgs(['--models-dir', 'x']);
      expect(withDir.modelsDir).toBe('x');
      expect(withDir.passthrough).toEqual([]);
      expect(scanAgentArgs(['--help']).help).toBe(true);
      expect(scanAgentArgs(['update']).update).toBe(true);
    });
  });

  describe('pi one-shot detection (--version / -v / --export)', () => {
    // pi answers these before any model resolution (main.js:404-407 prints
    // VERSION, :408-421 exports, both process.exit before session creation),
    // so they must bypass discovery + the first-run wizard.
    it('detects --version and -v in option-name position, forwarding them verbatim', () => {
      for (const flag of ['--version', '-v']) {
        const scan = scanAgentArgs([flag]);
        expect(scan.piOneShot).toBe(true);
        expect(scan.passthrough).toEqual([flag]);
      }
      expect(scanAgentArgs(['--mode', 'json', '--version']).piOneShot).toBe(true);
    });

    it('detects --export only when it consumed a value', () => {
      const scan = scanAgentArgs(['--export', 'session.jsonl']);
      expect(scan.piOneShot).toBe(true);
      expect(scan.passthrough).toEqual(['--export', 'session.jsonl']);
    });

    it('does not treat a trailing bare --export as a one-shot (pi sees an unknown flag)', () => {
      const scan = scanAgentArgs(['--export']);
      expect(scan.piOneShot).toBe(false);
      expect(scan.passthrough).toEqual(['--export']);
    });

    it('does not treat an empty --export value as a one-shot (pi stores "" but skips the export path)', () => {
      // pi consumes '' as the value (args.js: `result.export = args[++i]`),
      // then `if (parsed.export)` at main.js:408 is FALSY for '' — no
      // export-and-exit, pi falls into normal session startup. Bypassing the
      // wizard here would hand pi zero models and end in a silent no-op.
      const scan = scanAgentArgs(['--export', '']);
      expect(scan.piOneShot).toBe(false);
      expect(scan.passthrough).toEqual(['--export', '']);
    });

    it('does not trip on a --version consumed as the VALUE of --system-prompt', () => {
      const scan = scanAgentArgs(['--system-prompt', '--version']);
      expect(scan.piOneShot).toBe(false);
      expect(scan.passthrough).toEqual(['--system-prompt', '--version']);
    });

    it('leaves --list-models undetected — it stays on the wizard path', () => {
      const scan = scanAgentArgs(['--list-models']);
      expect(scan.piOneShot).toBe(false);
      expect(scan.passthrough).toEqual(['--list-models']);
    });
  });
});

describe('withDefaultModel', () => {
  it('prepends --model mlx/<id> to a fresh run', () => {
    expect(withDefaultModel(['-p', 'hi'], 'qwen3.5-0.8b-mlx-bf16')).toEqual([
      '--model',
      'mlx/qwen3.5-0.8b-mlx-bf16',
      '-p',
      'hi',
    ]);
    expect(withDefaultModel([], 'some-model')).toEqual(['--model', 'mlx/some-model']);
  });

  it('respects an explicit --model, --models scope, or --provider', () => {
    const withModel = ['--model', 'mlx/other-model', '-p', 'hi'];
    expect(withDefaultModel(withModel, 'default-model')).toBe(withModel);
    const withScope = ['--models', 'mlx/a,mlx/b'];
    expect(withDefaultModel(withScope, 'default-model')).toBe(withScope);
    const withProvider = ['--provider', 'mlx', '--model', 'x'];
    expect(withDefaultModel(withProvider, 'default-model')).toBe(withProvider);
  });

  it('scopes a provider-only run to mlx/* instead of a bare --model (no bogus cloud model)', () => {
    // Injecting `--model mlx/<default>` next to a user `--provider groq` makes
    // pi resolve a bogus CLOUD `groq/mlx/<default>` custom model
    // (buildFallbackModel), the exact off-machine route the local-first policy
    // fights. A `--models mlx/*` scope carries no provider, so pi cannot mint a
    // cloud model — it forces an mlx model at the CLI-arg layer (above settings
    // + cloud fallback), immune to a seed-write failure or ambient keys.
    const providerOnly = ['--provider', 'groq'];
    expect(withDefaultModel(providerOnly, 'default-model')).toEqual(['--models', 'mlx/*', '--provider', 'groq']);
    expect(withDefaultModel(providerOnly, 'default-model')).not.toContain('--model');
  });

  it('scopes a session-carrying run to mlx/* (restore preserved, cloud default blocked)', () => {
    // --models is a scope, not a model: pi restores an EXISTING session's saved
    // model (options.model stays undefined) and only picks an mlx model for a
    // new / unknown / empty session — never a cloud default. So prepend the
    // scope, never a bare --model that would fight the restore. (--fork is a
    // full suppressor, covered separately below.)
    for (const args of [['-c'], ['--continue'], ['-r'], ['--resume'], ['--session', 'abc'], ['--session-id', 'abc']]) {
      const out = withDefaultModel(args, 'default-model');
      expect(out).toEqual(['--models', 'mlx/*', ...args]);
      expect(out).not.toContain('--model');
    }
  });

  it('leaves a --fork run unchanged — the forked session restores its own model', () => {
    const argv = ['--fork', 'abc123', '-p', 'continue where we left off'];
    expect(withDefaultModel(argv, 'default-model')).toBe(argv);
  });

  it('does not treat prompt text as a flag', () => {
    const argv = ['-p', 'please run --continue for me'];
    expect(withDefaultModel(argv, 'm')).toEqual(['--model', 'mlx/m', '-p', 'please run --continue for me']);
  });

  describe('value-aware scan: a sentinel consumed as a VALUE must not suppress injection', () => {
    it('injects a local model when --model is the VALUE of --system-prompt (the leak this fix closes)', () => {
      // pi sets systemPrompt="--model" and leaves parsed.model UNSET, so without
      // a local injection pi resolves a cloud default. A raw membership scan that
      // saw the `--model` token would wrongly forward this unchanged. The value-
      // aware scan skips `--model` (it is --system-prompt's value) → injects.
      // Mutation guard: reverting to `passthrough.some(FULL_SUPPRESS_ARGS.has)`
      // makes this expect the unchanged argv and the test fails.
      expect(withDefaultModel(['--system-prompt', '--model', '-p', 'hi'], 'd')).toEqual([
        '--model',
        'mlx/d',
        '--system-prompt',
        '--model',
        '-p',
        'hi',
      ]);
    });

    it('injects a local model when --provider is the VALUE of --append-system-prompt', () => {
      expect(withDefaultModel(['--append-system-prompt', '--provider'], 'd')).toEqual([
        '--model',
        'mlx/d',
        '--append-system-prompt',
        '--provider',
      ]);
    });

    it('injects a local model when a carrier (-c) is consumed as the VALUE of --name', () => {
      // --name consumes `-c` as its value (pi: args[++i]); the benign reverse
      // direction — the leftover run is a plain fresh run → concrete --model.
      expect(withDefaultModel(['--name', '-c'], 'd')).toEqual(['--model', 'mlx/d', '--name', '-c']);
    });

    it('still classifies a REAL option name after its consumer sentinel skips its own value', () => {
      // A real `--session-id foo` (consumer + carrier) still scopes to mlx/*, and
      // a real explicit `--model x` (consumer + full-suppress) still forwards
      // unchanged — the sentinel classifies AND skips its value in one pass.
      expect(withDefaultModel(['--session-id', 'foo'], 'd')).toEqual(['--models', 'mlx/*', '--session-id', 'foo']);
      const explicit = ['--model', 'x', '-p', 'hi'];
      expect(withDefaultModel(explicit, 'd')).toBe(explicit);
    });
  });
});

/**
 * Mirror of pi 0.80.6 `getAgentDir` → `normalizePath` (default options)
 * on the `PI_CODING_AGENT_DIR` value: lone `~` and leading `~/` expand,
 * `file://` URLs resolve, `~user` and everything else pass verbatim.
 */
describe('expandPiAgentDir', () => {
  it('expands a lone ~ and a leading ~/ against the home dir', () => {
    expect(expandPiAgentDir('~', '/home/u')).toBe('/home/u');
    expect(expandPiAgentDir('~/tilde-agent', '/home/u')).toBe(join('/home/u', 'tilde-agent'));
    expect(expandPiAgentDir('~/a b/agent', '/home/u')).toBe(join('/home/u', 'a b/agent'));
  });

  it('does NOT expand ~user (pi does not either) and passes other values verbatim', () => {
    expect(expandPiAgentDir('~user/agent', '/home/u')).toBe('~user/agent');
    expect(expandPiAgentDir('/abs/agent', '/home/u')).toBe('/abs/agent');
    expect(expandPiAgentDir('relative/agent', '/home/u')).toBe('relative/agent');
    // No trim: pi's normalizePath default options leave whitespace alone,
    // so a padded value stays a literal (weird) path — parity over polish.
    expect(expandPiAgentDir(' ~/padded', '/home/u')).toBe(' ~/padded');
  });

  it('resolves file:// URLs like pi', () => {
    expect(expandPiAgentDir('file:///abs/agent', '/home/u')).toBe('/abs/agent');
  });
});

/**
 * End-to-end argv ROUTING through `run()`: pi's `parsePackageCommand`
 * and `handleConfigCommand` read ONLY args[0], so package commands and
 * `config` must reach `runAgent` verbatim — no `--model` injection
 * ahead of them and no first-run wizard.
 */
describe('run() argv routing', () => {
  function fakeModel(name: string): MlxModelInfo {
    return { discovered: { name } } as unknown as MlxModelInfo;
  }

  /**
   * Injected fakes for run(): `discoverBatches[i]` is the result of the
   * i-th discovery call (last batch repeats). Records every call.
   */
  function makeDeps(discoverBatches: MlxModelInfo[][] = [[fakeModel('fake-model')]]) {
    const calls = {
      discover: [] as string[],
      wizard: [] as string[],
      runAgent: [] as Array<{ modelsDir: string; models: MlxModelInfo[]; argv: string[] }>,
      writes: [] as Array<{ provider: string; modelId: string }>,
    };
    const deps: AgentRunDeps = {
      resolveModelsDir: (explicit) => explicit ?? '/fake/models',
      discoverMlxModels: (modelsDir) => {
        calls.discover.push(modelsDir);
        return Promise.resolve(discoverBatches[Math.min(calls.discover.length - 1, discoverBatches.length - 1)]!);
      },
      runAgent: (opts) => {
        calls.runAgent.push({ modelsDir: opts.modelsDir, models: opts.models, argv: opts.argv });
        return Promise.resolve();
      },
      wizard: (modelsDir) => {
        calls.wizard.push(modelsDir);
        return Promise.resolve();
      },
      // Hermetic defaults: never read/write the developer's real settings.json.
      readPersistedDefault: () => undefined,
      writePersistedDefault: (provider, modelId) => {
        calls.writes.push({ provider, modelId });
      },
    };
    return { deps, calls };
  }

  it('hands each package command to pi verbatim at argv[0] — no --model, no discovery, no wizard', async () => {
    for (const command of ['install', 'remove', 'uninstall', 'list']) {
      const { deps, calls } = makeDeps();
      await run([command, 'npm:some-extension'], deps);
      expect(calls.runAgent).toHaveLength(1);
      expect(calls.runAgent[0]!.argv).toEqual([command, 'npm:some-extension']);
      expect(calls.runAgent[0]!.argv).not.toContain('--model');
      expect(calls.runAgent[0]!.models).toEqual([]);
      expect(calls.discover).toHaveLength(0);
      expect(calls.wizard).toHaveLength(0);
    }
  });

  it('hands config to pi verbatim at argv[0] — no --model, no discovery, no wizard', async () => {
    const { deps, calls } = makeDeps();
    await run(['config', '--local'], deps);
    expect(calls.runAgent).toHaveLength(1);
    expect(calls.runAgent[0]!.argv).toEqual(['config', '--local']);
    expect(calls.runAgent[0]!.argv).not.toContain('--model');
    expect(calls.runAgent[0]!.models).toEqual([]);
    expect(calls.discover).toHaveLength(0);
    expect(calls.wizard).toHaveLength(0);
  });

  it('routes a pass-through command without any model present (empty dir, wizard stays out)', async () => {
    for (const argv of [['list'], ['config']]) {
      const { deps, calls } = makeDeps([[]]);
      await run(argv, deps);
      expect(calls.runAgent).toHaveLength(1);
      expect(calls.runAgent[0]!.argv).toEqual(argv);
      expect(calls.wizard).toHaveLength(0);
    }
  });

  it('routes a package command behind a stripped --models-dir pair', async () => {
    const { deps, calls } = makeDeps();
    await run(['--models-dir', '/x', 'install', 'npm:foo'], deps);
    expect(calls.runAgent).toHaveLength(1);
    expect(calls.runAgent[0]!.argv).toEqual(['install', 'npm:foo']);
    expect(calls.runAgent[0]!.modelsDir).toBe('/x');
    expect(calls.discover).toHaveLength(0);
  });

  it('exits 1 on a valueless --models-dir instead of consuming the next flag', async () => {
    for (const argv of [
      ['install', 'npm:x', '--models-dir', '--local'],
      ['--models-dir', '--help'],
      ['--models-dir', '--no-session', '-p', 'hi'],
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
      // Nothing ran: no pi handoff (help or otherwise), no discovery, no wizard.
      expect(calls.runAgent).toHaveLength(0);
      expect(calls.discover).toHaveLength(0);
      expect(calls.wizard).toHaveLength(0);
    }
  });

  it('still blocks update with exit code 1 before anything runs', async () => {
    const { deps, calls } = makeDeps();
    const prevExitCode = process.exitCode;
    try {
      await run(['update'], deps);
      expect(process.exitCode).toBe(1);
    } finally {
      process.exitCode = prevExitCode;
    }
    expect(calls.runAgent).toHaveLength(0);
    expect(calls.discover).toHaveLength(0);
    expect(calls.wizard).toHaveLength(0);
  });

  it('still injects --model mlx/<id> on a fresh agent run', async () => {
    const { deps, calls } = makeDeps();
    await run(['-p', 'hi'], deps);
    expect(calls.discover).toHaveLength(1);
    expect(calls.wizard).toHaveLength(0);
    expect(calls.runAgent).toHaveLength(1);
    expect(calls.runAgent[0]!.argv).toEqual(['--model', 'mlx/fake-model', '-p', 'hi']);
    expect(calls.runAgent[0]!.models.map((m) => m.discovered.name)).toEqual(['fake-model']);
  });

  it('still skips injection when the run already carries --model', async () => {
    const { deps, calls } = makeDeps();
    await run(['--model', 'mlx/other', '-p', 'hi'], deps);
    expect(calls.runAgent).toHaveLength(1);
    expect(calls.runAgent[0]!.argv).toEqual(['--model', 'mlx/other', '-p', 'hi']);
  });

  it('still runs the wizard for a fresh agent run with no models, then injects the downloaded one', async () => {
    const { deps, calls } = makeDeps([[], [fakeModel('downloaded-model')]]);
    await run(['-p', 'hi'], deps);
    expect(calls.wizard).toHaveLength(1);
    expect(calls.discover).toHaveLength(2);
    expect(calls.runAgent).toHaveLength(1);
    expect(calls.runAgent[0]!.argv).toEqual(['--model', 'mlx/downloaded-model', '-p', 'hi']);
  });

  it('forwards a --fork run unchanged so pi restores the forked session model', async () => {
    const { deps, calls } = makeDeps();
    await run(['--fork', 'abc123', '-p', 'hi'], deps);
    expect(calls.runAgent).toHaveLength(1);
    expect(calls.runAgent[0]!.argv).toEqual(['--fork', 'abc123', '-p', 'hi']);
    expect(calls.runAgent[0]!.argv).not.toContain('--model');
  });

  it('forwards pi one-shots (--version / -v / --export) verbatim — no discovery, no wizard, no injection', async () => {
    // pi prints VERSION (main.js:404-407) or exports (main.js:408-421) and
    // exits BEFORE model resolution, so a fresh install with zero models must
    // not fall into the first-run wizard ahead of them.
    for (const argv of [['--version'], ['-v'], ['--export', 'session.jsonl']]) {
      const { deps, calls } = makeDeps([[]]);
      await run(argv, deps);
      expect(calls.runAgent).toHaveLength(1);
      expect(calls.runAgent[0]!.argv).toEqual(argv);
      expect(calls.runAgent[0]!.models).toEqual([]);
      expect(calls.discover).toHaveLength(0);
      expect(calls.wizard).toHaveLength(0);
    }
  });

  it('still blocks update even when --version rides along', async () => {
    const { deps, calls } = makeDeps();
    const prevExitCode = process.exitCode;
    try {
      await run(['update', '--version'], deps);
      expect(process.exitCode).toBe(1);
    } finally {
      process.exitCode = prevExitCode;
    }
    expect(calls.runAgent).toHaveLength(0);
  });

  it('keeps --list-models on the wizard path with zero models (pi would print a dead-end /login hint)', async () => {
    const { deps, calls } = makeDeps([[], [fakeModel('downloaded-model')]]);
    await run(['--list-models'], deps);
    expect(calls.wizard).toHaveLength(1);
    expect(calls.runAgent).toHaveLength(1);
    expect(calls.runAgent[0]!.argv).toEqual(['--model', 'mlx/downloaded-model', '--list-models']);
  });

  it('keeps an empty --export value on the wizard path with zero models (pi would silently no-op)', async () => {
    const { deps, calls } = makeDeps([[], [fakeModel('downloaded-model')]]);
    await run(['--export', ''], deps);
    expect(calls.wizard).toHaveLength(1);
    expect(calls.runAgent).toHaveLength(1);
    expect(calls.runAgent[0]!.argv).toEqual(['--model', 'mlx/downloaded-model', '--export', '']);
  });

  /**
   * Fresh-run injection vs pi's persisted `/model` default
   * (`<agentDir>/settings.json` → `defaultProvider` + `defaultModel`,
   * written by pi's SettingsManager): a still-discovered mlx default
   * must win over the lexicographic first; a non-mlx default is
   * overridden (local-first policy) with a stderr notice.
   */
  describe('persisted /model default', () => {
    async function withTempSettings(settings: unknown, fn: (agentDir: string) => Promise<void> | void): Promise<void> {
      const agentDir = await mkdtemp(join(tmpdir(), 'mlx-agent-settings-'));
      try {
        if (settings !== undefined) {
          await writeFile(
            join(agentDir, 'settings.json'),
            typeof settings === 'string' ? settings : JSON.stringify(settings),
          );
        }
        await fn(agentDir);
      } finally {
        await rm(agentDir, { recursive: true, force: true });
      }
    }

    const twoModels = () => [[fakeModel('model-a'), fakeModel('model-b')]];

    it('prepends a persisted mlx default that is still discovered', async () => {
      await withTempSettings({ defaultProvider: 'mlx', defaultModel: 'model-b' }, async (agentDir) => {
        const { deps, calls } = makeDeps(twoModels());
        deps.readPersistedDefault = () => readPersistedDefaultModel(agentDir);
        await run(['-p', 'hi'], deps);
        expect(calls.runAgent[0]!.argv).toEqual(['--model', 'mlx/model-b', '-p', 'hi']);
      });
    });

    it('falls back to the first discovered model when the persisted mlx default is gone', async () => {
      await withTempSettings({ defaultProvider: 'mlx', defaultModel: 'deleted-model' }, async (agentDir) => {
        const { deps, calls } = makeDeps(twoModels());
        deps.readPersistedDefault = () => readPersistedDefaultModel(agentDir);
        await run(['-p', 'hi'], deps);
        expect(calls.runAgent[0]!.argv).toEqual(['--model', 'mlx/model-a', '-p', 'hi']);
      });
    });

    it('overrides a persisted non-mlx default with the first local model AND a stderr notice', async () => {
      await withTempSettings({ defaultProvider: 'anthropic', defaultModel: 'claude-x' }, async (agentDir) => {
        const { deps, calls } = makeDeps(twoModels());
        deps.readPersistedDefault = () => readPersistedDefaultModel(agentDir);
        const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
        let notices = '';
        try {
          await run(['-p', 'hi'], deps);
          notices = errorSpy.mock.calls.flat().join('\n');
        } finally {
          errorSpy.mockRestore();
        }
        expect(calls.runAgent[0]!.argv).toEqual(['--model', 'mlx/model-a', '-p', 'hi']);
        expect(notices).toContain('anthropic/claude-x');
        expect(notices).toContain('mlx/model-a');
      });
    });

    it('stays silent about a non-mlx default when no injection happens (--model run)', async () => {
      await withTempSettings({ defaultProvider: 'anthropic', defaultModel: 'claude-x' }, async (agentDir) => {
        const { deps, calls } = makeDeps(twoModels());
        deps.readPersistedDefault = () => readPersistedDefaultModel(agentDir);
        const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
        let errorCallCount = -1;
        try {
          await run(['--model', 'mlx/model-b', '-p', 'hi'], deps);
          errorCallCount = errorSpy.mock.calls.length;
        } finally {
          errorSpy.mockRestore();
        }
        expect(calls.runAgent[0]!.argv).toEqual(['--model', 'mlx/model-b', '-p', 'hi']);
        expect(errorCallCount).toBe(0);
      });
    });

    it('treats a missing or malformed settings file as no persisted default', async () => {
      await withTempSettings(undefined, (agentDir) => {
        expect(readPersistedDefaultModel(agentDir)).toBeUndefined();
      });
      await withTempSettings('{not json', (agentDir) => {
        expect(readPersistedDefaultModel(agentDir)).toBeUndefined();
      });
      await withTempSettings({ defaultProvider: 'mlx' }, (agentDir) => {
        expect(readPersistedDefaultModel(agentDir)).toBeUndefined();
      });
    });

    it('resolves the agent dir from PI_CODING_AGENT_DIR when no dir is passed (runAgent parity)', async () => {
      await withTempSettings({ defaultProvider: 'mlx', defaultModel: 'model-b' }, (agentDir) => {
        const prev = process.env.PI_CODING_AGENT_DIR;
        process.env.PI_CODING_AGENT_DIR = agentDir;
        try {
          expect(readPersistedDefaultModel()).toEqual({ provider: 'mlx', modelId: 'model-b' });
        } finally {
          if (prev === undefined) {
            delete process.env.PI_CODING_AGENT_DIR;
          } else {
            process.env.PI_CODING_AGENT_DIR = prev;
          }
        }
      });
    });

    it('tilde-expands PI_CODING_AGENT_DIR exactly like pi before reading settings.json', async () => {
      // pi expands a leading ~/ in PI_CODING_AGENT_DIR (getAgentDir →
      // normalizePath), so the reader must open settings.json under the
      // REAL home dir — a literal ./~/... would silently miss the user's
      // saved default. Throwaway dir directly under $HOME, no mocks.
      const realHomeDir = await mkdtemp(join(homedir(), '.mlx-agent-tilde-test-'));
      const prev = process.env.PI_CODING_AGENT_DIR;
      try {
        await writeFile(
          join(realHomeDir, 'settings.json'),
          JSON.stringify({ defaultProvider: 'mlx', defaultModel: 'tilde-model' }),
        );
        process.env.PI_CODING_AGENT_DIR = `~/${basename(realHomeDir)}`;
        expect(readPersistedDefaultModel()).toEqual({ provider: 'mlx', modelId: 'tilde-model' });
      } finally {
        if (prev === undefined) {
          delete process.env.PI_CODING_AGENT_DIR;
        } else {
          process.env.PI_CODING_AGENT_DIR = prev;
        }
        await rm(realHomeDir, { recursive: true, force: true });
      }
    });

    it('never throws on an unresolvable PI_CODING_AGENT_DIR (expansion failure = no default)', () => {
      // fileURLToPath rejects a non-localhost host; pi itself would crash
      // on this value, but the reader must stay failure-tolerant.
      const prev = process.env.PI_CODING_AGENT_DIR;
      process.env.PI_CODING_AGENT_DIR = 'file://not-localhost/agent';
      try {
        expect(readPersistedDefaultModel()).toBeUndefined();
      } finally {
        if (prev === undefined) {
          delete process.env.PI_CODING_AGENT_DIR;
        } else {
          process.env.PI_CODING_AGENT_DIR = prev;
        }
      }
    });

    /**
     * FIX-A (write-once persisted mlx default): the session / provider-only
     * launch paths suppress the `--model` injection, so pi's findInitialModel
     * step-3 default is the only slot that keeps them on-machine. `run()` must
     * seed `defaultProvider: mlx` + the mlx default into settings.json when
     * none is persisted — and only then (never over a user `/model` pick).
     */
    it('scopes a suppressed session/provider launch to mlx/* AND seeds the mlx default (belt)', async () => {
      for (const argv of [['-c'], ['--provider', 'groq'], ['--session-id', 'unknown-xyz']]) {
        await withTempSettings(undefined, async (agentDir) => {
          const { deps, calls } = makeDeps(twoModels());
          deps.readPersistedDefault = () => readPersistedDefaultModel(agentDir);
          deps.writePersistedDefault = (provider, modelId) => writePersistedDefaultModel(provider, modelId, agentDir);
          await run(argv, deps);
          // Authoritative guard: the local-only scope is injected ahead of the
          // carrier, never a bare --model (which would mint a cloud model next
          // to --provider).
          expect(calls.runAgent[0]!.argv).toEqual(['--models', 'mlx/*', ...argv]);
          expect(calls.runAgent[0]!.argv).not.toContain('--model');
          // Belt: settings.json also carries the local default (step-3 slot).
          expect(readPersistedDefaultModel(agentDir)).toEqual({ provider: 'mlx', modelId: 'model-a' });
        });
      }
    });

    it('write-once: never overwrites an already-persisted default (or its sibling fields)', async () => {
      await withTempSettings({ defaultProvider: 'mlx', defaultModel: 'model-b', theme: 'dark' }, async (agentDir) => {
        const { deps, calls } = makeDeps(twoModels());
        deps.readPersistedDefault = () => readPersistedDefaultModel(agentDir);
        // Recording writer (not the real one) makes any stray write observable.
        await run(['-c'], deps);
        expect(calls.writes).toHaveLength(0);
        const raw: unknown = JSON.parse(readFileSync(join(agentDir, 'settings.json'), 'utf8'));
        expect(raw).toEqual({ defaultProvider: 'mlx', defaultModel: 'model-b', theme: 'dark' });
      });
    });

    it('preserves unrelated settings fields when seeding the default', async () => {
      await withTempSettings({ theme: 'dark', quietStartup: true }, async (agentDir) => {
        const { deps } = makeDeps(twoModels());
        deps.readPersistedDefault = () => readPersistedDefaultModel(agentDir);
        deps.writePersistedDefault = (provider, modelId) => writePersistedDefaultModel(provider, modelId, agentDir);
        await run(['-c'], deps);
        const raw: unknown = JSON.parse(readFileSync(join(agentDir, 'settings.json'), 'utf8'));
        expect(raw).toEqual({ theme: 'dark', quietStartup: true, defaultProvider: 'mlx', defaultModel: 'model-a' });
      });
    });

    it('does not crash the launch when settings.json is unwritable (scope still applied)', async () => {
      // A seed-write failure must not crash the launch AND must not silently
      // drop the on-machine guard: the CLI-arg injection (`--model` fresh,
      // `--models mlx/*` for carriers) is applied regardless of whether the
      // belt seed persisted.
      const cases: Array<{ argv: string[]; expected: string[] }> = [
        { argv: ['-p', 'hi'], expected: ['--model', 'mlx/fake-model', '-p', 'hi'] },
        { argv: ['-c'], expected: ['--models', 'mlx/*', '-c'] },
        { argv: ['--session-id', 'unknown-xyz'], expected: ['--models', 'mlx/*', '--session-id', 'unknown-xyz'] },
        { argv: ['--provider', 'groq'], expected: ['--models', 'mlx/*', '--provider', 'groq'] },
      ];
      for (const { argv, expected } of cases) {
        await withTempSettings(undefined, async (agentDir) => {
          // Make `<agentDir>/not-a-dir` a FILE, then point the writer under it so
          // mkdirSync/writeFileSync fail with ENOTDIR — the writer must swallow it.
          await writeFile(join(agentDir, 'not-a-dir'), 'x');
          const unwritable = join(agentDir, 'not-a-dir', 'deeper');
          const { deps, calls } = makeDeps();
          deps.readPersistedDefault = () => undefined;
          deps.writePersistedDefault = (provider, modelId) => writePersistedDefaultModel(provider, modelId, unwritable);
          await run(argv, deps);
          expect(calls.runAgent).toHaveLength(1);
          expect(calls.runAgent[0]!.argv).toEqual(expected);
        });
      }
    });

    it('chooseDefaultModel is pure over the three policy branches', () => {
      const models = [fakeModel('model-a'), fakeModel('model-b')];
      expect(chooseDefaultModel(models, undefined)).toEqual({ modelId: 'model-a' });
      expect(chooseDefaultModel(models, { provider: 'mlx', modelId: 'model-b' })).toEqual({ modelId: 'model-b' });
      expect(chooseDefaultModel(models, { provider: 'mlx', modelId: 'gone' })).toEqual({ modelId: 'model-a' });
      const overridden = chooseDefaultModel(models, { provider: 'groq', modelId: 'llama' });
      expect(overridden.modelId).toBe('model-a');
      expect(overridden.notice).toContain('groq/llama');
    });
  });
});

/**
 * WB-3: the write-once seed must never clobber a present-but-recoverable
 * settings.json. It writes ONLY when the file is genuinely absent (ENOENT) or
 * already a valid settings object; a malformed / unreadable / non-object file is
 * left byte-for-byte untouched (pi treats those as load errors, not "start
 * empty"). readPersistedDefaultModel returns undefined for a malformed file, so
 * run() auto-seeds — which is exactly why the writer itself must refuse.
 */
describe('writePersistedDefaultModel (WB-3: never clobber a recoverable settings.json)', () => {
  async function withDir(fn: (dir: string) => Promise<void> | void): Promise<void> {
    const dir = await mkdtemp(join(tmpdir(), 'mlx-agent-write-'));
    try {
      await fn(dir);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }

  it('SKIPS a present-but-malformed settings.json, leaving its bytes untouched', async () => {
    await withDir(async (dir) => {
      const path = join(dir, 'settings.json');
      await writeFile(path, '{ oops');
      // Mutation guard: reverting WB-3 rewrites this to {defaultProvider,defaultModel}.
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      try {
        writePersistedDefaultModel('mlx', 'model-x', dir);
      } finally {
        errorSpy.mockRestore();
      }
      expect(readFileSync(path, 'utf8')).toBe('{ oops');
    });
  });

  it('SKIPS a valid-JSON non-object (array) settings.json, leaving it untouched', async () => {
    await withDir(async (dir) => {
      const path = join(dir, 'settings.json');
      await writeFile(path, '[]');
      writePersistedDefaultModel('mlx', 'model-x', dir);
      expect(readFileSync(path, 'utf8')).toBe('[]');
    });
  });

  it('creates a fresh file with exactly the two fields when settings.json is absent', async () => {
    await withDir((dir) => {
      writePersistedDefaultModel('mlx', 'model-x', dir);
      const raw: unknown = JSON.parse(readFileSync(join(dir, 'settings.json'), 'utf8'));
      expect(raw).toEqual({ defaultProvider: 'mlx', defaultModel: 'model-x' });
    });
  });

  it('merges the two fields into a valid settings object, preserving existing keys', async () => {
    await withDir(async (dir) => {
      const path = join(dir, 'settings.json');
      await writeFile(path, JSON.stringify({ theme: 'dark' }));
      writePersistedDefaultModel('mlx', 'model-x', dir);
      const raw: unknown = JSON.parse(readFileSync(path, 'utf8'));
      expect(raw).toEqual({ theme: 'dark', defaultProvider: 'mlx', defaultModel: 'model-x' });
    });
  });
});
