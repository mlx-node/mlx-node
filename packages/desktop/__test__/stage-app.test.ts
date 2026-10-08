/**
 * What the bundle is allowed to contain.
 *
 * `runtimeClosure` decides, package by package, what gets copied into the
 * shipped app. Its exclusions are not size tuning — they are the difference
 * between a bundle that notarizes and one that does not, or between an app
 * that is mostly itself and one that is mostly other people's cloud SDKs. Both
 * were found by running the release gate or measuring the artifact rather than
 * by reading code:
 *
 *  - `@mlx-node/core-*` is napi's published prebuilt. Staging it shipped the
 *    239 MB native payload a second time (993 MB total) even though nothing
 *    loads it.
 *  - The cloud-LLM provider SDKs behind `@earendil-works/pi-ai` were 114 MB of
 *    an app that exists to run models locally, reachable only from a
 *    `stream()` call this app never makes.
 *
 * The exclusions are silent by nature: the app still builds, still launches,
 * and still passes its own tests with any one wrong. Only a gate, a notary, or
 * `du` says otherwise, and all of those are minutes-to-hours away from the
 * edit. So the rules are pinned here, at the point where they are cheap to
 * check.
 *
 * These tests drive the real entry points — `runtimeClosure`, `stageApp`, and
 * `pruneExcludedNested` — over CONSTRUCTED fixture workspaces and assert on
 * the staged output: which packages land in `stage/node_modules` and which do
 * not. Nothing below reads the installed repo tree's layout; an assertion that
 * did would pin whatever dependency graph upstream happened to ship today and
 * break on every reshuffle. The exclusion lists themselves are the contract.
 */

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

import {
  isNonRuntimeFile,
  pruneExcludedNested,
  runtimeClosure,
  scopeCoreNativeOverride,
  stageApp,
  stageRuntimeBuildFiles,
} from '../scripts/stage-app.js';

// Only the out-of-process provider-SDK probe below runs against the real repo —
// it imports the installed pi-coding-agent to prove loading it resolves no SDK.
// No other test may read this tree's node_modules.
const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');

interface SeedDeps {
  dependencies?: Record<string, string>;
  optionalDependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
}

/** A registry package under `<root>/node_modules/<name>`, manifest and entry point included. */
function seedPkg(root: string, name: string, deps: SeedDeps = {}): void {
  const dir = join(root, 'node_modules', name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'package.json'), JSON.stringify({ name, version: '1.0.0', ...deps }));
  writeFileSync(join(dir, 'index.js'), 'module.exports = {};');
}

/** A workspace package under `<root>/packages/<dir>` — where `@mlx-node/*` names resolve. */
function seedWorkspace(root: string, dir: string, name: string, deps: SeedDeps = {}): void {
  const pkgDir = join(root, 'packages', dir);
  mkdirSync(join(pkgDir, 'dist'), { recursive: true });
  // stageApp copies the dashboard's `assets` unconditionally (the offline
  // tokenizer vocabulary), so a fixture dashboard must have one.
  if (name === '@mlx-node/dashboard') mkdirSync(join(pkgDir, 'assets'), { recursive: true });
  writeFileSync(join(pkgDir, 'package.json'), JSON.stringify({ name, version: '1.0.0', ...deps }));
  writeFileSync(join(pkgDir, 'dist', 'index.js'), 'export {};');
}

/** The `dependencies` of pi-ai that are provider SDKs — the names the exclusion rule must both drop and report. */
const SDK_ROOTS = [
  '@anthropic-ai/sdk',
  '@aws-sdk/client-bedrock-runtime',
  '@google/genai',
  '@smithy/node-http-handler',
  'openai',
];

/**
 * One fixture workspace covering the whole dashboard-side graph: the three
 * workspace roots, an external root, a pi-ai-like package declaring every
 * provider SDK (all five are seeded — the dashboard closure must refuse them
 * by name, and the CLI reuse of this graph needs them resolvable), a
 * platform-gated napi optional, and a devDependency that must never be queued.
 */
function seedDashboardGraph(root: string): string[] {
  seedWorkspace(root, 'dashboard', '@mlx-node/dashboard', {
    dependencies: { tokenizers: '1.0.0', other: '1.0.0' },
  });
  seedWorkspace(root, 'server', '@mlx-node/server', {
    dependencies: { '@earendil-works/pi-ai': '1.0.0' },
    optionalDependencies: {
      '@mlx-node/core-darwin-arm64': '1.0.0',
      'platform-only-optional': '1.0.0',
    },
  });
  seedWorkspace(root, 'lm', '@mlx-node/lm');
  // All five SDKs are seeded, not just referenced: the dashboard closure must
  // refuse them by name before resolution, but the CLI test reuses this graph
  // with `@mlx-node/cli` in the roots, where the exclusion is switched off and
  // every one of them has to resolve. `openai` alone gets a dependency, so the
  // walk-not-deny-list half of the rule has a subtree to shed.
  seedPkg(root, 'openai', { dependencies: { zod: '1.0.0' } });
  for (const name of SDK_ROOTS.filter((n) => n !== 'openai')) seedPkg(root, name);
  seedPkg(root, '@earendil-works/pi-ai', {
    dependencies: Object.fromEntries(SDK_ROOTS.map((name) => [name, '1.0.0'])),
  });
  seedPkg(root, 'tokenizers');
  seedPkg(root, 'other');
  seedPkg(root, 'zod');
  // A devDependency of a staged package: installable, never part of a runtime
  // closure.
  seedPkg(root, 'electron-updater', { devDependencies: { 'app-builder-lib': '1.0.0' } });
  seedPkg(root, 'app-builder-lib');
  return ['@mlx-node/dashboard', '@mlx-node/server', '@mlx-node/lm', 'electron-updater'];
}

/** A minimal Electron-app-shaped `desktopDir` for `stageApp`. */
function seedDesktop(root: string): string {
  const desktop = join(root, 'desktop');
  mkdirSync(join(desktop, 'dist'), { recursive: true });
  mkdirSync(join(desktop, 'build'), { recursive: true });
  writeFileSync(join(desktop, 'dist', 'index.js'), 'export {};');
  for (const name of ['iconTemplate.png', 'iconTemplate@2x.png']) writeFileSync(join(desktop, 'build', name), name);
  writeFileSync(join(desktop, 'package.json'), JSON.stringify({ name: '@mlx-node/desktop', version: '0.0.1' }));
  return desktop;
}

describe('runtime build assets', () => {
  it('stages the tray images without generator inputs', () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-desktop-build-'));
    const desktop = join(root, 'desktop');
    const stage = join(root, 'stage');
    try {
      mkdirSync(join(desktop, 'build'), { recursive: true });
      for (const name of [
        'iconTemplate.png',
        'iconTemplate@2x.png',
        'icon.icns',
        'make-icons.ts',
        'tray-icon-source.png',
      ]) {
        writeFileSync(join(desktop, 'build', name), name);
      }
      mkdirSync(join(desktop, 'build', 'icon.iconset'));
      writeFileSync(join(desktop, 'build', 'icon.iconset', 'icon_16x16.png'), 'generated');

      stageRuntimeBuildFiles(desktop, stage);

      expect(readdirSync(join(stage, 'build')).sort()).toEqual(['iconTemplate.png', 'iconTemplate@2x.png'].sort());
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

it('rejects a changed generated core loader instead of silently shipping a broken override', () => {
  expect(() => scopeCoreNativeOverride('module.exports = require("./native.node")')).toThrow(
    'Unrecognized core native binding override',
  );
});

describe('packaged update eligibility', () => {
  it.each([undefined, false, true])('enables updates only when the packager explicitly signs: %s', (autoUpdates) => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-update-stage-'));
    const desktop = join(root, 'desktop');
    const stage = join(root, 'stage');
    try {
      mkdirSync(join(desktop, 'dist'), { recursive: true });
      mkdirSync(join(desktop, 'build'), { recursive: true });
      writeFileSync(join(desktop, 'dist', 'index.js'), 'export {};');
      for (const name of ['iconTemplate.png', 'iconTemplate@2x.png']) {
        writeFileSync(join(desktop, 'build', name), name);
      }
      // The source manifest must not be able to opt an unsigned package in.
      writeFileSync(join(desktop, 'package.json'), JSON.stringify({ version: '0.0.13', autoUpdates: true }));
      mkdirSync(join(root, 'node_modules', 'fixture'), { recursive: true });
      writeFileSync(join(root, 'node_modules', 'fixture', 'package.json'), JSON.stringify({ name: 'fixture' }));
      stageApp({ repoRoot: root, desktopDir: desktop, stageDir: stage, roots: ['fixture'], autoUpdates });
      const manifest = JSON.parse(readFileSync(join(stage, 'package.json'), 'utf-8')) as { autoUpdates: boolean };
      expect(manifest.autoUpdates).toBe(autoUpdates === true);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

it('stages the offline vocabulary and only the macOS arm64 tokenizer binary', () => {
  const root = mkdtempSync(join(tmpdir(), 'mlx-tokenizer-stage-'));
  const desktop = join(root, 'desktop');
  const stage = join(root, 'stage');
  const dashboard = join(root, 'packages', 'dashboard');
  try {
    mkdirSync(join(desktop, 'dist'), { recursive: true });
    mkdirSync(join(desktop, 'build'), { recursive: true });
    for (const name of ['iconTemplate.png', 'iconTemplate@2x.png']) writeFileSync(join(desktop, 'build', name), name);
    mkdirSync(join(dashboard, 'dist'), { recursive: true });
    mkdirSync(join(dashboard, 'assets'), { recursive: true });
    writeFileSync(
      join(dashboard, 'package.json'),
      JSON.stringify({ name: '@mlx-node/dashboard', dependencies: { tokenizers: '0.23.2', other: '1.0.0' } }),
    );
    writeFileSync(join(dashboard, 'assets', 'o200k_base.json.gz'), 'vocabulary');
    writeFileSync(join(dashboard, 'assets', 'tiktoken-LICENSE'), 'license');
    for (const name of ['tokenizers', 'other']) {
      const dir = join(root, 'node_modules', name);
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, 'package.json'), JSON.stringify({ name }));
      writeFileSync(join(dir, 'index.js'), 'module.exports = {};');
      for (const file of [
        'tokenizers.darwin-arm64.node',
        'tokenizers.darwin-universal.node',
        'tokenizers.linux-x64-gnu.node',
      ])
        writeFileSync(join(dir, file), file);
    }
    stageApp({ repoRoot: root, desktopDir: desktop, stageDir: stage, roots: ['@mlx-node/dashboard'] });
    const modules = join(stage, 'node_modules');
    expect(readdirSync(join(modules, 'tokenizers')).filter((name) => name.endsWith('.node'))).toEqual([
      'tokenizers.darwin-arm64.node',
    ]);
    expect(readdirSync(join(modules, 'other')).filter((name) => name.endsWith('.node'))).toHaveLength(3);
    expect(readFileSync(join(modules, '@mlx-node/dashboard/assets/o200k_base.json.gz'), 'utf8')).toBe('vocabulary');
    expect(readFileSync(join(modules, '@mlx-node/dashboard/assets/tiktoken-LICENSE'), 'utf8')).toBe('license');
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

describe('runtimeClosure', () => {
  it('resolves workspace packages under packages/ and externals under node_modules', () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-closure-'));
    try {
      const roots = seedDashboardGraph(root);
      const closure = runtimeClosure(root, roots);
      expect(closure.workspace).toEqual(['@mlx-node/dashboard', '@mlx-node/lm', '@mlx-node/server']);
      expect(closure.external).toContain('tokenizers');
      expect(closure.external).toContain('other');
      expect(closure.external).toContain('electron-updater');
      expect(closure.external).toContain('@earendil-works/pi-ai');
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('never queues a devDependency', () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-closure-'));
    try {
      const roots = seedDashboardGraph(root);
      const closure = runtimeClosure(root, roots);
      // `app-builder-lib` is installed and resolvable; only its declaration
      // kind keeps it out.
      expect(closure.external).not.toContain('app-builder-lib');
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('reports a missing optional as skippedOptional rather than failing', () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-closure-'));
    try {
      const roots = seedDashboardGraph(root);
      const closure = runtimeClosure(root, roots);
      expect(closure.skippedOptional).toEqual(['platform-only-optional']);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('throws on a missing hard dependency — absence is never silent', () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-closure-'));
    try {
      seedWorkspace(root, 'dashboard', '@mlx-node/dashboard', { dependencies: { gone: '1.0.0' } });
      expect(() => runtimeClosure(root, ['@mlx-node/dashboard'])).toThrow('Cannot resolve "gone"');
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('never stages the napi prebuilt that would duplicate the native payload', () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-closure-'));
    try {
      const roots = seedDashboardGraph(root);
      const closure = runtimeClosure(root, roots);
      // The assertion is on `external` — the list that is actually COPIED — and
      // on the report beside it. One without the other is a rule that either
      // does nothing or hides what it did.
      expect(closure.external.filter((name) => name.startsWith('@mlx-node/core-'))).toEqual([]);
      expect(closure.excludedPrebuilt).toEqual(['@mlx-node/core-darwin-arm64']);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('keeps the complete CLI runtime, including lazy providers used by agent options', () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-closure-'));
    try {
      const roots = seedDashboardGraph(root);
      // `@mlx-node/cli` in the roots switches the provider-SDK exclusion off:
      // the bundled CLI's agent options may reach those providers. `openai` is
      // both a pi-ai SDK and on the exclusion list, so it pins the
      // roots.includes('@mlx-node/cli') gate rather than the name set.
      seedWorkspace(root, 'cli', '@mlx-node/cli', { dependencies: { openai: '1.0.0' } });
      const cli = runtimeClosure(root, [...roots, '@mlx-node/cli']);
      expect(cli.workspace).toContain('@mlx-node/cli');
      expect(cli.external).toContain('openai');
      expect(cli.external).toContain('zod');
      expect(cli.excludedProviderSdk).toEqual([]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

/**
 * The cloud-LLM SDKs.
 *
 * `@earendil-works/pi-ai` declares five provider SDKs as hard dependencies and
 * puts every one of them behind a `*.lazy.js` shim, so the module is fetched by
 * the `stream()` call that needs it and at no other time. This app parses pi
 * session FILES; it never asks pi to talk to a model. So the SDKs are dead
 * weight — ~72 MB of it, 38% of the `node_modules` that would otherwise ship.
 *
 * The rule names five packages and drops each one's whole subtree, because the
 * packages behind them stop being reachable. Both halves are asserted on the
 * fixture: naming without dropping would mean the walk kept a path in through
 * something else, and dropping without naming would mean the exclusion came
 * from somewhere this rule cannot defend. Whether upstream still declares the
 * same five is deliberately NOT checked — the rule is the contract, and the
 * probe at the end of this block is what keeps the premise honest.
 */
describe('cloud provider SDK exclusion', () => {
  it('stages none of the five provider SDKs', () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-sdk-'));
    try {
      const roots = seedDashboardGraph(root);
      const closure = runtimeClosure(root, roots);
      // On `external` — the list that is actually COPIED. A rule that reports
      // an exclusion it does not perform would satisfy an assertion on
      // `excludedProviderSdk` alone.
      expect(closure.external.filter((name) => SDK_ROOTS.includes(name))).toEqual([]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('drops the subtree behind them, not just the five names', () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-sdk-'));
    try {
      const roots = seedDashboardGraph(root);
      const closure = runtimeClosure(root, roots);
      // `zod` is installed and declared only by `openai`. It leaves because
      // nothing reachable still depends on it — the same way the real tree
      // sheds protobufjs, google-auth-library, and the rest of the SDK
      // support casts. If the walk stopped deciding and only the named five
      // were filtered, this package would still ship.
      expect(closure.external).not.toContain('zod');
      expect(closure.external.filter((name) => name.startsWith('@smithy/'))).toEqual([]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('reports the exclusion instead of dropping it silently', () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-sdk-'));
    try {
      const roots = seedDashboardGraph(root);
      const closure = runtimeClosure(root, roots);
      expect(closure.excludedProviderSdk.sort()).toEqual([...SDK_ROOTS].sort());
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('still reaches pi-ai — the SDKs are excluded, the library that lazy-loads them is not', () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-sdk-'));
    try {
      const roots = seedDashboardGraph(root);
      const closure = runtimeClosure(root, roots);
      expect(closure.external).toContain('@earendil-works/pi-ai');
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('stages a complete app: closure, exclusion and nested pruning all land in stage/node_modules', () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-e2e-stage-'));
    try {
      const roots = seedDashboardGraph(root);
      const desktop = seedDesktop(root);
      const stage = join(root, 'stage');
      // `keep` asks for two provider SDKs, one of which Yarn nested instead of
      // hoisting, a normal nested package that must survive the same walk, and
      // a platform-gated napi optional.
      seedPkg(root, 'keep', {
        dependencies: { openai: '1.0.0', '@smithy/node-http-handler': '1.0.0', chalk: '1.0.0' },
        optionalDependencies: { '@mlx-node/core-darwin-arm64': '1.0.0' },
      });
      seedPkg(root, 'chalk');
      const nested = join(root, 'node_modules', 'keep', 'node_modules');
      mkdirSync(join(nested, '@smithy', 'node-http-handler'), { recursive: true });
      writeFileSync(
        join(nested, '@smithy', 'node-http-handler', 'package.json'),
        JSON.stringify({ name: '@smithy/node-http-handler', version: '9.9.9' }),
      );
      mkdirSync(join(nested, 'chalk'), { recursive: true });
      writeFileSync(join(nested, 'chalk', 'package.json'), JSON.stringify({ name: 'chalk', version: '9.9.9' }));
      writeFileSync(join(nested, 'chalk', 'index.js'), 'module.exports = {};');

      const staged = stageApp({ repoRoot: root, desktopDir: desktop, stageDir: stage, roots: [...roots, 'keep'] });
      const modules = join(stage, 'node_modules');

      // What ships: `keep`, its hoisted `chalk`, and the rest of the dashboard
      // graph. What does not: every name on an exclusion list, top-level AND
      // nested.
      expect(existsSync(join(modules, 'keep'))).toBe(true);
      expect(existsSync(join(modules, 'chalk'))).toBe(true);
      expect(existsSync(join(modules, '@earendil-works', 'pi-ai'))).toBe(true);
      expect(existsSync(join(modules, 'openai'))).toBe(false);
      expect(existsSync(join(modules, '@smithy'))).toBe(false);
      expect(existsSync(join(modules, '@mlx-node', 'core-darwin-arm64'))).toBe(false);
      expect(existsSync(join(modules, 'keep', 'node_modules', '@smithy', 'node-http-handler'))).toBe(false);
      expect(existsSync(join(modules, 'keep', 'node_modules', 'chalk'))).toBe(true);

      // And each outcome is reported, not silent. `keep` and `pi-ai` between
      // them reach all five SDK names.
      expect(staged.excludedProviderSdk.sort()).toEqual([...SDK_ROOTS].sort());
      expect(staged.excludedPrebuilt).toEqual(['@mlx-node/core-darwin-arm64']);
      expect(staged.prunedNested).toEqual(['@smithy/node-http-handler']);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('is safe because pi-ai loads no provider SDK until a stream() call', () => {
    // The assertion the whole exclusion rests on, measured rather than argued.
    //
    // A separate `node` process imports the pi barrel exactly the way
    // packages/dashboard does, with a `module.registerHooks` resolve hook
    // recording every file URL Node resolves. If ANY provider SDK is reached
    // while merely loading the module — a static import upstream added, an
    // eager `import()` at module scope — its path shows up here.
    //
    // Out of process on purpose: the hook has to see Node's own resolver, and
    // inside vitest the module graph is the runner's, not the app's.
    const probe = `
      import { registerHooks } from 'node:module';
      const hits = new Set();
      registerHooks({
        resolve(spec, ctx, next) {
          const r = next(spec, ctx);
          // Greedy prefix so this lands on the LAST node_modules segment. The
          // obvious non-greedy form names the OUTER package for anything nested
          // — and @earendil-works/pi-ai nests — which would report every nested
          // provider SDK as pi-ai and pass no matter what got loaded.
          const m = /^.*\\/node_modules\\/((?:@[^/]+\\/)?[^/]+)\\//.exec(r?.url ?? '');
          if (m !== null) hits.add(m[1]);
          return r;
        },
      });
      const pi = await import('@earendil-works/pi-coding-agent');
      for (const name of ['SessionManager', 'parseSessionEntries', 'buildContextEntries', 'migrateSessionEntries']) {
        if (pi[name] === undefined) throw new Error('missing export: ' + name);
      }
      console.log(JSON.stringify([...hits]));
    `;
    const out = execFileSync(process.execPath, ['--input-type=module', '-e', probe], {
      cwd: repoRoot,
      encoding: 'utf-8',
      maxBuffer: 16 * 1024 * 1024,
    });
    const resolved = JSON.parse(out.trim().split('\n').at(-1) as string) as string[];

    // Sanity: the probe really did load the library, so "no SDK" cannot mean
    // "nothing was loaded".
    expect(resolved).toContain('@earendil-works/pi-ai');
    expect(resolved.filter((name) => SDK_ROOTS.includes(name))).toEqual([]);
  });
});

/**
 * The exclusion list decides by NAME; the staging copy runs by DIRECTORY.
 *
 * Wherever Yarn could not hoist, those two disagree, and the disagreement is
 * silent in exactly the direction that matters: `runtimeClosure` reports the
 * package as excluded and a copy ships anyway. pi-ai once nested
 * `@smithy/node-http-handler` this way; today it does not, so the rule is
 * insurance against the next version bump that re-nests an SDK.
 *
 * The interesting inputs — an excluded package nested two levels down, a
 * legitimate one beside it — have to be constructed, so this runs on a fixture
 * like everything else here.
 */
describe('pruneExcludedNested', () => {
  it('removes excluded packages from nested node_modules and leaves the rest', () => {
    const root = mkdtempSync(join(tmpdir(), 'stage-nested-'));
    try {
      const seed = (rel: string): void => {
        mkdirSync(join(root, rel), { recursive: true });
        writeFileSync(join(root, rel, 'package.json'), '{}');
      };
      // Scoped and unscoped, one and two levels deep, plus the survivors.
      seed('@earendil-works/pi-ai/node_modules/@smithy/node-http-handler');
      seed('@earendil-works/pi-ai/node_modules/typebox');
      seed('@earendil-works/pi-coding-agent/node_modules/openai');
      seed('@earendil-works/pi-coding-agent/node_modules/undici');
      seed('drizzle-orm/node_modules/@mlx-node/core-darwin-arm64');
      seed('a/node_modules/b/node_modules/@anthropic-ai/sdk');
      seed('a/node_modules/b/node_modules/chalk');

      const removed = pruneExcludedNested(root);

      expect(removed).toEqual([
        '@anthropic-ai/sdk',
        '@mlx-node/core-darwin-arm64',
        '@smithy/node-http-handler',
        'openai',
      ]);
      for (const gone of [
        '@earendil-works/pi-ai/node_modules/@smithy/node-http-handler',
        '@earendil-works/pi-coding-agent/node_modules/openai',
        'drizzle-orm/node_modules/@mlx-node/core-darwin-arm64',
        'a/node_modules/b/node_modules/@anthropic-ai/sdk',
      ]) {
        expect(existsSync(join(root, gone)), gone).toBe(false);
      }
      for (const kept of [
        '@earendil-works/pi-ai/node_modules/typebox',
        '@earendil-works/pi-coding-agent/node_modules/undici',
        'a/node_modules/b/node_modules/chalk',
      ]) {
        expect(existsSync(join(root, kept)), kept).toBe(true);
      }
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

/**
 * Files no Node runtime opens — 75 MB of the staged tree before this rule.
 *
 * The boundaries are the whole rule. `.ts` must survive (Node executes it;
 * `.d.ts` has no runtime form), and `.map` must only match behind a code
 * extension (`foo.js.map` cannot be anything but a source map; a bare `foo.map`
 * could be a package's own data).
 */
describe('isNonRuntimeFile', () => {
  it('removes declarations, source maps and build state', () => {
    for (const name of [
      'index.d.ts',
      'index.d.cts',
      'index.d.mts',
      'index.js.map',
      'index.cjs.map',
      'index.mjs.map',
      'index.d.ts.map',
      'style.css.map',
      'tsconfig.tsbuildinfo',
    ]) {
      expect(isNonRuntimeFile(name), name).toBe(true);
    }
  });

  it('keeps everything a runtime can load', () => {
    for (const name of [
      'index.js',
      'index.cjs',
      'index.mjs',
      // Node 22+ runs these directly, and packages do resolve entry points into
      // `src/*.ts`. Widening the rule to plain `.ts` breaks those silently.
      'index.ts',
      'index.cts',
      'index.mts',
      // pi-coding-agent's image pipeline resolves this by name at runtime.
      'photon_rs_bg.wasm',
      'package.json',
      'dark.json',
      'template.html',
      'LICENSE',
      // Not a source map. Nothing in the tree ships one today, and the rule
      // must not start deleting one the day something does.
      'terrain.map',
    ]) {
      expect(isNonRuntimeFile(name), name).toBe(false);
    }
  });
});

/**
 * `isNonRuntimeFile` pins the boundary; this pins the sweep that uses it.
 *
 * `pruneNonRuntime` walks the staged node_modules — nested copies included —
 * and then the app's own `dist`, deleting by filename and by directory name.
 * If that step were deleted outright every other test here stays green: the
 * staging assertions name packages, not what a package happens to ship inside
 * itself. So the fixture plants what the rule exists to remove — declaration
 * files, source maps, tsbuildinfo, a `docs/` and a nested `examples/` — and
 * the assertion lands on both halves of the contract: the staged tree lacks
 * them, and `prunedDirs`/`prunedFiles` say so.
 */
describe('non-runtime pruning', () => {
  it('removes declarations, maps, build state and doc dirs — staged tree and report agree', () => {
    const root = mkdtempSync(join(tmpdir(), 'mlx-prune-stage-'));
    try {
      const roots = seedDashboardGraph(root);
      const desktop = seedDesktop(root);
      const stage = join(root, 'stage');
      const other = join(root, 'node_modules', 'other');

      // File-level pruning inside a staged package — one specimen per pattern.
      for (const name of ['index.d.ts', 'index.js.map', 'tsconfig.tsbuildinfo']) {
        writeFileSync(join(other, name), name);
      }
      // A plain `.ts` beside them: the rule must not touch what Node can run.
      writeFileSync(join(other, 'tool.ts'), 'export {};');
      // Dir-level pruning at the package root…
      mkdirSync(join(other, 'docs'), { recursive: true });
      writeFileSync(join(other, 'docs', 'guide.md'), 'readme');
      // …and inside a nested node_modules, which the walk must recurse into.
      const inner = join(other, 'node_modules', 'inner');
      mkdirSync(join(inner, 'examples'), { recursive: true });
      writeFileSync(join(inner, 'package.json'), JSON.stringify({ name: 'inner' }));
      writeFileSync(join(inner, 'index.js'), 'module.exports = {};');
      writeFileSync(join(inner, 'index.d.ts'), 'declare const x: number;');
      writeFileSync(join(inner, 'examples', 'demo.js'), 'module.exports = {};');
      // The app's own dist is swept too: `tsc -b` emits these beside every entry.
      writeFileSync(join(desktop, 'dist', 'index.d.ts'), 'export {};');
      writeFileSync(join(desktop, 'dist', 'index.js.map'), '{}');

      const staged = stageApp({ repoRoot: root, desktopDir: desktop, stageDir: stage, roots });
      const modules = join(stage, 'node_modules');
      const stagedOther = join(modules, 'other');
      const stagedInner = join(stagedOther, 'node_modules', 'inner');

      // Gone from the staged tree — files, whole dirs, and the app's own dist.
      for (const gone of [
        join(stagedOther, 'index.d.ts'),
        join(stagedOther, 'index.js.map'),
        join(stagedOther, 'tsconfig.tsbuildinfo'),
        join(stagedOther, 'docs'),
        join(stagedInner, 'index.d.ts'),
        join(stagedInner, 'examples'),
        join(stage, 'dist', 'index.d.ts'),
        join(stage, 'dist', 'index.js.map'),
      ]) {
        expect(existsSync(gone), gone).toBe(false);
      }
      // Kept: everything a runtime can load, including the nested package and
      // the `.ts` file the file rule deliberately does not match.
      for (const kept of [
        join(stagedOther, 'index.js'),
        join(stagedOther, 'tool.ts'),
        join(stagedInner, 'index.js'),
        join(stagedInner, 'package.json'),
        join(stage, 'dist', 'index.js'),
      ]) {
        expect(existsSync(kept), kept).toBe(true);
      }

      // Reported, not silent: three files + two dirs under `other`, two files
      // under the app's dist. An exact count pins both directions — a sweep
      // that deleted nothing reports 0, and one that over-deleted would have
      // tripped the `kept` assertions above.
      expect(staged.prunedDirs).toBe(2);
      expect(staged.prunedFiles).toBe(6);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
