/**
 * What the bundle is allowed to contain.
 *
 * `runtimeClosure` decides, package by package, what gets copied into the
 * shipped app. Three of its rules are not size tuning — they are the difference
 * between a bundle that notarizes and one that does not, or between an app that
 * is mostly itself and one that is mostly other people's cloud SDKs. All three
 * were found by running the release gate or measuring the artifact rather than
 * by reading code:
 *
 *  - `@mlx-node/core-*` is napi's published prebuilt. Staging it shipped the
 *    239 MB native payload a second time (993 MB total) even though nothing
 *    loads it.
 *  - `@mariozechner/clipboard*` bakes upstream's CI home into its load commands,
 *    which `verify-bundle.ts` step [3/5] refuses outright.
 *  - The cloud-LLM provider SDKs behind `@earendil-works/pi-ai` were 114 MB of
 *    an app that exists to run models locally, reachable only from a
 *    `stream()` call this app never makes.
 *
 * All three exclusions are silent by nature: the app still builds, still
 * launches, and still passes its own tests with any one wrong. Only a gate, a
 * notary, or `du` says otherwise, and all of those are minutes-to-hours away
 * from the edit. So the rules are pinned here, at the point where they are cheap
 * to check.
 *
 * These run against the REAL repo tree rather than a fixture. A fixture would
 * pin the shape of the walk while saying nothing about the dependency graph we
 * actually ship, and the graph is the part that moves under us — clipboard
 * arrived transitively, through a dependency nobody added for it.
 */

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

import { isNonRuntimeFile, pruneExcludedNested, runtimeClosure, stageRuntimeBuildFiles } from '../scripts/stage-app.js';

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');

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

// The same roots packaging uses: what the three entries import, not what
// packages/desktop/package.json happens to declare.
const ROOTS = ['@mlx-node/dashboard', '@mlx-node/server', '@mlx-node/lm'];

describe('runtimeClosure', () => {
  const closure = runtimeClosure(repoRoot, ROOTS);

  it('reaches the workspace packages the app actually runs', () => {
    expect(closure.workspace).toContain('@mlx-node/dashboard');
    expect(closure.workspace).toContain('@mlx-node/server');
    expect(closure.external.length).toBeGreaterThan(0);
  });

  it('excludes the clipboard prebuilts that leak a build path', () => {
    // The assertion is on `external` rather than on the excluded list, because
    // `external` is what gets COPIED. Reporting a package as excluded while
    // still staging it would satisfy a check on the excluded list alone.
    const staged = closure.external.filter((name) => name.startsWith('@mariozechner/clipboard'));
    expect(staged).toEqual([]);
  });

  it('reports the exclusion instead of dropping it silently', () => {
    // A silent exclusion reads as "this dependency was never here" to the next
    // person wondering why clipboard support does nothing.
    expect(closure.excludedThirdParty).toContain('@mariozechner/clipboard');
  });

  it('still reaches clipboard in the graph — the exclusion is a decision, not an accident', () => {
    // Guard-the-guard. If pi-coding-agent ever drops the dependency, the two
    // tests above start passing for a reason that has nothing to do with the
    // exclusion rule, and the rule could then be deleted without any test
    // noticing. This fails loudly when that day comes.
    expect(existsSync(join(repoRoot, 'node_modules', '@mariozechner', 'clipboard'))).toBe(true);
  });

  it('never stages the napi prebuilt that would duplicate the native payload', () => {
    expect(closure.external.filter((name) => name.startsWith('@mlx-node/core-'))).toEqual([]);
  });
});

/**
 * The cloud-LLM SDKs.
 *
 * `@earendil-works/pi-ai` declares seven provider SDKs as hard dependencies and
 * puts every one of them behind a `*.lazy.js` shim, so the module is fetched by
 * the `stream()` call that needs it and at no other time. This app parses pi
 * session FILES; it never asks pi to talk to a model. So the SDKs are dead
 * weight — 114 MB of it, over half the shipped `node_modules`.
 *
 * The rule names seven packages. It drops 81, because the other 74 stop being
 * reachable. Both halves are asserted: naming without dropping would mean the
 * walk kept a path in through something else, and dropping without naming would
 * mean the size came from somewhere this rule cannot defend.
 */
describe('cloud provider SDK exclusion', () => {
  const closure = runtimeClosure(repoRoot, ROOTS);

  // The seven `dependencies` of @earendil-works/pi-ai that are provider SDKs.
  const SDK_ROOTS = [
    '@anthropic-ai/sdk',
    '@aws-sdk/client-bedrock-runtime',
    '@google/genai',
    '@mistralai/mistralai',
    '@opentelemetry/api',
    '@smithy/node-http-handler',
    'openai',
  ];

  it('stages none of the seven provider SDKs', () => {
    // On `external` — the list that is actually COPIED — for the same reason as
    // clipboard above: a rule that reports an exclusion it does not perform
    // would satisfy an assertion on `excludedProviderSdk` alone.
    expect(closure.external.filter((name) => SDK_ROOTS.includes(name))).toEqual([]);
  });

  it('drops the subtree behind them, not just the seven names', () => {
    // These are the expensive ones, and not one of them is named by the rule:
    // they leave because nothing reachable still depends on them. If a future
    // edit hard-codes a deny-list instead, this keeps passing while the walk
    // silently stops being the thing that decides.
    const collateral = closure.external.filter(
      (name) =>
        name === 'zod' ||
        name === 'protobufjs' ||
        name === 'web-streams-polyfill' ||
        name === 'google-auth-library' ||
        name.startsWith('@aws-sdk/') ||
        name.startsWith('@smithy/') ||
        name.startsWith('@opentelemetry/'),
    );
    expect(collateral).toEqual([]);
  });

  it('reports the exclusion instead of dropping it silently', () => {
    expect(closure.excludedProviderSdk.sort()).toEqual([...SDK_ROOTS].sort());
  });

  it('still reaches pi-ai — the SDKs are excluded, the library that lazy-loads them is not', () => {
    // Guard-the-guard, two ways at once. `pi-ai` itself IS eagerly loaded and
    // must keep shipping; and if upstream ever drops a provider SDK from its
    // dependencies, the assertions above would start passing for a reason that
    // has nothing to do with this rule.
    expect(closure.external).toContain('@earendil-works/pi-ai');
    const piAi = JSON.parse(
      readFileSync(join(repoRoot, 'node_modules', '@earendil-works', 'pi-ai', 'package.json'), 'utf-8'),
    ) as { dependencies?: Record<string, string> };
    expect(Object.keys(piAi.dependencies ?? {}).sort()).toEqual(expect.arrayContaining([...SDK_ROOTS].sort()));
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
 * package as excluded and a copy ships anyway. `@earendil-works/pi-ai` really
 * does nest `@smithy/node-http-handler`, so this is a live case rather than a
 * hypothetical one.
 *
 * On a fixture, not the repo tree — unlike the closure rules above, this one is
 * about a filesystem operation, and the interesting inputs (an excluded package
 * nested two levels down, a legitimate one beside it) have to be constructed.
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

  it('still finds a nested copy in the real tree — the gap this closes is live', () => {
    // Guard-the-guard. If pi-ai ever stops nesting, the fixture test above keeps
    // passing while the rule protects nothing real, and someone deletes it.
    expect(existsSync(join(repoRoot, 'node_modules', '@earendil-works', 'pi-ai', 'node_modules', '@smithy'))).toBe(
      true,
    );
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
