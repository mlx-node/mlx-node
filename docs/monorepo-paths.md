# Workspace paths: how `@mlx-node/*` specifiers resolve

Every cross-package import in this repo (`@mlx-node/lm`, `@mlx-node/server/host/env-policy`, …)
resolves through the `exports` map of the package that owns it. No `resolve.alias` entry names a
workspace package, no `tsconfig` `paths` entry names one, and there is no per-consumer list of
subpaths to keep in sync.

The short version:

| Surface                         | Mechanism | Condition that matches | Lands on                    |
| ------------------------------- | --------- | ---------------------- | --------------------------- |
| `tsc` (all projects)            | `exports` | `types`                | `packages/*/dist/**/*.d.ts` |
| Vitest / Vite dev-server        | `exports` | `@mlx-node/source`     | `packages/*/src/**/*.ts`    |
| `oxnode` (`yarn mlx`, examples) | `exports` | `import`               | `packages/*/dist/**/*.js`   |
| Node, published consumers       | `exports` | `import` / `default`   | `packages/*/dist/**/*.js`   |

## What was there before, and what it cost

The root `vite.config.ts` carried a hand-maintained `resolve.alias` table that rewrote each
specifier — including every subpath — to that package's TypeScript source. Tests therefore ran
against `src` while Node ran against `dist`, and the alias list was a second copy of each
package's `exports` map that nothing validated. It had already drifted:

- `@mlx-node/lm/model-discovery` and `./model-detection` were declared in `exports` and imported
  by `packages/agent/src/provider/models.ts` and `packages/server/src/host/discover.ts`, but no
  corresponding `dist` file existed. Every test passed, because the alias table sent those imports
  to `src`. Only a runtime that went through `exports` — `yarn mlx`, a published install, and CI
  before its `yarn build:ts` step — could see the gap.
- `@mlx-node/vlm` was imported by `examples/` and two test files but had no alias entry at all, so
  it silently resolved to `dist` while its siblings resolved to `src`.
- Alias matching is prefix-based and first-match-wins (`matches(pattern, importee)` returns true for
  `importee === pattern` or `importee.startsWith(pattern + '/')`, the first entry wins, and the
  rewrite is a plain string replace). A bare `@mlx-node/server` key placed above its subpaths would
  rewrite `@mlx-node/server/host` to `<abs>/packages/server/src/index.ts/host`. The table was
  ordered correctly, with a comment explaining the hazard; the constraint is gone with it.

## The replacement

Each package declares the condition itself, so the subpath list lives in exactly one place — next
to the subpath it names:

```jsonc
// packages/lm/package.json
"exports": {
  ".": {
    "@mlx-node/source": "./src/index.ts",
    "types": "./dist/index.d.ts",
    "import": "./dist/index.js"
  },
  "./model-discovery": {
    "@mlx-node/source": "./src/model-discovery.ts",
    "types": "./dist/model-discovery.d.ts",
    "import": "./dist/model-discovery.js"
  }
}
```

The root config opts the dev/test resolver into that condition:

```ts
const SOURCE_CONDITION = '@mlx-node/source';

ssr: {
  resolve: {
    conditions: ['module', 'node', 'development|production', SOURCE_CONDITION],
  },
},
```

Four properties of that block are load-bearing, each verified against the pinned toolchain
(`@voidzero-dev/vite-plus-core` 0.3.0, Vitest 4.1.11) rather than assumed:

1. **`ssr.resolve` replaces Vite's defaults, it does not extend them.** With no user conditions the
   environments resolve as client `['module','browser','development|production']` and ssr
   `['module','node','development|production']`. Hence the defaults are repeated above on purpose:
   dropping `node` or `module` would change how third-party packages resolve inside the test
   environment.
2. **It has to be `ssr.resolve.conditions`.** The root `resolve.conditions` key configures the
   client environment only and does not reach the resolver Vitest uses — measured: with the
   condition there, `@mlx-node/lm` resolved to `dist/index.js`. `environments.ssr.resolve` is not an
   alternative either: Vite mirrors it back into `ssr.resolve`, so it behaves the same as (3).
3. **Vitest mirrors this list onto its Node processes as real `--conditions` flags**
   (`resolveConditions()` → `execArgv`). So the condition is not merely a bundler setting: any
   in-process module that Node itself loads is resolved with it too. Node can only run TypeScript it
   can strip, and `packages/server/src/host/index.ts` uses a parameter property — enough to kill a
   worker thread at startup with _"TypeScript parameter property is not supported in strip-only
   mode"_.
4. **Therefore spawns pass `execArgv: []`.** The dashboard's SQLite worker thread is the case that
   bit: it loads built JavaScript, and `new Worker()` inherits the parent's flags.
   `packages/dashboard/src/worker/client.ts` now clears them, the same rule the desktop sidecar's
   `fork` already followed in `supervisor/child-node.ts`. Any future nested Node spawn that loads a
   workspace package needs the same treatment — this is the one sharp edge of the arrangement.

A silent half is worth knowing about: the condition is inert until it is wired. With the key present
in `exports` but nothing in `ssr.resolve.conditions`, a specifier whose `dist` file exists resolves
to **stale `dist` without failing** (measured: `@mlx-node/lm` loaded 42 exports from `dist` instead
of the source build's 41), while a specifier whose `dist` file is missing fails outright.

## Rules that follow from the mechanics

- **Never pass `@mlx-node/source` to Node.** Node refuses to strip types for files under a
  `node_modules` path (`ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING`), and workspace packages are
  linked into `node_modules`. The condition is for bundlers and test runners only.
- **`types` still points at `dist`.** Type-checking is unchanged: `tsc` reads the published
  declaration files and project references keep reordering builds. `customConditions` was
  deliberately _not_ adopted — it would move type-checking onto `src` for every project, which in
  the composite packages (`rootDir: src`) risks pulling files from outside `rootDir`, and it would
  stop exercising the published `.d.ts` surface at all. Revisit only together with a published-types
  gate (below).
- **The map stays ESM-only.** `@mlx-node/core` publishes `require` because it is CJS; the TypeScript
  packages declare `types` + `import` (+ `default` on some entries) and nothing else, so a CJS
  consumer calling `require('@mlx-node/lm')` gets `ERR_PACKAGE_PATH_NOT_EXPORTED`. That is
  pre-existing behaviour, not a consequence of this change.
- **Inside the test runtime, `require.resolve` and `import.meta.resolve` are piped through Vite.**
  They report the source path, so they cannot be used to check what a consumer sees — use a child
  `node` process for that.
- **One alias remains**, and it crosses no package boundary: the dashboard SPA's `@/` →
  `packages/dashboard/ui/src`, repeated in the root config because 29 test imports use the same
  specifiers the SPA's own `vite.config.ts` resolves. `@mlx-node/core` no longer has an alias either:
  its `exports` already point `import` and `require` at the same `packages/core/index.cjs`, so the
  alias was resolving to exactly the file the package boundary resolves to.

## Published surface

Because the condition ships in the manifest, the targets it names must ship too, so every published
package gained `"src"` in `files`. The alternative — stripping the condition at pack time — is not
available on Yarn 4: its `publishConfig` supports a fixed field list that does not include `exports`
or `types` (pnpm rewrites both). A `prepack` rewrite of `package.json` is possible but adds a moving
part between the repo manifest and the published one.

Nothing enforces the map's consistency automatically, by choice: the checks are `yarn typecheck`
(does every declared `types` target exist) and a build before tests in CI. If drift becomes a
problem again, the candidates are, in increasing cost: a test that walks each package's `exports`
and asserts every target exists on disk; `publint` for `exports` → `files` coverage;
`@arethetypeswrong/cli` for the published type surface (its `--pack` flag is npm-only, so a Yarn
repo must pack first, and any `exports` key containing `*` returns early as a wildcard).

## Adding a subpath

1. Add the module under `packages/<pkg>/src/`.
2. Add the `exports` entry with `@mlx-node/source` first, then `types`, then `import`/`default`.
   Node treats key order as normative: `types` is matched before runtime conditions and a `default`
   belongs last.
3. Nothing else. There is no second list to update — that is the point of this arrangement.
4. `yarn build:ts` so `dist` matches, otherwise the runtime and published paths for the new subpath
   do not exist yet even though the map declares them.

## What was verified for this change

- `yarn typecheck` (full `tsc -b`) clean; `yarn mlx --help` starts the CLI from source through
  `exports` → `dist`, so the oxnode path is intact.
- `packages/dashboard/__test__` 710/710 (it was 82 failures deep before the `execArgv` fix), and a
  74-file / 1297-test slice across `agent`, `privacy`, `dashboard`, `server/host`, `models` and
  `core` at 1289 passed / 8 skipped.
- Resolution direction, not just pass/fail: with the condition wired, `import.meta.resolve` inside
  Vitest reports `packages/*/src/**`, and a real child `node` process reports `packages/*/dist/**`.

## References

- Node.js — [Modules: Packages](https://nodejs.org/api/packages.html) (conditional exports, subpath
  patterns, condition ordering) and
  [Modules: TypeScript](https://nodejs.org/api/typescript.html) (type stripping is refused under
  `node_modules`).
- Vite — `resolve.conditions`, `ssr.resolve.conditions`, `resolve.tsconfigPaths` (native but opt-in;
  the `vite-tsconfig-paths` plugin is now redundant).
- Vitest — `resolveConditions` → `execArgv`: the forwarding described above.
- TypeScript — `customConditions` (valid under `node16`/`nodenext`/`bundler`) and
  [paths](https://www.typescriptlang.org/tsconfig/paths.html), which never rewrites emitted
  specifiers and therefore cannot be the mechanism on its own.
- Turborepo — [Internal Packages](https://turborepo.dev/docs/core-concepts/internal-packages):
  just-in-time source exports require a transpiling consumer and cannot be cached; compiled packages
  pair a `types` condition with a built `default`.
- Nx — [switch to workspaces and project references](https://nx.dev/docs/kb/switch-to-workspaces-project-references):
  path aliases "were not designed for project linking"; imports should resolve through
  `node_modules` with bundler conditions mirroring `customConditions`.
