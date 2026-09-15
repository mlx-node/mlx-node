# Workspace paths: how `@mlx-node/*` specifiers resolve

Every cross-package import in this repo (`@mlx-node/lm`, `@mlx-node/server/host/env-policy`, …)
resolves through the `exports` map of the package that owns it. No alias entry names a workspace
package, no `tsconfig` `paths` entry names one, and there is no per-consumer list of subpaths to keep
in sync.

The short version:

| Surface                         | Mechanism                                                                     | Lands on                    |
| ------------------------------- | ----------------------------------------------------------------------------- | --------------------------- |
| `tsc` (all projects)            | `exports` → `types`                                                           | `packages/*/dist/**/*.d.ts` |
| Vitest / Vite dev-server        | `workspaceSource()` plugin, reading the `@mlx-node/source` entry in `exports` | `packages/*/src/**/*.ts`    |
| `oxnode` (`yarn mlx`, examples) | `exports` → `import`                                                          | `packages/*/dist/**/*.js`   |
| Node, published consumers       | `exports` → `import` / `default`                                              | `packages/*/dist/**/*.js`   |

## What was there before, and what it cost

The root `vite.config.ts` carried a hand-maintained `resolve.alias` table that rewrote each
specifier — including every subpath — to that package's TypeScript source. Tests therefore ran
against `src` while Node ran against `dist`, and the alias list was a second copy of each package's
`exports` map that nothing validated. It had already drifted:

- `@mlx-node/lm/model-discovery` and `./model-detection` were declared in `exports` and imported by
  `packages/agent/src/provider/models.ts` and `packages/server/src/host/discover.ts`, but no
  corresponding `dist` file existed. Every test passed, because the alias table sent those imports to
  `src`. Only a runtime that went through `exports` — `yarn mlx`, a published install, and CI before
  its `yarn build:ts` step — could see the gap.
- `@mlx-node/vlm` was imported by `examples/` and two test files but had no alias entry at all, so it
  silently resolved to `dist` while its siblings resolved to `src`.
- Alias matching is prefix-based and first-match-wins (`matches(pattern, importee)` returns true for
  `importee === pattern` or `importee.startsWith(pattern + '/')`, the first entry wins, and the
  rewrite is a plain string replace). A bare `@mlx-node/server` key placed above its subpaths would
  rewrite `@mlx-node/server/host` to `<abs>/packages/server/src/index.ts/host`. The table was ordered
  correctly, with a comment explaining the hazard; the constraint is gone with it.

## The replacement

Each package declares its source entry next to the subpath it belongs to, as the first key of that
`exports` entry:

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

The root config reads those entries and resolves to them for the dev/test pipeline only:

```ts
function workspaceSource(): Plugin {
  // one map built at config load: package name -> { dir, exports }
  return {
    name: 'mlx-node:workspace-source',
    enforce: 'pre',
    resolveId(source) {
      // `@mlx-node/lm/model-discovery` -> pkg.exports['./model-discovery']['@mlx-node/source']
      return; /* absolute path to ./src/model-discovery.ts */
    },
  };
}
```

So the subpath list exists exactly once — in the package that owns it — and adding a subpath needs no
config change at all.

The map is built at config load, and a workspace `package.json` is not a config file, so the plugin
also restarts the dev server when one changes (verified: touching a manifest logs _server
restarted_). Without that, a watch session would keep resolving a subpath that moved, miss one that
is new, and fall back to `dist` for it.

## Why not an export condition

The obvious alternative is to wire the condition into the resolver instead: add it to
`ssr.resolve.conditions` (the only key that reaches the test resolver; the root `resolve.conditions`
key configures the client environment and does not) and let Vite match `@mlx-node/source` itself.
That was built and measured, and it is wrong for this repo:

- **Vitest mirrors `ssr.resolve.conditions` onto its Node processes as real `--conditions` flags**
  (`resolveConditions()` → `execArgv`). With the condition wired, every Vitest process carried
  `--conditions @mlx-node/source`, so _Node's own_ resolution of `@mlx-node/*` returned TypeScript
  files — not just the bundler's.
- **Node can only run the TypeScript it can strip**, and this repo's source is not all erasable:
  `packages/server/src/host/index.ts` uses a parameter property. Two loads died on it:
  - the dashboard's SQLite worker thread (which inherits `execArgv`), taking 82 dashboard tests with
    it — fixed at the time with `execArgv: []`;
  - the acceptance test's fork of the built desktop sidecar entry, taking all four
    `sidecar-e2e.test.ts` cases with it — a defect an adversarial review caught and this change then
    removed at the source.
- A condition also cannot be scope-limited: `environments.ssr.resolve.conditions` is mirrored back
  into `ssr.resolve`, and `resolve.conditions` does not reach the test resolver at all.

Resolving inside Vite keeps the whole mechanism on the bundler side, where it belongs: Node,
`oxnode`, the published `exports` map and every spawned process see exactly what they saw before.
The `resolveId`-with-`enforce: 'pre'` shape is also the direction Vite itself points at — an
`resolve.alias` entry with a `customResolver` is deprecated in favour of it.

## Rules that follow from the mechanics

- **The condition is data, not a wired condition.** Nothing sets `@mlx-node/source` as a resolution
  condition, so nothing Node loads can select it. Do not add it to `resolve.conditions` or
  `ssr.resolve.conditions`: that is what reintroduces the failure described above.
- **`types` still points at `dist`.** Type-checking is unchanged: `tsc` reads the published
  declaration files and project references keep reordering builds. `customConditions` was
  deliberately _not_ adopted — it would move type-checking onto `src` for every project, which in the
  composite packages (`rootDir: src`) risks pulling files outside `rootDir`, and it would stop
  exercising the published `.d.ts` surface at all. Revisit only together with a published-types gate.
- **`import.meta.resolve` in-process is not a Vite oracle.** The runner carries
  `--experimental-import-meta-resolve`, so that call reports Node's resolution (`dist`), not the
  plugin's. To observe what the test runtime actually loads, import the module and check a
  source-only value; to observe what a consumer sees, use a child `node` process.
- **The map stays ESM-only.** `@mlx-node/core` publishes `require` because it is CJS; the TypeScript
  packages declare `types` + `import` (+ `default` on some entries) and nothing else, so a CJS
  consumer calling `require('@mlx-node/lm')` gets `ERR_PACKAGE_PATH_NOT_EXPORTED`. That is
  pre-existing behaviour, not a consequence of this change.
- **One alias remains**, and it crosses no package boundary: the dashboard SPA's `@/` →
  `packages/dashboard/ui/src`, repeated in the root config because 29 test imports use the same
  specifiers the SPA's own `vite.config.ts` resolves. `@mlx-node/core` has no alias either: its
  `exports` already point `import` and `require` at the same `packages/core/index.cjs`, so an alias
  was resolving to exactly the file the package boundary resolves to.

## Published surface

Because the condition ships in the manifest, the targets it names must ship too, so every published
package gained `"src"` in `files`. The alternative — stripping the condition at pack time — is not
available on Yarn 4: its `publishConfig` supports a fixed field list that does not include `exports`
or `types` (pnpm rewrites both). A `prepack` rewrite of `package.json` is possible but adds a moving
part between the repo manifest and the published one.

Nothing enforces the map's consistency automatically, by choice: the checks are `yarn typecheck`
(does every declared `types` target exist) and a build before tests in CI. If drift becomes a problem
again, the candidates are, in increasing cost: a test that walks each package's `exports` and asserts
every target exists on disk; `publint` for `exports` → `files` coverage; `@arethetypeswrong/cli` for
the published type surface (its `--pack` flag is npm-only, so a Yarn repo must pack first, and any
`exports` key containing `*` returns early as a wildcard).

## Adding a subpath

1. Add the module under `packages/<pkg>/src/`.
2. Add the `exports` entry with `@mlx-node/source` first, then `types`, then `import`/`default`.
   Node treats key order as normative: `types` is matched before runtime conditions and a `default`
   belongs last. The plugin picks the entry up from the manifest — no config change.
3. `yarn build:ts` so `dist` matches, otherwise the runtime and published paths for the new subpath do
   not exist yet even though the map declares them.

## What was verified for this change

- `yarn typecheck` (full `tsc -b`) clean, and `yarn mlx --help` still starts the CLI from source
  through `exports` → `dist`.
- The two load paths that broke under the condition wiring: `packages/dashboard/__test__` 710/710 and
  `packages/desktop/__test__/sidecar-e2e.test.ts` green.
- Two slices covering the packages whose resolution changed plus the model-dependent suites:
  80 files / 1375 tests, and 221 files / 3691 tests.
- Resolution direction, not just pass/fail: a source-only marker added to a package module was
  observable after importing it through the boundary, and no Vitest process carried a
  `--conditions @mlx-node/source` flag.

## References

- Node.js — [Modules: Packages](https://nodejs.org/api/packages.html) (conditional exports, subpath
  patterns, condition ordering) and
  [Modules: TypeScript](https://nodejs.org/api/typescript.html) (type stripping is refused under
  `node_modules`).
- Vite — `resolve.conditions` (client environment), `ssr.resolve.conditions` (the key Vitest reads),
  `resolve.alias[].customResolver` deprecation in favour of `enforce: 'pre'` `resolveId` plugins.
- Vitest — `resolveConditions()` → `execArgv`: the forwarding described above.
- TypeScript — `customConditions` (valid under `node16`/`nodenext`/`bundler`) and
  [paths](https://www.typescriptlang.org/tsconfig/paths.html), which never rewrites emitted
  specifiers and therefore cannot be the mechanism on its own.
- Turborepo — [Internal Packages](https://turborepo.dev/docs/core-concepts/internal-packages):
  just-in-time source exports require a transpiling consumer and cannot be cached; compiled packages
  pair a `types` condition with a built `default`.
- Nx — [switch to workspaces and project references](https://nx.dev/docs/kb/switch-to-workspaces-project-references):
  path aliases "were not designed for project linking"; imports should resolve through `node_modules`
  with bundler conditions mirroring `customConditions`.
