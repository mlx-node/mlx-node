import { existsSync, globSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

import { parseSync } from 'oxc-parser';
import type { Plugin } from 'vite-plus';
import { defineConfig } from 'vite-plus';

/**
 * Test lanes. Every Vitest file goes to one lane:
 *   - `addon`: loads the native addon (`@mlx-node/core`), so it runs serially on macOS.
 *   - `darwin-ts`: no addon, but needs macOS (the `DARWIN_TS` list).
 *   - `pure-ts`: everything else; runs in parallel, also on Linux CI.
 *
 * A file loads the addon when one of its runtime imports (`import`, `export … from`,
 * `import()` or `require()` with a string literal) leads to `@mlx-node/core`.
 * `import type` is skipped. A top-level `vi.mock(spec, () => …)` that never calls
 * `importActual`/`importOriginal` cuts the ESM edges to that module for that test;
 * `require()` goes around Vitest's mocks, so its edges are never cut.
 *
 * See a lane: `vp test list --project addon`.
 */

const ROOT = import.meta.dirname;
const ADDON = '@mlx-node/core';
const CORE_DIR = join(ROOT, 'packages/core');

/** Files the import graph cannot judge: a child process loads the addon, or a free-port-then-bind race. */
const SERIAL_BY_HAND = [
  '__test__/cli/agent-e2e.test.ts', // spawns the `mlx` CLI, which runs a real model
  '__test__/server/host/addon-free-subpaths.test.ts', // `node -e` child requires packages/core/index.cjs
  '__test__/server/host/logger.test.ts', // listen(0) → close → bind again
  '__test__/server/host/net.test.ts', // listen(0) → close → bind again
  '__test__/utils/dataset.test.ts', // reads data that CI makes with the addon
  'packages/desktop/__test__/child-entries.test.ts', // child processes require the addon
  'packages/desktop/__test__/sidecar-e2e.test.ts', // forks the inference entry, which loads the addon
];

/** Addon-free files that need macOS: APFS case folding, `/bin/zsh`, Mach-O binaries, the built macOS addon. */
const DARWIN_TS = [
  '__test__/core/metallib-select.test.ts',
  'packages/agent/__test__/cold-tier-root.test.ts',
  'packages/dashboard/__test__/api-prefix-cache.test.ts',
  'packages/dashboard/__test__/chat-work-landing.test.ts',
  'packages/dashboard/__test__/chat-work-path.test.ts',
  'packages/dashboard/__test__/download.test.ts',
  'packages/dashboard/__test__/prefix-cache.test.ts',
  'packages/desktop/__test__/payload.test.ts',
  'packages/desktop/__test__/verify-bundle.test.ts',
];

const TEST_GLOBS = [
  '__test__/**/*.{test,spec}.ts',
  'examples/**/*.{test,spec}.ts',
  'packages/*/__test__/**/*.{test,spec}.ts',
];

type Exports = Record<string, string | Record<string, string>>;

/** Each workspace package's name → its folder and `exports` map. */
const WORKSPACE = new Map<string, { dir: string; exports: Exports }>();
for (const entry of readdirSync(join(ROOT, 'packages'))) {
  const manifestPath = join(ROOT, 'packages', entry, 'package.json');
  if (!existsSync(manifestPath)) continue;
  const { name, exports } = JSON.parse(readFileSync(manifestPath, 'utf-8')) as { name?: string; exports?: Exports };
  if (name !== undefined && exports !== undefined) WORKSPACE.set(name, { dir: dirname(manifestPath), exports });
}

/** The TypeScript source an `@mlx-node/*` specifier names through its `@mlx-node/source` export, or null. */
function workspaceSourcePath(spec: string): string | null {
  const parts = spec.split('/');
  if (parts.length < 2 || parts[0] !== '@mlx-node') return null;
  const pkg = WORKSPACE.get(`${parts[0]}/${parts[1]}`);
  const entry = pkg?.exports[parts.length > 2 ? `./${parts.slice(2).join('/')}` : '.'];
  const target = entry !== null && typeof entry === 'object' ? entry['@mlx-node/source'] : undefined;
  return pkg === undefined || target === undefined ? null : resolve(pkg.dir, target);
}

interface Edge {
  spec: string;
  viaRequire: boolean;
}

interface Parsed {
  edges: Edge[];
  mocked: string[];
}

const parsedCache = new Map<string, Parsed>();

function literal(node: any): string | null {
  if (node?.type === 'Literal' && typeof node.value === 'string') return node.value;
  if (node?.type === 'TemplateLiteral' && node.expressions.length === 0) return node.quasis[0].value.cooked;
  return null;
}

function walk(node: any, visit: (n: any) => void): void {
  if (Array.isArray(node)) for (const item of node) walk(item, visit);
  else if (typeof node === 'object' && node !== null) {
    if (typeof node.type === 'string') visit(node);
    for (const value of Object.values(node)) walk(value, visit);
  }
}

function parse(file: string): Parsed {
  const cached = parsedCache.get(file);
  if (cached !== undefined) return cached;
  const source = readFileSync(file, 'utf-8');
  const { module, program } = parseSync(file, source);
  const edges: Edge[] = [];
  for (const imp of module.staticImports) {
    if (imp.entries.length === 0 || imp.entries.some((e) => !e.isType)) {
      edges.push({ spec: imp.moduleRequest.value, viaRequire: false });
    }
  }
  for (const exp of module.staticExports) {
    for (const e of exp.entries) {
      if (!e.isType && e.moduleRequest) edges.push({ spec: e.moduleRequest.value, viaRequire: false });
    }
  }
  walk(program, (n) => {
    const viaRequire = n.type === 'CallExpression' && n.callee.type === 'Identifier' && n.callee.name === 'require';
    if (n.type !== 'ImportExpression' && !viaRequire) return;
    const spec = literal(viaRequire ? n.arguments[0] : n.source);
    if (spec !== null) edges.push({ spec, viaRequire });
  });
  const mocked: string[] = [];
  for (const stmt of program.body as any[]) {
    const call = stmt.type === 'ExpressionStatement' ? stmt.expression : null;
    const callee = call?.type === 'CallExpression' ? call.callee : null;
    const isViMock =
      callee?.type === 'MemberExpression' &&
      callee.computed === false &&
      (callee.object.name === 'vi' || callee.object.name === 'vitest') &&
      callee.property.name === 'mock';
    const factory = isViMock ? call.arguments[1] : undefined;
    const isFactoryFn = factory?.type === 'ArrowFunctionExpression' || factory?.type === 'FunctionExpression';
    if (!isFactoryFn || factory.params === undefined || factory.params.length > 0) continue;
    const body = source.slice(factory.start, factory.end);
    const spec = literal(call.arguments[0]);
    if (spec !== null && !body.includes('importActual') && !body.includes('importOriginal')) mocked.push(spec);
  }
  const parsed = { edges, mocked };
  parsedCache.set(file, parsed);
  return parsed;
}

const EXTENSIONS = ['.ts', '.tsx', '.mts', '.cts', '.js', '.mjs', '.cjs'];

function findFile(base: string): string | null {
  if (existsSync(base) && statSync(base).isFile()) return base;
  // Source imports `./foo.js` for a file that is `foo.ts` on disk.
  const stem = ['.js', '.mjs', '.cjs'].reduce((s, ext) => (s.endsWith(ext) ? s.slice(0, -ext.length) : s), base);
  for (const candidate of [base, stem]) {
    for (const ext of EXTENSIONS) if (existsSync(candidate + ext)) return candidate + ext;
  }
  for (const ext of EXTENSIONS) if (existsSync(join(base, `index${ext}`))) return join(base, `index${ext}`);
  return null;
}

/** A file path, `ADDON`, or null for a builtin or third-party module. */
function resolveSpec(spec: string, from: string): string | null {
  if (spec === ADDON || spec.startsWith(`${ADDON}/`) || spec.endsWith('.node')) return ADDON;
  if (spec.startsWith('.') || spec.startsWith('/')) {
    const abs = resolve(dirname(from), spec);
    if (
      abs === join(CORE_DIR, 'index.cjs') ||
      abs === join(CORE_DIR, 'index') ||
      abs.startsWith(join(CORE_DIR, 'npm/'))
    ) {
      return ADDON;
    }
    return findFile(abs);
  }
  if (spec.startsWith('@/')) return findFile(join(ROOT, 'packages/dashboard/ui/src', spec.slice(2)));
  const source = workspaceSourcePath(spec);
  return source === null ? null : findFile(source);
}

function reachesAddon(file: string, mocked: ReadonlySet<string>, seen: Set<string>): boolean {
  if (seen.has(file)) return false;
  seen.add(file);
  for (const { spec, viaRequire } of parse(file).edges) {
    const target = resolveSpec(spec, file);
    if (target === null || (!viaRequire && mocked.has(target))) continue;
    if (target === ADDON || reachesAddon(target, mocked, seen)) return true;
  }
  return false;
}

function classify(): { addon: string[]; darwinTs: string[]; pureTs: string[] } {
  const lanes = { addon: [] as string[], darwinTs: [] as string[], pureTs: [] as string[] };
  const files = globSync(TEST_GLOBS, {
    cwd: ROOT,
    exclude: (path) =>
      path
        .replaceAll('\\', '/')
        .split('/')
        .some((part) => part === 'node_modules' || part === 'dist' || part.startsWith('.')),
  })
    .map((rel) => rel.replaceAll('\\', '/'))
    .sort();
  for (const rel of files) {
    const abs = join(ROOT, rel);
    const mocked = new Set(parse(abs).mocked.flatMap((spec) => resolveSpec(spec, abs) ?? []));
    if (SERIAL_BY_HAND.includes(rel) || reachesAddon(abs, mocked, new Set())) lanes.addon.push(rel);
    else if (DARWIN_TS.includes(rel)) lanes.darwinTs.push(rel);
    else lanes.pureTs.push(rel);
  }
  return lanes;
}

// `@mlx-node/*` resolves to each package's `@mlx-node/source` export, so tests run `src`
// while Node, `tsc` and published consumers keep `dist`. Not `resolve.conditions`:
// Vitest passes those to Node as `--conditions`, and Node cannot run all of our TypeScript.
const workspaceSource: Plugin = {
  name: 'mlx-node:workspace-source',
  enforce: 'pre',
  resolveId: (source) => workspaceSourcePath(source),
  // The map is read once; Vite restarts on a config edit but not on a package.json edit.
  configureServer(server) {
    const restart = (file: string): void => {
      if (file.startsWith(join(ROOT, 'packages/')) && file.endsWith('package.json')) void server.restart();
    };
    for (const event of ['add', 'change', 'unlink'] as const) server.watcher.on(event, restart);
  },
};

const lanes = classify();

export default defineConfig({
  fmt: {
    printWidth: 120,
    tabWidth: 2,
    singleQuote: true,
    sortPackageJson: true,
    sortImports: {
      groups: [
        ['type-import'],
        ['type-builtin', 'value-builtin'],
        ['type-external', 'value-external', 'type-internal', 'value-internal'],
        ['type-parent', 'type-sibling', 'type-index', 'value-parent', 'value-sibling', 'value-index'],
        ['unknown'],
      ],
      newlinesBetween: true,
      order: 'asc',
    },
    ignorePatterns: [
      '**/dist/**',
      'packages/dashboard/web/**',
      '**/tests/**',
      '**/generated/**',
      '**/fixtures/**',
      '.yarn/**',
      'index.d.cts',
      'index.cjs',
      '/trl',
      '/transformers',
      '/mlx-lm',
      '/mlx-rs',
      '/crates/mlx-sys/mlx',
      // Byte-for-byte upstream excerpts; kquant_ggml_parity verifies them verbatim.
      '/crates/mlx-core/vendor',
      // Generated converter outputs; parity tests compare their raw bytes.
      '/crates/mlx-core/src/utils/testdata/gguf_import_golden',
      // Shipped skill files are pinned by sha256 in the manifest; formatting one fails its install check.
      'packages/dashboard/skills/own/**',
      'packages/dashboard/skills/vendor/**',
      'packages/dashboard/skills/manifest.json',
      '**/*.metal.inc',
      '**/*.cu.inc',
      // Vendored Arc files stay byte-comparable with upstream (PROVENANCE.md lists every local change).
      'packages/dashboard/ui/src/components/arc/**',
    ],
  },
  lint: {
    ignorePatterns: ['packages/dashboard/ui/src/components/arc/**'],
    options: {
      typeAware: true,
      typeCheck: true,
    },
  },
  staged: {
    '*': ['vp check --fix'],
    '*.rs': ['cargo fmt --all --'],
  },
  test: {
    globals: true,
    environment: 'node',
    // Tests must never write inference telemetry to the developer's real
    // ~/.mlx-node. metrics-trace.test.ts manages this var itself (temp dirs).
    env: { MLX_AGENT_METRICS: '0' },
    // Node 26 on arm64: two workers that type-strip .ts at once can SIGILL in V8's wasm-to-JS wrapper tier-up
    // (nodejs/node#67467). Never tiering the wrapper removes the race; drop this once that issue is fixed.
    execArgv: ['--wasm-wrapper-tiering-budget=1000000000'],
    watch: false,
    // A `beforeAll` that loads a model needs as long as a test.
    testTimeout: 120_000,
    hookTimeout: 120_000,
    // `pure-ts` and `darwin-ts` finish before the first `addon` file starts (`groupOrder`).
    projects: [
      { extends: true, test: { name: 'pure-ts', include: lanes.pureTs, sequence: { groupOrder: 0 } } },
      { extends: true, test: { name: 'darwin-ts', include: lanes.darwinTs, sequence: { groupOrder: 0 } } },
      {
        extends: true,
        // One GPU user at a time: parallel Metal work trips the GPU watchdog (SIGABRT).
        test: { name: 'addon', include: lanes.addon, maxWorkers: 1, maxConcurrency: 1, sequence: { groupOrder: 1 } },
      },
    ],
  },
  plugins: [workspaceSource],
  resolve: {
    // The dashboard SPA's own `@/` alias, for its tests.
    alias: { '@/': `${import.meta.dirname}/packages/dashboard/ui/src/` },
  },
});
