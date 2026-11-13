import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { beforeAll, describe, expect, it } from 'vite-plus/test';

// The published tarball must ship the built Vite SPA under `web/`, not just the server `dist/`. `tsc -b` (what CI
// runs before publish) never builds `web/`, and `web/` is gitignored, so this asserts both that `build:ui` produces
// the SPA and that `npm publish`'s file list would include it.

const here = dirname(fileURLToPath(import.meta.url));
const dashboardDir = join(here, '..');
const webDir = join(dashboardDir, 'web');
const assetsDir = join(webDir, 'assets');

/**
 * Size caps in bytes, set about 1 % above the measured build. A cap that fails means the bundle grew: find what
 * grew before raising it.
 */
const JS_CAP_BYTES = 1_516_288;
const CSS_CAP_BYTES = 278_528;
const FONTS_CAP_BYTES = 230_000;

/** Strings only the dev harness carries. None may reach `web/`. */
const HARNESS_STRINGS = ['__gallery', 'dev-fixture', '.dev-home', 'mlx:dev-', 'dev-open'];

/**
 * Every URL the shipped JS may hold, by prefix. None is fetched at run time (CSP `connect-src 'self'`): XML namespace
 * names, the parse base of a router, library help text, and the Hugging Face model page and the web search providers'
 * key and billing pages that links open in the system browser. A new URL fails here so somebody looks at it.
 */
const JS_URL_PREFIXES = [
  'http://www.w3.org/',
  'http://localhost',
  'https://react.dev/errors/',
  'https://reactrouter.com/en/main/routers/picking-a-router',
  'https://github.com/ungap/url-search-params',
  'https://github.com/syntax-tree/hast-util-to-jsx-runtime',
  'https://huggingface.co/',
  'https://dashboard.exa.ai',
  'https://platform.deepseek.com',
  'https://api-dashboard.search.brave.com',
  'https://www.firecrawl.dev/app',
];

function endsWith(name: string, suffix: string): boolean {
  return name.length > suffix.length && name.slice(name.length - suffix.length) === suffix;
}

function assetNames(): string[] {
  return existsSync(assetsDir) ? readdirSync(assetsDir) : [];
}

function filesUnder(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...filesUnder(path));
    else out.push(path);
  }
  return out;
}

/**
 * The text with comments cut out. `cssComments` cuts every `/* … *\/`; otherwise only license banners (`/*!`) go,
 * because a minified script may hold `/*` inside a string.
 */
function withoutComments(text: string, cssComments: boolean): string {
  let out = '';
  let at = 0;
  while (at < text.length) {
    const open = text.indexOf('/*', at);
    if (open === -1) break;
    const banner = text[open + 2] === '!';
    const close = text.indexOf('*/', open + 2);
    if (!cssComments && !banner) {
      out += text.slice(at, open + 2);
      at = open + 2;
      continue;
    }
    out += text.slice(at, open);
    at = close === -1 ? text.length : close + 2;
  }
  return out + text.slice(at);
}

/** Each `http://` / `https://` URL in `text`, up to the first quote, backquote, space, paren or `$`. */
function urlsIn(text: string): string[] {
  const found: string[] = [];
  const stops = new Set(['"', "'", '`', ' ', ')', '$', '\n', '<', '>']);
  for (const scheme of ['http://', 'https://']) {
    let at = text.indexOf(scheme);
    while (at !== -1) {
      let end = at + scheme.length;
      while (end < text.length && !stops.has(text[end])) end += 1;
      found.push(text.slice(at, end));
      at = text.indexOf(scheme, end);
    }
  }
  return found;
}

/**
 * The value of the `src` attribute of the module script `web/index.html` references, without a leading `/`.
 *
 * Resolving through the HTML rather than globbing `web/assets/*.js` matters: a stale sibling asset left by an earlier
 * build would satisfy a glob while the browser loads something else.
 */
function attributeAfter(html: string, marker: string, attribute: string): string | null {
  const tag = html.indexOf(marker);
  if (tag === -1) return null;
  const tagEnd = html.indexOf('>', tag);
  const key = `${attribute}="`;
  const start = html.indexOf(key, tag);
  if (start === -1 || (tagEnd !== -1 && start > tagEnd)) return null;
  const end = html.indexOf('"', start + key.length);
  if (end === -1) return null;
  const value = html.slice(start + key.length, end);
  return value.startsWith('/') ? value.slice(1) : value;
}

function servedFile(marker: string, attribute: string): { path: string; text: string } {
  const html = readFileSync(join(webDir, 'index.html'), 'utf8');
  const relative = attributeAfter(html, marker, attribute);
  if (relative === null || !relative.startsWith('assets/')) {
    throw new Error(`web/index.html references no hashed asset for ${marker}:\n${html}`);
  }
  const path = join(webDir, relative);
  return { path, text: readFileSync(path, 'utf8') };
}

function servedBundle(): string {
  return servedFile('<script type="module"', 'src').text;
}

function servedStyles(): { path: string; text: string } {
  return servedFile('<link rel="stylesheet"', 'href');
}

/** Every package the dashboard uses. A package that is added or comes back must be added here on purpose. */
const DEPENDENCIES = [
  '@huggingface/hub',
  '@mlx-node/agent',
  '@mlx-node/lm',
  '@mlx-node/server',
  '@napi-rs/keyring',
  'drizzle-orm',
];
const DEV_DEPENDENCIES = [
  '@fontsource-variable/geist',
  '@fontsource-variable/geist-mono',
  '@fontsource-variable/inter',
  '@radix-ui/react-dialog',
  '@radix-ui/react-popover',
  '@radix-ui/react-select',
  '@radix-ui/react-switch',
  '@radix-ui/react-tabs',
  '@radix-ui/react-tooltip',
  '@tailwindcss/vite',
  '@types/node',
  '@types/react',
  '@types/react-dom',
  '@vitejs/plugin-react',
  'clsx',
  'happy-dom',
  'lucide-react',
  'motion',
  'oxc-parser',
  'playwright-core',
  'react',
  'react-dom',
  'react-markdown',
  'react-router-dom',
  'remark-gfm',
  'tailwindcss',
  'typescript',
  'vite',
  'vite-plus',
];

describe('dashboard package dependencies', () => {
  it('lists exactly the packages the dashboard uses', () => {
    const manifest = JSON.parse(readFileSync(join(dashboardDir, 'package.json'), 'utf8')) as {
      dependencies?: Record<string, string>;
      devDependencies?: Record<string, string>;
      peerDependencies?: Record<string, string>;
      optionalDependencies?: Record<string, string>;
    };
    expect(Object.keys(manifest.dependencies ?? {}).sort()).toEqual(DEPENDENCIES);
    expect(Object.keys(manifest.devDependencies ?? {}).sort()).toEqual(DEV_DEPENDENCIES);
    expect(Object.keys(manifest.peerDependencies ?? {})).toEqual([]);
    expect(Object.keys(manifest.optionalDependencies ?? {})).toEqual([]);
  });
});

describe('dashboard npm pack ships the SPA', () => {
  beforeAll(() => {
    // Build the SPA exactly as `prepack` / the CI publish step does. The runner's NODE_ENV=test would give Vite a
    // development React build.
    execFileSync('yarn', ['build:ui'], {
      cwd: dashboardDir,
      stdio: 'pipe',
      env: { ...process.env, NODE_ENV: 'production' },
    });
  }, 180_000);

  it('build:ui emits web/index.html and a hashed asset', () => {
    expect(existsSync(join(webDir, 'index.html'))).toBe(true);
    expect(assetNames().some((name) => endsWith(name, '.js') || endsWith(name, '.css'))).toBe(true);
  });

  it('npm pack file list includes the SPA', () => {
    const stdout = execFileSync('npm', ['pack', '--dry-run', '--json', '--ignore-scripts'], {
      cwd: dashboardDir,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    // npm's --json shape has moved across releases: an array of results, a
    // bare result object, or (npm 12) a map keyed by package name.
    const parsed = JSON.parse(stdout) as unknown;
    const results: Array<{ files: Array<{ path: string }> }> = Array.isArray(parsed)
      ? parsed
      : 'files' in (parsed as object)
        ? [parsed as { files: Array<{ path: string }> }]
        : Object.values(parsed as Record<string, { files: Array<{ path: string }> }>);
    const packed = results[0]!.files.map((f) => f.path);

    expect(packed).toContain('web/index.html');
    expect(packed.some((p) => p.startsWith('web/assets/') && p.length > 'web/assets/'.length)).toBe(true);
    // The Chat skills bundle and the web tool server the Chat agent starts.
    expect(packed).toContain('skills/manifest.json');
    expect(packed).toContain('skills/own/chat-web/skills/web-tools/SKILL.md');
    expect(packed).toContain('skills/own/deepseek/skills/deepseek-search/SKILL.md');
    for (const vendor of ['exa', 'brave', 'firecrawl']) {
      expect(packed).toContain(`skills/vendor/${vendor}/LICENSE`);
      expect(packed).toContain(`skills/vendor/${vendor}/PROVENANCE.md`);
    }
    expect(packed).toContain('dist/chat/web/server.js');
    expect(packed).toContain('dist/chat/web/adapters/firecrawl.js');
  });

  it('bundles exactly six woff2 fonts and nothing loads a font from elsewhere', () => {
    const fonts = assetNames().filter((name) => endsWith(name, '.woff2'));
    expect(fonts).toHaveLength(6);
    const bytes = fonts.reduce((sum, name) => sum + statSync(join(assetsDir, name)).size, 0);
    expect(bytes).toBeLessThanOrEqual(FONTS_CAP_BYTES);
    const css = servedStyles().text;
    for (const name of fonts) expect(css).toContain(`/assets/${name}`);
  });

  it('holds no remote URL in the built CSS outside a comment', () => {
    expect(urlsIn(withoutComments(servedStyles().text, true))).toEqual([]);
  });

  it('holds no URL in the built JS other than the listed non-request ones', () => {
    const urls = urlsIn(withoutComments(servedBundle(), false));
    const unlisted = urls.filter((url) => {
      if (url === 'http://' || url === 'https://') return false;
      return !JS_URL_PREFIXES.some((prefix) => url.startsWith(prefix));
    });
    expect(unlisted).toEqual([]);
  });

  it('carries nothing of the dev harness', () => {
    expect(existsSync(join(webDir, 'dev'))).toBe(false);
    for (const file of filesUnder(webDir)) {
      if (endsWith(file, '.woff2')) continue;
      const text = readFileSync(file, 'utf8');
      for (const marker of HARNESS_STRINGS)
        expect({ file, marker, found: text.includes(marker) }).toEqual({ file, marker, found: false });
    }
  });

  it('stays inside the size caps: one JS chunk and one stylesheet', () => {
    const scripts = assetNames().filter((name) => endsWith(name, '.js'));
    const sheets = assetNames().filter((name) => endsWith(name, '.css'));
    expect(scripts).toHaveLength(1);
    expect(sheets).toHaveLength(1);
    expect(statSync(join(assetsDir, scripts[0]!)).size).toBeLessThanOrEqual(JS_CAP_BYTES);
    expect(statSync(servedStyles().path).size).toBeLessThanOrEqual(CSS_CAP_BYTES);
  });
});

/**
 * The Cache-page fixes must survive the BUILD, not merely exist in `ui/src`.
 *
 * This suite exists because of how the defects were found: the served bundle
 * was searched for `sidecarCount`, `sidecarBytes`, `coldHits`, `coldMisses` and
 * `coldBytesRestored` and contained ZERO occurrences of any of them, while the
 * API had been serving all five for months. Over half the bytes on the Usage
 * tile were unlabelled because the SPA never asked. Page tests mount the
 * components straight from `ui/src`, so they pass on source alone and cannot
 * notice a `web/` that was never rebuilt.
 *
 * Assertions are POSITIVE first. An absence-only suite passes against an empty
 * file, a 404 page, or a build that silently produced nothing.
 *
 * Every string below is a plain literal in `cache.tsx`, chosen so minification
 * cannot dissolve it: mangling renames bindings, never string contents, and
 * none of these sit inside a template hole. Model family names are deliberately
 * NOT asserted: the empty-state hint builds them from the API's
 * `restoreFamilies`, so they do not appear in the bundle at all.
 */
describe('the served bundle carries the Cache-page fixes', () => {
  it('names both cold-tier object kinds in the shipped JS', () => {
    const bundle = servedBundle();
    // The tile and the chart count the same thing, and the chart says so in its title.
    expect(bundle).toContain('Objects by age');
    // `sidecarCount` and `sidecarBytes` reach the browser as rendered text.
    expect(bundle).toContain('blocks and state sidecars');
    expect(bundle).toContain('state sidecar');
    expect(bundle).toContain(' sidecars');
    // The cold counters the API serves.
    expect(bundle).toContain('Cold-tier health');
    expect(bundle).toContain('acceptance bar: 0');
  });

  it('ships the scoped hit rate and the saturating percent rules', () => {
    const bundle = servedBundle();
    // The hit rate is labelled with the scope it was computed over.
    expect(bundle).toContain('this cache only');
    expect(bundle).toContain('no lookups recorded for this cache');
    // The saturation literals only exist if the rounding fix is compiled in.
    expect(bundle).toContain('>99%');
    expect(bundle).toContain('<1%');
  });

  it('no longer hardcodes a single restore family in the empty state', () => {
    const bundle = servedBundle();
    // The family list is served over the wire so it cannot drift from the native gate.
    expect(bundle).not.toContain('qwen3 dense');
    expect(bundle).toContain('restore-eligible family');
  });
});
