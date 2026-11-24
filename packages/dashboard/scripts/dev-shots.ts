/**
 * Screenshots of the running dev harness (`yarn workspace @mlx-node/dashboard dev`), in the installed Chrome.
 *
 *   oxnode scripts/dev-shots.ts [--base <url>] [--out <dir>] [--routes /a,/b] [--states palette,settings] [--motion]
 *     [--clip-check] [--card-check]
 *
 * Default: light and dark x 1180x800 and 720x480 x every route, reduced motion, into `.dev-shots/<theme>-<width>/`.
 * `--motion`: frames of `/`, `/metrics`, `/engine` at fixed times after navigation, dark, motion on.
 * `--clip-check`: no screenshots; lists every table cell, at both sizes, whose text is cut although its column does not
 * set `truncate`, or whose content is taller than its row. Exits 1 when the list is not empty.
 * `--card-check`: on `/models` and on the gallery's catalog card section, light and dark x both sizes, reads the box of
 * every catalog card; exits 1 when two cards of one page differ in height, two cards of one grid row differ in width, a
 * row of a card is not at the offset and height of the same row of the page's first card, a row that is not line-clamped
 * holds more than it shows, or a build switch scrolls. Saves a picture of each measured grid into `--out`.
 */

import { mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { type Browser, type BrowserContext, chromium, type Page } from 'playwright-core';

import { FIRST_SESSION_ID } from './dev-fixture.js';
import { GALLERY_PATH } from './dev/harness-plugin.js';

const DEFAULT_BASE = 'http://127.0.0.1:5179';
const DEFAULT_OUT = join(import.meta.dirname, '../.dev-shots');
const ROUTES = [
  '/',
  '/sessions',
  `/sessions/${FIRST_SESSION_ID}`,
  '/metrics',
  '/engine',
  '/cache',
  '/models',
  '/coding-agents',
  '/welcome',
];
const THEMES = ['light', 'dark'] as const;
const VIEWPORTS = [
  { width: 1180, height: 800 },
  { width: 720, height: 480 },
] as const;
const MOTION_ROUTES = ['/', '/metrics', '/engine'];
const MOTION_FRAMES_MS = [0, 120, 240, 480, 840, 1500];
/** No loading region for this long = settled. A shimmering label (`data-state="active"`) is not loading. */
const QUIET_MS = 300;
const SETTLE_TIMEOUT_MS = 20_000;

type State = 'palette' | 'settings';

interface Args {
  base: string;
  out: string;
  routes: string[];
  states: State[];
  motion: boolean;
  clipCheck: boolean;
  cardCheck: boolean;
}

function parseArgs(argv: readonly string[]): Args {
  const args: Args = {
    base: DEFAULT_BASE,
    out: DEFAULT_OUT,
    routes: ROUTES,
    states: [],
    motion: false,
    clipCheck: false,
    cardCheck: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === '--motion') args.motion = true;
    else if (arg === '--clip-check') args.clipCheck = true;
    else if (arg === '--card-check') args.cardCheck = true;
    else if (arg === '--base' && next !== undefined) args.base = argv[(i += 1)]!;
    else if (arg === '--out' && next !== undefined) args.out = resolve(argv[(i += 1)]!);
    else if (arg === '--routes' && next !== undefined)
      args.routes = argv[(i += 1)]!.split(',').filter((route) => route.length > 0);
    else if (arg === '--states' && next !== undefined)
      args.states = argv[(i += 1)]!.split(',').filter(
        (state): state is State => state === 'palette' || state === 'settings',
      );
  }
  if (args.base.endsWith('/')) args.base = args.base.slice(0, -1);
  return args;
}

/** `/` → `overview`, `/sessions/<id>` → `sessions-detail`, else the path's words joined by `-`. */
export function routeSlug(route: string): string {
  const parts = route.split('/').filter((part) => part.length > 0);
  if (parts.length === 0) return 'overview';
  if (parts[0] === 'sessions' && parts.length > 1) return 'sessions-detail';
  return parts.join('-');
}

async function reachable(base: string): Promise<boolean> {
  try {
    const response = await fetch(`${base}/`);
    return response.ok;
  } catch {
    return false;
  }
}

/** Fonts loaded, the shell (or `/welcome`) on screen, and no loading region for {@link QUIET_MS}. */
async function settle(page: Page): Promise<void> {
  await page.waitForFunction(
    () =>
      document.fonts.status === 'loaded' &&
      (document.querySelector('[data-scroll-root]') !== null ||
        (location.pathname === '/welcome' && document.querySelector('h1, h2') !== null)),
    undefined,
    { timeout: SETTLE_TIMEOUT_MS },
  );
  await page.evaluate(() => document.fonts.ready.then(() => undefined));
  const deadline = Date.now() + SETTLE_TIMEOUT_MS;
  let quietSince = Date.now();
  while (Date.now() < deadline) {
    const busy = await page.evaluate(
      () => document.querySelector('[aria-busy="true"]:not([data-state="active"])') !== null,
    );
    if (busy) quietSince = Date.now();
    else if (Date.now() - quietSince >= QUIET_MS) return;
    await page.waitForTimeout(50);
  }
  process.stderr.write(`  ${page.url()}: still busy after ${SETTLE_TIMEOUT_MS} ms, shooting anyway\n`);
}

function watchConsole(page: Page, label: () => string): void {
  page.on('console', (message) => {
    if (message.type() === 'error' || message.type() === 'warning') {
      process.stderr.write(`  [${label()}] console.${message.type()}: ${message.text()}\n`);
    }
  });
  page.on('pageerror', (error) => process.stderr.write(`  [${label()}] page error: ${error.message}\n`));
}

async function shootState(page: Page, state: State, path: string): Promise<void> {
  if (state === 'palette') {
    await page.keyboard.press('Meta+k');
    await page.waitForSelector('[role="dialog"][aria-label="Command palette"]');
  } else {
    await page.keyboard.press('Meta+Comma');
    await page.waitForSelector('[role="dialog"][aria-label="Settings"]');
  }
  await page.waitForTimeout(QUIET_MS);
  await page.screenshot({ path });
  await page.keyboard.press('Escape');
  await page.waitForTimeout(QUIET_MS);
}

async function matrix(browser: Browser, args: Args): Promise<number> {
  let count = 0;
  const routes = args.routes.includes('/') ? ['/', ...args.routes.filter((route) => route !== '/')] : args.routes;
  for (const theme of THEMES) {
    for (const viewport of VIEWPORTS) {
      const dir = join(args.out, `${theme}-${viewport.width}`);
      mkdirSync(dir, { recursive: true });
      const context: BrowserContext = await browser.newContext({
        viewport,
        colorScheme: theme,
        reducedMotion: 'reduce',
        deviceScaleFactor: 2,
      });
      const page = await context.newPage();
      let current = '';
      watchConsole(page, () => `${theme}-${viewport.width} ${current}`);
      for (const route of routes) {
        current = route;
        await page.goto(`${args.base}${route}`, { waitUntil: 'domcontentloaded' });
        await settle(page);
        const slug = routeSlug(route);
        await page.screenshot({ path: join(dir, `${slug}.png`), fullPage: true });
        count += 1;
        for (const state of args.states) {
          await shootState(page, state, join(dir, `${slug}+${state}.png`));
          count += 1;
        }
      }
      await context.close();
    }
  }
  return count;
}

/** Every cut or overflowing table cell of the page, as `<caption> · <column> · row <n>: <text>`. */
async function clippedCells(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const out: string[] = [];
    for (const table of document.querySelectorAll('table')) {
      const caption = table.querySelector('caption')?.textContent ?? '(no caption)';
      const heads = [...table.querySelectorAll('thead th')].map((th) => (th.textContent ?? '').trim());
      table.querySelectorAll('tbody tr').forEach((tr, rowIndex) => {
        [...tr.children].forEach((cell, columnIndex) => {
          if (!(cell instanceof HTMLTableCellElement) || cell.colSpan > 1) return;
          const cut = cell.scrollWidth > cell.clientWidth + 1 && !cell.hasAttribute('data-truncate');
          const tall = cell.scrollHeight > cell.clientHeight + 1;
          if (!cut && !tall) return;
          const text = (cell.textContent ?? '').trim().slice(0, 60);
          out.push(
            `${caption} · ${heads[columnIndex] ?? columnIndex} · row ${rowIndex + 1}: ${cut ? 'cut' : 'taller than its row'} "${text}"`,
          );
        });
      });
    }
    return out;
  });
}

async function clipPass(browser: Browser, args: Args): Promise<number> {
  let hits = 0;
  for (const viewport of VIEWPORTS) {
    const context = await browser.newContext({ viewport, colorScheme: 'light', reducedMotion: 'reduce' });
    const page = await context.newPage();
    for (const route of args.routes) {
      await page.goto(`${args.base}${route}`, { waitUntil: 'domcontentloaded' });
      await settle(page);
      for (const line of await clippedCells(page)) {
        process.stdout.write(`${viewport.width}x${viewport.height} ${route} · ${line}\n`);
        hits += 1;
      }
    }
    await context.close();
  }
  return hits;
}

/** The section of the gallery the card check measures. */
const GALLERY_CARDS = 'Catalog card';
const CARD_VIEW_HEIGHT = 2400;

interface CardReport {
  /** One line per measured grid: its name, card count, heights and widths. */
  sizes: string[];
  problems: string[];
}

/**
 * The boxes of every catalog card on the page, by grid. A grid stretches its cards to its tallest one, so heights are
 * also compared across the grids of the page, and each row of a card against the same row of the page's first card.
 */
async function measureCards(page: Page): Promise<CardReport> {
  return page.evaluate(() => {
    const round = (value: number): number => Math.round(value * 10) / 10;
    const sizes: string[] = [];
    const problems: string[] = [];
    const pageHeights = new Set<number>();
    let firstRows: { where: string; rows: Map<string, { top: number; height: number }> } | undefined;
    document.querySelectorAll('[data-catalog-grid]').forEach((grid) => {
      const holder = grid.closest('[data-gallery]') ?? grid.closest('[id]');
      const name = holder?.getAttribute('data-gallery') ?? holder?.id ?? '(grid)';
      const cards = [...grid.querySelectorAll<HTMLElement>('[data-catalog-card]')];
      const boxes = cards.map((card) => card.getBoundingClientRect());
      const heights = [...new Set(boxes.map((box) => round(box.height)))];
      const widths = [...new Set(boxes.map((box) => round(box.width)))];
      sizes.push(`${name}: ${cards.length} cards, height ${heights.join(' / ')} px, width ${widths.join(' / ')} px`);
      if (heights.length > 1) problems.push(`${name}: card heights differ (${heights.join(', ')} px)`);
      for (const height of heights) pageHeights.add(height);
      const rows = new Map<number, number[]>();
      for (const box of boxes) rows.set(round(box.top), [...(rows.get(round(box.top)) ?? []), round(box.width)]);
      for (const [top, row] of rows) {
        if (new Set(row).size > 1) problems.push(`${name}: widths differ in the row at ${top} px (${row.join(', ')})`);
      }
      cards.forEach((card, index) => {
        const label = card.querySelector('[data-row="title"] a')?.textContent?.trim() ?? `card ${index + 1}`;
        const state = card.closest('[data-gallery-card]')?.getAttribute('data-gallery-card');
        const where = `${name} · ${state ?? label}`;
        if (card.querySelector('[data-row="build"] [data-fade-end], [data-row="build"] [data-fade-start]') !== null) {
          problems.push(`${where}: the build switch scrolls`);
        }
        const cardTop = card.getBoundingClientRect().top;
        const cardRows = new Map<string, { top: number; height: number }>();
        for (const cell of card.querySelectorAll<HTMLElement>('[data-row]')) {
          const row = cell.getAttribute('data-row') ?? '';
          const box = cell.getBoundingClientRect();
          cardRows.set(row, { top: round(box.top - cardTop), height: round(box.height) });
          // A line clamp hides its last lines on purpose; the full text is in its title.
          const clamped = getComputedStyle(cell).webkitLineClamp !== 'none';
          if (!clamped && cell.scrollHeight > cell.clientHeight + 1) {
            problems.push(`${where}: ${row} is ${cell.scrollHeight} px of content in ${cell.clientHeight} px`);
          }
          if (cell.scrollWidth > cell.clientWidth + 1) {
            problems.push(`${where}: ${row} is ${cell.scrollWidth} px wide in ${cell.clientWidth} px`);
          }
        }
        if (cardRows.size === 0) return;
        if (firstRows === undefined) {
          firstRows = { where, rows: cardRows };
          return;
        }
        for (const [row, box] of cardRows) {
          const first = firstRows.rows.get(row);
          if (first === undefined || first.top !== box.top || first.height !== box.height) {
            problems.push(
              `${where}: ${row} is ${box.height} px at ${box.top} px, ${firstRows.where} has ${first?.height ?? '-'} px at ${first?.top ?? '-'} px`,
            );
          }
        }
      });
    });
    if (pageHeights.size > 1) problems.push(`card heights differ across the page (${[...pageHeights].join(', ')} px)`);
    return { sizes, problems };
  });
}

async function cardPass(browser: Browser, args: Args): Promise<number> {
  let hits = 0;
  mkdirSync(args.out, { recursive: true });
  for (const theme of THEMES) {
    for (const viewport of VIEWPORTS) {
      // Tall, so every grid is on screen at once: its picture is not cut by the page's own scroll area.
      const context = await browser.newContext({
        viewport: { width: viewport.width, height: CARD_VIEW_HEIGHT },
        colorScheme: theme,
        reducedMotion: 'reduce',
        deviceScaleFactor: 2,
      });
      const page = await context.newPage();
      const tag = `${theme}-${viewport.width}`;
      watchConsole(page, () => tag);
      for (const route of ['/models', GALLERY_PATH]) {
        await page.goto(`${args.base}${route}`, { waitUntil: 'domcontentloaded' });
        if (route === GALLERY_PATH) {
          await page.waitForSelector(`[data-gallery="${GALLERY_CARDS}"] [data-catalog-card]`, {
            timeout: SETTLE_TIMEOUT_MS,
          });
          await page.evaluate(() => document.fonts.ready.then(() => undefined));
          await page.waitForTimeout(QUIET_MS);
        } else {
          await settle(page);
        }
        const report = await measureCards(page);
        for (const line of report.sizes) process.stdout.write(`${tag} ${route} · ${line}\n`);
        for (const line of report.problems) {
          process.stdout.write(`${tag} ${route} · PROBLEM ${line}\n`);
          hits += 1;
        }
        const grids = page.locator('[data-catalog-grid]');
        for (let index = 0; index < (await grids.count()); index += 1) {
          const name = route === GALLERY_PATH ? 'gallery' : `models-${index + 1}`;
          await grids.nth(index).screenshot({ path: join(args.out, `cards-${tag}-${name}.png`) });
        }
      }
      await context.close();
    }
  }
  return hits;
}

async function motionPass(browser: Browser, args: Args): Promise<number> {
  const dir = join(args.out, 'motion');
  mkdirSync(dir, { recursive: true });
  const context = await browser.newContext({
    viewport: VIEWPORTS[0],
    colorScheme: 'dark',
    reducedMotion: 'no-preference',
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();
  watchConsole(page, () => 'motion');
  let count = 0;
  for (const route of MOTION_ROUTES) {
    await page.goto(`${args.base}${route}`, { waitUntil: 'domcontentloaded' });
    const start = Date.now();
    for (const at of MOTION_FRAMES_MS) {
      const wait = start + at - Date.now();
      if (wait > 0) await page.waitForTimeout(wait);
      await page.screenshot({ path: join(dir, `${routeSlug(route)}-${at}.png`) });
      count += 1;
    }
  }
  await context.close();
  return count;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (!(await reachable(args.base))) {
    throw new Error(
      `the dev harness is not reachable at ${args.base}/ — start it first: yarn workspace @mlx-node/dashboard dev`,
    );
  }
  const browser = await chromium.launch({ channel: 'chrome' });
  try {
    if (args.cardCheck) {
      const hits = await cardPass(browser, args);
      process.stdout.write(`${hits} card problems\n`);
      if (hits > 0) process.exitCode = 1;
      return;
    }
    if (args.clipCheck) {
      const hits = await clipPass(browser, args);
      process.stdout.write(`${hits} clipped cells\n`);
      if (hits > 0) process.exitCode = 1;
      return;
    }
    const count = args.motion ? await motionPass(browser, args) : await matrix(browser, args);
    process.stdout.write(`${count} screenshots in ${args.out}\n`);
  } finally {
    await browser.close();
  }
}

main().catch((err: unknown) => {
  process.stderr.write(`dev-shots failed: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
