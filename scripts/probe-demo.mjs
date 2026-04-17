#!/usr/bin/env node
// Probe the dev-server demo: navigate headed, dump status + any errors.
import { chromium } from 'playwright';

const URL = 'http://localhost:5173/?profile=1';
const HEADED = process.env.HEADED !== '0';

const browser = await chromium.launch({
  headless: !HEADED,
  args: [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan,SharedArrayBuffer',
    '--disable-dawn-features=disallow_unsafe_apis',
  ],
});
const ctx = await browser.newContext();
const page = await ctx.newPage();
page.on('console', (msg) => console.log(`[console:${msg.type()}]`, msg.text()));
page.on('pageerror', (e) => console.error('[pageerror]', e.message));
page.on('requestfailed', (r) => console.error('[reqfail]', r.url(), r.failure()?.errorText));

console.log('goto', URL);
await page.goto(URL, { waitUntil: 'domcontentloaded' });
const PROBE_SECS = Number(process.env.PROBE_SECS ?? 300);
console.log(`waiting up to ${PROBE_SECS}s for Ready...`);
const start = Date.now();
const deadline = start + PROBE_SECS * 1000;
let lastStatus = '';
while (Date.now() < deadline) {
  const status = await page.evaluate(() => document.getElementById('status')?.textContent);
  if (status !== lastStatus) {
    const t = ((Date.now() - start) / 1000).toFixed(1);
    console.log(`[${t}s] status= ${status}`);
    lastStatus = status ?? '';
  }
  if (status && (status.includes('Ready') || status.includes('Error') || status.toLowerCase().includes('failed'))) {
    break;
  }
  await new Promise((r) => setTimeout(r, 2500));
}
const finalStatus = await page.evaluate(() => document.getElementById('status')?.textContent);
const logLines = await page.evaluate(() =>
  Array.from(document.getElementById('log')?.children ?? []).map((c) => c.textContent),
);
console.log('FINAL status:', finalStatus);
console.log('LOG lines:');
for (const l of logLines) console.log('  ', l);

// Keep browser open briefly so user can see
if (HEADED) {
  console.log('(leaving browser open 20s for manual inspection)');
  await new Promise((r) => setTimeout(r, 20_000));
}
await browser.close();
