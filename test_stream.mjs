import { chromium } from 'playwright';
const browser = await chromium.launch({ headless: false, channel: 'chrome' });
const page = await browser.newPage();
const errors = [];
page.on('console', (msg) => {
  const text = msg.text();
  if (
    text.includes('mlx_stream') ||
    text.includes('import') ||
    text.includes('LinkError') ||
    text.includes('not a function')
  ) {
    errors.push(text);
    console.log(`[ERR] ${text}`);
  }
});
page.on('pageerror', (e) => console.log(`[PAGE_ERR] ${e.message}`));
await page.goto('http://localhost:5173/');
for (let i = 0; i < 60; i++) {
  await new Promise((r) => setTimeout(r, 2000));
  const ready = await page.evaluate(() => {
    const btn = document.getElementById('send');
    return btn && !btn.disabled;
  });
  if (ready) {
    console.log('Model ready!');
    break;
  }
  if (i % 10 === 0) console.log(`${i * 2}s...`);
}
if (errors.length) {
  console.log('\nErrors during init:');
  errors.forEach((e) => console.log(e));
}
await page.fill('#prompt', 'Hi');
await page.click('#send');
// Wait and check for streaming
for (let i = 0; i < 60; i++) {
  await new Promise((r) => setTimeout(r, 1000));
  const logText = await page.evaluate(() => document.getElementById('log')?.innerText || '');
  if (logText.includes('chatFn returned') || logText.includes('Error')) {
    const resp = await page.evaluate(() => document.querySelector('.response-content')?.textContent || '');
    const think = await page.evaluate(() => document.querySelector('.thinking-content')?.textContent || '');
    console.log(`\nThinking: "${think}"`);
    console.log(`Response: "${resp?.slice(0, 200)}"`);
    break;
  }
}
await browser.close();
