import { chromium } from 'playwright';
const browser = await chromium.launch({ headless: false, channel: 'chrome' });
const page = await browser.newPage();
page.on('console', msg => {
  const text = msg.text();
  if (text.includes('Error') || text.includes('memory access')) console.log(`[ERR] ${text}`);
});
await page.goto('http://localhost:5173/');
for (let i = 0; i < 60; i++) {
  await new Promise(r => setTimeout(r, 2000));
  const ready = await page.evaluate(() => { const btn = document.getElementById('send'); return btn && !btn.disabled; });
  if (ready) { console.log('Model ready!'); break; }
  if (i % 10 === 0) console.log(`${i*2}s...`);
}
await page.fill('#prompt', 'Hello, who are u?');
await page.click('#send');
console.log('Sent message, waiting...');
for (let i = 0; i < 120; i++) {
  await new Promise(r => setTimeout(r, 2000));
  const logText = await page.evaluate(() => document.getElementById('log')?.innerText || '');
  if (logText.includes('chatFn returned') || logText.includes('Error:')) {
    const chatText = await page.evaluate(() => {
      const resp = document.querySelector('.response-content');
      return resp ? resp.textContent : document.getElementById('chat')?.innerText || '';
    });
    console.log(`\n=== RESPONSE ===\n${chatText?.slice(0, 500)}`);
    if (logText.includes('Error:')) console.log('\n=== ERROR FOUND ===');
    const match = logText.match(/chatFn returned in ([\d.]+)s/);
    if (match) console.log(`Time: ${match[1]}s`);
    break;
  }
  if (i % 10 === 0) console.log(`${i*2}s...`);
}
await browser.close();
