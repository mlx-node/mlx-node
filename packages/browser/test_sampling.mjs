import { withWebGPUPage } from "../../scripts/webgpu-browser.mjs";

const fails = [];
await withWebGPUPage({ headed: true, channel: "chrome" }, async (page) => {
  page.on("console", (msg) => {
    const text = msg.text();
    if (text.includes("categorical") || text.includes("FAIL")) {
      fails.push(text);
      console.log(`[LOG] ${text}`);
    }
  });
  await page.goto("http://localhost:5173/test-ops.html");
  let last = "",
    stable = 0;
  for (let i = 0; i < 300; i++) {
    await new Promise((r) => setTimeout(r, 2000));
    const t = await page.evaluate(() => document.body?.innerText || "");
    const p = t.match(/(\d+)\s*passed/i),
      f = t.match(/(\d+)\s*failed/i);
    if (p || f) {
      if (t === last) {
        stable++;
        if (stable >= 3) {
          console.log(`\nTests: P=${p?.[1] || 0}, F=${f?.[1] || 0}`);
          if (f?.[1] > 0) {
            for (const l of t.split("\n"))
              if (l.includes("✗")) console.log("  " + l.trim());
          }
          break;
        }
      } else {
        stable = 0;
        last = t;
      }
    }
  }
});
