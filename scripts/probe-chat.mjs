#!/usr/bin/env node
// Probe: open demo, wait for Ready, fire a chat, watch log panel for 2 min.
import { withWebGPUPage } from "./webgpu-browser.mjs";

const URL = "http://localhost:5173/?profile=1";
const HEADED = process.env.HEADED !== "0";

await withWebGPUPage({ headed: HEADED }, async (page) => {
  page.setDefaultTimeout(180_000);
  page.on("console", (msg) => {
    const t = msg.type();
    if (t === "error" || t === "warning")
      console.log(`[console:${t}]`, msg.text());
  });
  page.on("pageerror", (e) => console.error("[pageerror]", e.message));

  await page.goto(URL, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() =>
    document.getElementById("status")?.textContent?.includes("Ready"),
  );
  console.log("Ready. Firing chat.");

  await page.fill(
    "#prompt",
    "Explain in careful detail how an atom is structured. Cover the nucleus with protons and neutrons, the electron cloud and orbitals, isotopes, chemical bonding, and the periodic table. Be precise, specific, and give at least five distinct paragraphs of content.",
  );
  await page.click("#send");

  const deadline = Date.now() + 120_000;
  let lastLen = 0;
  while (Date.now() < deadline) {
    const info = await page.evaluate(() => ({
      status: document.getElementById("status")?.textContent,
      numLines: document.getElementById("log")?.children.length ?? 0,
      lines: Array.from(document.getElementById("log")?.children ?? [])
        .slice(-10)
        .map((c) => c.textContent),
    }));
    if (info.numLines !== lastLen) {
      console.log(
        `[${((Date.now() - (Date.now() - 120_000)) / 1000).toFixed(1)}s] status=${info.status}, lines=${info.numLines}`,
      );
      for (const l of info.lines) console.log("  ", l);
      lastLen = info.numLines;
    }
    await new Promise((r) => setTimeout(r, 2500));
  }
});
