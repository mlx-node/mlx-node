// device-capability.ts — decide whether this device can run the in-browser model
// WITHOUT crashing the tab.
//
// The live demos upload the real model into WebGPU (this build f32-expands the
// bf16 weights on upload → ~3.2 GB of GPU buffers) plus a 256 MiB WASM heap and
// per-shard staging copies. iOS Safari caps a single tab at roughly 2 GB of
// memory (WebKit's "jetsam" process limit) regardless of the phone's physical
// RAM, so the OS kills the tab mid-load — an UNCATCHABLE crash (no JS error
// handler runs). This is why opening a chapter on an iPhone blanks/reloads the
// page. Detect the devices that will OOM so callers can show a "read here, run
// on desktop" message instead of auto-starting the load that crashes them.
//
// Only the live model demos need this gate — the chapter prose renders fine on
// any device and is never gated.

export type ModelBlockReason = 'ios' | 'low-memory' | 'no-webgpu' | null;

/**
 * Returns why the in-browser model should NOT be loaded on this device, or
 * `null` when it is safe to load (the common desktop case). Pure and cheap;
 * safe to call on every render. Returns `null` under SSR/prerender (no
 * navigator) — the client re-evaluates on mount.
 */
export function detectModelBlockReason(): ModelBlockReason {
  if (typeof navigator === 'undefined') return null;

  // iOS / iPadOS — including iPad's desktop-mode UA (platform "MacIntel" +
  // touch). WebKit gives a tab ~2 GB max, well below the ~3.2 GB this build
  // needs, so the tab is jetsam-killed during the WebGPU upload. A real Mac has
  // maxTouchPoints 0, so it is never matched here.
  const ua = navigator.userAgent || '';
  const isIOS =
    /iP(hone|ad|od)/.test(ua) || (navigator.platform === 'MacIntel' && (navigator.maxTouchPoints ?? 0) > 1);
  if (isIOS) return 'ios';

  // Low-RAM devices. Chrome/Chromium expose deviceMemory in GiB (rounded, capped
  // at 8); Safari/Firefox omit it, so this only fires where the signal exists.
  // < 6 GiB can't hold the WASM heap + GPU buffers + staging.
  const deviceMemory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  if (typeof deviceMemory === 'number' && deviceMemory > 0 && deviceMemory < 6) {
    return 'low-memory';
  }

  // No WebGPU at all → nothing to run. Say so instead of spinning forever.
  if (!('gpu' in navigator)) return 'no-webgpu';

  return null;
}

/** Convenience: true when it is safe to auto-start the in-browser model load. */
export function canAutoLoadModel(): boolean {
  return detectModelBlockReason() === null;
}
