
import { useCallback, useEffect, useRef, useState } from "react";

export type CopyFeedbackState = "idle" | "copied" | "error";

/**
 * Writes text to the clipboard. Uses the async Clipboard API and falls back to a hidden textarea with
 * document.execCommand("copy") when the API is missing or refused (sandboxed or cross-origin iframes, Safari
 * inside embedded previews, insecure contexts). Must be called from a user gesture. Rejects only when both fail.
 */
export async function writeClipboard(text: string): Promise<void> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
  } catch {
    // Fall through to the legacy path below.
  }
  const active = document.activeElement as HTMLElement | null;
  const selection = document.getSelection();
  const ranges = selection ? Array.from({ length: selection.rangeCount }, (_, index) => selection.getRangeAt(index)) : [];
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.cssText = "position:fixed;top:0;left:0;width:1px;height:1px;opacity:0;pointer-events:none;";
  document.body.appendChild(area);
  area.select();
  area.setSelectionRange(0, text.length);
  let ok = false;
  try { ok = document.execCommand("copy"); } catch { ok = false; }
  area.remove();
  if (selection) { selection.removeAllRanges(); ranges.forEach(range => selection.addRange(range)); }
  active?.focus?.({ preventScroll: true });
  if (!ok) throw new Error("Copy failed");
}

/** Shared clipboard state for actions that render their own button or menu. */
export function useCopyFeedback(duration = 1900) {
  const [state, setState] = useState<CopyFeedbackState>("idle");
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const timeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const reset = useCallback(() => {
    if (timeout.current) clearTimeout(timeout.current);
    timeout.current = null;
    setState("idle");
    setActiveKey(null);
  }, []);

  useEffect(() => () => {
    if (timeout.current) clearTimeout(timeout.current);
  }, []);

  const copy = useCallback(async (value: string, key = "default") => {
    if (timeout.current) clearTimeout(timeout.current);
    setActiveKey(key);
    try {
      await writeClipboard(value);
      setState("copied");
      timeout.current = setTimeout(reset, duration);
      return true;
    } catch {
      setState("error");
      timeout.current = setTimeout(reset, duration);
      return false;
    }
  }, [duration, reset]);

  return { state, activeKey, copy, reset };
}
