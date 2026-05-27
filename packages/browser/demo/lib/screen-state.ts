// Screen-state machine for the demo SPA.
//
// Today's flow (still intact):
//   landing --load_kickoff--> loading --model_ready--> chat
//                                       --model_error--> landing
//
// New "Inspector" lesson flow (Education App pivot):
//   landing --start_learning--> chapter_index
//   chapter_index --open_chapter--> chapter
//   chapter --back_to_index--> chapter_index
//   chapter_index --back_to_landing--> landing
//   any-learn-state --open_free_chat--> loading (then chat via model_ready)
//
// The lesson states (chapter_index, chapter) do NOT require the model to be
// loaded — chapters can run self-contained widgets. Tapping
// "Open free chat" from anywhere in learn mode kicks off the model load if
// it hasn't been loaded already (delegated to the existing load_kickoff path).
//
// `model_ready` only transitions to `chat` from `loading`. If a background
// model load completes while the user is mid-lesson (chapter_index or
// chapter), it must NOT yank them out of the lesson — `model_ready` is a
// no-op in those states. The user can still get into chat by dispatching
// `open_free_chat` explicitly.
import { CHAPTERS } from "../learn/chapters";

export type ScreenState =
  | { kind: "landing" }
  | { kind: "loading" }
  | { kind: "chat" }
  | { kind: "chapter_index" }
  | { kind: "chapter"; chapterId: string };

export type ScreenEvent =
  | { type: "init" }
  | { type: "load_kickoff" }
  | { type: "model_ready" }
  | { type: "model_error" }
  | { type: "reset_chat" }
  | { type: "start_learning" }
  | { type: "open_chapter"; chapterId: string }
  | { type: "back_to_index" }
  | { type: "back_to_landing" }
  | { type: "open_free_chat" }
  // `restore` directly sets the screen state without going through the normal
  // transition rules. Used by the URL bridge (popstate handling and the
  // initial-state path in app.tsx) to restore a screen the reducer's
  // transitions wouldn't otherwise reach (e.g. jumping straight to a chapter
  // from a deep link). The reducer simply trusts the payload — URL parsing
  // has already validated it via parseScreenFromUrl.
  | { type: "restore"; screen: ScreenState };

export function reduceScreen(
  state: ScreenState | undefined,
  event: ScreenEvent,
): ScreenState {
  if (state === undefined || event.type === "init") return { kind: "landing" };
  if (event.type === "restore") return event.screen;
  switch (state.kind) {
    case "landing":
      if (event.type === "load_kickoff") return { kind: "loading" };
      if (event.type === "start_learning") return { kind: "chapter_index" };
      return state;
    case "loading":
      if (event.type === "model_ready") return { kind: "chat" };
      if (event.type === "model_error") return { kind: "landing" };
      return state;
    case "chat":
      return state;
    case "chapter_index":
      if (event.type === "open_chapter")
        return { kind: "chapter", chapterId: event.chapterId };
      if (event.type === "back_to_landing") return { kind: "landing" };
      if (event.type === "open_free_chat" || event.type === "load_kickoff")
        return { kind: "loading" };
      // `model_ready` is intentionally a no-op here: if the model finishes
      // loading in the background while the user is browsing chapters, do
      // NOT yank them out of the lesson flow.
      return state;
    case "chapter":
      if (event.type === "back_to_index") return { kind: "chapter_index" };
      if (event.type === "open_chapter")
        return { kind: "chapter", chapterId: event.chapterId };
      if (event.type === "back_to_landing") return { kind: "landing" };
      if (event.type === "open_free_chat" || event.type === "load_kickoff")
        return { kind: "loading" };
      // `model_ready` is intentionally a no-op here — see chapter_index above.
      return state;
  }
}

/**
 * Parse the current `window.location.search` into a {@link ScreenState} for
 * cold-load deep linking. Returns `null` for an unrecognized / transient /
 * invalid URL — the caller falls back to the default landing screen.
 *
 * Rules:
 * - `?screen=landing` → `{ kind: 'landing' }`
 * - `?screen=loading` → `null` (loading is transient; treat like landing)
 * - `?screen=chat` → `{ kind: 'chat' }`
 * - `?screen=chapter_index` → `{ kind: 'chapter_index' }`
 * - `?screen=chapter&chapterId=<id>` → `{ kind: 'chapter', chapterId }` iff
 *   the id exists in the {@link CHAPTERS} registry AND `available !== false`.
 *   Otherwise `null` (falls back to landing).
 * - Anything else (or no `screen` at all): `null`.
 *
 * A malformed URL must never crash the app, so this is wrapped in try/catch.
 */
export function parseScreenFromUrl(): ScreenState | null {
  try {
    if (typeof window === "undefined") return null;
    const params = new URLSearchParams(window.location.search);
    const kind = params.get("screen");
    if (!kind) return null;
    switch (kind) {
      case "landing":
        return { kind: "landing" };
      case "loading":
        // Transient state — do not restore. Caller falls back to landing.
        return null;
      case "chat":
        return { kind: "chat" };
      case "chapter_index":
        return { kind: "chapter_index" };
      case "chapter": {
        const chapterId = params.get("chapterId");
        if (!chapterId) return null;
        const match = CHAPTERS.find((c) => c.id === chapterId);
        if (!match || match.available === false) return null;
        return { kind: "chapter", chapterId: match.id };
      }
      default:
        return null;
    }
  } catch {
    return null;
  }
}

/**
 * Serialize a {@link ScreenState} into the URL query string component (no
 * leading `?`). Returns `null` to signal the URL must NOT be updated for this
 * screen (currently only `loading`, which is transient).
 */
export function screenToUrlQuery(screen: ScreenState): string | null {
  if (screen.kind === "loading") return null;
  const params = new URLSearchParams();
  params.set("screen", screen.kind);
  if (screen.kind === "chapter") params.set("chapterId", screen.chapterId);
  return params.toString();
}

export type ProfileLikeStats = {
  numTokens?: number;
  gpuRpcCount?: number;
  poolHits?: number;
  poolMisses?: number;
};

export type TelemetryView = {
  decodeTokPerSec: string;
  prefillTokPerSec: string;
  gpuRpc: string;
  pool: string;
  modelLine: string;
};

function formatTokRate(
  label: string,
  value: number | null | undefined,
): string {
  const rate = value ?? 0;
  return rate > 0 ? `${label} ${Math.round(rate)} tok/s` : `${label} —`;
}

/**
 * Formats the telemetry footer strip values.
 *
 * `stats` comes from the demo's ProfileStats reducer (worker counters).
 * `prefillTokensPerSecond` and `decodeTokensPerSecond` come from
 * ChatResult.performance (chat-stream API).
 * The two sources are merged at the call site (see chat screen wiring).
 */
export function formatTelemetry(
  stats: ProfileLikeStats | null | undefined,
  prefillTokensPerSecond: number | null | undefined,
  decodeTokensPerSecond: number | null | undefined,
  modelLine: string,
): TelemetryView {
  const decodeTokPerSec = formatTokRate("decode", decodeTokensPerSecond);
  const prefillTokPerSec = formatTokRate("prefill", prefillTokensPerSecond);

  let gpuRpc = "—";
  let pool = "—";

  if (stats && stats.numTokens) {
    const rpcPerTok =
      stats.gpuRpcCount && stats.numTokens
        ? Math.round(stats.gpuRpcCount / stats.numTokens)
        : null;
    if (rpcPerTok != null) {
      gpuRpc = `${rpcPerTok.toLocaleString()} gpu-rpc/tok`;
    }

    const hits = stats.poolHits ?? 0;
    const misses = stats.poolMisses ?? 0;
    const total = hits + misses;
    if (total > 0) {
      pool = `pool ${Math.round((hits / total) * 100)}%`;
    }
  }

  return { decodeTokPerSec, prefillTokPerSec, gpuRpc, pool, modelLine };
}

export function formatLoadingText(status: string | null): string {
  return status && status.trim().length > 0 ? status : "Initializing model…";
}

export type ReasoningEffort = "off" | "low" | "medium" | "high";

const REASONING_CYCLE: ReasoningEffort[] = ["off", "low", "medium", "high"];

export function cycleReasoningEffort(
  current: ReasoningEffort,
): ReasoningEffort {
  const i = REASONING_CYCLE.indexOf(current);
  return REASONING_CYCLE[(i + 1) % REASONING_CYCLE.length];
}
