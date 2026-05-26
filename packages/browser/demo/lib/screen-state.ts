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
// loaded — chapters can run mocked or self-contained widgets. Tapping
// "Open free chat" from anywhere in learn mode kicks off the model load if
// it hasn't been loaded already (delegated to the existing load_kickoff path).
export type ScreenState =
  | "landing"
  | "loading"
  | "chat"
  | "chapter_index"
  | "chapter";

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
  | { type: "open_free_chat" };

export function reduceScreen(
  state: ScreenState | undefined,
  event: ScreenEvent,
): ScreenState {
  if (state === undefined || event.type === "init") return "landing";
  switch (state) {
    case "landing":
      if (event.type === "load_kickoff") return "loading";
      if (event.type === "start_learning") return "chapter_index";
      return "landing";
    case "loading":
      if (event.type === "model_ready") return "chat";
      if (event.type === "model_error") return "landing";
      return "loading";
    case "chat":
      return "chat";
    case "chapter_index":
      if (event.type === "open_chapter") return "chapter";
      if (event.type === "back_to_landing") return "landing";
      if (event.type === "open_free_chat" || event.type === "load_kickoff")
        return "loading";
      if (event.type === "model_ready") return "chat";
      return "chapter_index";
    case "chapter":
      if (event.type === "back_to_index") return "chapter_index";
      if (event.type === "open_chapter") return "chapter";
      if (event.type === "back_to_landing") return "landing";
      if (event.type === "open_free_chat" || event.type === "load_kickoff")
        return "loading";
      if (event.type === "model_ready") return "chat";
      return "chapter";
  }
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
