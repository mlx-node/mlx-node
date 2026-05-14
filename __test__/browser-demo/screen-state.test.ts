import { describe, it, expect } from "vitest";

import {
  type ScreenState,
  type ScreenEvent,
  reduceScreen,
  formatTelemetry,
  formatLoadingText,
  cycleReasoningEffort,
  type ReasoningEffort,
} from "../../packages/browser/demo/lib/screen-state";

describe("reduceScreen", () => {
  it("starts on landing", () => {
    expect(
      reduceScreen(undefined as unknown as ScreenState, { type: "init" }),
    ).toBe("landing");
  });

  it("landing → loading on LOAD_KICKOFF", () => {
    expect(reduceScreen("landing", { type: "load_kickoff" })).toBe("loading");
  });

  it("loading → chat on MODEL_READY", () => {
    expect(reduceScreen("loading", { type: "model_ready" })).toBe("chat");
  });

  it("loading → landing on MODEL_ERROR", () => {
    expect(reduceScreen("loading", { type: "model_error" })).toBe("landing");
  });

  it("chat stays on chat on RESET_CHAT (Reset clears messages, not screen)", () => {
    expect(reduceScreen("chat", { type: "reset_chat" })).toBe("chat");
  });

  it("ignores unrelated events", () => {
    expect(reduceScreen("chat", { type: "load_kickoff" })).toBe("chat");
    expect(reduceScreen("landing", { type: "model_ready" })).toBe("landing");
  });
});

describe("formatTelemetry", () => {
  it("renders dashes when no stats and no perf", () => {
    expect(formatTelemetry(null, null, null, "qwen3.5-0.8b-mlx-bf16")).toEqual({
      decodeTokPerSec: "decode —",
      prefillTokPerSec: "prefill —",
      gpuRpc: "—",
      pool: "—",
      modelLine: "qwen3.5-0.8b-mlx-bf16",
    });
  });

  it("renders prefill and decode tok/s from perf even without stats", () => {
    expect(formatTelemetry(null, 144.8, 21.42, "x")).toEqual({
      decodeTokPerSec: "decode 21 tok/s",
      prefillTokPerSec: "prefill 145 tok/s",
      gpuRpc: "—",
      pool: "—",
      modelLine: "x",
    });
  });

  it("formats tok/s, gpu-rpc/tok, pool from stats + perf", () => {
    const stats = {
      numTokens: 100,
      gpuRpcCount: 207000,
      poolHits: 624,
      poolMisses: 376,
    };
    expect(
      formatTelemetry(stats, 144.8, 21.42, "qwen3.5-0.8b-mlx-bf16"),
    ).toEqual({
      decodeTokPerSec: "decode 21 tok/s",
      prefillTokPerSec: "prefill 145 tok/s",
      gpuRpc: "2,070 gpu-rpc/tok",
      pool: "pool 62%",
      modelLine: "qwen3.5-0.8b-mlx-bf16",
    });
  });

  it("hides pool when no pool data", () => {
    const stats = { numTokens: 50, gpuRpcCount: 5000 };
    expect(formatTelemetry(stats, null, 10, "x").pool).toBe("—");
  });
});

describe("formatLoadingText", () => {
  it("returns the raw status when set", () => {
    expect(formatLoadingText("Compiling shaders…")).toBe("Compiling shaders…");
  });

  it("falls back to default when null", () => {
    expect(formatLoadingText(null)).toBe("Initializing model…");
  });
});

describe("cycleReasoningEffort", () => {
  it("cycles off → low → medium → high → off", () => {
    const seq: ReasoningEffort[] = ["off", "low", "medium", "high", "off"];
    let cur: ReasoningEffort = "off";
    for (let i = 1; i < seq.length; i++) {
      cur = cycleReasoningEffort(cur);
      expect(cur).toBe(seq[i]);
    }
  });
});
