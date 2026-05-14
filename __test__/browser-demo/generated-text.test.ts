import { describe, expect, it } from "vitest";

import {
  sanitizeAssistantText,
  sanitizeThinkingText,
  splitAssistantThinking,
} from "../../packages/browser/src/generated-text";

describe("generated-text thinking parsing", () => {
  it("splits a normal think block", () => {
    expect(splitAssistantThinking("<think>A</think> B")).toEqual({
      thinking: "A",
      text: "B",
    });
  });

  it("uses the first matching close tag after an opening think tag", () => {
    expect(
      splitAssistantThinking(
        "<think>A</think> Answer mentions </think> literally",
      ),
    ).toEqual({
      thinking: "A",
      text: "Answer mentions </think> literally",
    });
  });

  it("handles a close tag when the opening tag was not present in the buffer", () => {
    expect(splitAssistantThinking("A</think> B")).toEqual({
      thinking: "A",
      text: "B",
    });
  });

  it("does not keep an empty close-tag boundary in assistant text", () => {
    expect(splitAssistantThinking("</think> B")).toEqual({
      thinking: "",
      text: "B",
    });
  });

  it("unwraps generated thought-process disclosure markup", () => {
    expect(
      splitAssistantThinking(
        "<details><summary>Thought process</summary>A</details>\n\nB",
      ),
    ).toEqual({
      thinking: "A",
      text: "B",
    });
  });

  it("removes partial thought-process disclosure wrappers from thinking", () => {
    expect(
      sanitizeThinkingText(
        "<details open><summary>Thought process</summary>Thinking Process:\nA",
      ),
    ).toBe("Thinking Process:\nA");
  });

  it("keeps assistant text clean when thought-process markup prefixes the result", () => {
    expect(
      sanitizeAssistantText(
        "<details><summary>Thought process</summary>A</details>\n\nFinal answer",
      ),
    ).toBe("Final answer");
  });
});
