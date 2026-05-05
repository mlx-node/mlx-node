import { ArrowUp, ImagePlus, Mic, Square } from "lucide-react";
import { type FormEvent, type KeyboardEvent, type RefObject } from "react";

import { type ReasoningEffort } from "../../lib/screen-state";

import { PillStepper } from "./PillStepper";

export type PowerComposerProps = {
  textareaRef: RefObject<HTMLTextAreaElement | null>;
  sendRef: RefObject<HTMLButtonElement | null>;
  imageButtonRef: RefObject<HTMLButtonElement | null>;
  imageInputRef: RefObject<HTMLInputElement | null>;
  reasoningEffort: ReasoningEffort;
  onCycleReasoning: () => void;
  temperature: number;
  onTemperatureChange: (v: number) => void;
  maxTokens: number;
  onMaxTokensChange: (v: number) => void;
  toolsEnabled: boolean;
  onToggleTools: () => void;
  generating: boolean;
  onSendOrStop: () => void;
  sendDisabled: boolean;
};

const MAX_TOKEN_LIMIT = 36864;

export function PowerComposer(props: PowerComposerProps) {
  const {
    textareaRef,
    sendRef,
    imageButtonRef,
    imageInputRef,
    reasoningEffort,
    onCycleReasoning,
    temperature,
    onTemperatureChange,
    maxTokens,
    onMaxTokensChange,
    toolsEnabled,
    onToggleTools,
    generating,
    onSendOrStop,
    sendDisabled,
  } = props;

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!sendDisabled) onSendOrStop();
    }
  }

  function onTextareaInput(e: FormEvent<HTMLTextAreaElement>) {
    const ta = e.currentTarget;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 132)}px`;
  }

  return (
    <div className="composer-frame">
      <div className="composer-card">
        <div className="composer-row">
          <button
            ref={imageButtonRef}
            type="button"
            className="composer-icon-btn"
            aria-label="Attach image"
            disabled
          >
            <ImagePlus size={18} aria-hidden="true" />
          </button>
          <input
            ref={imageInputRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
          />
          <textarea
            ref={textareaRef}
            id="prompt"
            rows={1}
            placeholder="Ask Qwen anything…"
            className="composer-textarea"
            onKeyDown={onKeyDown}
            onInput={onTextareaInput}
          />
          <button
            type="button"
            className="composer-icon-btn"
            aria-label="Voice input"
          >
            <Mic size={18} aria-hidden="true" />
          </button>
          <button
            ref={sendRef}
            type="button"
            id="send"
            className={`composer-send${generating ? " stopping" : ""}`}
            aria-label={generating ? "Stop" : "Send"}
            disabled={sendDisabled}
            onClick={onSendOrStop}
          >
            {generating ? <Square size={14} fill="currentColor" /> : <ArrowUp size={18} />}
          </button>
        </div>
        <div className="composer-pills">
          <button
            type="button"
            className={`pill${reasoningEffort !== "off" ? " active" : ""}`}
            onClick={onCycleReasoning}
          >
            think · {reasoningEffort}
          </button>
          <PillStepper
            label="temp"
            value={temperature}
            min={0}
            max={2}
            step={0.1}
            onChange={onTemperatureChange}
          />
          <PillStepper
            label="max"
            value={maxTokens}
            min={1}
            max={MAX_TOKEN_LIMIT}
            step={1}
            onChange={onMaxTokensChange}
          />
          <button
            type="button"
            className={`pill qwen${toolsEnabled ? " active" : ""}`}
            onClick={onToggleTools}
          >
            tools · {toolsEnabled ? "on" : "off"}
          </button>
        </div>
      </div>
    </div>
  );
}
