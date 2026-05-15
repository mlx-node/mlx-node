import { ArrowUp, ImagePlus, Mic } from "lucide-react";
import { type FormEvent, type RefObject } from "react";

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
    sendDisabled,
  } = props;

  function onTextareaInput(e: FormEvent<HTMLTextAreaElement>) {
    const ta = e.currentTarget;
    ta.style.height = "auto";
    const next = Math.min(ta.scrollHeight, 132);
    ta.style.height = `${next}px`;
    ta.dataset.overflow = ta.scrollHeight > 132 ? "true" : "false";
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
            className="composer-send"
            aria-label={generating ? "Generating" : "Send"}
            disabled={sendDisabled || generating}
          >
            <ArrowUp size={18} aria-hidden="true" />
          </button>
        </div>
        <div className="composer-pills">
          <button
            type="button"
            className={`pill${reasoningEffort !== "off" ? " active" : ""}`}
            onClick={onCycleReasoning}
            disabled={generating}
          >
            think · {reasoningEffort}
          </button>
          <PillStepper
            label="temp"
            value={temperature}
            min={0}
            max={2}
            step={0.1}
            disabled={generating}
            onChange={onTemperatureChange}
          />
          <PillStepper
            label="max"
            value={maxTokens}
            min={1}
            max={MAX_TOKEN_LIMIT}
            step={1}
            disabled={generating}
            onChange={onMaxTokensChange}
          />
          <button
            type="button"
            className={`pill qwen${toolsEnabled ? " active" : ""}`}
            onClick={onToggleTools}
            disabled={generating}
          >
            tools · {toolsEnabled ? "on" : "off"}
          </button>
        </div>
      </div>
    </div>
  );
}
