// ChatLayerOverlay.tsx
//
// Always-mounted overlay that hosts the free-chat UI (chat header, message
// list, telemetry strip, power composer). Visibility is toggled by the
// `visible` prop — the JSX must stay mounted across navigation because the
// big imperative `useEffect` in app.tsx writes directly into the chat DOM via
// `chatRef` / `promptRef`. Unmounting would invalidate those refs.
//
// Wired in Phase 2.C alongside the RouterProvider mount. Visibility is
// controlled from __root.tsx based on the current pathname (=== '/chat').

import { triggerLocalPicker } from '../lib/local-model-picker';
import { useFreeChat } from '../providers/free-chat';
import { useModelLoader } from '../providers/model-loader';
import { useTelemetry } from '../providers/telemetry';
import { ChatHeader } from './chat/ChatHeader';
import { InspectorDrawer } from './chat/InspectorDrawer';
import { PowerComposer } from './chat/PowerComposer';
import { TelemetryStrip } from './chat/TelemetryStrip';
import { PanelLoading } from './loading/PanelLoading';

export function ChatLayerOverlay({ visible }: { visible: boolean }) {
  const {
    promptRef,
    chatRef,
    sendRef,
    imageButtonRef,
    imageInputRef,
    reasoningEffort,
    cycleReasoning,
    temperature,
    setTemperature,
    maxTokens,
    setMaxTokens,
    toolsEnabled,
    setToolsEnabled,
    generating,
    sendDisabled,
    resetChat,
    inspectedPrompt,
    setInspectedPrompt,
    mlxWorkerRef,
    inspectorAbortRef,
  } = useFreeChat();
  const { stats, prefillTokensPerSecond, decodeTokensPerSecond, modelLine } = useTelemetry();
  const { status, loadingText, loadingProgress, hostedModelAvailable, kickoffLoad } = useModelLoader();

  // The chat needs the full model. Opening /chat no longer auto-downloads it,
  // so when it isn't ready we surface an explicit "Load model to chat"
  // affordance (or a loading panel) over the message area rather than leaving
  // the composer silently disabled.
  const modelReady = status === 'ready';

  return (
    <div className={`chat-layer ${visible ? 'visible' : ''}`}>
      <div className="app-shell">
        <ChatHeader onReset={resetChat} />
        {/*
          Model-consent affordance. Opening /chat no longer auto-downloads the
          model, so until it's ready we show an explicit "Load model to chat"
          panel (or a loading panel) ABOVE the message list. This is a sibling
          of the #chat node — NOT a child — because app.tsx writes message DOM
          directly into #chat via chatRef (appendChild / replaceChildren), and
          React-owned children there would fight those imperative writes.
        */}
        {!modelReady ? (
          <div className="shrink-0 px-4 pt-4">
            {status === 'loading' ? (
              <PanelLoading status={loadingText || null} progress={loadingProgress} />
            ) : (
              <div className="mx-auto flex max-w-md flex-col items-center gap-3 rounded-lg border border-border bg-muted/20 p-6 text-center">
                <p className="text-sm font-medium text-foreground">Load the model to chat</p>
                <p
                  className="text-xs leading-relaxed text-muted-foreground"
                  role={status === 'error' ? 'alert' : undefined}
                >
                  {status === 'error'
                    ? 'Loading failed — please try again.'
                    : 'Chatting loads the real Qwen3.5-0.8B (~1.6 GB) and runs entirely on your device; nothing leaves your browser.'}
                </p>
                <button
                  type="button"
                  onClick={() => {
                    if (hostedModelAvailable === false) {
                      triggerLocalPicker();
                      return;
                    }
                    kickoffLoad();
                  }}
                  className="rounded-lg border border-primary/50 bg-primary/10 px-4 py-2 text-sm font-medium text-primary transition-colors hover:bg-primary/20 focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
                >
                  {hostedModelAvailable === false ? 'Choose local model' : status === 'error' ? 'Retry' : 'Load model'}
                </button>
              </div>
            )}
          </div>
        ) : null}
        {/*
          Status ref host kept for the legacy useEffect's statusEl writes —
          see the comment in app.tsx for full context. Hidden visually; the
          ChatHeader renders its own static "Ready on WebGPU" indicator.
        */}
        <div className="chat-messages" id="chat" ref={chatRef} />

        <TelemetryStrip
          stats={stats}
          prefillTokensPerSecond={prefillTokensPerSecond}
          decodeTokensPerSecond={decodeTokensPerSecond}
          modelLine={modelLine ?? ''}
        />

        <PowerComposer
          textareaRef={promptRef}
          sendRef={sendRef}
          imageButtonRef={imageButtonRef}
          imageInputRef={imageInputRef}
          reasoningEffort={reasoningEffort}
          onCycleReasoning={cycleReasoning}
          temperature={temperature}
          onTemperatureChange={setTemperature}
          maxTokens={maxTokens}
          onMaxTokensChange={setMaxTokens}
          toolsEnabled={toolsEnabled}
          onToggleTools={() => setToolsEnabled(!toolsEnabled)}
          generating={generating}
          sendDisabled={sendDisabled}
        />
      </div>
      {visible && inspectedPrompt !== null ? (
        <InspectorDrawer
          prompt={inspectedPrompt}
          workerRef={mlxWorkerRef}
          abortRef={inspectorAbortRef}
          onClose={() => setInspectedPrompt(null)}
        />
      ) : null}
    </div>
  );
}
