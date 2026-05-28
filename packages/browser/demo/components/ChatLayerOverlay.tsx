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

import { ChatHeader } from './chat/ChatHeader';
import { InspectorDrawer } from './chat/InspectorDrawer';
import { PowerComposer } from './chat/PowerComposer';
import { TelemetryStrip } from './chat/TelemetryStrip';
import { useFreeChat } from '../providers/free-chat';
import { useTelemetry } from '../providers/telemetry';

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

  return (
    <div className={`chat-layer ${visible ? 'visible' : ''}`}>
      <div className="app-shell">
        <ChatHeader onReset={resetChat} />
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
