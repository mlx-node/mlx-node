// routes/chat.tsx — Free chat route (/chat).
//
// Phase 2.C: this route is intentionally minimal. The actual chat UI lives
// in <ChatLayerOverlay /> mounted from __root.tsx — that overlay stays
// always-mounted because the big imperative `useEffect` in app.tsx writes
// directly into the chat DOM via refs (chatRef/promptRef). Unmounting it
// would break those refs. Visibility is driven by the pathname (=== '/chat').
//
// The route itself just kicks off model loading on mount if needed and
// renders an empty layer-marker div.

import { createFileRoute } from '@tanstack/react-router';
import { useEffect } from 'react';

import { useModelLoader } from '../providers/model-loader';

function ChatRouteComponent() {
  const { status, hostedModelAvailable, kickoffLoad } = useModelLoader();

  // Direct landings on /chat (bookmark, deep link, hard reload) need the
  // model to come up. kickoffLoad is idempotent at the App level — if a
  // kickoff has already fired (e.g. we got here from a chapter that
  // triggered the load) the second call is a no-op.
  useEffect(() => {
    if (status === 'ready') return;
    if (hostedModelAvailable === false) return; // local-picker flow
    kickoffLoad();
  }, [status, hostedModelAvailable, kickoffLoad]);

  // Sentinel for the layout reserved by ChatLayerOverlay (rendered in
  // __root.tsx). The overlay overlaps everything when visible so the empty
  // slot below is hidden in practice.
  return <div className="chat-route-slot" aria-hidden="true" />;
}

export const Route = createFileRoute('/chat')({
  component: ChatRouteComponent,
});
