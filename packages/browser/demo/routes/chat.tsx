// routes/chat.tsx — Free chat route (/chat).
//
// Phase 2.C: this route is intentionally minimal. The actual chat UI lives
// in <ChatLayerOverlay /> mounted from __root.tsx — that overlay stays
// always-mounted because the big imperative `useEffect` in app.tsx writes
// directly into the chat DOM via refs (chatRef/promptRef). Unmounting it
// would break those refs. Visibility is driven by the pathname (=== '/chat').
//
// The route component itself renders nothing — the always-mounted overlay
// owns the surface. Crucially, opening /chat does NOT auto-download the model:
// loading happens only via an explicit user action (the consent affordance the
// overlay shows when the model isn't ready, or the global header Load button).

import { createFileRoute } from '@tanstack/react-router';

function ChatRouteComponent() {
  // ChatLayerOverlay (in __root.tsx) covers the full viewport when visible;
  // this route component has nothing else to render. We deliberately do NOT
  // kick off a model load here — the overlay surfaces a "Load model to chat"
  // affordance instead (see ChatLayerOverlay), keeping the ~1.6 GB download
  // gated behind explicit user consent.
  return null;
}

export const Route = createFileRoute('/chat')({
  component: ChatRouteComponent,
});
