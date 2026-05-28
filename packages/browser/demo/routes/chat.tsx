// routes/chat.tsx — Free chat route (/chat).
// Phase 2.A skeleton: placeholder only. Real chat screen wired in Phase 2.C.

import { createFileRoute } from '@tanstack/react-router';

export const Route = createFileRoute('/chat')({
  component: () => <div>Chat route</div>,
});
