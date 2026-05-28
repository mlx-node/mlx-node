// routes/index.tsx — Landing route (/).
// Phase 2.A skeleton: placeholder only. Real Landing screen wired in Phase 2.C.

import { createFileRoute } from '@tanstack/react-router';

export const Route = createFileRoute('/')({
  component: () => <div>Landing route</div>,
});
