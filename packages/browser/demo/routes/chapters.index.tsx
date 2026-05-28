// routes/chapters.index.tsx — Chapters index route (/chapters).
// Phase 2.A skeleton: placeholder only. Real ChapterIndex wired in Phase 2.C.

import { createFileRoute } from '@tanstack/react-router';

export const Route = createFileRoute('/chapters/')({
  component: () => <div>Chapter index route</div>,
});
