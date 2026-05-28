// routes/chapters.tsx — Chapters layout route (/chapters).
// Phase 2.A skeleton: layout renders <Outlet /> so child routes can mount.
// Real chapter layout wired in Phase 2.C.

import { createFileRoute, Outlet } from '@tanstack/react-router';

export const Route = createFileRoute('/chapters')({
  component: () => <Outlet />,
});
