// routes/zh.chapters.tsx — Chinese chapters layout route (/zh/chapters).
// Mirror of routes/chapters.tsx: layout renders <Outlet /> so child routes can
// mount (the locale provider already wraps this via the /zh parent layout).

import { createFileRoute, Outlet } from '@tanstack/react-router';

export const Route = createFileRoute('/zh/chapters')({
  component: () => <Outlet />,
});
