import { createFileRoute } from '@tanstack/react-router';

// Route definition ONLY — the component lives in jspace.lazy.tsx so JSpaceApp and
// its baked starter JSON load only on /jspace, not in the shared entry every page
// downloads. EN-only SEO for /jspace is applied in __root (applySeoHead(getJspaceSeo())).
export const Route = createFileRoute('/jspace')({});
