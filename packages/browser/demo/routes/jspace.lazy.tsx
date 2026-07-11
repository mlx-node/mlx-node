import { createLazyFileRoute } from '@tanstack/react-router';

import JSpaceApp from '../jspace/JSpaceApp';

// Lazy component boundary: JSpaceApp statically imports the three baked starter
// JSON modules (demo/jspace/starters/*.json). Loading them through this lazy route
// keeps that /jspace-only payload OUT of the shared entry bundle.
export const Route = createLazyFileRoute('/jspace')({
  component: JSpaceApp,
});
