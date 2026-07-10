import { createFileRoute } from '@tanstack/react-router';

import JSpaceApp from '../jspace/JSpaceApp';

export const Route = createFileRoute('/jspace')({
  component: JSpaceApp,
});
