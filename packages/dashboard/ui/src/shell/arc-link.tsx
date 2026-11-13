import type { ReactElement } from 'react';
import { Link } from 'react-router-dom';

import type { ArcLinkProps } from '../components/arc/lib/arc-provider';

/** What Arc renders for a link inside the app: a router link, so `href` navigates without a reload. */
export function ArcLink({ href, ref, ...rest }: ArcLinkProps): ReactElement {
  return <Link to={href} ref={ref} {...rest} />;
}
