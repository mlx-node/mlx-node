import { PanelLeft, Search } from 'lucide-react';
import type { ReactElement } from 'react';

import { IconButton } from '../components/arc-x/icon-button/icon-button';
import { Breadcrumb, type BreadcrumbItem } from '../components/arc/breadcrumb/breadcrumb';
import { titleWhileCut } from '../lib/hover-title';
import { truncateMiddle } from '../lib/labels';
import styles from './app-shell.module.css';
import type { Crumb } from './crumbs';

const PARENT_CRUMB_CHARS = 24;

export interface TitleBarProps {
  crumbs: readonly Crumb[];
  /** Below 880 px the sidebar is always a rail: no toggle, and search shrinks to an icon. */
  narrow: boolean;
  rail: boolean;
  onToggleSidebar: () => void;
  onSearch: () => void;
}

export function breadcrumbItems(crumbs: readonly Crumb[]): BreadcrumbItem[] {
  return crumbs.map((crumb, index) => {
    const last = index === crumbs.length - 1;
    // The last crumb is cut by the width of the bar (app-shell.module.css), so it never wraps.
    const label = last ? crumb.label : truncateMiddle(crumb.label, PARENT_CRUMB_CHARS);
    return crumb.to === undefined || last ? { label } : { label, href: crumb.to };
  });
}

/** Full-width 40 px bar: traffic-light gap, sidebar toggle, breadcrumb, search. The bar drags the window. */
export function TitleBar({ crumbs, narrow, rail, onToggleSidebar, onSearch }: TitleBarProps): ReactElement {
  return (
    <header data-titlebar="" className={styles.titlebar}>
      <div className={styles.lights} aria-hidden="true" />
      {narrow ? null : (
        <IconButton
          className={styles.noDrag}
          label={rail ? 'Expand sidebar' : 'Collapse sidebar'}
          icon={<PanelLeft size={16} strokeWidth={1.75} aria-hidden />}
          onClick={onToggleSidebar}
        />
      )}
      <div
        className={styles.crumbs}
        onPointerOver={(event) => {
          const node = event.target instanceof Element ? event.target.closest('li:last-child > span') : null;
          if (node !== null) titleWhileCut(node, node.getAttribute('data-label') ?? '');
        }}
      >
        <Breadcrumb ariaLabel="Location" items={breadcrumbItems(crumbs)} />
      </div>
      {narrow ? (
        <IconButton
          className={styles.noDrag}
          label="Search"
          icon={<Search size={16} strokeWidth={1.75} aria-hidden />}
          onClick={onSearch}
        />
      ) : (
        <button type="button" className={styles.search} onClick={onSearch}>
          <Search size={14} strokeWidth={1.75} aria-hidden />
          <span className={styles.searchText}>Search or jump to…</span>
          <span className={styles.keys} aria-hidden="true">
            <kbd>⌘</kbd>
            <kbd>K</kbd>
          </span>
        </button>
      )}
    </header>
  );
}
