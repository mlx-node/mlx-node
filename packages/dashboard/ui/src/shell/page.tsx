import { type ReactElement, type ReactNode, useEffect, useLayoutEffect, useRef, useState } from 'react';

import { Placeholder } from '../components/arc-x/placeholder/placeholder';
import { type Crumb, useCrumbs } from './crumbs';
import styles from './page.module.css';
import { useScrollRoot } from './scroll-root';

export interface PageProps {
  children: ReactNode;
  /** Narrow reading column for forms: 720. Default full. */
  width?: 'full' | 'reading';
  busy?: boolean;
}

/** The page container: one column, max 1200 px, the page's own container for `@` variants. */
export function Page({ children, width = 'full', busy = false }: PageProps): ReactElement {
  return (
    <div data-page="" className={styles.page} data-width={width} aria-busy={busy ? 'true' : undefined}>
      {children}
    </div>
  );
}

export interface PageHeaderProps {
  title: string;
  /** One line under the title, max 72ch. */
  description?: ReactNode;
  /** Parent crumbs for detail pages, e.g. [{ label: 'Sessions', to: '/sessions' }]. The title is appended. */
  crumbs?: readonly Crumb[];
  /** Small inline items after the title: badges, ModelBadge. */
  meta?: ReactNode;
  /** View controls: SegmentedControl, CompactField(Select), SearchField. Left of actions. */
  controls?: ReactNode;
  /** Buttons. At most one variant="primary". */
  actions?: ReactNode;
  /** Keep the title row pinned while the page scrolls. Use on pages with a global range. */
  sticky?: boolean;
  /** The title is not known yet: the h1 holds a text placeholder of this width; `title` still names the window. */
  titlePlaceholder?: string;
}

const STUCK_HEIGHT = '--page-stuck-height';

/** The one `h1` of a page; it names the window and feeds the title-bar breadcrumb. */
export function PageHeader({
  title,
  description,
  crumbs = [],
  meta,
  controls,
  actions,
  sticky = false,
  titlePlaceholder,
}: PageHeaderProps): ReactElement {
  useCrumbs([...crumbs, { label: title }]);
  useEffect(() => {
    document.title = `${title} — mlx-node`;
  }, [title]);

  const root = useScrollRoot();
  const sentinel = useRef<HTMLDivElement>(null);
  const row = useRef<HTMLElement>(null);
  const [stuck, setStuck] = useState(false);

  useEffect(() => {
    const node = sentinel.current;
    if (!sticky || node === null || typeof IntersectionObserver === 'undefined') return;
    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[entries.length - 1];
        if (entry !== undefined) setStuck(!entry.isIntersecting);
      },
      { root },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [sticky, root]);

  useLayoutEffect(() => {
    const node = row.current;
    if (!sticky || node === null || root === null) return;
    const write = (): void => {
      root.style.setProperty(STUCK_HEIGHT, `${Math.round(node.getBoundingClientRect().height)}px`);
    };
    write();
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(write);
    observer?.observe(node, { box: 'border-box' });
    return () => {
      observer?.disconnect();
      root.style.removeProperty(STUCK_HEIGHT);
    };
  }, [sticky, root]);

  const hasTrailing = controls !== undefined || actions !== undefined;
  const titleRow = (
    <>
      <div className={styles.titleGroup}>
        <h1 className={styles.title}>
          {titlePlaceholder === undefined ? title : <Placeholder width={titlePlaceholder} inline />}
        </h1>
        {meta === undefined ? null : <div className={styles.meta}>{meta}</div>}
      </div>
      {hasTrailing ? (
        <div className={styles.trailing}>
          {controls === undefined ? null : <div className={styles.controls}>{controls}</div>}
          {actions === undefined ? null : <div className={styles.actions}>{actions}</div>}
        </div>
      ) : null}
    </>
  );

  if (!sticky) {
    return (
      <header className={styles.header}>
        <div className={styles.titleRow}>{titleRow}</div>
        {description === undefined ? null : <p className={styles.description}>{description}</p>}
      </header>
    );
  }
  // A sticky box sticks only inside its parent: the row must be a direct child of the page column, not of a wrapper.
  return (
    <>
      <div ref={sentinel} className={styles.sentinel} aria-hidden="true" />
      <header ref={row} className={styles.titleRow} data-sticky="" data-stuck={stuck ? '' : undefined}>
        {titleRow}
      </header>
      {description === undefined ? null : (
        <p className={`${styles.description} ${styles.afterSticky}`}>{description}</p>
      )}
    </>
  );
}

/** 12-column grid, 12 px gaps. Children set their span with container variants (`col-span-12 @md:col-span-6`). */
export function PageGrid({ children }: { children: ReactNode }): ReactElement {
  return <div className={styles.grid}>{children}</div>;
}

export function PageSection({
  title,
  description,
  actions,
  children,
}: {
  title?: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
}): ReactElement {
  const hasHead = title !== undefined || description !== undefined || actions !== undefined;
  return (
    <section className={styles.section} aria-label={title}>
      {hasHead ? (
        <div className={styles.sectionHead}>
          <div className={styles.sectionHeading}>
            {title === undefined ? null : <h2 className={styles.sectionTitle}>{title}</h2>}
            {description === undefined ? null : <p className={styles.sectionDescription}>{description}</p>}
          </div>
          {actions === undefined ? null : <div className={styles.sectionActions}>{actions}</div>}
        </div>
      ) : null}
      {children}
    </section>
  );
}
