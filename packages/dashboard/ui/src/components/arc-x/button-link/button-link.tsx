import type { MouseEvent, ReactElement, ReactNode } from 'react';
import { Link } from 'react-router-dom';

import { cn } from '../../../lib/cn';
import styles from './button-link.module.css';

interface ButtonLinkBase {
  /** Default 'secondary'. */
  variant?: 'primary' | 'secondary' | 'ghost';
  /** Default 'md'. */
  size?: 'sm' | 'md';
  children: ReactNode;
  className?: string;
  /** The accessible name, when the shown words alone do not say where the link goes. */
  label?: string;
  /** Runs before the link is followed (a popover closes itself); `preventDefault` keeps it from being followed. */
  onClick?: (event: MouseEvent<HTMLAnchorElement>) => void;
  /** A middle click, which follows an external link without a click. */
  onAuxClick?: (event: MouseEvent<HTMLAnchorElement>) => void;
}

/** `to`: an in-app route. `href`: an external URL, opened outside the app. */
export type ButtonLinkProps = ButtonLinkBase & ({ to: string; href?: never } | { href: string; to?: never });

/** A link that looks like an Arc Button (height, radius, colors, hover), without the press scale. */
export function ButtonLink(props: ButtonLinkProps): ReactElement {
  const className = cn(styles.root, props.className);
  const variant = props.variant ?? 'secondary';
  const size = props.size ?? 'md';
  if (props.href !== undefined) {
    return (
      <a
        className={className}
        data-variant={variant}
        data-size={size}
        href={props.href}
        target="_blank"
        rel="noreferrer"
        aria-label={props.label}
        onClick={props.onClick}
        onAuxClick={props.onAuxClick}
      >
        {props.children}
      </a>
    );
  }
  return (
    <Link
      className={className}
      data-variant={variant}
      data-size={size}
      to={props.to}
      aria-label={props.label}
      onClick={props.onClick}
    >
      {props.children}
    </Link>
  );
}
