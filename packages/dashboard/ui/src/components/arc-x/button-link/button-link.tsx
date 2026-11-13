import type { ReactElement, ReactNode } from 'react';
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
  /** Runs before the link is followed (a popover closes itself). */
  onClick?: () => void;
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
        onClick={props.onClick}
      >
        {props.children}
      </a>
    );
  }
  return (
    <Link className={className} data-variant={variant} data-size={size} to={props.to} onClick={props.onClick}>
      {props.children}
    </Link>
  );
}
