import { forwardRef, type ReactNode } from 'react';

import { cn } from '../../../lib/cn';
import { Button, type ButtonProps } from '../../arc/button/button';
import { Tooltip } from '../../arc/tooltip/tooltip';
import styles from './icon-button.module.css';

export interface IconButtonProps extends Omit<ButtonProps, 'children'> {
  /** The accessible name, and the tooltip text. */
  label: string;
  icon: ReactNode;
  /** true (default): the label as a tooltip; a string: that text instead (e.g. why it is disabled); false: none. */
  tooltip?: boolean | string;
}

/** A square Arc Button with an icon, an aria-label and a tooltip. */
export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { label, icon, tooltip = true, variant = 'ghost', size = 'sm', className, ...rest },
  ref,
) {
  const button = (
    <Button
      ref={ref}
      type="button"
      variant={variant}
      size={size}
      aria-label={label}
      className={cn(styles.icon, styles[size], className)}
      {...rest}
    >
      {icon}
    </Button>
  );
  if (tooltip === false) return button;
  return <Tooltip content={typeof tooltip === 'string' ? tooltip : label}>{button}</Tooltip>;
});
