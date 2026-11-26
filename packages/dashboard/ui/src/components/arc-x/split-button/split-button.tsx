import { ChevronDown } from 'lucide-react';
import { forwardRef, type ButtonHTMLAttributes, type ReactElement, type ReactNode, useState } from 'react';

import { Button, type ButtonVariant } from '../../arc/button/button';
import { Menu } from '../menu/menu';
import styles from './split-button.module.css';

export interface SplitButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> {
  /** The main press. */
  label: string;
  variant?: Exclude<ButtonVariant, 'danger'>;
  loading?: boolean;
  /** The arrow's name: "More ways to deny". */
  moreLabel: string;
  /** Name of the menu the arrow opens. */
  menuLabel: string;
  /** Rows of the menu: `MenuItem`s. */
  children: ReactNode;
}

/**
 * A main button and an arrow joined: the main press is the common action, the arrow opens a menu of nearby ones. Two
 * tab stops; the arrow is a menu button. The ref and the button props go to the main button.
 */
export const SplitButton = forwardRef<HTMLButtonElement, SplitButtonProps>(function SplitButton(
  { label, variant = 'secondary', loading, moreLabel, menuLabel, children, disabled, className, ...main },
  ref,
): ReactElement {
  const [open, setOpen] = useState(false);
  return (
    <span className={className === undefined ? styles.group : `${styles.group} ${className}`} data-split-button="">
      <Button
        {...main}
        ref={ref}
        type="button"
        variant={variant}
        size="sm"
        loading={loading}
        disabled={disabled}
        className={styles.main}
      >
        {label}
      </Button>
      <Menu
        label={menuLabel}
        open={open}
        onOpenChange={setOpen}
        side="bottom"
        align="start"
        trigger={
          <Button
            type="button"
            variant={variant}
            size="sm"
            disabled={disabled}
            aria-label={moreLabel}
            aria-expanded={open}
            className={styles.more}
          >
            <ChevronDown size={14} strokeWidth={1.75} aria-hidden="true" />
          </Button>
        }
      >
        {children}
      </Menu>
    </span>
  );
});
