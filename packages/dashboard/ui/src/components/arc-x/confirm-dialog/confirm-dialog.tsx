import { useId, useState, type ReactElement } from 'react';

import { errMessage } from '../../../lib/errors';
import { Alert } from '../../arc/alert/alert';
import { Button } from '../../arc/button/button';
import { Dialog, DialogContent } from '../../arc/dialog/dialog';
import { DEFAULT_ERROR_TITLE, type ConfirmContent } from '../confirm-popover/confirm-popover';
import styles from './confirm-dialog.module.css';

export interface ConfirmDialogProps extends ConfirmContent {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Where focus goes when the dialog closes; by default the element that had it when the dialog opened. */
  returnFocus?: () => HTMLElement | null;
}

/**
 * The centered form of a confirmation, for a confirmation the command palette started (no trigger to grow
 * from). Takes the same ConfirmContent as ConfirmPopover; the success label is not shown, the dialog closes.
 */
export function ConfirmDialog(props: ConfirmDialogProps): ReactElement {
  const descriptionId = useId();
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const tone = props.tone ?? 'danger';

  const onOpenChange = (next: boolean): void => {
    if (!next && working) return;
    if (!next) setError(null);
    props.onOpenChange(next);
  };

  const confirm = async (): Promise<void> => {
    if (working) return;
    setWorking(true);
    setError(null);
    try {
      await props.onConfirm();
    } catch (err) {
      setError(errMessage(err));
      setWorking(false);
      return;
    }
    setWorking(false);
    props.onOpenChange(false);
    props.onDone?.();
  };

  return (
    <Dialog open={props.open} onOpenChange={onOpenChange}>
      <DialogContent
        title={props.title}
        aria-describedby={descriptionId}
        onEscapeKeyDown={(event) => {
          if (working) event.preventDefault();
        }}
        onInteractOutside={(event) => {
          if (working) event.preventDefault();
        }}
        onCloseAutoFocus={(event) => {
          const target = props.returnFocus?.();
          if (target === null || target === undefined) return;
          event.preventDefault();
          target.focus({ preventScroll: true });
        }}
      >
        <div className={styles.body}>
          <p id={descriptionId} className={styles.description}>
            {props.description}
          </p>
          {props.notice !== undefined ? <div className={styles.notice}>{props.notice}</div> : null}
          {error !== null ? (
            <Alert tone="danger" title={props.errorTitle ?? DEFAULT_ERROR_TITLE}>
              {error}
            </Alert>
          ) : null}
          <div className={styles.footer}>
            <Button type="button" variant="ghost" size="md" disabled={working} onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button
              type="button"
              variant={tone === 'danger' ? 'danger' : 'primary'}
              size="md"
              loading={working}
              onClick={() => {
                void confirm();
              }}
            >
              {working ? props.workingLabel : props.confirmLabel}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
