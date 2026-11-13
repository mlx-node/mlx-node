import type { ReactElement } from 'react';

import { CHAT } from '../lib/chat-strings';
import { MorphLoader } from './arc-x/morph-loader/morph-loader';
import { Button } from './arc/button/button';
import { Tooltip } from './arc/tooltip/tooltip';
import styles from './unload-button.module.css';

/** The engine's Unload action, shared by the model picker and the thread's engine note. */
export interface UnloadControl {
  /** `working`: the call runs. `off`: an answer or a load holds the engine. */
  state: 'idle' | 'working' | 'off';
  onUnload: () => void;
}

export function UnloadButton({ control }: { control: UnloadControl }): ReactElement {
  const working = control.state === 'working';
  const button = (
    <Button
      type="button"
      variant="ghost"
      size="sm"
      className={styles.button}
      disabled={control.state === 'off'}
      aria-disabled={working || undefined}
      onClick={() => {
        if (control.state === 'idle') control.onUnload();
      }}
    >
      {working ? (
        <span className={styles.working}>
          <MorphLoader size={14} status="loading" />
          {CHAT.engine.unloading}
        </span>
      ) : (
        CHAT.engine.unload
      )}
    </Button>
  );
  if (control.state !== 'off') return button;
  return (
    <Tooltip content={CHAT.engine.unloadBusy}>
      <span className={styles.tipTarget}>{button}</span>
    </Tooltip>
  );
}
