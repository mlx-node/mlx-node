import { TriangleAlert } from 'lucide-react';
import { Fragment, type ReactElement } from 'react';

import { KeyValue } from '../../components/arc-x/key-value/key-value';
import { LiveDot } from '../../components/arc-x/live-dot/live-dot';
import { Panel } from '../../components/arc-x/panel/panel';
import { SwapText } from '../../components/arc-x/swap-text/swap-text';
import { PROVIDERS } from '../../lib/providers-strings';
import type { ProvidersResponse } from '../../lib/providers-types';
import { chatLines, workParts } from '../../lib/providers-view';

/** What Chat and Work get from the current selection, in two sentences. */
export function NowPanel({ data }: { data: ProvidersResponse }): ReactElement {
  const chat = chatLines(data);
  const work = workParts(data);
  return (
    <Panel title={PROVIDERS.now}>
      <div className="grid gap-3">
        <KeyValue
          layout="rows"
          ariaLabel={PROVIDERS.now}
          items={[
            {
              key: 'chat',
              label: PROVIDERS.nowChat,
              value: (
                <span className="grid gap-0.5">
                  {chat.map((line, index) => (
                    <span
                      key={index}
                      className={
                        line.tone === 'danger' ? 'text-danger' : line.tone === 'warning' ? 'text-warning' : undefined
                      }
                    >
                      <SwapText text={line.text} />
                    </span>
                  ))}
                </span>
              ),
            },
            {
              key: 'work',
              label: PROVIDERS.nowWork,
              value: (
                <span>
                  {work.map((part, index) => (
                    <Fragment key={part.text}>
                      {part.missing ? (
                        <span className="inline-flex items-center gap-1 text-danger">
                          <TriangleAlert size={12} aria-hidden="true" />
                          <SwapText text={part.text} />
                        </span>
                      ) : (
                        <SwapText text={part.text} />
                      )}
                      {index < work.length - 1 ? ', ' : part.text.endsWith('.') ? null : '.'}
                    </Fragment>
                  ))}
                </span>
              ),
            },
          ]}
        />
        {data.applyPending ? (
          <span className="text-body text-secondary">
            <LiveDot tone="accent" pulse label={PROVIDERS.pending} showLabel />
          </span>
        ) : null}
      </div>
    </Panel>
  );
}
