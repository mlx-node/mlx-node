// PromptTokens — the prompt rendered as clickable, keyboard-operable token
// spans that drive the position axis of the lens grid / cross-sections.
//
// `white-space: pre-wrap` so the RAW token text (leading spaces, newlines,
// tabs) lays out exactly as the model saw it — ASCII prompts keep their shape.
// The whitespace toggle flips each span between `cleanupTokenText` (whitespace
// made visible: '' → '∅', leading space → '·', '\n' → '↵', '\t' → '→') and the
// verbatim text. The `role="button" tabIndex aria-pressed onKeyDown(Enter/Space)`
// pattern is the established in-repo standard (RangePrecisionScatter.tsx). Pure
// presentational: no worker, no `window`/`document`, no data fetch.

import { cleanupTokenText } from '../learn/inspector/TopKBars';
import type { TokenInfo } from '../../src/inspector-types';

export function PromptTokens({
  tokens,
  selectedPos,
  showWhitespace,
  onSelectPos,
}: {
  tokens: TokenInfo[];
  selectedPos: number | null;
  showWhitespace: boolean;
  onSelectPos: (pos: number) => void;
}) {
  return (
    <div
      className="rounded-lg border border-border bg-background/50 p-3 font-mono text-[13px] leading-relaxed"
      style={{ whiteSpace: 'pre-wrap' }}
    >
      {tokens.map((token, pos) => (
        <span
          key={pos}
          role="button"
          tabIndex={0}
          aria-pressed={pos === selectedPos}
          onClick={() => onSelectPos(pos)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              onSelectPos(pos);
            }
          }}
          className={[
            'cursor-pointer rounded-sm',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
            pos === selectedPos ? 'bg-primary/20 text-primary' : 'hover:bg-muted',
          ].join(' ')}
        >
          {showWhitespace ? cleanupTokenText(token.text) : token.text}
        </span>
      ))}
    </div>
  );
}
