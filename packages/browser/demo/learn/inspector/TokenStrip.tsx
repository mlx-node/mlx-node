import { cn } from '@/lib/utils';
import * as React from 'react';

import type { TokenInfo } from '../../../src/inspector-types';

export type TokenStripProps = {
  tokens: TokenInfo[];
  /** Cell size in pixels — must match the heatmap so chips line up cell-to-cell. */
  cellSize: number;
  /** Render top (horizontal) or left (vertical column). */
  orientation: 'top' | 'left';
  /** Optional currently-highlighted token index (e.g. from heatmap hover). */
  highlightIndex?: number | null;
};

/**
 * Render Qwen tokenizer text safely for display:
 *
 *  - Replace the BPE leading-space marker "Ġ" with a literal space.
 *  - Convert leading whitespace to a faint glyph "·" so chips read clearly.
 *  - Truncate long tokens with an ellipsis but keep the full string in title.
 *  - Visualize newlines and special-token brackets verbatim.
 */
function renderTokenText(raw: string): string {
  if (!raw) return '∅';
  let text = raw.replace(/^Ġ/, ' ').replace(/Ġ/g, ' ');
  text = text.replace(/\n/g, '↵');
  text = text.replace(/\t/g, '→');
  if (text.startsWith(' ')) {
    text = '·' + text.slice(1);
  }
  if (text.length > 12) {
    text = text.slice(0, 11) + '…';
  }
  return text;
}

export function TokenStrip({ tokens, cellSize, orientation, highlightIndex = null }: TokenStripProps) {
  if (orientation === 'top') {
    return (
      <div className="flex select-none" aria-hidden="true">
        {tokens.map((tok, i) => (
          <TokenChip
            key={i}
            tok={tok}
            highlighted={i === highlightIndex}
            style={{
              width: cellSize,
              height: 28,
            }}
            stack="vertical"
          />
        ))}
      </div>
    );
  }

  return (
    <div className="flex flex-col select-none" aria-hidden="true">
      {tokens.map((tok, i) => (
        <TokenChip
          key={i}
          tok={tok}
          highlighted={i === highlightIndex}
          style={{
            width: 80,
            height: cellSize,
          }}
          stack="horizontal"
        />
      ))}
    </div>
  );
}

function TokenChip({
  tok,
  highlighted,
  style,
  stack,
}: {
  tok: TokenInfo;
  highlighted: boolean;
  style: React.CSSProperties;
  stack: 'vertical' | 'horizontal';
}) {
  return (
    <div
      title={`${tok.text} (id ${tok.id})`}
      style={style}
      className={cn(
        'flex items-center justify-center overflow-hidden border-b border-border/40 px-1 text-[0.68rem] leading-tight',
        stack === 'horizontal' && 'justify-end pr-2 text-right',
        highlighted ? 'bg-primary/20 text-primary' : 'text-muted-foreground',
      )}
    >
      <span className="truncate font-mono">{renderTokenText(tok.text)}</span>
    </div>
  );
}
