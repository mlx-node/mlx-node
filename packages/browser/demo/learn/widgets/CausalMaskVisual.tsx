import * as React from 'react';

/**
 * Visualises the 5x5 causal mask for the prompt
 * "The cat sat on the". Cells on or below the diagonal are allowed (green);
 * cells above the diagonal are masked (gray with a slash). Clicking a cell
 * shows a sentence describing what that cell means.
 */

const TOKENS = ['The', ' cat', ' sat', ' on', ' the'];

function renderToken(text: string): string {
  if (text.startsWith(' ')) return '·' + text.slice(1);
  return text;
}

function trimmed(text: string): string {
  return text.trim();
}

export function CausalMaskVisual() {
  const n = TOKENS.length;
  const [selected, setSelected] = React.useState<{ i: number; j: number } | null>({
    i: 4,
    j: 0,
  });

  function selectionMessage(): string {
    if (!selected) return 'Click any cell.';
    const { i, j } = selected;
    const qName = trimmed(TOKENS[i] ?? '');
    const kName = trimmed(TOKENS[j] ?? '');
    if (i >= j) {
      return `Row ${i} attends to col ${j}: "${qName}" can see "${kName}".`;
    }
    return `Row ${i} cannot attend to col ${j}: "${qName}" would have to peek at the future "${kName}".`;
  }

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">Causal mask · 5 × 5</div>
        <div className="text-[11px] text-muted-foreground">Green = allowed (i &gt;= j) · gray = masked (i &lt; j)</div>
      </div>

      <div className="overflow-x-auto">
        <table className="border-collapse font-mono text-[11px]">
          <thead>
            <tr>
              <th className="px-2 py-1" />
              {TOKENS.map((t, j) => (
                <th
                  key={`col-${j}`}
                  className="px-2 py-1 text-center text-muted-foreground"
                  title={`key position ${j}: ${JSON.stringify(t)}`}
                >
                  <div>K {j}</div>
                  <div className="text-foreground/70">{renderToken(t)}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {TOKENS.map((row, i) => (
              <tr key={`row-${i}`}>
                <th
                  className="px-2 py-1 text-left text-muted-foreground"
                  title={`query position ${i}: ${JSON.stringify(row)}`}
                >
                  <div>Q {i}</div>
                  <div className="text-foreground/70">{renderToken(row)}</div>
                </th>
                {TOKENS.map((_col, j) => {
                  const allowed = i >= j;
                  const isSel = selected && selected.i === i && selected.j === j;
                  return (
                    <td key={`cell-${i}-${j}`} className="border border-border/60 p-0">
                      <button
                        type="button"
                        onClick={() => setSelected({ i, j })}
                        title={
                          allowed
                            ? `(${i},${j}) allowed — "${trimmed(row)}" can see "${trimmed(TOKENS[j] ?? '')}"`
                            : `(${i},${j}) masked — would be the future`
                        }
                        className={[
                          'block h-10 w-10 text-[11px] font-mono leading-none transition-colors focus:outline-none',
                          allowed
                            ? 'bg-emerald-500/20 text-emerald-900 dark:text-emerald-100 hover:bg-emerald-500/35'
                            : 'bg-muted/60 text-muted-foreground hover:bg-muted',
                          isSel ? 'ring-2 ring-primary ring-offset-1' : '',
                        ].join(' ')}
                      >
                        {allowed ? '✓' : '/'}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="rounded-md bg-muted/40 px-3 py-2 text-xs text-foreground/85">{selectionMessage()}</div>

      <div className="space-y-1.5 text-[12px] text-foreground/85">
        <p>
          <strong>At training time</strong>, the mask makes teacher forcing safe: we feed the whole sentence in parallel
          and predict each next token, but no position can see its own future answer. Without the mask, next-token
          prediction would be a trivial copy.
        </p>
        <p>
          <strong>At inference time</strong>, the same mask is still applied even though token <em>N + 1</em> physically
          doesn&apos;t exist yet — this keeps the math identical between train and serve, which is what makes KV caching
          (chapter 12) valid.
        </p>
      </div>
    </div>
  );
}
