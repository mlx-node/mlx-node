import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Visualises the 5x5 causal mask for the prompt
 * "The cat sat on the". Cells on or below the diagonal are allowed (green);
 * cells above the diagonal are masked (gray with a slash). Clicking a cell
 * shows a sentence describing what that cell means.
 */

const TOKENS = ['The', ' cat', ' sat', ' on', ' the'];

// Per-locale copy. English strings are moved verbatim from the original JSX.
const COPY = {
  en: {
    heading: 'Causal mask · 5 × 5',
    legend: 'Green = allowed (i >= j) · gray = masked (i < j)',
    clickAny: 'Click any cell.',
    attends: (i: number, j: number, qName: string, kName: string) =>
      `Row ${i} attends to col ${j}: "${qName}" can see "${kName}".`,
    cannotAttend: (i: number, j: number, qName: string, kName: string) =>
      `Row ${i} cannot attend to col ${j}: "${qName}" would have to peek at the future "${kName}".`,
    keyTitle: (j: number, t: string) => `key position ${j}: ${JSON.stringify(t)}`,
    queryTitle: (i: number, t: string) => `query position ${i}: ${JSON.stringify(t)}`,
    allowedTitle: (i: number, j: number, q: string, k: string) => `(${i},${j}) allowed — "${q}" can see "${k}"`,
    maskedTitle: (i: number, j: number) => `(${i},${j}) masked — would be the future`,
    trainPara: (
      <>
        <strong>At training time</strong>, the mask makes teacher forcing safe: we feed the whole sentence in parallel
        and predict each next token, but no position can see its own future answer. Without the mask, next-token
        prediction would be a trivial copy.
      </>
    ),
    inferPara: (
      <>
        <strong>At inference time</strong>, the same mask is still applied even though token <em>N + 1</em> physically
        doesn&apos;t exist yet — this keeps the math identical between train and serve, which is what makes KV caching
        (chapter 12) valid.
      </>
    ),
  },
  zh: {
    heading: '因果掩码 · 5 × 5',
    legend: '绿色 = 允许（i >= j）· 灰色 = 掩掉（i < j）',
    clickAny: '点击任意格子。',
    attends: (i: number, j: number, qName: string, kName: string) =>
      `第 ${i} 行可以关注第 ${j} 列：“${qName}”能看到“${kName}”。`,
    cannotAttend: (i: number, j: number, qName: string, kName: string) =>
      `第 ${i} 行不能关注第 ${j} 列：“${qName}”得偷看未来的“${kName}”才行。`,
    keyTitle: (j: number, t: string) => `key 位置 ${j}：${JSON.stringify(t)}`,
    queryTitle: (i: number, t: string) => `query 位置 ${i}：${JSON.stringify(t)}`,
    allowedTitle: (i: number, j: number, q: string, k: string) => `(${i},${j}) 允许——“${q}”能看到“${k}”`,
    maskedTitle: (i: number, j: number) => `(${i},${j}) 已掩掉——那是未来`,
    trainPara: (
      <>
        <strong>训练时</strong>，掩码让 teacher forcing 变得安全：我们把整句并行喂入并预测每个下一
        token，但任何位置都看不到自己的未来答案。没有掩码，下一 token 预测就退化成简单的抄写。
      </>
    ),
    inferPara: (
      <>
        <strong>推理时</strong>，同样的掩码仍然在用，哪怕 token <em>N + 1</em>{' '}
        物理上还不存在——这让训练与推理的数学完全一致，而这正是 KV 缓存（第 12 章）成立的前提。
      </>
    ),
  },
} as const;

function renderToken(text: string): string {
  if (text.startsWith(' ')) return '·' + text.slice(1);
  return text;
}

function trimmed(text: string): string {
  return text.trim();
}

export function CausalMaskVisual() {
  const copy = COPY[useLocale()];
  const n = TOKENS.length;
  const [selected, setSelected] = React.useState<{ i: number; j: number } | null>({
    i: 4,
    j: 0,
  });

  function selectionMessage(): string {
    if (!selected) return copy.clickAny;
    const { i, j } = selected;
    const qName = trimmed(TOKENS[i] ?? '');
    const kName = trimmed(TOKENS[j] ?? '');
    if (i >= j) {
      return copy.attends(i, j, qName, kName);
    }
    return copy.cannotAttend(i, j, qName, kName);
  }

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        <div className="text-[11px] text-muted-foreground">{copy.legend}</div>
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
                  title={copy.keyTitle(j, t)}
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
                  title={copy.queryTitle(i, row)}
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
                            ? copy.allowedTitle(i, j, trimmed(row), trimmed(TOKENS[j] ?? ''))
                            : copy.maskedTitle(i, j)
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
        <p>{copy.trainPara}</p>
        <p>{copy.inferPara}</p>
      </div>
    </div>
  );
}
