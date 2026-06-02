import { cn } from '@/lib/utils';
import * as React from 'react';

/**
 * Chapter 6 supplement — the FELT problem that motivates RoPE: plain
 * self-attention is PERMUTATION-INVARIANT in its weights.
 *
 * The chapter opens by claiming a transformer "literally cannot tell 'the cat
 * sat on the mat' from 'mat the on sat cat the'". This widget makes that
 * claim something the learner SEES rather than just reads.
 *
 * Setup: a tiny bag of three tokens, each with a fixed 2-D "content" vector.
 * Crucially, position is NOT encoded anywhere — a token's vector is the same no
 * matter which slot it sits in. Toy attention weight A→B is the softmax (over
 * the whole bag) of the dot products of the CONTENT vectors, so the weight from
 * `cat` to `sat` depends ONLY on those two tokens' content, never on their
 * slots. Reorder the bag and every weight is the identical number — that
 * blindness to order is exactly the gap RoPE fills by rotating Q and K by
 * position.
 *
 * Self-contained: no model, worker, or network — pure formula-derived SVG/DOM.
 */

type TokenId = 'cat' | 'sat' | 'mat';

// Hand-picked 2-D content vectors. Chosen so the three pairwise similarities
// are clearly distinct (cat·sat differs from cat·mat differs from sat·mat),
// which makes the "same number after reordering" payoff legible. Position is
// deliberately absent: this is content only.
const CONTENT: Record<TokenId, readonly [number, number]> = {
  cat: [0.9, 0.2],
  sat: [0.5, 0.8],
  mat: [-0.3, 0.7],
};

// A handful of fixed orderings the "Shuffle order" button cycles through. Each
// is a permutation of the same bag — the slots change, the tokens don't.
const ORDERS: readonly (readonly TokenId[])[] = [
  ['cat', 'sat', 'mat'],
  ['mat', 'cat', 'sat'],
  ['sat', 'mat', 'cat'],
];

const COLOR: Record<TokenId, string> = {
  cat: 'oklch(0.72 0.18 350)', // pink
  sat: 'oklch(0.75 0.12 185)', // teal
  mat: 'oklch(0.78 0.14 90)', // amber
};

function dot(a: readonly [number, number], b: readonly [number, number]): number {
  return a[0] * b[0] + a[1] * b[1];
}

/**
 * Toy attention weights for one query token over the whole bag.
 *
 * weight(query → key) = softmax_over_keys( content(query) · content(key) ).
 *
 * The result is keyed by token IDENTITY, and it reads ONLY from CONTENT — it
 * never sees a slot index — so it is provably invariant to how the tokens are
 * ordered on screen. That invariance is the whole point of the widget, so we
 * compute identity→identity here and let the UI relabel slots around it.
 */
export function attentionWeightsFrom(query: TokenId, keys: readonly TokenId[]): Record<TokenId, number> {
  const logits = keys.map((k) => dot(CONTENT[query], CONTENT[k]));
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - max));
  const sum = exps.reduce((s, v) => s + v, 0);
  const out = {} as Record<TokenId, number>;
  keys.forEach((k, i) => {
    out[k] = exps[i]! / sum;
  });
  return out;
}

// All three tokens, used as the fixed key set. The key set is a SET, not an
// order, so attention weights computed from it are identical regardless of the
// on-screen ordering — which is exactly the property we are demonstrating.
const ALL_TOKENS: readonly TokenId[] = ['cat', 'sat', 'mat'];

function WeightBar({ token, weight, dimmed }: { token: TokenId; weight: number; dimmed: boolean }) {
  const pct = Math.round(weight * 100);
  return (
    <div className={cn('space-y-1', dimmed && 'opacity-45')}>
      <div className="flex items-baseline justify-between text-[11px]">
        <span className="font-mono text-foreground/85">{token}</span>
        <span className="font-mono tabular-nums text-muted-foreground">{weight.toFixed(2)}</span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: COLOR[token] }} />
      </div>
    </div>
  );
}

export function PermutationInvariance() {
  const [orderIdx, setOrderIdx] = React.useState(0);
  const [query, setQuery] = React.useState<TokenId>('cat');

  const order = ORDERS[orderIdx]!;

  // Weights are computed from the fixed token SET, not the on-screen order, so
  // they are the same object across every reordering. Memo keyed on `query`
  // alone makes that explicit: changing `orderIdx` does NOT recompute them.
  const weights = React.useMemo(() => attentionWeightsFrom(query, ALL_TOKENS), [query]);

  function shuffle() {
    setOrderIdx((i) => (i + 1) % ORDERS.length);
  }

  const orderLabel = order.join(' · ');

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Plain attention can&apos;t see order
        </div>
        <div className="text-[11px] text-muted-foreground">Reorder the bag — the weights don&apos;t move</div>
      </div>

      {/* Current on-screen order: the slots change when you shuffle, but each
          token keeps its identity (and its content vector). */}
      <div className="space-y-1.5">
        <div className="text-[11px] text-muted-foreground">
          Current order · <span className="font-mono text-foreground/85">{orderLabel}</span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {order.map((tok, slot) => {
            const active = tok === query;
            return (
              <button
                key={tok}
                type="button"
                onClick={() => setQuery(tok)}
                aria-pressed={active}
                className={cn(
                  'flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-left transition-colors',
                  active ? 'border-primary/60 bg-primary/10' : 'border-border/60 bg-muted/20 hover:bg-foreground/5',
                )}
              >
                <span className="font-mono text-[10px] tabular-nums text-muted-foreground">slot {slot}</span>
                <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: COLOR[tok] }} />
                <span className="font-mono text-[13px] text-foreground/90">{tok}</span>
              </button>
            );
          })}
          <button
            type="button"
            onClick={shuffle}
            className="ml-auto inline-flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1 text-xs font-medium text-foreground hover:bg-primary/20"
          >
            <span aria-hidden="true">⤭</span>
            Shuffle order
          </button>
        </div>
      </div>

      {/* The query token's attention weights over the bag. The bars are keyed by
          token identity, so they are the SAME numbers no matter where the
          tokens sit on screen above. */}
      <div className="space-y-2 rounded-md border border-border/60 bg-muted/20 p-3">
        <div className="text-[11px] text-muted-foreground">
          How much{' '}
          <span className="font-mono text-foreground/85" style={{ color: COLOR[query] }}>
            {query}
          </span>{' '}
          attends to each token · <span className="font-mono">softmax(content · content)</span>
        </div>
        <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
          {ALL_TOKENS.map((tok) => (
            <WeightBar key={tok} token={tok} weight={weights[tok]} dimmed={tok === query} />
          ))}
        </div>
      </div>

      <div className="rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1.5 text-[12px] text-foreground/90">
        <strong>Same weights, just relabeled positions.</strong> Shuffle the order and the bar for{' '}
        <span className="font-mono">{query}</span> → every token holds its exact value — the slots above moved, the
        numbers did not.
      </div>

      <p className="text-[12px] text-foreground/85">
        Plain attention only sees <em>which</em> tokens are present, not their order — swap the order and every weight
        is unchanged. That blindness is exactly the gap RoPE fills by rotating Q and K by position.
      </p>

      <p className="text-[10px] text-muted-foreground">
        Illustrative — three fixed tokens with hand-picked 2-D content vectors and no positional signal. The weight from
        one token to another is <span className="font-mono">softmax</span> of the dot product of their content vectors,
        which depends only on the two tokens, never on their slots — so it is the same across every ordering. The real
        model adds many dimensions, learned Q/K projections, and the RoPE rotation that breaks exactly this symmetry.
      </p>
    </div>
  );
}
