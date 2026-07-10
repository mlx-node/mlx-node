// PinManager — the /jspace pinned-token rail.
//
// Pins are SINGLE token ids the reader picks by clicking a cell and adding its
// top token (Constraint 8: `derivePins` — the concept-splitting predicate used by
// the lesson and the self-test — never appears here). Each pin gets a stable
// accent from `pinColor(i)`, shared with the RankHeatmap column and the RankChart
// line so a chip and its series read as the same thing. Clicking a chip makes it
// the ACTIVE series the heatmap renders.
//
// Hard cap: at `maxPins` (= LENS_MAX_PINNED) the Add control is DISABLED with a
// visible reason, and `handleAdd` refuses the 9th client-side — the worker's
// `LENS_MAX_PINNED` hard-error (inspector-types.ts:454) never gets a chance to
// fire. Passing no `onAddPin`/`onRemovePin` renders the rail READ-ONLY (the
// model-free starter frames carry fixed concept pins).
//
// Pure presentational: no worker, no `window`/`document`, no data fetch. It never
// nests a button inside a button and never stops event propagation — the chip's
// "select" and "remove" affordances are SIBLING buttons.

import { pinColor } from '../jlens-core/colors';
import { renderTokenDisplay } from '../learn/inspector/TopKBars';

export function PinManager({
  pins,
  labelOf,
  activePinIdx,
  onSelectActive,
  maxPins,
  addCandidate = null,
  onAddPin,
  onRemovePin,
}: {
  /** Pinned token ids, index-aligned with the run's `pinned` tracks + accents. */
  pins: number[];
  /** Display text for a pinned id (from the run's pinned track). */
  labelOf: (id: number) => string;
  /** Which pin is the active heatmap series (index into `pins`), or null. */
  activePinIdx: number | null;
  onSelectActive: (idx: number) => void;
  /** Hard cap — LENS_MAX_PINNED. Never a bare literal. */
  maxPins: number;
  /** The token the Add control would pin (the selected cell's top token). */
  addCandidate?: { id: number; text: string } | null;
  /** Omit to render the rail read-only (no Add control). */
  onAddPin?: (id: number) => void;
  /** Omit to render the rail read-only (no remove buttons). */
  onRemovePin?: (id: number) => void;
}) {
  const editable = onAddPin !== undefined;
  const atCap = pins.length >= maxPins;
  const alreadyPinned = addCandidate !== null && pins.includes(addCandidate.id);
  const canAdd = editable && addCandidate !== null && !atCap && !alreadyPinned;

  let addReason = '';
  if (atCap) addReason = `Maximum ${maxPins} pins reached — remove one to add another.`;
  else if (addCandidate === null) addReason = 'Select a cell to pin its top token.';
  else if (alreadyPinned) addReason = `"${renderTokenDisplay(addCandidate.text)}" is already pinned.`;

  function handleAdd() {
    if (onAddPin === undefined || addCandidate === null) return;
    // Client-side hard cap: refuse the 9th BEFORE the worker's LENS_MAX_PINNED
    // error can fire.
    if (pins.length >= maxPins) return;
    if (pins.includes(addCandidate.id)) return;
    onAddPin(addCandidate.id);
  }

  return (
    <section aria-label="Pinned tokens" className="space-y-2">
      <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Pinned token series">
        {pins.length === 0 ? (
          <span className="text-[12px] text-muted-foreground">
            No pins yet — click a cell, then add its top token to track its rank.
          </span>
        ) : (
          pins.map((id, i) => {
            const color = pinColor(i);
            const active = i === activePinIdx;
            const label = renderTokenDisplay(labelOf(id));
            return (
              <span
                key={`${id}-${i}`}
                className="inline-flex items-center overflow-hidden rounded-md border font-mono text-[12px]"
                style={{ borderColor: color, background: active ? `${color}26` : `${color}14` }}
              >
                {/* SELECT the active series. Sibling of the remove button — never
                    nested, so no event-propagation juggling is needed. */}
                <button
                  type="button"
                  aria-pressed={active}
                  onClick={() => onSelectActive(i)}
                  className="px-2 py-1 font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                  style={{ color }}
                  title={`Rank-track "${labelOf(id)}" — click to show its layer × position field`}
                >
                  {label}
                </button>
                {onRemovePin !== undefined ? (
                  <button
                    type="button"
                    aria-label={`Remove pin ${labelOf(id)}`}
                    onClick={() => onRemovePin(id)}
                    className="px-1.5 py-1 text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                    style={{ borderLeft: `1px solid ${color}` }}
                  >
                    ×
                  </button>
                ) : null}
              </span>
            );
          })
        )}
      </div>

      {editable ? (
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={!canAdd}
            onClick={handleAdd}
            className="rounded-md border border-border bg-background px-2.5 py-1 text-[12px] font-medium transition-colors hover:bg-muted disabled:pointer-events-none disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          >
            {addCandidate !== null && canAdd
              ? `+ Pin “${renderTokenDisplay(addCandidate.text)}”`
              : '+ Pin selected token'}
          </button>
          <span className="text-[11px] text-muted-foreground">
            {pins.length} / {maxPins} pins
          </span>
          {!canAdd && addReason !== '' ? (
            <span className="text-[11px] text-muted-foreground" aria-live="polite">
              {addReason}
            </span>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
