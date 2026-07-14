import { useEffect, useLayoutEffect, useRef, useState } from 'react';

export type PillStepperProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  onChange: (next: number) => void;
};

export function PillStepper({ label, value, min, max, step, disabled = false, onChange }: PillStepperProps) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const [shiftX, setShiftX] = useState(0);
  const [maxW, setMaxW] = useState<number | undefined>(undefined);

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  // Keep the popup inside the *visible* viewport. It is anchored at left:0 of the
  // trigger pill, so when that pill sits near an edge the popup overflows and
  // gets clipped by the root overflow:hidden (the enlarged coarse-pointer +/-
  // buttons make it wider still). Re-run on every viewport change so a popup
  // opened in one state isn't left with a stale offset after a rotate,
  // split-view resize, or pinch-zoom + pan.
  useLayoutEffect(() => {
    if (!open) {
      setShiftX(0);
      setMaxW(undefined);
      return;
    }
    const measure = () => {
      const wrap = wrapperRef.current;
      const el = popupRef.current;
      if (!wrap || !el) return;
      const margin = 8;
      // Clamp against the VISUAL viewport, not window.innerWidth: pinch-zoom
      // shrinks and offsets the visual viewport without reflowing the layout
      // viewport, so innerWidth would place the popup off-screen at >1x zoom.
      // offsetLeft/width are layout-viewport-relative CSS px, the same space as
      // getBoundingClientRect() on every engine that exposes visualViewport
      // (Chrome 61+/Safari 13+/Firefox 91+ all return layout-viewport rects,
      // unaffected by pinch scale); engines without it take the innerWidth
      // fallback, so the two coordinate systems never mix. `scroll` fires on
      // pan (which changes offsetLeft); `resize` on zoom/rotate.
      // Bound: the popup can shrink to ~116px (two 44px buttons + a zero-width
      // input); below that visible width (>~2.8x zoom on a phone) a sliver of
      // the far control clips, but it stays left-aligned and pannable. Not
      // worth a stacked-layout reflow for that self-inflicted extreme.
      const vv = window.visualViewport;
      const viewLeft = vv ? vv.offsetLeft : 0;
      const viewWidth = vv ? vv.width : window.innerWidth;
      const viewRight = viewLeft + viewWidth;
      // Cap the popup to the visible width (the input shrinks via minWidth:0) so
      // it can never be wider than what the user can see; use the capped width
      // for the position math so shift and cap settle together in one pass.
      const cap = Math.max(0, viewWidth - 2 * margin);
      setMaxW(cap);
      // Measure from the UNTRANSFORMED anchor so re-clamping is idempotent: the
      // popup is position:absolute; left:0 of the (never-transformed) wrapper, so
      // wrapper.left is its natural left edge, independent of the translateX we
      // apply. Reading the popup's own rect would fold in the current shift.
      const left0 = wrap.getBoundingClientRect().left;
      const right0 = left0 + Math.min(el.offsetWidth, cap);
      let dx = 0;
      if (right0 > viewRight - margin) dx = viewRight - margin - right0;
      if (left0 + dx < viewLeft + margin) dx = viewLeft + margin - left0;
      setShiftX(dx);
    };
    measure();
    const vv = window.visualViewport;
    window.addEventListener('resize', measure);
    vv?.addEventListener('resize', measure);
    vv?.addEventListener('scroll', measure);
    return () => {
      window.removeEventListener('resize', measure);
      vv?.removeEventListener('resize', measure);
      vv?.removeEventListener('scroll', measure);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function onClickOutside(e: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', onClickOutside);
    return () => document.removeEventListener('mousedown', onClickOutside);
  }, [open]);

  const fmt = (n: number) => (Number.isInteger(step) ? `${Math.round(n)}` : `${Math.round(n * 100) / 100}`);

  function clamp(n: number) {
    return Math.min(max, Math.max(min, n));
  }

  return (
    <div ref={wrapperRef} style={{ position: 'relative', display: 'inline-flex' }}>
      <button
        type="button"
        className="pill"
        onClick={() => {
          if (!disabled) setOpen((o) => !o);
        }}
        aria-haspopup="dialog"
        aria-expanded={open}
        disabled={disabled}
      >
        {label} · {fmt(value)}
      </button>
      {open && !disabled && (
        <div
          ref={popupRef}
          role="dialog"
          style={{
            position: 'absolute',
            bottom: 'calc(100% + 8px)',
            left: 0,
            transform: shiftX ? `translateX(${shiftX}px)` : undefined,
            // Never let the popup exceed the VISIBLE viewport (measured live,
            // incl. pinch-zoom); otherwise the clamp can only fit one edge and
            // the opposite control gets cut off.
            maxWidth: maxW,
            boxSizing: 'border-box',
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 10,
            padding: 8,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            zIndex: 30,
            boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
          }}
        >
          <button
            type="button"
            className="composer-icon-btn"
            disabled={disabled}
            onClick={() => onChange(clamp(value - step))}
          >
            −
          </button>
          <input
            type="number"
            min={min}
            max={max}
            step={step}
            value={value}
            disabled={disabled}
            onChange={(e) => {
              const n = Number.parseFloat(e.target.value);
              if (Number.isFinite(n)) onChange(clamp(n));
            }}
            style={{
              width: 80,
              // Let the input shrink (below its 80px basis) when the popup is
              // capped to a very small visible viewport under pinch-zoom; the
              // +/- buttons keep their tap size and the input gives way first.
              minWidth: 0,
              flexShrink: 1,
              padding: '6px 8px',
              borderRadius: 6,
              background: 'var(--surface-2)',
              border: '1px solid var(--border)',
              color: 'var(--text)',
              fontFamily: 'var(--font-mono)',
              // 16px so iOS Safari doesn't auto-zoom the page when this numeric
              // input is focused (any <16px input triggers the zoom).
              fontSize: 16,
              textAlign: 'center',
            }}
          />
          <button
            type="button"
            className="composer-icon-btn"
            disabled={disabled}
            onClick={() => onChange(clamp(value + step))}
          >
            +
          </button>
        </div>
      )}
    </div>
  );
}
