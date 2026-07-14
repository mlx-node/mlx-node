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

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  // Keep the popup inside the viewport. It is anchored at left:0 of the trigger
  // pill, so when that pill sits near the right edge on a narrow screen the
  // popup overflows and gets clipped by the root overflow:hidden (the enlarged
  // coarse-pointer +/- buttons make it wider still). Nudge it back in while open,
  // and re-run on viewport changes so a popup opened in one orientation isn't
  // left with a stale offset after a rotate / split-view resize / zoom.
  useLayoutEffect(() => {
    if (!open) {
      setShiftX(0);
      return;
    }
    // Measure from the UNTRANSFORMED anchor so re-clamping is idempotent: the
    // popup is position:absolute; left:0 of the (never-transformed) wrapper, so
    // wrapper.left is its natural left edge and popup.offsetWidth is its layout
    // width — both independent of the translateX we apply. Reading the popup's
    // own getBoundingClientRect() would fold in the current shift and drift on
    // every recompute.
    const measure = () => {
      const wrap = wrapperRef.current;
      const el = popupRef.current;
      if (!wrap || !el) return;
      const margin = 8;
      const left0 = wrap.getBoundingClientRect().left;
      const right0 = left0 + el.offsetWidth;
      let dx = 0;
      if (right0 > window.innerWidth - margin) dx = window.innerWidth - margin - right0;
      if (left0 + dx < margin) dx = margin - left0;
      setShiftX(dx);
    };
    measure();
    window.addEventListener('resize', measure);
    window.visualViewport?.addEventListener('resize', measure);
    return () => {
      window.removeEventListener('resize', measure);
      window.visualViewport?.removeEventListener('resize', measure);
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
            // Never let the popup exceed the viewport; otherwise the clamp above
            // can only fit one edge and the opposite control gets cut off.
            maxWidth: 'calc(100vw - 16px)',
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
