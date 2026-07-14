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
  // coarse-pointer +/- buttons make it wider still). Measure on open and nudge
  // it back in; reset when closed.
  useLayoutEffect(() => {
    if (!open) {
      setShiftX(0);
      return;
    }
    const el = popupRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const margin = 8;
    let dx = 0;
    if (r.right > window.innerWidth - margin) dx = window.innerWidth - margin - r.right;
    if (r.left + dx < margin) dx = margin - r.left;
    setShiftX(dx);
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
