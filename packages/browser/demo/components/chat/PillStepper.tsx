import { useEffect, useRef, useState } from "react";

export type PillStepperProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  onChange: (next: number) => void;
};

export function PillStepper({
  label,
  value,
  min,
  max,
  step,
  disabled = false,
  onChange,
}: PillStepperProps) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  useEffect(() => {
    if (!open) return;
    function onClickOutside(e: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, [open]);

  const fmt = (n: number) =>
    Number.isInteger(step) ? `${Math.round(n)}` : `${Math.round(n * 100) / 100}`;

  function clamp(n: number) {
    return Math.min(max, Math.max(min, n));
  }

  return (
    <div ref={wrapperRef} style={{ position: "relative", display: "inline-flex" }}>
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
          role="dialog"
          style={{
            position: "absolute",
            bottom: "calc(100% + 8px)",
            left: 0,
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: 10,
            padding: 8,
            display: "flex",
            alignItems: "center",
            gap: 6,
            zIndex: 30,
            boxShadow: "0 8px 24px rgba(0,0,0,0.4)",
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
              padding: "6px 8px",
              borderRadius: 6,
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              color: "var(--text)",
              fontFamily: "var(--font-mono)",
              fontSize: 12,
              textAlign: "center",
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
