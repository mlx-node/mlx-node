import { useLayoutEffect, useState, type RefObject } from 'react';

/**
 * The element's content width, observed with a ResizeObserver and coalesced to one animation frame. A measured width
 * of 0 (a hidden tab, happy-dom) gives `fallbackWidth`, so marks always exist.
 */
export function useChartSize(ref: RefObject<Element | null>, fallbackWidth = 640): number {
  const [width, setWidth] = useState(0);

  useLayoutEffect(() => {
    const element = ref.current;
    if (element === null) return;
    const read = (): void => {
      const next = Math.round(element.getBoundingClientRect().width);
      setWidth((current) => (current === next ? current : next));
    };
    read();
    if (typeof ResizeObserver !== 'function') return;
    let frame = 0;
    const observer = new ResizeObserver(() => {
      if (frame !== 0) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        read();
      });
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
      if (frame !== 0) cancelAnimationFrame(frame);
    };
  }, [ref]);

  return width > 0 ? width : fallbackWidth;
}
