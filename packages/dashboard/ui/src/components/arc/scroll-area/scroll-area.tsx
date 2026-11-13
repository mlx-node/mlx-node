
import { forwardRef, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useRef } from "react";
import type { CSSProperties, HTMLAttributes, PointerEvent as ReactPointerEvent, ReactNode, Ref, UIEvent } from "react";
import styles from "./scroll-area.module.css";

type Axis = "y" | "x";

/** Which edges still have content beyond them. */
export interface ScrollAreaEdges { top: boolean; bottom: boolean; left: boolean; right: boolean }

/**
 * A native scroll container with thin overlay scrollbars that appear while scrolling or on hover and fade away at rest.
 * Each edge fades out only when there is more content beyond it, in proportion to how much. Scrolling stays native, so
 * keyboard, touch momentum, and scroll snap all behave as the platform expects.
 */
export interface ScrollAreaProps extends Omit<HTMLAttributes<HTMLDivElement>, "onScroll"> {
  children?: ReactNode;
  /** Axes that scroll. Defaults to vertical. */
  orientation?: "vertical" | "horizontal" | "both";
  /** Length of the edge fade in px. 0 turns the fades off. */
  fade?: number;
  /** "auto" shows scrollbars while scrolling or hovering; "always" keeps them visible when content overflows. */
  scrollbars?: "auto" | "always";
  /** How long scrollbars linger after scrolling stops, in ms. */
  hideDelay?: number;
  /** Maximum height of the viewport, for vertical areas that should grow with their content. */
  maxHeight?: CSSProperties["maxHeight"];
  /** Passed to the viewport's `scroll-snap-type`, for example "x mandatory". Children set their own `scroll-snap-align`. */
  snap?: CSSProperties["scrollSnapType"];
  /** Accessible name. The viewport becomes a labelled, focusable region that arrow and page keys scroll. */
  label?: string;
  /** Turns vertical wheel movement into horizontal scrolling for horizontal areas. Defaults to true. */
  wheelToHorizontal?: boolean;
  viewportClassName?: string;
  viewportStyle?: CSSProperties;
  viewportRef?: Ref<HTMLDivElement>;
  onScroll?: (event: UIEvent<HTMLDivElement>) => void;
  /** Called when content starts or stops extending past an edge. */
  onEdgeChange?: (edges: ScrollAreaEdges) => void;
}

const MIN_THUMB = 28;

export const ScrollArea = forwardRef<HTMLDivElement, ScrollAreaProps>(function ScrollArea({
  children, orientation = "vertical", fade = 28, scrollbars = "auto", hideDelay = 900, maxHeight, snap, label,
  wheelToHorizontal = true, viewportClassName, viewportStyle, viewportRef, onScroll, onEdgeChange, className, ...rest
}, forwardedRef) {
  const rootRef = useRef<HTMLDivElement>(null);
  const viewport = useRef<HTMLDivElement>(null);
  const tracks = useRef<Record<Axis, HTMLDivElement | null>>({ x: null, y: null });
  const thumbs = useRef<Record<Axis, HTMLDivElement | null>>({ x: null, y: null });
  const hideTimer = useRef<number | undefined>(undefined);
  const lastEdges = useRef("");
  const onEdgeRef = useRef(onEdgeChange);
  useImperativeHandle(forwardedRef, () => rootRef.current as HTMLDivElement);
  useImperativeHandle(viewportRef, () => viewport.current as HTMLDivElement);
  useEffect(() => { onEdgeRef.current = onEdgeChange; }, [onEdgeChange]);

  const vertical = orientation !== "horizontal", horizontal = orientation !== "vertical";

  /* Everything visual is written straight to the DOM on each scroll frame, so scrolling never re-renders React. */
  const sync = useCallback(() => {
    const node = viewport.current, root = rootRef.current;
    if (!node || !root) return;
    const { scrollTop, scrollLeft, scrollHeight, scrollWidth, clientHeight, clientWidth } = node;
    const maxY = Math.max(0, scrollHeight - clientHeight), maxX = Math.max(0, scrollWidth - clientWidth);
    // RTL layouts report negative scrollLeft; distance from the start edge is its magnitude.
    const left = Math.abs(scrollLeft);
    const edges: ScrollAreaEdges = { top: vertical && scrollTop > .5, bottom: vertical && maxY - scrollTop > .5, left: horizontal && left > .5, right: horizontal && maxX - left > .5 };

    if (fade > 0) {
      node.style.setProperty("--fade-top", `${vertical ? Math.min(fade, scrollTop) : 0}px`);
      node.style.setProperty("--fade-bottom", `${vertical ? Math.min(fade, maxY - scrollTop) : 0}px`);
      node.style.setProperty("--fade-left", `${horizontal ? Math.min(fade, left) : 0}px`);
      node.style.setProperty("--fade-right", `${horizontal ? Math.min(fade, maxX - left) : 0}px`);
    }

    const place = (axis: Axis, scroll: number, max: number, client: number, total: number) => {
      const track = tracks.current[axis], thumb = thumbs.current[axis];
      if (!track || !thumb) return;
      const overflowing = max > .5;
      track.toggleAttribute("data-hidden", !overflowing);
      if (!overflowing) return;
      const length = axis === "y" ? track.clientHeight : track.clientWidth;
      const size = Math.max(MIN_THUMB, length * client / total);
      const offset = (length - size) * (scroll / max);
      thumb.style[axis === "y" ? "height" : "width"] = `${size}px`;
      thumb.style.transform = axis === "y" ? `translate3d(0, ${offset}px, 0)` : `translate3d(${offset}px, 0, 0)`;
    };
    if (vertical) place("y", scrollTop, maxY, clientHeight, scrollHeight);
    if (horizontal) place("x", left, maxX, clientWidth, scrollWidth);
    root.toggleAttribute("data-overflow", maxY > .5 || maxX > .5);

    const key = `${edges.top}${edges.bottom}${edges.left}${edges.right}`;
    if (key !== lastEdges.current) { lastEdges.current = key; onEdgeRef.current?.(edges); }
  }, [fade, horizontal, vertical]);

  useLayoutEffect(() => {
    const node = viewport.current;
    if (!node) return;
    sync();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => sync());
    observer.observe(node);
    Array.from(node.children).forEach(child => observer.observe(child));
    const mutations = new MutationObserver(() => { observer.disconnect(); observer.observe(node); Array.from(node.children).forEach(child => observer.observe(child)); sync(); });
    mutations.observe(node, { childList: true });
    return () => { observer.disconnect(); mutations.disconnect(); };
  }, [sync]);

  useEffect(() => () => window.clearTimeout(hideTimer.current), []);

  const wake = () => {
    const root = rootRef.current;
    if (!root) return;
    root.setAttribute("data-scrolling", "");
    window.clearTimeout(hideTimer.current);
    hideTimer.current = window.setTimeout(() => root.removeAttribute("data-scrolling"), hideDelay);
  };

  const handleScroll = (event: UIEvent<HTMLDivElement>) => {
    sync();
    wake();
    onScroll?.(event);
  };

  // A mouse wheel over a horizontal strip moves it sideways, but only while there is room, so the page still scrolls at the ends.
  useEffect(() => {
    const node = viewport.current;
    if (!node || orientation !== "horizontal" || !wheelToHorizontal) return;
    const wheel = (event: WheelEvent) => {
      if (event.ctrlKey || Math.abs(event.deltaX) >= Math.abs(event.deltaY)) return;
      const max = node.scrollWidth - node.clientWidth, at = Math.abs(node.scrollLeft);
      if (max <= 0 || (event.deltaY < 0 && at <= 0) || (event.deltaY > 0 && at >= max - .5)) return;
      event.preventDefault();
      node.scrollLeft += event.deltaY * (getComputedStyle(node).direction === "rtl" ? -1 : 1);
    };
    node.addEventListener("wheel", wheel, { passive: false });
    return () => node.removeEventListener("wheel", wheel);
  }, [orientation, wheelToHorizontal]);

  /* Dragging a thumb maps pointer travel to scroll distance; pressing the track pages toward the pointer. */
  const drag = useRef<{ axis: Axis; start: number; scroll: number; ratio: number } | null>(null);
  const onThumbDown = (event: ReactPointerEvent<HTMLDivElement>, axis: Axis) => {
    const node = viewport.current, track = tracks.current[axis], thumb = thumbs.current[axis];
    if (!node || !track || !thumb || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const length = axis === "y" ? track.clientHeight : track.clientWidth, size = axis === "y" ? thumb.offsetHeight : thumb.offsetWidth;
    const max = axis === "y" ? node.scrollHeight - node.clientHeight : node.scrollWidth - node.clientWidth;
    drag.current = { axis, start: axis === "y" ? event.clientY : event.clientX, scroll: axis === "y" ? node.scrollTop : node.scrollLeft, ratio: max / Math.max(1, length - size) };
    rootRef.current?.setAttribute("data-dragging", axis);
    node.style.scrollSnapType = "none";
  };
  const onThumbMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const state = drag.current, node = viewport.current;
    if (!state || !node) return;
    const delta = ((state.axis === "y" ? event.clientY : event.clientX) - state.start) * state.ratio;
    if (state.axis === "y") node.scrollTop = state.scroll + delta; else node.scrollLeft = state.scroll + delta;
  };
  const onThumbUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!drag.current) return;
    drag.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    rootRef.current?.removeAttribute("data-dragging");
    if (viewport.current) viewport.current.style.scrollSnapType = "";
    wake();
  };
  const onTrackDown = (event: ReactPointerEvent<HTMLDivElement>, axis: Axis) => {
    const node = viewport.current, thumb = thumbs.current[axis];
    if (!node || !thumb || event.button !== 0 || event.target !== event.currentTarget) return;
    const rect = thumb.getBoundingClientRect();
    const before = axis === "y" ? event.clientY < rect.top : event.clientX < rect.left;
    const page = (axis === "y" ? node.clientHeight : node.clientWidth) * .9 * (before ? -1 : 1);
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    node.scrollBy({ [axis === "y" ? "top" : "left"]: page, behavior: reduce ? "auto" : "smooth" });
  };

  const bar = (axis: Axis) => <div ref={node => { tracks.current[axis] = node; }} className={styles.track} data-axis={axis} data-hidden="" aria-hidden="true"
    onPointerDown={event => onTrackDown(event, axis)}>
    <div ref={node => { thumbs.current[axis] = node; }} className={styles.thumb}
      onPointerDown={event => onThumbDown(event, axis)} onPointerMove={onThumbMove} onPointerUp={onThumbUp} onPointerCancel={onThumbUp} />
  </div>;

  return <div ref={rootRef} className={[styles.root, className].filter(Boolean).join(" ")} data-orientation={orientation} data-scrollbars={scrollbars} {...rest}>
    <div ref={viewport} className={[styles.viewport, viewportClassName].filter(Boolean).join(" ")} data-fade={fade > 0 ? orientation : undefined}
      style={{ maxHeight, scrollSnapType: snap, ...viewportStyle }} tabIndex={0} role={label ? "region" : undefined} aria-label={label} onScroll={handleScroll}>
      {children}
    </div>
    {vertical && bar("y")}
    {horizontal && bar("x")}
  </div>;
});

export default ScrollArea;
