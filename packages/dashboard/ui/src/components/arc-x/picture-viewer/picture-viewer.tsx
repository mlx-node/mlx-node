import * as DialogPrimitive from '@radix-ui/react-dialog';
import { ChevronLeft, ChevronRight, ImageOff, Maximize, X, ZoomIn, ZoomOut } from 'lucide-react';
import { useRef, useState, type KeyboardEvent, type ReactElement } from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import { IconButton } from '../icon-button/icon-button';
import styles from './picture-viewer.module.css';

/** One picture of the viewer: its blob URL once read, else where its read stands. */
export interface ViewerPicture {
  key: number;
  url: string | null;
  state: 'loading' | 'ready' | 'gone' | 'too-big' | 'not-shown';
  /** Names the source: the app does not know what a picture shows. */
  alt: string;
  /** The header line: the file, and the page of a PDF. */
  title: string;
  /** The thumbnail's name: "Open page 5", "Open picture 1 of 3". */
  open: string;
}

export interface PictureViewerProps {
  open: boolean;
  pictures: readonly ViewerPicture[];
  index: number;
  onIndex: (index: number) => void;
  /** Close, Esc or a press outside. */
  onClose: () => void;
  /** Where focus goes when the viewer closes on the picture at `index`; null: Radix's own. */
  returnFocus?: (index: number) => HTMLElement | null;
}

/** Zoom steps; 1 fits the picture in the window. */
const ZOOMS = [1, 1.5, 2, 3, 4] as const;

function stateText(state: ViewerPicture['state']): string {
  const p = CHAT.pictures;
  if (state === 'loading') return p.loading;
  if (state === 'too-big') return p.tooBig;
  if (state === 'not-shown') return p.notShown;
  return p.gone;
}

/**
 * A full-window viewer for the pictures of one tool call: arrows (and ← →) move, + − 0 zoom, Esc closes, focus starts
 * on Close and stays inside. A polite counter says where it is. The fallback for Arc Pro `lightbox-gallery`.
 */
export function PictureViewer(props: PictureViewerProps): ReactElement {
  const p = CHAT.pictures;
  const close = useRef<HTMLButtonElement>(null);
  const [zoom, setZoom] = useState(0);
  const [fit, setFit] = useState<{ width: number; height: number } | null>(null);
  const [shownIndex, setShownIndex] = useState(props.index);
  if (shownIndex !== props.index) {
    setShownIndex(props.index);
    setZoom(0);
    setFit(null);
  }
  const total = props.pictures.length;
  const index = Math.min(Math.max(props.index, 0), Math.max(total - 1, 0));
  const current = props.pictures[index];
  const scale = ZOOMS[zoom]!;
  const move = (step: number): void => {
    const next = index + step;
    if (next >= 0 && next < total) props.onIndex(next);
  };
  const zoomBy = (step: number): void => {
    if (current?.url == null) return;
    setZoom((value) => Math.min(Math.max(value + step, 0), ZOOMS.length - 1));
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const keys: Record<string, () => void> = {
      ArrowLeft: () => move(-1),
      ArrowRight: () => move(1),
      '+': () => zoomBy(1),
      '=': () => zoomBy(1),
      '-': () => zoomBy(-1),
      '0': () => setZoom(0),
    };
    const run = keys[event.key];
    if (run === undefined) return;
    event.preventDefault();
    run();
  };

  return (
    <DialogPrimitive.Root
      open={props.open}
      onOpenChange={(open) => {
        if (!open) props.onClose();
      }}
    >
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className={styles.overlay} />
        <DialogPrimitive.Content
          className={styles.content}
          aria-modal="true"
          aria-describedby={undefined}
          onKeyDown={onKeyDown}
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            close.current?.focus();
          }}
          onCloseAutoFocus={(event) => {
            const target = props.returnFocus?.(index) ?? null;
            if (target === null || !target.isConnected) return;
            event.preventDefault();
            target.focus({ preventScroll: true });
          }}
        >
          <DialogPrimitive.Title className={styles.srOnly}>{p.viewer}</DialogPrimitive.Title>
          <div className={styles.bar}>
            <p className={styles.title}>{current?.title ?? ''}</p>
            <span className={styles.count} aria-live="polite" aria-atomic="true">
              {total > 0 ? fill(p.count, { n: index + 1, total }) : ''}
            </span>
            <IconButton
              label={p.zoomOut}
              icon={<ZoomOut size={16} strokeWidth={1.75} />}
              disabled={zoom === 0}
              onClick={() => zoomBy(-1)}
            />
            <IconButton
              label={p.zoomIn}
              icon={<ZoomIn size={16} strokeWidth={1.75} />}
              disabled={zoom === ZOOMS.length - 1 || current?.url == null}
              onClick={() => zoomBy(1)}
            />
            <IconButton
              label={p.zoomReset}
              icon={<Maximize size={16} strokeWidth={1.75} />}
              disabled={zoom === 0}
              onClick={() => setZoom(0)}
            />
            <DialogPrimitive.Close asChild>
              <IconButton ref={close} label={p.close} icon={<X size={16} strokeWidth={1.75} />} tooltip={false} />
            </DialogPrimitive.Close>
          </div>
          <div className={styles.stage}>
            <IconButton
              label={p.prev}
              icon={<ChevronLeft size={20} strokeWidth={1.75} />}
              size="md"
              disabled={index === 0}
              onClick={() => move(-1)}
            />
            <div className={styles.frame} data-zoomed={zoom > 0 ? '' : undefined}>
              {current?.url != null ? (
                <img
                  key={current.key}
                  className={styles.picture}
                  src={current.url}
                  alt={current.alt}
                  draggable={false}
                  style={
                    zoom > 0 && fit !== null
                      ? { width: fit.width * scale, height: fit.height * scale, maxWidth: 'none', maxHeight: 'none' }
                      : undefined
                  }
                  onLoad={(event) => {
                    const img = event.currentTarget;
                    if (zoom === 0) setFit({ width: img.clientWidth, height: img.clientHeight });
                  }}
                />
              ) : current !== undefined ? (
                <div className={styles.missing}>
                  {current.state === 'loading' ? null : <ImageOff size={20} strokeWidth={1.75} aria-hidden="true" />}
                  <span>{stateText(current.state)}</span>
                </div>
              ) : null}
            </div>
            <IconButton
              label={p.next}
              icon={<ChevronRight size={20} strokeWidth={1.75} />}
              size="md"
              disabled={index >= total - 1}
              onClick={() => move(1)}
            />
          </div>
          {total > 1 ? (
            <div className={styles.thumbs}>
              {props.pictures.map((picture, i) => (
                <button
                  key={picture.key}
                  type="button"
                  className={styles.thumb}
                  aria-label={picture.open}
                  aria-current={i === index ? 'true' : undefined}
                  onClick={() => props.onIndex(i)}
                >
                  {picture.url !== null ? <img src={picture.url} alt="" draggable={false} /> : null}
                </button>
              ))}
            </div>
          ) : null}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
