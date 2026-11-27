import { ImageOff } from 'lucide-react';
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type ReactElement,
  type ReactNode,
} from 'react';

import { DashboardApiError, getJson } from '../../../lib/api';
import { CHAT, fill } from '../../../lib/chat-strings';
import { formatBytes } from '../../../lib/format';
import type { SessionPictureResponse, TranscriptEntry } from '../../../lib/types';
import { Button } from '../../arc/button/button';
import { PictureViewer, type ViewerPicture } from '../picture-viewer/picture-viewer';
import styles from './picture-strip.module.css';

export type ToolPicture = NonNullable<Extract<TranscriptEntry, { kind: 'tool' }>['pictures']>[number];

/** Where one picture stands: read, being read, or why it cannot be shown. */
export type PictureLoad =
  | { state: 'loading' }
  | { state: 'ready'; url: string }
  | { state: 'gone' }
  | { state: 'too-big' }
  | { state: 'not-shown' };

/** The wait before the one retry of a picture the reader did not find: the agent writes its line within a buffer delay. */
export const PICTURE_RETRY_MS = 1_000;

/** A read row's count and size: "1 picture · 412 KB", "3 pictures · 1.2 MB", a PDF "3 of 12 pages · 1.4 MB". */
export function picturesDetail(pictures: readonly ToolPicture[], pdfPages: number | undefined): string {
  const p = CHAT.pictures;
  const n = pictures.length;
  const count = pdfPages !== undefined ? fill(p.pages, { n, total: pdfPages }) : n === 1 ? p.one : fill(p.many, { n });
  return `${count} · ${formatBytes(pictures.reduce((sum, picture) => sum + picture.bytes, 0))}`;
}

function pictureKey(call: string, n: number): string {
  return `${n}:${call}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** A picture's bytes as a blob URL; null when `data` is no base64. */
function blobUrl(mediaType: string, data: string): string | null {
  let binary: string;
  try {
    binary = atob(data);
  } catch {
    return null;
  }
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: mediaType }));
}

/**
 * The pictures of one session the page has read, as blob URLs, until the session closes. Each picture is asked for
 * once; a 404 is asked again once after {@link PICTURE_RETRY_MS}.
 */
export class PictureStore {
  private readonly items = new Map<string, PictureLoad>();
  private readonly running = new Map<string, Promise<void>>();
  private readonly listeners = new Set<() => void>();
  /** Bumped by {@link dispose}: a read that started before it makes no URL. */
  private generation = 0;
  version = 0;

  constructor(readonly session: string) {}

  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  get(call: string, n: number): PictureLoad | undefined {
    return this.items.get(pictureKey(call, n));
  }

  /** Reads the picture unless it is read, being read, or failed. */
  ensure(call: string, n: number): Promise<void> {
    const key = pictureKey(call, n);
    const known = this.items.get(key);
    if (known !== undefined && known.state !== 'loading') return Promise.resolve();
    const running = this.running.get(key);
    if (running !== undefined) return running;
    const generation = this.generation;
    this.set(key, { state: 'loading' });
    const read = this.read(call, n).then((load) => {
      if (this.running.get(key) === read) this.running.delete(key);
      if (generation !== this.generation) {
        if (load.state === 'ready') URL.revokeObjectURL(load.url);
        return;
      }
      this.set(key, load);
    });
    this.running.set(key, read);
    return read;
  }

  /** Forgets a picture that failed and reads it again. */
  retry(call: string, n: number): Promise<void> {
    const key = pictureKey(call, n);
    const known = this.items.get(key);
    if (known !== undefined && known.state !== 'loading' && known.state !== 'ready') this.items.delete(key);
    return this.ensure(call, n);
  }

  /** Revokes every URL and forgets every picture; reads in flight end with no URL. */
  dispose(): void {
    this.generation += 1;
    for (const load of this.items.values()) if (load.state === 'ready') URL.revokeObjectURL(load.url);
    this.items.clear();
    this.running.clear();
    this.bump();
  }

  private async read(call: string, n: number): Promise<PictureLoad> {
    const path = `/sessions/${encodeURIComponent(this.session)}/pictures/${encodeURIComponent(call)}/${n}`;
    for (let tries = 0; ; tries++) {
      try {
        const body = await getJson<SessionPictureResponse>(path);
        const url = blobUrl(body.mediaType, body.data);
        return url === null ? { state: 'gone' } : { state: 'ready', url };
      } catch (err) {
        const status = err instanceof DashboardApiError ? err.status : 0;
        if (status === 413) return { state: 'too-big' };
        if (status === 415) return { state: 'not-shown' };
        if (status !== 404 || tries > 0) return { state: 'gone' };
        await sleep(PICTURE_RETRY_MS);
      }
    }
  }

  private set(key: string, load: PictureLoad): void {
    this.items.set(key, load);
    this.bump();
  }

  private bump(): void {
    this.version += 1;
    for (const listener of this.listeners) listener();
  }
}

const PictureContext = createContext<PictureStore | null>(null);

const NO_STORE = (): (() => void) => () => {};

/** The pictures of one session: rows under it share what they read; every URL is revoked when it closes. */
export function PictureScope({ session, children }: { session: string | null; children: ReactNode }): ReactElement {
  const store = useMemo(() => (session === null ? null : new PictureStore(session)), [session]);
  useEffect(() => () => store?.dispose(), [store]);
  return <PictureContext.Provider value={store}>{children}</PictureContext.Provider>;
}

export interface PictureStripProps {
  /** The tool call that read the pictures. */
  call: string;
  pictures: readonly ToolPicture[];
  /** The file the call read, as the row names it. */
  path: string;
}

function openName(picture: ToolPicture, total: number): string {
  const p = CHAT.pictures;
  return picture.page !== undefined
    ? fill(p.openPage, { page: picture.page })
    : fill(p.open, { n: picture.n + 1, total });
}

/**
 * The pictures a tool call read, as 96 px tiles read from the session on disk when the strip mounts (its row opened);
 * a tile opens the viewer. Needs a {@link PictureScope}; without one it shows nothing.
 */
export function PictureStrip({ call, pictures, path }: PictureStripProps): ReactElement | null {
  const store = useContext(PictureContext);
  useSyncExternalStore(store?.subscribe ?? NO_STORE, () => store?.version ?? 0);
  const [viewer, setViewer] = useState({ open: false, index: 0 });
  const tiles = useRef(new Map<number, HTMLButtonElement>());
  const order = pictures.map((picture) => picture.n).join(',');

  useEffect(() => {
    if (store === null || order === '') return;
    let live = true;
    void (async () => {
      for (const n of order.split(',')) {
        if (!live) return;
        await store.ensure(call, Number(n));
      }
    })();
    return () => {
      live = false;
    };
  }, [store, call, order]);

  if (store === null || pictures.length === 0) return null;
  const p = CHAT.pictures;
  const total = pictures.length;
  const loads = pictures.map((picture) => store.get(call, picture.n) ?? { state: 'loading' as const });
  const viewed: ViewerPicture[] = pictures.map((picture, i) => {
    const load = loads[i]!;
    return {
      key: picture.n,
      url: load.state === 'ready' ? load.url : null,
      state: load.state,
      alt: picture.page !== undefined ? fill(p.altPage, { page: picture.page, path }) : fill(p.alt, { path }),
      title: picture.page !== undefined ? `${path} · ${fill(p.page, { page: picture.page })}` : path,
      open: openName(picture, total),
    };
  });
  const gone = pictures.filter((_, i) => loads[i]!.state === 'gone');
  const failures: Array<'too-big' | 'not-shown'> = (['too-big', 'not-shown'] as const).filter((state) =>
    loads.some((load) => load.state === state),
  );

  return (
    <div className={styles.root}>
      <ul className={styles.list} aria-label={p.label}>
        {pictures.map((picture, i) => {
          const load = loads[i]!;
          const caption =
            picture.page !== undefined ? (
              <span className={styles.caption} aria-hidden="true">
                {fill(p.page, { page: picture.page })}
              </span>
            ) : null;
          if (load.state === 'ready') {
            return (
              <li key={picture.n} className={styles.item}>
                <button
                  type="button"
                  className={styles.tile}
                  aria-label={viewed[i]!.open}
                  ref={(node) => {
                    if (node === null) tiles.current.delete(i);
                    else tiles.current.set(i, node);
                  }}
                  onClick={() => setViewer({ open: true, index: i })}
                >
                  <img className={styles.thumb} src={load.url} alt="" draggable={false} />
                </button>
                {caption}
              </li>
            );
          }
          return (
            <li key={picture.n} className={styles.item}>
              {load.state === 'loading' ? (
                <span className={styles.tile} data-state="loading">
                  <span className={styles.srOnly}>{p.loading}</span>
                </span>
              ) : (
                <span className={styles.tile} data-state="failed" aria-hidden="true">
                  <ImageOff size={16} strokeWidth={1.75} />
                </span>
              )}
              {caption}
            </li>
          );
        })}
      </ul>
      {gone.length > 0 ? (
        <p className={styles.note}>
          <span>{p.gone}</span>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => {
              for (const picture of gone) void store.retry(call, picture.n);
            }}
          >
            {p.again}
          </Button>
        </p>
      ) : null}
      {failures.map((state) => (
        <p key={state} className={styles.note}>
          {state === 'too-big' ? p.tooBig : p.notShown}
        </p>
      ))}
      <PictureViewer
        open={viewer.open}
        pictures={viewed}
        index={viewer.index}
        onIndex={(index) => setViewer({ open: true, index })}
        onClose={() => setViewer((shown) => ({ ...shown, open: false }))}
        returnFocus={(index) => tiles.current.get(index) ?? null}
      />
    </div>
  );
}
