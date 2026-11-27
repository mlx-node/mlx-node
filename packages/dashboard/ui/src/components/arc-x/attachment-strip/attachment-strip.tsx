import { FileText, Image as ImageIcon, X } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { type MouseEvent, type ReactElement } from 'react';

import type { PdfWay, UserAttachment } from '../../../../../src/chat/attachment-rule.js';
import { CHAT, fill } from '../../../lib/chat-strings';
import { formatBytes } from '../../../lib/format';
import { IconButton } from '../icon-button/icon-button';
import { MiddleText } from '../middle-text/middle-text';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import { Meter } from '../split-bar/split-bar';
import styles from './attachment-strip.module.css';

export interface AttachmentChipItem {
  id: string;
  kind: 'file' | 'image' | 'pdf';
  /** A thread picture has no name and reads `CHAT.attach.image`. */
  name?: string;
  /** A picture's type, e.g. `image/jpeg`. */
  mediaType?: string;
  bytes: number;
  /** A composer picture's data URL, or a composer PDF's first page, drawn as the chip's thumbnail. */
  thumbnail?: string;
  /** A PDF's pages. */
  pages?: number;
  /** A sent PDF read by the PDF tool rather than carried in its message. */
  way?: PdfWay;
  /** A PDF whose pages are still being drawn. */
  drawing?: boolean;
  /** Why this chip keeps Send off; shown in place of the size. */
  problem?: string;
}

export interface AttachmentStripProps {
  items: readonly AttachmentChipItem[];
  onRemove?: (id: string) => void;
  /** Thread chips: no remove button and no thumbnail. */
  readOnly?: boolean;
  /** One line under the chips: why the last files were not added. */
  note?: string;
  caption?: { text: string; tone: 'muted' | 'warning' };
  /** A bar under the chips: the pages of a PDF drawn so far; `text` is its value in words. */
  progress?: { label: string; text: string; done: number; total: number };
  /** Name of the list. Default `CHAT.attach.list`. */
  label?: string;
}

/** The chips of a sent message: a file or a PDF by name and size, a picture by type and size (the agent keeps no name). */
export function sentChips(attachments: readonly UserAttachment[]): AttachmentChipItem[] {
  return attachments.map((item, index) => {
    if (item.kind === 'image') return { id: `${index}`, kind: 'image', mediaType: item.mediaType, bytes: item.bytes };
    if (item.kind === 'pdf') {
      return { id: `${index}`, kind: 'pdf', name: item.name, bytes: item.bytes, pages: item.pages, way: item.way };
    }
    return { id: `${index}`, kind: 'file', name: item.name, bytes: item.bytes };
  });
}

/** `JPEG` for `image/jpeg`; '' for a type with no subtype. */
function typeLabel(mediaType: string | undefined): string {
  if (mediaType === undefined) return '';
  const slash = mediaType.indexOf('/');
  return slash === -1 ? '' : mediaType.slice(slash + 1).toUpperCase();
}

function metaOf(item: AttachmentChipItem): string {
  const a = CHAT.attach;
  const size = formatBytes(item.bytes);
  if (item.kind === 'file') return size;
  if (item.kind === 'pdf') {
    if (item.drawing === true) return a.pdfDrawing;
    const n = item.pages ?? 0;
    if (item.way === 'tool') return n === 1 ? a.pdfToolOne : fill(a.pdfTool, { n });
    return n === 1 ? fill(a.pdfPage, { size }) : fill(a.pdfPages, { n, size });
  }
  const type = typeLabel(item.mediaType);
  return type === '' ? size : `${type} · ${size}`;
}

/** A sent PDF's spoken name: its name, its pages and its size in one phrase. */
function spokenPdf(item: AttachmentChipItem, name: string): string {
  const a = CHAT.attach;
  const n = item.pages ?? 0;
  const size = formatBytes(item.bytes);
  if (item.way === 'tool') return fill(a.pdfSpokenTool, { name, n });
  return n === 1 ? fill(a.pdfSpokenOne, { name, size }) : fill(a.pdfSpoken, { name, n, size });
}

/** The files of a message as chips: thumbnail or icon, name, size and remove; one status line; a token caption. */
export function AttachmentStrip({
  items,
  onRemove,
  readOnly = false,
  note,
  caption,
  progress,
  label = CHAT.attach.list,
}: AttachmentStripProps): ReactElement | null {
  const staticMotion = useStaticMotion();
  if (items.length === 0 && note === undefined) return null;
  const ids = items.map((item) => item.id).join(' ');

  // Focus goes to the next chip's remove button, else to the message field of the form around the strip.
  const remove = (event: MouseEvent<HTMLButtonElement>, index: number): void => {
    const button = event.currentTarget;
    const nextId = items[index + 1]?.id;
    let target: HTMLElement | null = null;
    for (const chip of button.closest('[role="list"]')?.querySelectorAll<HTMLElement>('[role="listitem"]') ?? []) {
      if (nextId !== undefined && chip.dataset.id === nextId) target = chip.querySelector<HTMLElement>('[data-remove]');
    }
    target ??= button.closest('form')?.querySelector('textarea') ?? null;
    onRemove?.(items[index]!.id);
    target?.focus();
  };

  return (
    <div className={styles.root}>
      {items.length > 0 ? (
        <div className={styles.row}>
          <div className={styles.list} role="list" aria-label={label}>
            <AnimatePresence initial={false}>
              {items.map((item, index) => {
                const name = item.name ?? CHAT.attach.image;
                const thumb = !readOnly && item.kind !== 'file' && item.thumbnail !== undefined;
                const Icon = item.kind === 'image' ? ImageIcon : FileText;
                return (
                  <motion.div
                    key={item.id}
                    role="listitem"
                    aria-label={readOnly && item.kind === 'pdf' ? spokenPdf(item, name) : undefined}
                    data-id={item.id}
                    className={styles.chip}
                    data-problem={item.problem !== undefined ? '' : undefined}
                    layout={!staticMotion}
                    layoutDependency={ids}
                    initial={staticMotion ? false : { opacity: 0, scale: 0.97 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={staticMotion ? { opacity: 0, transition: { duration: 0 } } : { opacity: 0, scale: 0.95 }}
                    transition={staticMotion ? { duration: 0 } : motionX.spring.expand}
                  >
                    {thumb && item.kind === 'pdf' ? (
                      <motion.img
                        src={item.thumbnail}
                        alt=""
                        className={styles.thumb}
                        data-page=""
                        initial={staticMotion ? false : { opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={staticMotion ? { duration: 0 } : motionX.spring.expand}
                      />
                    ) : thumb ? (
                      <img src={item.thumbnail} alt="" className={styles.thumb} />
                    ) : (
                      <Icon className={styles.icon} size={14} strokeWidth={1.75} aria-hidden="true" />
                    )}
                    <span className={styles.text}>
                      <MiddleText text={name} className={styles.name} />
                      <span className={styles.meta}>{item.problem ?? metaOf(item)}</span>
                    </span>
                    {readOnly ? null : (
                      <IconButton
                        data-remove=""
                        label={fill(CHAT.attach.remove, { name })}
                        icon={<X size={12} strokeWidth={1.75} aria-hidden="true" />}
                        onClick={(event) => remove(event, index)}
                      />
                    )}
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
          {caption !== undefined ? (
            <span className={styles.caption} data-tone={caption.tone}>
              {caption.text}
            </span>
          ) : null}
        </div>
      ) : null}
      {progress !== undefined ? (
        <Meter
          kind="progress"
          label={progress.label}
          value={progress.done}
          max={progress.total}
          formatValue={() => progress.text}
          size="dense"
          warnAt={2}
          dangerAt={2}
          className={styles.progress}
        />
      ) : null}
      {note !== undefined ? (
        <p className={styles.note} role="status">
          {note}
        </p>
      ) : null}
    </div>
  );
}
