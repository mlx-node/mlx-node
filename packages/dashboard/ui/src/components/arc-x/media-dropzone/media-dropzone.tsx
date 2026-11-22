import { FileText, Paperclip, RotateCw, Trash2 } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import {
  type ChangeEvent,
  type ClipboardEvent,
  type DragEvent,
  type ReactElement,
  type ReactNode,
  useRef,
  useState,
} from 'react';

import { cn } from '../../../lib/cn';
import { formatBytes, formatNumber } from '../../../lib/format';
import {
  CLAIRIC_MAX_IMAGES,
  CLAIRIC_MAX_MEDIA_BYTES,
  CLAIRIC_MAX_TOTAL_MEDIA_BYTES,
  type ClairicImageRotation,
  type MediaAttachment,
  nextDraftId,
} from '../../../lib/system-one';
import { Alert } from '../../arc/alert/alert';
import { Button } from '../../arc/button/button';
import { IconButton } from '../icon-button/icon-button';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './media-dropzone.module.css';

const ACCEPTED_MIME = new Set(['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'application/pdf']);

export interface MediaDropzoneProps {
  attachments: readonly MediaAttachment[];
  onChange: (next: MediaAttachment[]) => void;
  disabled?: boolean;
  /** Optional trailing action inside the dropzone bar. */
  extraAction?: ReactNode;
  className?: string;
  /** When set alongside `onTextChange`, renders a single unified text + media composer surface. */
  text?: string;
  onTextChange?: (next: string) => void;
  textId?: string;
  placeholder?: string;
  /** Allow dropping/pasting/attaching images & PDFs. Defaults to true. */
  allowMedia?: boolean;
  onClear?: () => void;
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') resolve(reader.result);
      else reject(new Error('Could not read file'));
    };
    reader.onerror = () => reject(reader.error ?? new Error('Could not read file'));
    reader.readAsDataURL(file);
  });
}

function nextRotation(current: ClairicImageRotation): ClairicImageRotation {
  if (current === 0) return 90;
  if (current === 90) return 180;
  if (current === 180) return 270;
  return 0;
}

function normalizeMime(file: File): string {
  if (file.type !== '') return file.type.toLowerCase();
  const lower = file.name.toLowerCase();
  if (lower.endsWith('.png')) return 'image/png';
  if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) return 'image/jpeg';
  if (lower.endsWith('.webp')) return 'image/webp';
  if (lower.endsWith('.pdf')) return 'application/pdf';
  return '';
}

/**
 * Compact Clairic media dropzone: single-row dashed bar when empty, horizontal thumbnail strip when attached.
 * Enforces up to 4 images or PDFs, 4 MiB per file and 8 MiB combined, with per-image 90-degree rotation.
 */
export function MediaDropzone({
  attachments,
  onChange,
  disabled = false,
  extraAction,
  className,
  text,
  onTextChange,
  textId,
  placeholder,
  allowMedia = true,
  onClear,
}: MediaDropzoneProps): ReactElement {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const staticMotion = useStaticMotion();
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const totalBytes = attachments.reduce((sum, item) => sum + item.sizeBytes, 0);
  const isComposer = text !== undefined && onTextChange !== undefined;

  const ingestFiles = async (fileList: FileList | File[]): Promise<void> => {
    setError(null);
    const incoming = Array.from(fileList);
    if (incoming.length === 0) return;

    const added: MediaAttachment[] = [];
    const errors: string[] = [];
    let runningCount = attachments.length;
    let runningBytes = totalBytes;

    for (const file of incoming) {
      if (runningCount >= CLAIRIC_MAX_IMAGES) {
        errors.push(`Up to ${String(CLAIRIC_MAX_IMAGES)} images or PDFs are supported per request.`);
        break;
      }
      const mime = normalizeMime(file);
      if (!ACCEPTED_MIME.has(mime)) {
        errors.push(`"${file.name}" is not a supported PNG, JPEG, WebP, or PDF file.`);
        continue;
      }
      if (file.size > CLAIRIC_MAX_MEDIA_BYTES) {
        errors.push(`"${file.name}" (${formatBytes(file.size)}) exceeds the 4 MiB per-file limit.`);
        continue;
      }
      if (runningBytes + file.size > CLAIRIC_MAX_TOTAL_MEDIA_BYTES) {
        errors.push(`Adding "${file.name}" exceeds the 8 MiB combined media limit.`);
        continue;
      }
      try {
        const dataUrl = await readFileAsDataUrl(file);
        const comma = dataUrl.indexOf(',');
        const base64 = comma === -1 ? '' : dataUrl.slice(comma + 1);
        const kind = mime === 'application/pdf' ? 'pdf' : 'image';
        added.push({
          id: nextDraftId(kind),
          name: file.name,
          contentType: mime === 'image/jpg' ? 'image/jpeg' : mime,
          sizeBytes: file.size,
          base64,
          dataUrl,
          rotation: 0,
          kind,
        });
        runningCount += 1;
        runningBytes += file.size;
      } catch {
        errors.push(`Could not read "${file.name}".`);
      }
    }

    if (errors.length > 0) setError(errors.join(' '));
    if (added.length > 0) onChange([...attachments, ...added]);
  };

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>): void => {
    const files = event.target.files;
    if (files !== null && files.length > 0) {
      void ingestFiles(files);
    }
    event.target.value = '';
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>): void => {
    event.preventDefault();
    setDragging(false);
    if (disabled || !allowMedia) return;
    if (event.dataTransfer.files.length > 0) {
      void ingestFiles(event.dataTransfer.files);
    }
  };

  const handlePaste = (event: ClipboardEvent<HTMLTextAreaElement>): void => {
    if (disabled || !allowMedia) return;
    const files: File[] = [];
    for (const item of Array.from(event.clipboardData.items)) {
      if (item.kind === 'file') {
        const file = item.getAsFile();
        if (file !== null) files.push(file);
      }
    }
    if (files.length > 0) {
      event.preventDefault();
      void ingestFiles(files);
    }
  };

  const rotateItem = (id: string): void => {
    onChange(attachments.map((item) => (item.id === id ? { ...item, rotation: nextRotation(item.rotation) } : item)));
  };

  const removeItem = (id: string): void => {
    setError(null);
    onChange(attachments.filter((item) => item.id !== id));
  };

  const summaryCopy =
    attachments.length === 0
      ? 'Drop images or PDF (up to 4, 8 MB max)'
      : `${String(attachments.length)}/${String(CLAIRIC_MAX_IMAGES)} attached · ${formatBytes(totalBytes)} / 8 MB`;

  const attachmentStrip =
    attachments.length > 0 ? (
      <div className={styles.strip} role="list" aria-label="Attached media">
        <AnimatePresence initial={false}>
          {attachments.map((item) => (
            <motion.div
              key={item.id}
              role="listitem"
              layout={!staticMotion}
              initial={staticMotion ? false : { opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={staticMotion ? { opacity: 0, transition: { duration: 0 } } : { opacity: 0, scale: 0.95 }}
              transition={staticMotion ? { duration: 0 } : motionX.spring.expand}
              className={styles.item}
            >
              <div className={styles.thumbWrap}>
                {item.kind === 'image' ? (
                  <>
                    <img
                      src={item.dataUrl}
                      alt={item.name}
                      className={styles.thumb}
                      style={{ transform: `rotate(${String(item.rotation)}deg)` }}
                    />
                    {item.rotation !== 0 ? (
                      <span className={styles.rotBadge}>{`${String(item.rotation)}°`}</span>
                    ) : null}
                  </>
                ) : (
                  <FileText size={16} strokeWidth={1.75} className={styles.pdfIcon} aria-hidden="true" />
                )}
              </div>
              <div className={styles.fileMeta}>
                <span className={styles.fileName} title={item.name}>
                  {item.name}
                </span>
                <span className={styles.fileSize}>
                  {item.kind === 'pdf' ? `PDF · ${formatBytes(item.sizeBytes)}` : formatBytes(item.sizeBytes)}
                </span>
              </div>
              <div className={styles.itemBtns}>
                {item.kind === 'image' ? (
                  <IconButton
                    label={`Rotate ${item.name} (${String(item.rotation)}°)`}
                    icon={<RotateCw size={13} strokeWidth={1.75} aria-hidden="true" />}
                    onClick={() => rotateItem(item.id)}
                    disabled={disabled}
                  />
                ) : null}
                <IconButton
                  label={`Remove ${item.name}`}
                  icon={<Trash2 size={13} strokeWidth={1.75} aria-hidden="true" />}
                  onClick={() => removeItem(item.id)}
                  disabled={disabled}
                />
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    ) : null;

  if (isComposer) {
    const hasContent = text.length > 0 || attachments.length > 0;
    const compactText = allowMedia && attachments.length > 0 && text.length === 0;
    const resolvedPlaceholder = compactText
      ? 'Optional note or extra context for the attached file…'
      : (placeholder ?? 'Paste text, or drop / paste an image or PDF here…');

    return (
      <div className={cn(styles.root, className)}>
        <div
          className={styles.composer}
          data-dragging={dragging ? 'true' : undefined}
          data-disabled={disabled ? 'true' : undefined}
          onDragOver={(event) => {
            if (!allowMedia) return;
            event.preventDefault();
            if (!disabled) setDragging(true);
          }}
          onDragLeave={(event) => {
            if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
            setDragging(false);
          }}
          onDrop={handleDrop}
        >
          {allowMedia ? (
            <input
              ref={inputRef}
              type="file"
              className={styles.hiddenInput}
              accept="image/png,image/jpeg,image/webp,application/pdf"
              multiple
              disabled={disabled}
              onChange={handleInputChange}
              aria-label="Attach images or PDFs"
            />
          ) : null}

          <textarea
            id={textId}
            className={styles.composerTextarea}
            data-compact={compactText ? 'true' : undefined}
            value={text}
            disabled={disabled}
            placeholder={resolvedPlaceholder}
            onPaste={handlePaste}
            onChange={(event) => onTextChange(event.target.value)}
          />

          {allowMedia && attachmentStrip !== null ? (
            <div className={styles.composerMedia}>{attachmentStrip}</div>
          ) : null}

          <div className={styles.composerBar}>
            <div className={styles.composerBarLeft}>
              {allowMedia ? (
                <>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    disabled={disabled || attachments.length >= CLAIRIC_MAX_IMAGES}
                    onClick={() => inputRef.current?.click()}
                    className={styles.attachBtn}
                  >
                    <Paperclip size={13} strokeWidth={1.75} aria-hidden="true" />
                    {attachments.length > 0 ? 'Add file' : 'Attach image or PDF'}
                  </Button>
                  <span className={styles.copy}>
                    {attachments.length > 0
                      ? `${String(attachments.length)}/${String(CLAIRIC_MAX_IMAGES)} · ${formatBytes(totalBytes)}`
                      : 'or drag & drop / ⌘V'}
                  </span>
                </>
              ) : (
                <span className={styles.copy}>Plain text or JSON</span>
              )}
            </div>

            <div className={styles.composerBarRight}>
              {text.length > 0 ? <span className={styles.copy}>{`${formatNumber(text.length)} chars`}</span> : null}
              {hasContent && onClear !== undefined ? (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  disabled={disabled}
                  onClick={onClear}
                  className={styles.attachBtn}
                >
                  Clear
                </Button>
              ) : null}
            </div>
          </div>
        </div>

        {error !== null ? (
          <Alert tone="warning" title="Media limit">
            {error}
          </Alert>
        ) : null}
      </div>
    );
  }

  return (
    <div className={cn(styles.root, className)}>
      <div
        className={styles.bar}
        data-dragging={dragging ? 'true' : undefined}
        data-disabled={disabled ? 'true' : undefined}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={(event) => {
          if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
          setDragging(false);
        }}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          className={styles.hiddenInput}
          accept="image/png,image/jpeg,image/webp,application/pdf"
          multiple
          disabled={disabled}
          onChange={handleInputChange}
          aria-label="Attach images or PDFs"
        />
        <span className={styles.prompt}>
          <Paperclip size={14} strokeWidth={1.75} className={styles.icon} aria-hidden="true" />
          <span className={styles.copy}>{summaryCopy}</span>
        </span>
        <span className={styles.actions}>
          {extraAction}
          <Button
            type="button"
            variant="ghost"
            size="sm"
            disabled={disabled || attachments.length >= CLAIRIC_MAX_IMAGES}
            onClick={() => inputRef.current?.click()}
            className={styles.attachBtn}
          >
            Attach
          </Button>
        </span>
      </div>

      {error !== null ? (
        <Alert tone="warning" title="Media limit">
          {error}
        </Alert>
      ) : null}

      {attachmentStrip}
    </div>
  );
}
