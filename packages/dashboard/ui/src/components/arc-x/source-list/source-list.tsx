import * as PopoverPrimitive from '@radix-ui/react-popover';
import { ChevronDown } from 'lucide-react';
import { motion } from 'motion/react';
import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactElement,
  type ReactNode,
} from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import { Popover, PopoverContent } from '../../arc/popover/popover';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import { SiteMark } from '../site-mark/site-mark';
import styles from './source-list.module.css';

export { SiteMark } from '../site-mark/site-mark';

export interface SourceItem {
  url: string;
  host: string;
  title: string;
  date?: string | null;
  /** Cited by the answer; drawn first. */
  cited?: boolean;
  /** The number the answer used, when it used one. */
  n?: number;
}

export interface SourceListProps {
  sources: readonly SourceItem[];
  /** 'block': chips row that opens to the list. 'list': the list only (inside a tool row). Default 'block'. */
  layout?: 'block' | 'list';
  /** Default 'Sources'. */
  label?: string;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  /** The row to focus after a citation chip was activated. */
  focusUrl?: string | null;
}

const CHIPS = 4;
const INSTANT = { duration: 0 } as const;

function isHttp(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}

/** An external link for an http(s) address; any other address is plain text. */
function SourceLink(props: {
  url: string;
  className?: string;
  children: ReactNode;
  linkRef?: (node: HTMLAnchorElement | null) => void;
}): ReactElement {
  if (!isHttp(props.url)) return <span className={props.className}>{props.children}</span>;
  return (
    <a
      ref={props.linkRef}
      className={props.className}
      href={props.url}
      title={props.url}
      target="_blank"
      rel="noreferrer"
    >
      {props.children}
    </a>
  );
}

/** Cited sources first, in number order; then the rest in arrival order. */
function ordered(sources: readonly SourceItem[]): { cited: SourceItem[]; rest: SourceItem[] } {
  const cited = sources
    .filter((source) => source.cited === true)
    .sort((a, b) => (a.n ?? Number.MAX_SAFE_INTEGER) - (b.n ?? Number.MAX_SAFE_INTEGER));
  const rest = sources.filter((source) => source.cited !== true);
  return { cited, rest };
}

function Row({
  source,
  linkRef,
}: {
  source: SourceItem;
  linkRef: (node: HTMLAnchorElement | null) => void;
}): ReactElement {
  const facts = [source.host, source.date ?? null].filter((part): part is string => part !== null && part !== '');
  return (
    <li className={styles.row}>
      <SiteMark host={source.host} />
      <SourceLink url={source.url} className={styles.rowTitle} linkRef={linkRef}>
        {source.title !== '' ? source.title : source.host}
      </SourceLink>
      <span className={styles.rowFacts}>{facts.join(' · ')}</span>
    </li>
  );
}

function SourceRows(props: {
  sources: readonly SourceItem[];
  split: boolean;
  focusUrl: string | null | undefined;
}): ReactElement {
  const links = useRef(new Map<string, HTMLAnchorElement>());
  const { cited, rest } = props.split ? ordered(props.sources) : { cited: [], rest: [...props.sources] };
  const ref =
    (url: string) =>
    (node: HTMLAnchorElement | null): void => {
      if (node === null) links.current.delete(url);
      else links.current.set(url, node);
    };
  useLayoutEffect(() => {
    if (props.focusUrl === null || props.focusUrl === undefined) return;
    links.current.get(props.focusUrl)?.focus();
  }, [props.focusUrl]);
  return (
    <>
      {cited.length > 0 ? (
        <ul className={styles.rows}>
          {cited.map((source) => (
            <Row key={source.url} source={source} linkRef={ref(source.url)} />
          ))}
        </ul>
      ) : null}
      {cited.length > 0 && rest.length > 0 ? (
        <p className={styles.also}>
          <span className={styles.rule} aria-hidden="true" />
          {CHAT.sourcesAlso}
        </p>
      ) : null}
      {rest.length > 0 ? (
        <ul className={styles.rows}>
          {rest.map((source) => (
            <Row key={source.url} source={source} linkRef={ref(source.url)} />
          ))}
        </ul>
      ) : null}
    </>
  );
}

/** Sources of an answer, built from tool results only: chips that open to a list. Favicons are never loaded. */
export function SourceList(props: SourceListProps): ReactElement | null {
  const staticMotion = useStaticMotion();
  const listId = useId();
  const [ownOpen, setOwnOpen] = useState(false);
  const layout = props.layout ?? 'block';
  const open = props.open ?? ownOpen;
  const setOpen = (next: boolean): void => {
    if (props.open === undefined) setOwnOpen(next);
    props.onOpenChange?.(next);
  };
  if (props.sources.length === 0) return null;

  if (layout === 'list') {
    return (
      <div className={styles.list}>
        <SourceRows sources={props.sources} split={false} focusUrl={props.focusUrl} />
      </div>
    );
  }

  const { cited, rest } = ordered(props.sources);
  const chips = (cited.length > 0 ? cited : rest).slice(0, CHIPS);
  const more = props.sources.length - chips.length;
  return (
    <div className={styles.block} data-state={open ? 'open' : 'closed'}>
      <div className={styles.head}>
        <span className={styles.label}>{props.label ?? CHAT.sources}</span>
        {chips.map((source, index) => (
          <SourceLink key={source.url} url={source.url} className={styles.chip}>
            <span
              className={styles.chipInner}
              style={{ '--step': Math.min(index, motionX.reveal.maxSteps - 1) } as CSSProperties}
            >
              <SiteMark host={source.host} />
              <span className={styles.chipHost}>{source.host}</span>
            </span>
          </SourceLink>
        ))}
        <button
          type="button"
          className={styles.more}
          aria-expanded={open}
          aria-controls={listId}
          aria-label={fill(CHAT.sourcesAll, { n: props.sources.length })}
          onClick={() => setOpen(!open)}
        >
          {more > 0 ? `+${more}` : null}
          <ChevronDown className={styles.moreChevron} size={12} strokeWidth={1.75} aria-hidden="true" />
        </button>
      </div>
      {open ? (
        <motion.div
          id={listId}
          className={styles.openClip}
          initial={staticMotion ? false : { height: 0 }}
          animate={{ height: 'auto' }}
          transition={staticMotion ? INSTANT : motionX.spring.expand}
        >
          <SourceRows sources={props.sources} split focusUrl={props.focusUrl} />
        </motion.div>
      ) : null}
    </div>
  );
}

export interface CitationChipProps {
  n: number;
  title: string;
  host: string;
  onSelect: () => void;
}

/** A `[n]` mark of an answer that names a real source: hover or focus shows the title and host. */
export function CitationChip({ n, title, host, onSelect }: CitationChipProps): ReactElement {
  const [open, setOpen] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const clear = (): void => {
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = null;
  };
  useEffect(() => clear, []);
  const show = (delayed: boolean): void => {
    clear();
    if (!delayed) {
      setOpen(true);
      return;
    }
    timer.current = setTimeout(() => setOpen(true), motionX.hoverHold * 1000);
  };
  const hide = (): void => {
    clear();
    setOpen(false);
  };
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverPrimitive.Anchor asChild>
        <button
          type="button"
          className={styles.cite}
          aria-label={fill(CHAT.citation, { n, title, host })}
          onPointerEnter={() => show(true)}
          onPointerLeave={hide}
          onFocus={() => show(false)}
          onBlur={hide}
          onClick={() => {
            hide();
            onSelect();
          }}
        >
          {n}
        </button>
      </PopoverPrimitive.Anchor>
      <PopoverContent
        side="top"
        align="center"
        className={styles.citePanel}
        onOpenAutoFocus={(event) => event.preventDefault()}
        onCloseAutoFocus={(event) => event.preventDefault()}
      >
        <span className={styles.citeTitle}>{title}</span>
        <span className={styles.citeHost}>
          <SiteMark host={host} />
          {host}
        </span>
      </PopoverContent>
    </Popover>
  );
}
