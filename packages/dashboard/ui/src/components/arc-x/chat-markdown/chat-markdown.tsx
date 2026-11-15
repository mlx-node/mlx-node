import { Component, memo, useRef, type CSSProperties, type ReactElement, type ReactNode } from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import { cn } from '../../../lib/cn';
import { parseMarkdown, plainText, type MdBlock, type MdInline } from '../../../lib/markdown-parse';
import { repairTail } from '../../../lib/stream-repair';
import { settledSplit } from '../../../lib/stream-split';
import { displayHost, hostKey, httpUrl } from '../../../lib/web-tools';
import { CodePanel } from '../code-panel/code-panel';
import { useRichMotion } from '../motion/use-entrance';
import { CitationChip } from '../source-list/source-list';
import styles from './chat-markdown.module.css';

export type LinkRule = { mode: 'chat'; hosts: ReadonlySet<string> } | { mode: 'work' };

export interface ChatMarkdownProps {
  text: string;
  /** The answer still arrives: each settled block is parsed once, the open tail is repaired for each paint. */
  streaming?: boolean;
  /** Length of `text` already shown before this paint; text after it fades in. */
  freshFrom?: number;
  /**
   * 'chat': a link is a link only when its host is one of `hosts`; any other link is its words plus the host as
   * text. 'work': every http(s) link is a link, followed by its host when the words are not the address.
   */
  links: LinkRule;
  /** `[n]` marks that name a real source; any other `[n]` stays text. */
  citations?: ReadonlyMap<number, { title: string; host: string }>;
  onCite?: (n: number) => void;
  className?: string;
}

interface RenderContext {
  links: LinkRule;
  citations: ReadonlyMap<number, { title: string; host: string }> | undefined;
  onCite: ((n: number) => void) | undefined;
  /** Text at or after this offset fades in; undefined: nothing fades. */
  fresh: number | undefined;
}

/** Bidi embedding, override and isolate controls and direction marks: they could make a host or a word read in another order. */
function isBidiControl(code: number): boolean {
  return (
    (code >= 0x202a && code <= 0x202e) ||
    (code >= 0x2066 && code <= 0x2069) ||
    code === 0x200e ||
    code === 0x200f ||
    code === 0x061c
  );
}

/** The text without bidi controls. */
export function visibleText(text: string): string {
  let out = '';
  let start = 0;
  for (let i = 0; i < text.length; i += 1) {
    if (isBidiControl(text.charCodeAt(i))) {
      out += text.slice(start, i);
      start = i + 1;
    }
  }
  return start === 0 ? text : out + text.slice(start);
}

function textNode(node: Extract<MdInline, { type: 'text' }>, ctx: RenderContext, key: string): ReactNode {
  const fresh = ctx.fresh;
  if (fresh === undefined || node.at + node.len <= fresh) return visibleText(node.value);
  if (node.at >= fresh || node.len !== node.value.length) {
    return (
      <span key={`${key}-f${fresh}`} className={styles.fresh} data-fresh="">
        {visibleText(node.value)}
      </span>
    );
  }
  const cut = fresh - node.at;
  return (
    <span key={key}>
      {visibleText(node.value.slice(0, cut))}
      <span key={`f${fresh}`} className={styles.fresh} data-fresh="">
        {visibleText(node.value.slice(cut))}
      </span>
    </span>
  );
}

function renderLink(href: string, children: ReactNode, words: string, ctx: RenderContext, key: string): ReactNode {
  const url = httpUrl(href.trim());
  if (url === null) return <span key={key}>{children}</span>;
  const isAddress = words.trim() === href.trim();
  const host = (
    <bdi key="host" className={styles.host}>
      {` (${displayHost(url)})`}
    </bdi>
  );
  const active = ctx.links.mode === 'work' || ctx.links.hosts.has(hostKey(url));
  if (!active) {
    return (
      <span key={key}>
        {children}
        {isAddress ? null : host}
      </span>
    );
  }
  const showHost = !isAddress && (ctx.links.mode === 'work' || namesOtherHost(words, url));
  return (
    <span key={key}>
      <a className={styles.link} href={url.href} title={visibleText(url.href)} target="_blank" rel="noreferrer">
        {children}
      </a>
      {showHost ? host : null}
    </span>
  );
}

const MAX_NAMED_WORDS = 64;

/** A word of link text that reads as an address of another host than the link opens (or as one that does not parse). */
function wordNamesOtherHost(raw: string, url: URL): boolean {
  let from = 0;
  let to = raw.length;
  while (from < to && '("\'<'.includes(raw[from]!)) from += 1;
  while (to > from && '.,;:!?)"\'>'.includes(raw[to - 1]!)) to -= 1;
  const word = raw.slice(from, to);
  const scheme = word.includes('://');
  if (!scheme && !word.includes('.')) return false;
  const named = httpUrl(scheme ? word : `https://${word}`);
  if (named === null) return scheme;
  return displayHost(named) !== displayHost(url);
}

/** Any word of the link text names another host; past a fixed number of address-like words, assume one does. */
function namesOtherHost(words: string, url: URL): boolean {
  const text = visibleText(words);
  let seen = 0;
  let start = 0;
  for (let i = 0; i <= text.length; i += 1) {
    const ch = text[i];
    if (ch !== undefined && ch !== ' ' && ch !== '\t' && ch !== '\n') continue;
    const word = text.slice(start, i);
    start = i + 1;
    if (!word.includes('.') && !word.includes('://')) continue;
    seen += 1;
    if (seen > MAX_NAMED_WORDS || wordNamesOtherHost(word, url)) return true;
  }
  return false;
}

function inlines(nodes: readonly MdInline[], ctx: RenderContext, prefix: string): ReactNode[] {
  return nodes.map((node, index) => {
    const key = `${prefix}${index}`;
    switch (node.type) {
      case 'text':
        return textNode(node, ctx, key);
      case 'code':
        return (
          <code key={key} className={styles.inlineCode}>
            {visibleText(node.value)}
          </code>
        );
      case 'em':
        return <em key={key}>{inlines(node.children, ctx, `${key}.`)}</em>;
      case 'strong':
        return <strong key={key}>{inlines(node.children, ctx, `${key}.`)}</strong>;
      case 'del':
        return <del key={key}>{inlines(node.children, ctx, `${key}.`)}</del>;
      case 'link':
        return renderLink(node.href, inlines(node.children, ctx, `${key}.`), plainText(node.children), ctx, key);
      case 'image': {
        const alt = visibleText(node.alt).trim();
        const words = alt === '' ? CHAT.imageLinkBare : fill(CHAT.imageLink, { alt });
        return renderLink(node.href, words, words, ctx, key);
      }
      case 'cite': {
        const source = ctx.citations?.get(node.n);
        if (source === undefined || ctx.onCite === undefined) return `[${node.n}]`;
        const onCite = ctx.onCite;
        return (
          <CitationChip key={key} n={node.n} title={source.title} host={source.host} onSelect={() => onCite(node.n)} />
        );
      }
      case 'break':
        return <br key={key} />;
      case 'soft':
        return '\n';
    }
  });
}

const HEADING_TAGS = ['h3', 'h4', 'h5', 'h6', 'h6', 'h6'] as const;
const HEADING_CLASS = [styles.h1, styles.h2, styles.h3, styles.h3, styles.h3, styles.h3];

function blocks(list: readonly MdBlock[], ctx: RenderContext, prefix: string, tight = false): ReactNode[] {
  return list.map((block, index) => {
    const key = `${prefix}${index}`;
    const freshBlock = ctx.fresh !== undefined && block.at >= ctx.fresh ? '' : undefined;
    switch (block.type) {
      case 'paragraph':
        if (tight) return <span key={key}>{inlines(block.children, ctx, `${key}.`)}</span>;
        return (
          <p key={key} className={styles.p}>
            {inlines(block.children, ctx, `${key}.`)}
          </p>
        );
      case 'heading': {
        const Tag = HEADING_TAGS[Math.min(5, block.level - 1)] ?? 'h6';
        return (
          <Tag key={key} className={HEADING_CLASS[Math.min(5, block.level - 1)]}>
            {inlines(block.children, ctx, `${key}.`)}
          </Tag>
        );
      }
      case 'code':
        return (
          <div key={key} className={cn(styles.block, freshBlock !== undefined && styles.fresh)} data-fresh={freshBlock}>
            <CodePanel code={visibleText(block.value)} label="code" wrap />
          </div>
        );
      case 'quote':
        return (
          <blockquote key={key} className={styles.quote}>
            {blocks(block.children, ctx, `${key}.`)}
          </blockquote>
        );
      case 'list': {
        const items = block.items.map((item, itemIndex) => (
          <li key={itemIndex} className={styles.item}>
            {blocks(item, ctx, `${key}.${itemIndex}.`, block.tight)}
          </li>
        ));
        return block.ordered ? (
          <ol key={key} className={styles.list} start={block.start !== 1 ? block.start : undefined}>
            {items}
          </ol>
        ) : (
          <ul key={key} className={styles.list}>
            {items}
          </ul>
        );
      }
      case 'rule':
        return <hr key={key} className={styles.rule} />;
      case 'table':
        return (
          <div key={key} className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  {block.head.map((cell, c) => (
                    <th key={c} style={alignStyle(block.align[c])}>
                      {inlines(cell, ctx, `${key}.h${c}.`)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {block.rows.map((row, r) => (
                  <tr key={r}>
                    {row.map((cell, c) => (
                      <td key={c} style={alignStyle(block.align[c])}>
                        {inlines(cell, ctx, `${key}.${r}.${c}.`)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
    }
  });
}

function alignStyle(align: 'left' | 'center' | 'right' | null | undefined): CSSProperties | undefined {
  return align === null || align === undefined ? undefined : { textAlign: align };
}

/** One settled block: it re-renders only when its own text, the link rule or the citations change. */
const HeadBlock = memo(function HeadBlock(props: {
  block: MdBlock;
  links: LinkRule;
  citations: RenderContext['citations'];
  onCite: RenderContext['onCite'];
  fresh: number | undefined;
}): ReactElement {
  const ctx: RenderContext = {
    links: props.links,
    citations: props.citations,
    onCite: props.onCite,
    fresh: props.fresh,
  };
  return <>{blocks([props.block], ctx, `h${props.block.at}.`)}</>;
});

interface Settled {
  /** The text up to the last cut, and its blocks. */
  head: string;
  blocks: MdBlock[];
  /** `head` ends at a cut, so the next head may be parsed from there on. */
  atCut: boolean;
}

/** The blocks of `head`, parsed from the end of the earlier head when this one only grew past a cut. */
function settledBlocks(earlier: Settled, head: string): MdBlock[] {
  if (head === earlier.head) return earlier.blocks;
  if (earlier.atCut && earlier.head !== '' && head.length > earlier.head.length && head.startsWith(earlier.head))
    return earlier.blocks.concat(parseMarkdown(head.slice(earlier.head.length), earlier.head.length));
  return parseMarkdown(head);
}

function MarkdownBody(props: ChatMarkdownProps): ReactElement {
  const rich = useRichMotion();
  const fresh = rich && props.streaming === true ? props.freshFrom : undefined;
  const streaming = props.streaming === true;
  const settled = useRef<Settled>({ head: '', blocks: [], atCut: false });
  const earlier = settled.current;
  const from = earlier.atCut && props.text.startsWith(earlier.head) ? earlier.head.length : 0;
  const split = streaming ? settledSplit(props.text, from) : props.text.length;
  const head = props.text.slice(0, split);
  const headBlocks = settledBlocks(earlier, head);
  settled.current = { head, blocks: headBlocks, atCut: streaming };
  const { links, citations, onCite } = props;
  const tail = streaming ? repairTail(props.text.slice(split)) : '';
  const tailBlocks = streaming && tail !== '' ? parseMarkdown(tail, split) : [];
  const tailCtx: RenderContext = { links, citations, onCite, fresh };
  return (
    <div className={cn(styles.root, props.className)}>
      {headBlocks.map((block, index) => {
        const end = headBlocks[index + 1]?.at ?? split;
        return (
          <HeadBlock
            key={block.at}
            block={block}
            links={links}
            citations={citations}
            onCite={onCite}
            fresh={fresh !== undefined && fresh < end ? fresh : undefined}
          />
        );
      })}
      {blocks(tailBlocks, tailCtx, 't')}
    </div>
  );
}

/** An answer that cannot be drawn as Markdown is shown as its plain text, so it never blanks the page. */
class PlainOnError extends Component<{ text: string; className?: string; children: ReactNode }, { failed: boolean }> {
  override state = { failed: false };

  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }

  override render(): ReactNode {
    if (!this.state.failed) return this.props.children;
    return (
      <div className={cn(styles.root, this.props.className)}>
        <p className={cn(styles.p, styles.plain)}>{visibleText(this.props.text)}</p>
      </div>
    );
  }
}

/**
 * Model text as Markdown, parsed by hand into React elements: no HTML is ever produced from it, `javascript:` and
 * every address that is not http(s) stay text, and in Chat only hosts of the conversation become links.
 */
export function ChatMarkdown(props: ChatMarkdownProps): ReactElement {
  return (
    <PlainOnError text={props.text} className={props.className}>
      <MarkdownBody {...props} />
    </PlainOnError>
  );
}
