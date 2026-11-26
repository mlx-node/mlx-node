import { TextNode, type CustomComponentMap, type NodeComponentProps } from 'markstream-react';
import { createContext, useContext, type CSSProperties, type ReactElement, type ReactNode } from 'react';
import type {
  CodeBlockNode,
  HeadingNode,
  ImageNode,
  InlineCodeNode,
  LinkNode,
  ListItemNode,
  ListNode,
  ParsedNode,
  ReferenceNode,
  TableCellNode,
  TableNode,
  TextNode as TextNodeData,
} from 'stream-markdown-parser';

import { CHAT, fill } from '../../../lib/chat-strings';
import { citeNumber, citePieces, linkView, type LinkRule, type LinkView } from '../../../lib/link-rule';
import { nodeWords } from '../../../lib/markstream-config';
import { CodePanel } from '../code-panel/code-panel';
import { CitationChip } from '../source-list/source-list';
import styles from './chat-markdown.module.css';

/** What one ChatMarkdown decides for its nodes; markstream's registry is global, so it reaches them by context. */
export interface ChatMd {
  links: LinkRule;
  citations: ReadonlyMap<number, { title: string; host: string }> | undefined;
  onCite: ((n: number) => void) | undefined;
  imagesAsAlt: boolean;
}

export const ChatMdContext = createContext<ChatMd>({
  links: { mode: 'chat', hosts: new Set() },
  citations: undefined,
  onCite: undefined,
  imagesAsAlt: true,
});

/** Inside link words: no anchor, chip or image link is drawn there. */
const InLinkContext = createContext(false);

type NodeProps = NodeComponentProps<ParsedNode>;

/** The children of a node through markstream's dispatch, so our overrides draw them too. */
function children(nodes: readonly ParsedNode[] | undefined, props: NodeProps, prefix: string): ReactNode {
  if (nodes === undefined || nodes.length === 0) return null;
  const { ctx, renderNode } = props;
  if (ctx === undefined || renderNode === undefined) return nodeWords(nodes);
  return nodes.map((child, index) => renderNode(child, `${prefix}-${index}`, ctx) as ReactNode);
}

function keyOf(props: NodeProps, fallback: string): string {
  return props.indexKey === undefined ? fallback : String(props.indexKey);
}

function Code(props: NodeProps): ReactNode {
  const node = props.node as CodeBlockNode;
  const code = String(node.diff === true ? node.raw : (node.code ?? ''));
  // A fence whose first line is still arriving draws nothing yet.
  if (node.loading === true && code === '') return null;
  return (
    <div className={styles.block}>
      <CodePanel code={code.endsWith('\n') ? code.slice(0, -1) : code} label="code" wrap />
    </div>
  );
}

/**
 * The override for `type`. markstream routes a fence by its language before its type, so a ```link fence reaches the
 * `link` entry: any code block is drawn as code, any other stranger as its source text.
 */
function only(type: string, View: (props: NodeProps) => ReactNode): (props: NodeProps) => ReactNode {
  function Only(props: NodeProps): ReactNode {
    const node = props.node;
    if (node.type === type) return <View {...props} />;
    if (node.type === 'code_block') return <Code {...props} />;
    return String(node.raw ?? '');
  }
  Only.displayName = `ChatMd(${type})`;
  return Only;
}

function Cite({ n }: { n: number }): ReactNode {
  const chat = useContext(ChatMdContext);
  const source = chat.citations?.get(n);
  const onCite = chat.onCite;
  if (source === undefined || onCite === undefined) return `[${n}]`;
  return <CitationChip n={n} title={source.title} host={source.host} onSelect={() => onCite(n)} />;
}

/** Plain text keeps markstream's word fade; a `[n]` that names a source becomes a chip between its pieces. */
function Text(props: NodeProps): ReactNode {
  const chat = useContext(ChatMdContext);
  const inLink = useContext(InLinkContext);
  const node = props.node as TextNodeData;
  const citations = chat.citations;
  const plain = (
    <TextNode node={node} ctx={props.ctx} indexKey={props.indexKey} typewriter={props.typewriter} fade={props.fade} />
  );
  if (inLink || citations === undefined || chat.onCite === undefined) return plain;
  const pieces = citePieces(String(node.content ?? ''), (n) => citations.has(n));
  if (pieces.length === 1 && pieces[0]!.kind === 'text') return plain;
  const key = keyOf(props, 'text');
  return pieces.map((piece, index) =>
    piece.kind === 'cite' ? (
      <Cite key={`c${index}`} n={piece.n} />
    ) : (
      <TextNode
        key={`t${index}`}
        node={{ type: 'text', content: piece.text }}
        ctx={props.ctx}
        indexKey={index === 0 ? props.indexKey : `${key}-t${index}`}
        typewriter={props.typewriter}
        fade={props.fade}
      />
    ),
  );
}

function Reference(props: NodeProps): ReactNode {
  const inLink = useContext(InLinkContext);
  const id = String((props.node as ReferenceNode).id);
  const n = citeNumber(id);
  return n === null || inLink ? `[${id}]` : <Cite n={n} />;
}

/** Words, then ` (host)` when the view names one; an anchor only when the view opens one. */
function LinkOut({ view, words }: { view: LinkView; words: ReactNode }): ReactElement {
  const host = view.host === null ? null : <bdi className={styles.host}>{` (${view.host})`}</bdi>;
  if (view.href === null) {
    return (
      <span>
        {words}
        {host}
      </span>
    );
  }
  return (
    <span>
      <a className={styles.link} href={view.href} title={view.title} target="_blank" rel="noreferrer">
        {words}
      </a>
      {host}
    </span>
  );
}

function Link(props: NodeProps): ReactNode {
  const chat = useContext(ChatMdContext);
  const inLink = useContext(InLinkContext);
  const node = props.node as LinkNode;
  const words = (
    <InLinkContext.Provider value={true}>{children(node.children, props, keyOf(props, 'link'))}</InLinkContext.Provider>
  );
  // Half a link, or a link inside link words: its words only.
  if (node.loading === true || inLink) return <span>{words}</span>;
  // Inline HTML is never parsed, so no link comes from it; should one, it is its source.
  if (node.markup === 'html') return String(node.raw ?? '');
  return <LinkOut view={linkView(String(node.href ?? ''), nodeWords(node.children), chat.links)} words={words} />;
}

function Image(props: NodeProps): ReactNode {
  const chat = useContext(ChatMdContext);
  const inLink = useContext(InLinkContext);
  const node = props.node as ImageNode;
  const alt = String(node.alt ?? '').trim();
  if (chat.imagesAsAlt || inLink) return <span>{alt}</span>;
  const words = alt === '' ? CHAT.imageLinkBare : fill(CHAT.imageLink, { alt });
  return <LinkOut view={linkView(String(node.src ?? ''), words, chat.links)} words={words} />;
}

/** Raw HTML, math and anything else markstream would draw as markup: its source, as text. */
function Source(props: NodeProps): ReactNode {
  const node = props.node as ParsedNode & { content?: unknown };
  return String(node.raw ?? node.content ?? '');
}

function SourceBlock(props: NodeProps): ReactElement {
  return <p className={styles.source}>{Source(props)}</p>;
}

const HEADING_TAGS = ['h3', 'h4', 'h5', 'h6', 'h6', 'h6'] as const;
const HEADING_CLASS = [styles.h1, styles.h2, styles.h3, styles.h3, styles.h3, styles.h3];

function Heading(props: NodeProps): ReactElement {
  const node = props.node as HeadingNode;
  const index = Math.min(5, Math.max(0, (Number(node.level) || 1) - 1));
  const Tag = HEADING_TAGS[index] ?? 'h6';
  return <Tag className={HEADING_CLASS[index]}>{children(node.children, props, keyOf(props, 'heading'))}</Tag>;
}

function List(props: NodeProps): ReactElement {
  const node = props.node as ListNode;
  const items = children(node.items, props, keyOf(props, 'list'));
  if (!node.ordered) return <ul className={styles.list}>{items}</ul>;
  return (
    <ol className={styles.list} start={node.start !== undefined && node.start !== 1 ? node.start : undefined}>
      {items}
    </ol>
  );
}

function ListItem(props: NodeProps): ReactElement {
  const node = props.node as ListItemNode;
  return <li className={styles.item}>{children(node.children, props, keyOf(props, 'item'))}</li>;
}

/** A nested renderer of markstream's own would run with its defaults; quotes draw their children here instead. */
function Blockquote(props: NodeProps): ReactElement {
  const node = props.node as ParsedNode & { children?: ParsedNode[] };
  return <blockquote className={styles.quote}>{children(node.children, props, keyOf(props, 'quote'))}</blockquote>;
}

function alignStyle(cell: TableCellNode): CSSProperties | undefined {
  return cell.align === undefined ? undefined : { textAlign: cell.align };
}

function Table(props: NodeProps): ReactElement {
  const node = props.node as TableNode;
  const key = keyOf(props, 'table');
  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            {node.header.cells.map((cell, c) => (
              <th key={c} style={alignStyle(cell)}>
                {children(cell.children, props, `${key}-h${c}`)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {node.rows.map((row, r) => (
            <tr key={r}>
              {row.cells.map((cell, c) => (
                <td key={c} className={cell.align === 'right' ? styles.figures : undefined} style={alignStyle(cell)}>
                  {children(cell.children, props, `${key}-${r}-${c}`)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function InlineCode(props: NodeProps): ReactElement {
  return <code className={styles.inlineCode}>{String((props.node as InlineCodeNode).code ?? '')}</code>;
}

function Rule(): ReactElement {
  return <hr className={styles.rule} />;
}

/**
 * Every node type ChatMarkdown draws itself. markstream keeps paragraph, strong, emphasis, strikethrough and hard
 * break; its text component draws the words (through `Text`). Code, Mermaid, D2 and infographic fences are all code.
 */
export const NODES: CustomComponentMap = {
  text: only('text', Text),
  link: only('link', Link),
  image: only('image', Image),
  reference: only('reference', Reference),
  code_block: Code,
  mermaid: Code,
  d2: Code,
  infographic: Code,
  html_inline: only('html_inline', Source),
  html_block: only('html_block', SourceBlock),
  math_inline: only('math_inline', Source),
  math_block: only('math_block', SourceBlock),
  heading: only('heading', Heading),
  list: only('list', List),
  list_item: only('list_item', ListItem),
  blockquote: only('blockquote', Blockquote),
  table: only('table', Table),
  inline_code: only('inline_code', InlineCode),
  thematic_break: only('thematic_break', Rule),
};
