import type { ComponentProps, ReactElement } from 'react';
import ReactMarkdown, { type Components, type ExtraProps, type Options } from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { cn } from '../../../lib/cn';
import { CodePanel } from '../code-panel/code-panel';
import { useRichMotion } from '../motion/use-entrance';
import { freshBlocks, type HastNodeLike } from './fresh-blocks';
import styles from './markdown.module.css';

export interface MarkdownProps {
  text: string;
  /** Offset into `text`: top-level blocks that start at or after it fade in (`chunk`). Omitted: nothing fades. */
  freshFrom?: number;
  className?: string;
}

const REMARK_PLUGINS = [remarkGfm];

/** All text under a hast node. */
function textOf(node: HastNodeLike | undefined): string {
  if (node === undefined) return '';
  if (node.type === 'text') {
    const value: unknown = (node as { value?: unknown }).value;
    return typeof value === 'string' ? value : '';
  }
  let out = '';
  for (const child of node.children ?? []) out += textOf(child);
  return out;
}

function stripTrailingNewline(text: string): string {
  return text.endsWith('\n') ? text.slice(0, -1) : text;
}

/** A data attribute react-markdown passed through from the hast node, for a renderer whose root is another tag. */
function dataAttr(props: object, name: `data-${string}`): string | undefined {
  const value: unknown = (props as Record<string, unknown>)[name];
  return typeof value === 'string' ? value : undefined;
}

type Props<K extends keyof HTMLElementTagNameMap> = ComponentProps<K> & ExtraProps;

const COMPONENTS: Components = {
  p: ({ node: _node, className, ...rest }: Props<'p'>) => <p {...rest} className={cn(styles.p, className)} />,
  h1: ({ node: _node, className, ...rest }: Props<'h1'>) => <h3 {...rest} className={cn(styles.h1, className)} />,
  h2: ({ node: _node, className, ...rest }: Props<'h2'>) => <h4 {...rest} className={cn(styles.h2, className)} />,
  h3: ({ node: _node, className, ...rest }: Props<'h3'>) => <h5 {...rest} className={cn(styles.h3, className)} />,
  h4: ({ node: _node, className, ...rest }: Props<'h4'>) => <h6 {...rest} className={cn(styles.h3, className)} />,
  h5: ({ node: _node, className, ...rest }: Props<'h5'>) => <h6 {...rest} className={cn(styles.h3, className)} />,
  h6: ({ node: _node, className, ...rest }: Props<'h6'>) => <h6 {...rest} className={cn(styles.h3, className)} />,
  ul: ({ node: _node, className, ...rest }: Props<'ul'>) => <ul {...rest} className={cn(styles.list, className)} />,
  ol: ({ node: _node, className, ...rest }: Props<'ol'>) => <ol {...rest} className={cn(styles.list, className)} />,
  blockquote: ({ node: _node, className, ...rest }: Props<'blockquote'>) => (
    <blockquote {...rest} className={cn(styles.quote, className)} />
  ),
  hr: ({ node: _node, className, ...rest }: Props<'hr'>) => <hr {...rest} className={cn(styles.rule, className)} />,
  table: ({ node: _node, className, ...rest }: Props<'table'>) => (
    <table {...rest} className={cn(styles.table, className)} />
  ),
  a: ({ node: _node, className, ...rest }: Props<'a'>) => (
    <a {...rest} className={cn(styles.link, className)} target="_blank" rel="noreferrer" />
  ),
  code: ({ node: _node, className, ...rest }: Props<'code'>) => (
    <code {...rest} className={cn(styles.inlineCode, className)} />
  ),
  pre: ({ node, className, ...rest }: Props<'pre'>) => (
    <div className={cn(styles.block, className)} data-fresh={dataAttr(rest, 'data-fresh')}>
      <CodePanel code={stripTrailingNewline(textOf(node as HastNodeLike | undefined))} wrap />
    </div>
  ),
};

/** Markdown on Arc tokens: no raw HTML, code blocks as CodePanel, links open outside the app. */
export function Markdown({ text, freshFrom, className }: MarkdownProps): ReactElement {
  const rich = useRichMotion();
  const rehypePlugins: Options['rehypePlugins'] =
    freshFrom !== undefined && rich ? [[freshBlocks, freshFrom]] : undefined;
  return (
    <div className={cn(styles.root, className)}>
      <ReactMarkdown remarkPlugins={REMARK_PLUGINS} rehypePlugins={rehypePlugins} components={COMPONENTS} skipHtml>
        {text}
      </ReactMarkdown>
    </div>
  );
}
