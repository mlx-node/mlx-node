/**
 * The one parser setup every ChatMarkdown uses. Model text is untrusted: raw HTML stays text, only written links and
 * schemed addresses link, and syntax the answers do not need (math, footnotes, containers, marks, task boxes) is off.
 * Rule names are markstream's own; the corpus test in `chat-markdown-lib.test.ts` fails when a bump renames one.
 */

import type { MarkdownIt, ParsedNode, ParseOptions } from 'stream-markdown-parser';

/** The `:::name` containers stream-markdown-parser registers. */
const CONTAINERS = ['admonition', 'info', 'warning', 'error', 'tip', 'danger', 'note', 'caution'] as const;

interface LinkifySettings {
  set: (options: { fuzzyLink?: boolean; fuzzyEmail?: boolean }) => unknown;
}

function isLinkify(value: unknown): value is LinkifySettings {
  return typeof value === 'object' && value !== null && typeof (value as { set?: unknown }).set === 'function';
}

/**
 * Turns the factory's instance into the chat one. A module-level function: the renderer memoizes on it. Never calls
 * `md.use()`, which would turn the stream tokenizer off.
 */
export function chatMarkdownIt(md: MarkdownIt): MarkdownIt {
  md.set({ html: false, typographer: false });
  const linkify = md.linkify;
  if (!isLinkify(linkify)) throw new Error('markdown-it linkify has no set()');
  linkify.set({ fuzzyLink: false, fuzzyEmail: false });
  md.core.ruler.disable(
    ['fix_html_inline_streaming', 'fix_html_inline_tokens', 'github-task-lists', 'footnote_tail'],
    true,
  );
  // Block 'reference' is link definitions: `[1]: https://...` lines stay visible.
  md.block.ruler.disable(
    [
      'reference',
      'math_block',
      'explicit_math_block',
      'footnote_def',
      'pending_footnote_def',
      'vmr_container_fallback',
      ...CONTAINERS.map((name) => `container_${name}`),
    ],
    true,
  );
  // The inline 'reference' rule is markstream's `[n]` mark: its tokens stop the parser from reusing settled blocks,
  // so it is off while an answer streams (see `chatCiteMarkdownIt`).
  md.inline.ruler.disable(['reference', 'math', 'ins', 'mark', 'sub', 'sup', 'footnote_ref', 'footnote_inline'], true);
  // Emphasis pairs after markdown-it's nesting cap, and the tree builder recurses once per level (quadratic, then a
  // stack overflow), so a run nested deeper than the cap stays its source text.
  md.core.ruler.push('chat_depth_cap', (state) => {
    const max = state.md.options.maxNesting ?? 100;
    for (const token of state.tokens) {
      if (token.type !== 'inline' || token.children === null) continue;
      let depth = 0;
      let deep = false;
      for (const child of token.children) {
        if (child.nesting === 1 && ++depth > max) {
          deep = true;
          break;
        }
        if (child.nesting === -1) depth--;
      }
      if (!deep) continue;
      const text = new state.Token('text', '', 0);
      text.content = token.content;
      token.children = [text];
    }
  });
  return md;
}

/**
 * The setup of a finished answer that cites sources: `chatMarkdownIt` with the `[n]` mark on, so a mark comes from the
 * parser and an escaped `\[1\]` stays text.
 */
export function chatCiteMarkdownIt(md: MarkdownIt): MarkdownIt {
  chatMarkdownIt(md);
  md.inline.ruler.enable(['reference']);
  return md;
}

export const CHAT_PARSE_OPTIONS: ParseOptions = { reuseStableTopLevelNodes: true };

/** The words of inline nodes, as a link's visible text. */
export function nodeWords(nodes: readonly ParsedNode[] | undefined): string {
  if (nodes === undefined) return '';
  let out = '';
  for (const node of nodes) {
    const fields = node as ParsedNode & { content?: unknown; code?: unknown; alt?: unknown; id?: unknown };
    switch (node.type) {
      case 'text':
      case 'html_inline':
        out += typeof fields.content === 'string' ? fields.content : '';
        break;
      case 'inline_code':
        out += typeof fields.code === 'string' ? fields.code : '';
        break;
      case 'image':
        out += typeof fields.alt === 'string' ? fields.alt : '';
        break;
      case 'reference':
        out += `[${String(fields.id)}]`;
        break;
      case 'hardbreak':
        out += ' ';
        break;
      default:
        if ('children' in node && Array.isArray(node.children)) out += nodeWords(node.children as ParsedNode[]);
    }
  }
  return out;
}

/** A streaming parse and render that fits in one frame lets the next text in at once. */
export const PACE_FRAME_MS = 16;
/** After a slower one, the next text waits this many times its cost: input and scrolling keep 4/5 of the time. */
export const PACE_HOLD_FACTOR = 4;
/** The longest wait, so a very slow answer still moves. Under it a parse of up to 3.3 s still leaves 3/4 free. */
export const PACE_HOLD_MAX_MS = 10_000;

/** How long the next streaming text waits after a parse and render that took `costMs`. */
export function paceHoldMs(costMs: number): number {
  if (!(costMs > PACE_FRAME_MS)) return 0;
  return Math.min(PACE_HOLD_FACTOR * costMs, PACE_HOLD_MAX_MS);
}
