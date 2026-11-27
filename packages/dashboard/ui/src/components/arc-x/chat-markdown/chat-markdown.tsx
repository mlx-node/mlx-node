import { NodeRenderer, setCustomComponents } from 'markstream-react';
import {
  Component,
  useEffect,
  useLayoutEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactElement,
  type ReactNode,
} from 'react';

import { cn } from '../../../lib/cn';
import { visibleText, type LinkRule } from '../../../lib/link-rule';
import { CHAT_PARSE_OPTIONS, chatCiteMarkdownIt, chatMarkdownIt, paceHoldMs } from '../../../lib/markstream-config';
import { useRichMotion } from '../motion/use-entrance';
import styles from './chat-markdown.module.css';
import { ChatMdContext, NODES, type ChatMd } from './md-nodes';

export type { LinkRule } from '../../../lib/link-rule';

/** The markstream registry entry every ChatMarkdown draws its nodes with. */
const CUSTOM_ID = 'mlx-chat';
setCustomComponents(CUSTOM_ID, NODES);

export interface ChatMarkdownProps {
  text: string;
  /** The answer still arrives: open marks, fences and links show their streaming state, new words fade in. */
  streaming?: boolean;
  /**
   * 'chat': a link is a link only when its host is one of `hosts`; any other link is its words plus the host as
   * text. 'work': every http(s) link is a link, followed by its host when the words are not the address.
   */
  links: LinkRule;
  /** `[n]` marks that name a real source; any other `[n]` stays text. */
  citations?: ReadonlyMap<number, { title: string; host: string }>;
  onCite?: (n: number) => void;
  /** 'alt': an image is drawn as its alt text, with no link. Default 'link': "Image: alt" as a link. */
  images?: 'link' | 'alt';
  className?: string;
}

/** The text the renderer was last handed, and until when the next streaming text waits. */
interface Pace {
  shown: string;
  holdUntil: number;
}

/**
 * The text to hand the renderer. While streaming, a parse and render that took longer than a frame holds the next
 * text for a few times its cost (`paceHoldMs`), so a slow answer cannot take the whole main thread; the held text
 * follows when the hold ends. The finished text is handed in at once.
 */
function usePacedText(latest: string, streaming: boolean): string {
  const pace = useRef<Pace>({ shown: '', holdUntil: 0 });
  const [tick, wake] = useReducer((n: number) => n + 1, 0);
  const renderStart = performance.now();
  const held = streaming && latest !== pace.current.shown && renderStart < pace.current.holdUntil;
  const shown = held ? pace.current.shown : latest;
  useLayoutEffect(() => {
    // Runs once the renderer's subtree is in the DOM; `renderStart` is from the render that handed `shown` in.
    const end = performance.now();
    pace.current = { shown, holdUntil: streaming ? end + paceHoldMs(end - renderStart) : 0 };
  }, [shown, streaming]);
  useEffect(() => {
    if (!held) return undefined;
    const timer = setTimeout(wake, Math.max(0, pace.current.holdUntil - performance.now()) + 1);
    return () => clearTimeout(timer);
  }, [held, latest, tick]);
  return shown;
}

function MarkdownBody(props: ChatMarkdownProps): ReactElement {
  const rich = useRichMotion();
  const streaming = props.streaming === true;
  const { links, citations, onCite } = props;
  const imagesAsAlt = props.images === 'alt';
  // Keyed on what the rule means, so a caller's `{ mode: 'work' }` literal does not redraw every node.
  const hosts = links.mode === 'chat' ? links.hosts : null;
  const chat = useMemo<ChatMd>(
    () => ({ links: hosts === null ? { mode: 'work' } : { mode: 'chat', hosts }, citations, onCite, imagesAsAlt }),
    [hosts, citations, onCite, imagesAsAlt],
  );
  const latest = useMemo(() => visibleText(props.text), [props.text]);
  const content = usePacedText(latest, streaming);
  const fade = rich && streaming;
  const cites = citations !== undefined;
  // The same element while the text is held, so the renderer does not render again. The renderer keeps a block whose
  // source did not change, so the citing setup mounts a new one.
  const body = useMemo(
    () => (
      <NodeRenderer
        key={cites ? 'cites' : 'plain'}
        content={content}
        final={!streaming}
        customId={CUSTOM_ID}
        customMarkdownIt={cites ? chatCiteMarkdownIt : chatMarkdownIt}
        parseOptions={CHAT_PARSE_OPTIONS}
        htmlPolicy="escape"
        renderCodeBlocksAsPre
        codeBlockStream={false}
        typewriter={false}
        fade={fade}
        smoothStreaming={false}
        batchRendering={false}
        maxLiveNodes={0}
        deferNodesUntilVisible={false}
        viewportPriority={false}
        showTooltips={false}
      />
    ),
    [content, streaming, fade, cites],
  );
  return (
    <div className={cn(styles.root, props.className)}>
      <ChatMdContext.Provider value={chat}>{body}</ChatMdContext.Provider>
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
        <p className={styles.plain}>{visibleText(this.props.text)}</p>
      </div>
    );
  }
}

/**
 * Model text as Markdown through markstream-react, on our own node components: raw HTML stays text, no image or
 * script is ever loaded, `javascript:` and every address that is not http(s) stay text, and in Chat only hosts of
 * the conversation become links. While streaming, settled blocks keep their objects and skip render.
 */
export function ChatMarkdown(props: ChatMarkdownProps): ReactElement {
  return (
    <PlainOnError text={props.text} className={props.className}>
      <MarkdownBody {...props} />
    </PlainOnError>
  );
}
