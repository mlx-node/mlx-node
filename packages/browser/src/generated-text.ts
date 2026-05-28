const GENERATED_ROLE_PREFIX = /^\s*(?:\((?:user|assistant|system)\)|(?:user|assistant|system))\s*:\s*/i;

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function stripGeneratedRolePrefix(text: string): string {
  let cleaned = text;
  for (let i = 0; i < 3; i++) {
    const next = cleaned.replace(GENERATED_ROLE_PREFIX, '');
    if (next === cleaned) break;
    cleaned = next;
  }
  return cleaned.trimStart();
}

function stripGeneratedPromptEcho(text: string, latestUserText?: string): string {
  const prompt = latestUserText?.trim();
  if (!prompt) return text;

  const escaped = escapeRegExp(prompt);
  const patterns = [
    new RegExp(`^\\s*(?:\\((?:user)\\)|user)\\s*:\\s*${escaped}(?:\\s*\\r?\\n)+\\s*`, 'i'),
    new RegExp(`^\\s*${escaped}(?:\\s*\\r?\\n)+\\s*`, 'i'),
    new RegExp(`^\\s*(?:\\((?:user)\\)|user)\\s*:\\s*${escaped}\\s*(?:[!?。！？]+\\s*)?`, 'i'),
    new RegExp(`^\\s*${escaped}\\s*(?:[!?。！？]+\\s*)?`, 'i'),
  ];

  let cleaned = text;
  for (const pattern of patterns) {
    const next = cleaned.replace(pattern, '');
    if (next !== cleaned) {
      cleaned = next;
      break;
    }
  }
  return cleaned.trimStart();
}

function stripLeadingDecodeFragment(text: string): string {
  return text.replace(/^\s*[A-Z]\s*(?:\r?\n){2,}(?=[A-Z])/u, '').replace(/^\s*[!?]+\s*(?=[A-Z0-9"'(])/u, '');
}

function normalizeSentenceForRepeat(sentence: string): string {
  return sentence
    .trim()
    .toLowerCase()
    .replace(/^(?:okay|sure|hello|hi)[,!]?\s+/u, '')
    .replace(/\s+/g, ' ');
}

function splitSentences(text: string): string[] {
  const parts: string[] = [];
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char !== '.' && char !== '!' && char !== '?') continue;

    const previous = text[i - 1] ?? '';
    const next = text[i + 1] ?? '';
    if (/\d/u.test(previous) && /\d/u.test(next)) continue;
    if (next && !/\s/u.test(next)) continue;

    let end = i + 1;
    while (end < text.length && /\s/u.test(text[end])) end++;
    parts.push(text.slice(start, end));
    start = end;
  }
  if (start < text.length) parts.push(text.slice(start));
  return parts;
}

function collapseRepeatedSentences(text: string): string {
  const parts = splitSentences(text);
  if (!parts || parts.length < 2) return text;

  const collapsed: string[] = [];
  let previous = '';
  for (const part of parts) {
    const normalized = normalizeSentenceForRepeat(part);
    if (normalized && normalized === previous) continue;
    collapsed.push(part);
    if (normalized) previous = normalized;
  }
  return collapsed.join('').trimStart();
}

function cleanAssistantContent(text: string, latestUserText?: string): string {
  return collapseRepeatedSentences(
    stripLeadingDecodeFragment(stripGeneratedPromptEcho(stripGeneratedRolePrefix(text), latestUserText)),
  );
}

const THINK_TAG_RE = /<\s*\/?\s*(?:think|longcat_think)\b[^>]*>/gi;
const THINK_OPEN_TAG_RE = /<\s*(?:think|longcat_think)\b[^>]*>/i;
const THINK_CLOSE_TAG_RE = /<\s*\/\s*(?:think|longcat_think)\s*>/i;
const LEADING_THOUGHT_DETAILS_RE =
  /^\s*<details\b[^>]*>\s*<summary\b[^>]*>\s*(?:[▾▼▶▸]\s*)?(?:thought\s+process|thinking\s+process|thinking)\s*:?\s*<\/summary>\s*([\s\S]*?)\s*<\/details>\s*/iu;
const THOUGHT_DETAILS_RE =
  /<details\b[^>]*>\s*<summary\b[^>]*>\s*(?:[▾▼▶▸]\s*)?(?:thought\s+process|thinking\s+process|thinking)\s*:?\s*<\/summary>\s*([\s\S]*?)\s*<\/details>/giu;
const PARTIAL_THOUGHT_DETAILS_OPEN_RE =
  /<details\b[^>]*>\s*<summary\b[^>]*>\s*(?:[▾▼▶▸]\s*)?(?:thought\s+process|thinking\s+process|thinking)\s*:?\s*<\/summary>\s*/giu;
const THOUGHT_SUMMARY_TAG_RE =
  /<summary\b[^>]*>\s*(?:[▾▼▶▸]\s*)?(?:thought\s+process|thinking\s+process|thinking)\s*:?\s*<\/summary>/giu;
const PARTIAL_THOUGHT_SUMMARY_START_RE =
  /<summary\b[^>]*>\s*(?:[▾▼▶▸]\s*)?(?:thought\s+process|thinking\s+process|thinking)\s*:?/giu;
const THOUGHT_DETAILS_START_RE =
  /<details\b[^>]*>\s*(?=<summary\b[^>]*>\s*(?:[▾▼▶▸]\s*)?(?:thought\s+process|thinking\s+process|thinking)\s*:?\s*(?:<\/summary>|$))/giu;
const REASONING_TEXT_RE =
  /(?:<\/\s*(?:think|longcat_think)\s*>|<details\b[\s\S]*<summary\b[\s\S]*(?:thought\s+process|thinking\s+process)|(?:^|\n)\s*(?:#{1,6}\s*)?(?:here'?s\s+a\s+)?(?:thought|thinking)\s+process\s*:|(?:^|\n)\s*(?:\d+\.\s*)?(?:\*\*)?(?:draft response|check against|final output|analy[sz]e|identify|refine|synthesi[sz]e|plan)\b)/iu;
const PLAIN_THOUGHT_START_RE =
  /^\s*(?:(?:#{1,6}\s*)?(?:here'?s\s+a\s+)?(?:thought|thinking)\s+process\s*:|(?:\d+\.\s*)?(?:\*\*)?(?:analy[sz]e\s+(?:the\s+)?request|identify\s+(?:the\s+)?(?:core\s+)?(?:concept|context|requirements)|determine\s+(?:the\s+)?configuration|refine\s+(?:the\s+)?(?:explanation|thought\s+process)|synthesi[sz]e\s+(?:the\s+)?(?:description|response)|plan)\b)/iu;
const FINAL_ANSWER_BOUNDARY_RE =
  /(?:^|\n)\s*(?:#{1,6}\s*)?(?:\*\*)?(?:final\s+answer|answer|response)\s*:?\s*(?:\*\*)?\s*(?:\r?\n|$)/iu;
const REASONING_PREFIX_LABELS = [
  'analyze',
  'analyse',
  'check against',
  'draft response',
  'final output',
  'here is a thinking process',
  'here is a thought process',
  "here's a thinking process",
  "here's a thought process",
  'identify',
  'plan',
  'refine',
  'synthesize',
  'synthesise',
  'thinking process',
  'thought process',
];

function unwrapThoughtDisclosure(text: string): string {
  let cleaned = text;
  for (let i = 0; i < 4; i++) {
    const next = cleaned.replace(THOUGHT_DETAILS_RE, '$1');
    if (next === cleaned) break;
    cleaned = next;
  }
  return cleaned
    .replace(THOUGHT_DETAILS_START_RE, '')
    .replace(PARTIAL_THOUGHT_DETAILS_OPEN_RE, '')
    .replace(THOUGHT_SUMMARY_TAG_RE, '')
    .replace(PARTIAL_THOUGHT_SUMMARY_START_RE, '')
    .replace(/<\/summary>/giu, '')
    .replace(/<\/details>/giu, '')
    .trim();
}

export function looksLikeReasoningText(text: string | null | undefined): boolean {
  return !!text && REASONING_TEXT_RE.test(text);
}

export function couldStillBeReasoningTextPrefix(text: string | null | undefined): boolean {
  if (!text) return false;
  const trimmed = text.trimStart().toLowerCase();
  if (!trimmed) return true;
  if (
    /^<\s*\/?\s*(?:t|th|thi|thin|think|l|lo|lon|long|longcat|longcat_?|longcat_t|longcat_th|longcat_thi|longcat_thin|longcat_think|d|de|det|deta|detai|detail|details|s|su|sum|summ|summa|summar|summary)?$/iu.test(
      trimmed,
    )
  ) {
    return true;
  }
  if (/^(?:#{1,6}\s*)?$/u.test(trimmed)) return true;
  if (/^(?:\d{1,3}\.?\s*)$/u.test(trimmed)) return true;

  const probe = trimmed
    .replace(/^(?:#{1,6}\s*)/u, '')
    .replace(/^(?:\d{1,3}\.?\s*)/u, '')
    .replace(/^\*{1,2}/u, '')
    .trimStart();
  if (!probe) return true;

  return REASONING_PREFIX_LABELS.some((label) => label.startsWith(probe) && probe.length < label.length);
}

function splitPlainThoughtText(text: string, latestUserText?: string): { text: string; thinking: string } | null {
  if (!PLAIN_THOUGHT_START_RE.test(text)) return null;

  const boundary = FINAL_ANSWER_BOUNDARY_RE.exec(text);
  if (!boundary || boundary.index <= 0) {
    return {
      thinking: sanitizeThinkingText(text),
      text: '',
    };
  }

  return {
    thinking: sanitizeThinkingText(text.slice(0, boundary.index)),
    text: cleanAssistantContent(text.slice(boundary.index + boundary[0].length), latestUserText),
  };
}

function splitLeadingThoughtDisclosure(
  text: string,
  latestUserText?: string,
): { text: string; thinking: string } | null {
  const match = LEADING_THOUGHT_DETAILS_RE.exec(text);
  if (!match) return null;
  return {
    text: cleanAssistantContent(text.slice(match[0].length), latestUserText),
    thinking: sanitizeThinkingText(match[1]),
  };
}

function splitThinkTaggedText(text: string, latestUserText?: string): { text: string; thinking: string } | null {
  const open = THINK_OPEN_TAG_RE.exec(text);
  const searchFrom = open ? open.index + open[0].length : 0;
  const close = THINK_CLOSE_TAG_RE.exec(text.slice(searchFrom));
  if (!close) return null;

  const closeIndex = searchFrom + close.index;
  return {
    text: cleanAssistantContent(text.slice(closeIndex + close[0].length), latestUserText),
    thinking: sanitizeThinkingText(text.slice(searchFrom, closeIndex)),
  };
}

export function sanitizeThinkingText(text: string | null | undefined): string {
  if (!text) return '';
  return unwrapThoughtDisclosure(text.replace(THINK_TAG_RE, ''));
}

export function splitAssistantThinking(
  text: string | null | undefined,
  latestUserText?: string,
): { text: string; thinking: string } {
  if (!text) return { text: '', thinking: '' };

  const disclosure = splitLeadingThoughtDisclosure(text, latestUserText);
  if (disclosure) return disclosure;

  const thinkTagged = splitThinkTaggedText(text, latestUserText);
  if (thinkTagged) return thinkTagged;

  const plainThought = splitPlainThoughtText(text, latestUserText);
  if (plainThought) return plainThought;

  return {
    text: cleanAssistantContent(text, latestUserText),
    thinking: '',
  };
}

export function sanitizeAssistantText(text: string | null | undefined, latestUserText?: string): string {
  return splitAssistantThinking(text, latestUserText).text;
}
