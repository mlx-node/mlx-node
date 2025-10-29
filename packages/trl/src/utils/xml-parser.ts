import { type XmlParseResult } from '../types';

const REASONING_OPEN = '<reasoning>';
const REASONING_CLOSE = '</reasoning>';
const ANSWER_OPEN = '<answer>';
const ANSWER_CLOSE = '</answer>';

type ParserState =
  | 'searchReasoningOpen'
  | 'readReasoning'
  | 'searchAnswerOpen'
  | 'readAnswer'
  | 'consumeTrailing'
  | 'done';

function isWhitespace(char: string): boolean {
  return char === ' ' || char === '\n' || char === '\r' || char === '\t' || char === '\f' || char === '\v';
}

function sliceWithTag(text: string, openTag: string, closeTag: string): string | null {
  const openIndex = text.indexOf(openTag);
  if (openIndex === -1) return null;
  const start = openIndex + openTag.length;
  const closeIndex = text.indexOf(closeTag, start);
  if (closeIndex === -1) return null;
  return text.slice(start, closeIndex).trim();
}

export function parseXmlCot(content: string): XmlParseResult {
  const text = content ?? '';
  const errors: string[] = [];

  let state: ParserState = 'searchReasoningOpen';
  let index = 0;
  const length = text.length;

  let reasoning: string | null = null;
  let answer: string | null = null;

  let reasoningFound = false;
  let answerFound = false;
  let reasoningClosed = false;
  let answerClosed = false;

  let hasLeadingText = false;
  let hasBetweenText = false;
  let hasTrailingText = false;

  let usedFallbackReasoning = false;
  let usedFallbackAnswer = false;

  while (state !== 'done') {
    switch (state) {
      case 'searchReasoningOpen': {
        while (index < length) {
          const char = text[index];
          if (isWhitespace(char)) {
            index += 1;
            continue;
          }
          if (text.startsWith(REASONING_OPEN, index)) {
            reasoningFound = true;
            index += REASONING_OPEN.length;
            state = 'readReasoning';
            break;
          }
          hasLeadingText = true;
          index += 1;
        }
        if (state === 'searchReasoningOpen') {
          state = 'done';
        }
        break;
      }
      case 'readReasoning': {
        const closingIndex = text.indexOf(REASONING_CLOSE, index);
        if (closingIndex === -1) {
          reasoning = text.slice(index).trim() || null;
          errors.push('Unterminated <reasoning>...</reasoning> section.');
          state = 'done';
          break;
        }
        reasoning = text.slice(index, closingIndex).trim();
        reasoningClosed = true;
        index = closingIndex + REASONING_CLOSE.length;
        state = 'searchAnswerOpen';
        break;
      }
      case 'searchAnswerOpen': {
        while (index < length) {
          const char = text[index];
          if (isWhitespace(char)) {
            index += 1;
            continue;
          }
          if (text.startsWith(ANSWER_OPEN, index)) {
            answerFound = true;
            index += ANSWER_OPEN.length;
            state = 'readAnswer';
            break;
          }
          hasBetweenText = true;
          index += 1;
        }
        if (state === 'searchAnswerOpen') {
          state = 'done';
        }
        break;
      }
      case 'readAnswer': {
        const closingIndex = text.indexOf(ANSWER_CLOSE, index);
        if (closingIndex === -1) {
          answer = text.slice(index).trim() || null;
          errors.push('Unterminated <answer>...</answer> section.');
          state = 'done';
          break;
        }
        answer = text.slice(index, closingIndex).trim();
        answerClosed = true;
        index = closingIndex + ANSWER_CLOSE.length;
        state = 'consumeTrailing';
        break;
      }
      case 'consumeTrailing': {
        while (index < length) {
          const char = text[index];
          if (!isWhitespace(char)) {
            hasTrailingText = true;
          }
          index += 1;
        }
        state = 'done';
        break;
      }
    }
  }

  if (!reasoningFound) {
    const fallback = sliceWithTag(text, REASONING_OPEN, REASONING_CLOSE);
    if (fallback !== null) {
      reasoning = fallback;
      reasoningFound = true;
      reasoningClosed = true;
      usedFallbackReasoning = true;
    }
  }

  if (!answerFound) {
    const fallback = sliceWithTag(text, ANSWER_OPEN, ANSWER_CLOSE);
    if (fallback !== null) {
      answer = fallback;
      answerFound = true;
      answerClosed = true;
      usedFallbackAnswer = true;
    }
  }

  if (!reasoningFound) {
    errors.push('Missing <reasoning>...</reasoning> section.');
  }
  if (!answerFound) {
    errors.push('Missing <answer>...</answer> section.');
  }

  if (hasLeadingText) {
    errors.push('XML format contains extra characters before <reasoning> section.');
  }
  if (hasBetweenText) {
    errors.push('XML format contains extra characters between reasoning and answer sections.');
  }
  if (hasTrailingText) {
    errors.push('XML format contains extra characters after </answer> section.');
  }

  const isSoftMatch = reasoningFound && answerFound && reasoningClosed && answerClosed;
  const isStrictMatch =
    isSoftMatch &&
    !hasLeadingText &&
    !hasBetweenText &&
    !hasTrailingText &&
    !usedFallbackReasoning &&
    !usedFallbackAnswer;

  return {
    reasoning: reasoning ?? null,
    answer: answer ?? null,
    isStrictMatch,
    isSoftMatch,
    errors,
  };
}

export function extractXmlAnswer(content: string): string | null {
  const { answer } = parseXmlCot(content);
  return answer;
}

export function extractXmlReasoning(content: string): string | null {
  const { reasoning } = parseXmlCot(content);
  return reasoning;
}

export function extractHashAnswer(text: string): string | null {
  if (!text) return null;
  const separator = text.indexOf('####');
  if (separator === -1) return null;
  const extracted = text.slice(separator + 4).trim();
  return extracted.length ? extracted : null;
}
