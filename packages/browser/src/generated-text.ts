const GENERATED_ROLE_PREFIX =
  /^\s*(?:\((?:user|assistant|system)\)|(?:user|assistant|system))\s*:\s*/i;

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function stripGeneratedRolePrefix(text: string): string {
  let cleaned = text;
  for (let i = 0; i < 3; i++) {
    const next = cleaned.replace(GENERATED_ROLE_PREFIX, "");
    if (next === cleaned) break;
    cleaned = next;
  }
  return cleaned.trimStart();
}

function stripGeneratedPromptEcho(
  text: string,
  latestUserText?: string,
): string {
  const prompt = latestUserText?.trim();
  if (!prompt) return text;

  const escaped = escapeRegExp(prompt);
  const patterns = [
    new RegExp(
      `^\\s*(?:\\((?:user)\\)|user)\\s*:\\s*${escaped}(?:\\s*\\r?\\n)+\\s*`,
      "i",
    ),
    new RegExp(`^\\s*${escaped}(?:\\s*\\r?\\n)+\\s*`, "i"),
    new RegExp(
      `^\\s*(?:\\((?:user)\\)|user)\\s*:\\s*${escaped}\\s*(?:[!?。！？]+\\s*)?`,
      "i",
    ),
    new RegExp(`^\\s*${escaped}\\s*(?:[!?。！？]+\\s*)?`, "i"),
  ];

  let cleaned = text;
  for (const pattern of patterns) {
    const next = cleaned.replace(pattern, "");
    if (next !== cleaned) {
      cleaned = next;
      break;
    }
  }
  return cleaned.trimStart();
}

function stripLeadingDecodeFragment(text: string): string {
  return text
    .replace(/^\s*[A-Z]\s*(?:\r?\n){2,}(?=[A-Z])/u, "")
    .replace(/^\s*[!?]+\s*(?=[A-Z0-9"'(])/u, "");
}

function normalizeSentenceForRepeat(sentence: string): string {
  return sentence
    .trim()
    .toLowerCase()
    .replace(/^(?:okay|sure|hello|hi)[,!]?\s+/u, "")
    .replace(/\s+/g, " ");
}

function splitSentences(text: string): string[] {
  const parts: string[] = [];
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char !== "." && char !== "!" && char !== "?") continue;

    const previous = text[i - 1] ?? "";
    const next = text[i + 1] ?? "";
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
  let previous = "";
  for (const part of parts) {
    const normalized = normalizeSentenceForRepeat(part);
    if (normalized && normalized === previous) continue;
    collapsed.push(part);
    if (normalized) previous = normalized;
  }
  return collapsed.join("").trimStart();
}

export function sanitizeAssistantText(
  text: string | null | undefined,
  latestUserText?: string,
): string {
  if (!text) return "";

  const thinkClose = "</think>";
  const longcatThinkClose = "</longcat_think>";
  const thinkEnd = Math.max(
    text.lastIndexOf(thinkClose) >= 0
      ? text.lastIndexOf(thinkClose) + thinkClose.length
      : -1,
    text.lastIndexOf(longcatThinkClose) >= 0
      ? text.lastIndexOf(longcatThinkClose) + longcatThinkClose.length
      : -1,
  );

  if (thinkEnd >= 0) {
    const thinking = text.slice(0, thinkEnd);
    const content = stripGeneratedPromptEcho(
      stripGeneratedRolePrefix(text.slice(thinkEnd)),
      latestUserText,
    );
    return content ? `${thinking}\n\n${content}` : thinking;
  }

  return collapseRepeatedSentences(
    stripLeadingDecodeFragment(
      stripGeneratedPromptEcho(stripGeneratedRolePrefix(text), latestUserText),
    ),
  );
}
