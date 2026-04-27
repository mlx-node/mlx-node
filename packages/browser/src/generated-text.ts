const GENERATED_ROLE_PREFIX =
  /^\s*(?:\((?:user|assistant|system)\)|(?:user|assistant|system))\s*:\s*/i;

function stripGeneratedRolePrefix(text: string): string {
  let cleaned = text;
  for (let i = 0; i < 3; i++) {
    const next = cleaned.replace(GENERATED_ROLE_PREFIX, "");
    if (next === cleaned) break;
    cleaned = next;
  }
  return cleaned.trimStart();
}

export function sanitizeAssistantText(text: string | null | undefined): string {
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
    const content = stripGeneratedRolePrefix(text.slice(thinkEnd));
    return content ? `${thinking}\n\n${content}` : thinking;
  }

  return stripGeneratedRolePrefix(text);
}
