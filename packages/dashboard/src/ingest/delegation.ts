import type { FileEntry, SessionEntry } from '@earendil-works/pi-coding-agent';

import { countDelegateTokens } from './tokenizer.js';

export type DelegationSummary =
  | {
      status: 'complete';
      tokenizer: 'o200k_base';
      evidenceTokens: number;
      handoffTokens: number;
      savedTokens: number;
      savingsRatio: number;
    }
  | {
      status: 'incomplete' | 'unavailable';
      reason:
        | 'no-final-handoff'
        | 'no-evidence'
        | 'unsupported-content'
        | 'legacy'
        | 'partial-record'
        | 'tokenizer-unavailable';
    };

function textContent(content: unknown): string | null {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return null;
  const text: string[] = [];
  for (const block of content) {
    if (!block || typeof block !== 'object') return null;
    if (block.type === 'thinking' || block.type === 'toolCall') continue;
    if (block.type !== 'text' || typeof block.text !== 'string') return null;
    text.push(block.text);
  }
  return text.join('\n');
}

/** Full ancestry of the displayed leaf, including evidence retained before compaction. */
function fullBranch(entries: FileEntry[], leafId: string | undefined): SessionEntry[] {
  const byId = new Map(entries.filter((e): e is SessionEntry => e.type !== 'session').map((e) => [e.id, e]));
  const branch: SessionEntry[] = [];
  const seen = new Set<string>();
  let entry = leafId ? byId.get(leafId) : undefined;
  while (entry && !seen.has(entry.id)) {
    seen.add(entry.id);
    branch.push(entry);
    entry = entry.parentId ? byId.get(entry.parentId) : undefined;
  }
  return branch.reverse();
}

/**
 * Evidence compression for explicitly marked delegate invocations only. Never
 * infer mode from a title, or convert the local model's usage into cloud savings.
 * The caller may reread transcripts or spend more tokens invoking/verifying the
 * worker; those costs are not observable in this session.
 */
export async function deriveDelegation(
  entries: FileEntry[],
  leafId: string | undefined,
  sessionId: string,
  completeFile: boolean,
): Promise<DelegationSummary | null> {
  if (
    !entries.some(
      (entry) =>
        entry.type === 'custom' &&
        (entry.customType === 'mlx-delegate-session' || entry.customType === 'mlx-delegate-handoff'),
    )
  )
    return null;
  const header = entries.find((entry) => entry.type === 'session');
  // Legacy handoffs have no session ID and are copied verbatim by a fork.
  // Only an unforked session can use one to establish its own delegate identity.
  const canIdentifyLegacy = header?.id === sessionId && !header.parentSession;
  let identified = false;
  let legacy = false;
  let issue: Exclude<DelegationSummary, { status: 'complete' }> | undefined;
  // The UI reports unique evidence for the whole session, including follow-ups.
  const evidence = new Set<string>();
  const handoffs: string[] = [];
  let run: { handoff: string | null; hasUser: boolean; unsupported: boolean; blocked: boolean } | undefined;

  const finish = (): void => {
    if (!run) return;
    if (run.unsupported) issue = { status: 'unavailable', reason: 'unsupported-content' };
    else if (run.blocked || !run.handoff?.trim()) issue = { status: 'incomplete', reason: 'no-final-handoff' };
    else handoffs.push(run.handoff);
    run = undefined;
  };

  for (const entry of fullBranch(entries, leafId)) {
    if (entry.type === 'custom' && entry.customType === 'mlx-delegate-session') {
      const data = entry.data as { version?: unknown; sessionId?: unknown } | null;
      // A fork copies old entries verbatim. It must not claim its parent's work.
      if (data?.version === 1 && data.sessionId === sessionId) {
        finish();
        identified = true;
        run = { handoff: null, hasUser: false, unsupported: false, blocked: false };
      }
      continue;
    }
    if (entry.type === 'custom' && entry.customType === 'mlx-delegate-handoff') {
      if (canIdentifyLegacy) legacy = true;
      if (run) run.blocked = true;
      continue;
    }
    if (!run || entry.type !== 'message') continue;
    const msg = entry.message as unknown as {
      role?: string;
      content?: unknown;
      isError?: boolean;
      stopReason?: string;
    };
    if (msg.role === 'user') {
      // A later ordinary-agent resume has no new marker. Do not count its work.
      if (run.hasUser) finish();
      else run.hasUser = true;
    } else if (msg.role === 'toolResult') {
      run.handoff = null;
      if (msg.isError) continue;
      const text = textContent(msg.content);
      if (text === null) run.unsupported = true;
      else if (text.trim()) evidence.add(text);
    } else if (msg.role === 'assistant') {
      const callsTools = Array.isArray(msg.content) && msg.content.some((b) => b?.type === 'toolCall');
      // Length-limited, aborted and failed generations are not completed handoffs.
      run.handoff = msg.stopReason === 'stop' && !callsTools ? textContent(msg.content) : null;
      if (msg.stopReason === 'stop' && !callsTools && run.handoff === null) run.unsupported = true;
    }
  }
  finish();
  if (!identified) return legacy ? { status: 'unavailable', reason: 'legacy' } : null;
  if (!completeFile) return { status: 'incomplete', reason: 'partial-record' };
  if (issue) return issue;
  if (evidence.size === 0) return { status: 'unavailable', reason: 'no-evidence' };
  let counts: number[];
  try {
    counts = await countDelegateTokens([...evidence, ...handoffs]);
  } catch {
    // A missing addon/asset must not hide the session or advertise false savings.
    return { status: 'unavailable', reason: 'tokenizer-unavailable' };
  }
  const evidenceTokens = counts.slice(0, evidence.size).reduce((sum, count) => sum + count, 0);
  const handoffTokens = counts.slice(evidence.size).reduce((sum, count) => sum + count, 0);
  if (evidenceTokens === 0) return { status: 'unavailable', reason: 'no-evidence' };
  const savedTokens = evidenceTokens - handoffTokens;
  return {
    status: 'complete',
    tokenizer: 'o200k_base',
    evidenceTokens,
    handoffTokens,
    savedTokens,
    savingsRatio: savedTokens / evidenceTokens,
  };
}
