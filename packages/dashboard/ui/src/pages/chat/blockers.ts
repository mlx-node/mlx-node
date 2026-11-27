/**
 * Why Send is off, first match wins, and what the page shows for it: the reason beside the button, the composer's
 * notice strip, and on the New screen the panel that takes the hero's place. Every text is fixed; a failure's own
 * message is used only for a code this table does not know.
 */

import {
  type AttachmentInput,
  attachmentView,
  CHAT_PICTURES_MAX_BASE64,
  estimateTokens,
  MESSAGE_FILES_MAX_BYTES,
  numberedBytes,
  pdfNotes,
  type PdfPlan,
  type PdfWay,
  planPdfs,
  storedPictureChars,
  type UserAttachment,
  utf8Bytes,
  WINDOW_SHARE,
} from '../../../../src/chat/attachment-rule.js';
import { type DraftAttachment, draftInput, draftTokens } from '../../lib/chat-attachments';
import { CHAT, fill } from '../../lib/chat-strings';
import type { ChatFailure, ChatFailureCode, ChatHostState, ChatMode } from '../../lib/chat-types';
import { scratchShown } from '../../lib/paths';
import type { TranscriptEntry } from '../../lib/types';

export type BlockActionKind =
  | 'install'
  | 'restart'
  | 'open-engine'
  | 'open-models'
  | 'open-providers'
  | 'copy-details'
  | 'retry'
  | 'pick-model'
  | 'choose-folder'
  | 'go-to-ask';

export interface BlockAction {
  label: string;
  kind: BlockActionKind;
}

export type BlockIcon = 'power' | 'unplug' | 'boxes' | 'download' | 'loader' | 'alert';

export interface StripNotice {
  tone: 'neutral' | 'warning' | 'danger';
  text: string;
  /** The whole sentence when `text` is the short one: the strip's tooltip. */
  title?: string;
  working?: boolean;
  action?: BlockAction;
  secondary?: BlockAction;
}

export interface Blocker {
  /** The reason beside the button and in its tooltip. */
  reason: string;
  notice?: StripNotice;
  /** The New screen's panel in place of the hero. */
  panel?: { title: string; description: string; icon: BlockIcon; action?: BlockAction; secondary?: BlockAction };
}

type Row = readonly [title: string, sentence: string, heroAddition: string, action: string];

function rowBlocker(
  row: Row,
  tone: StripNotice['tone'],
  icon: BlockIcon,
  kind: BlockActionKind | null,
  secondary?: BlockAction,
): Blocker {
  const [title, sentence, addition, label] = row;
  const action = kind === null || label === '' ? undefined : { label, kind };
  return {
    reason: sentence,
    notice: { tone, text: sentence, action, secondary },
    panel: {
      title,
      description: addition === '' ? sentence : `${sentence} ${addition}`,
      icon,
      action,
      secondary,
    },
  };
}

function modeTitle(mode: ChatMode): string {
  return mode === 'work' ? CHAT.modeWork : CHAT.modeChat;
}

function blockedBy(failure: ChatFailure | null, mode: ChatMode, canInstall = true): Blocker {
  const b = CHAT.blockers;
  switch (failure?.code) {
    case 'inference-failed':
      return rowBlocker(b['inference-failed'], 'danger', 'unplug', 'open-engine');
    case 'agent-incompatible':
      return rowBlocker(b['agent-incompatible'], 'danger', 'alert', canInstall ? 'install' : null, {
        label: b.copyDetails,
        kind: 'copy-details',
      });
    case 'skill-bundle-corrupt':
      return rowBlocker(b['skill-bundle-corrupt'], 'danger', 'alert', null);
    default: {
      const text = failure?.message ?? b['agent-failed'][1];
      return rowBlocker([fill(CHAT.blockedTitle, { mode: modeTitle(mode) }), text, '', ''], 'danger', 'alert', null);
    }
  }
}

/** Rows 1 to 7: the states of the transport, the agent and the engine that stop every message of a mode. */
export function modeBlocker(connected: boolean, host: ChatHostState | null, mode: ChatMode): Blocker | null {
  const b = CHAT.blockers;
  if (!connected) return { reason: b.reconnecting };
  if (host === null) return { reason: CHAT.startingTip };
  const canInstall = host.agent.canInstall !== false;
  if (host.agent.binary === 'missing') {
    if (host.agent.install === 'running') {
      const text = b['agent-installing'];
      return {
        reason: text,
        notice: { tone: 'neutral', text, working: true },
        panel: { title: b['agent-missing'][0], description: text, icon: 'loader' },
      };
    }
    if (!canInstall) return rowBlocker(b['agent-install-unavailable'], 'neutral', 'alert', null);
    if (host.agent.install === 'failed') return rowBlocker(b['agent-install-failed'], 'danger', 'alert', 'install');
    return rowBlocker(b['agent-missing'], 'neutral', 'download', 'install');
  }
  if (host.agent.binary === 'too-old') {
    return canInstall
      ? rowBlocker(b['agent-too-old'], 'neutral', 'download', 'install')
      : rowBlocker(b['agent-too-old-unavailable'], 'neutral', 'alert', null);
  }
  const state = host.modes[mode];
  if (state.state === 'blocked') return blockedBy(state.failure, mode, canInstall);
  if (state.state === 'failed') return rowBlocker(b['agent-failed'], 'danger', 'alert', 'restart');
  if (host.inference === 'stopped') return rowBlocker(b['inference-stopped'], 'neutral', 'power', 'open-engine');
  if (host.models.length === 0) return rowBlocker(b['no-models'], 'neutral', 'boxes', 'open-models');
  return null;
}

/** Strips that block nothing: the engine starting, an agent that sent something unreadable. */
export function infoNotice(host: ChatHostState | null, mode: ChatMode): StripNotice | null {
  if (host === null) return null;
  if (host.inference === 'starting') return { tone: 'neutral', text: CHAT.blockers['inference-starting'] };
  const state = host.modes[mode];
  if (state.state === 'crashed' && state.failure?.code === 'agent-protocol-error') {
    return { tone: 'warning', text: CHAT.blockers['agent-protocol-error'] };
  }
  return null;
}

/**
 * Work: the engine restarted under live commands and monitors. The agent process is shared, so the count is every
 * Work conversation's; the next message restarts the agent and stops them all.
 */
export function engineLostNotice(
  host: ChatHostState | null,
  mode: ChatMode,
  conversationId: string | null,
): StripNotice | null {
  if (host === null || mode !== 'work') return null;
  const { engineLost, tasks } = host.modes.work;
  if (!engineLost || tasks === 0) return null;
  const here =
    conversationId === null
      ? 0
      : (host.active.find((entry) => entry.mode === 'work' && entry.conversationId === conversationId)?.tasks ?? 0);
  const others = tasks - here;
  return {
    tone: 'warning',
    text: tasks === 1 ? CHAT.engineRestartStopsOne : fill(CHAT.engineRestartStops, { n: tasks }),
    title:
      others > 0 ? `${CHAT.engineRestartTip} ${fill(CHAT.engineRestartOthers, { n: others })}` : CHAT.engineRestartTip,
  };
}

/** "agent {tag} ({agentVersion}) · needs {minTag}": versions only. */
export function copyDetails(host: ChatHostState, mode: ChatMode): string {
  return fill(CHAT.copyDetailsText, {
    tag: host.agent.tag ?? 'none',
    version: host.modes[mode].agentVersion ?? 'unknown',
    minTag: host.agent.minTag,
  });
}

/** The page's sentence for a refusal that names why the agent or the engine cannot act. */
export const BLOCKER_SENTENCES: Partial<Record<ChatFailureCode, string>> = {
  'inference-stopped': CHAT.blockers['inference-stopped'][1],
  'inference-failed': CHAT.blockers['inference-failed'][1],
  'no-models': CHAT.blockers['no-models'][1],
  'agent-missing': CHAT.blockers['agent-missing'][1],
  'agent-too-old': CHAT.blockers['agent-too-old'][1],
  'agent-incompatible': CHAT.blockers['agent-incompatible'][1],
  'agent-failed': CHAT.blockers['agent-failed'][1],
  'skill-bundle-corrupt': CHAT.blockers['skill-bundle-corrupt'][1],
  'agent-crashed': CHAT.outcomes['agent-crashed'],
  'engine-lost': CHAT.history.engineLost,
  'load-slow': CHAT.history.loadSlow,
};

/** The page's own sentence for a folder the Control Panel refused, found by the server's fixed sentence. */
export function cwdProblemText(message: string | undefined): string | null {
  if (message === undefined) return null;
  for (const sentence of Object.values(CHAT.cwdProblems)) if (sentence === message) return sentence;
  return null;
}

/**
 * The strip after a send that the server refused, or a turn that failed before it ran. `toast` instead of a strip
 * for a busy conversation (the controls that could cause it are off, so it is a race).
 */
export function sendFailure(
  reason: string | undefined,
  code: string | undefined,
  mode: ChatMode,
  message?: string,
  /** The conversation's model, for the sentence of a load that failed. */
  model?: string,
  canInstall = true,
  /** Where Work without a folder makes its folders, as people read it. */
  root?: string,
): { strip: StripNotice } | { toast: string } {
  const e = CHAT.sendErrors;
  const strip = (
    row: readonly [string, string],
    kind: BlockActionKind | null,
    tone: StripNotice['tone'] = 'danger',
  ) => ({
    strip: { tone, text: row[0], action: kind === null || row[1] === '' ? undefined : { label: row[1], kind } },
  });
  switch (reason) {
    case 'agent-slow':
      return strip(e['agent-slow'], 'retry');
    case 'load-slow':
      return strip([CHAT.history.loadSlow, CHAT.history.tryAgain], 'retry');
    case 'profile-mismatch':
      return strip(
        mode === 'work' ? e['profile-mismatch-work'] : e['profile-mismatch-chat'],
        canInstall ? 'install' : null,
      );
    case 'model-unavailable':
      return strip(e['model-unavailable'], 'pick-model');
    case 'model-load-failed':
      return strip(
        [fill(e['model-load-failed'][0], { model: model ?? CHAT.model }), e['model-load-failed'][1]],
        'retry',
      );
    case 'cwd-invalid':
      return strip([cwdProblemText(message) ?? e['cwd-invalid'][0], e['cwd-invalid'][1]], 'choose-folder');
    case 'cwd-mismatch':
      return strip(e['cwd-mismatch'], 'retry');
    case 'folder-not-made':
      return strip(
        [fill(e['folder-not-made'][0], { root: root ?? scratchShown(null, null) }), e['folder-not-made'][1]],
        'choose-folder',
      );
    case 'conversation-busy':
      return { toast: e['conversation-busy'][0] };
    case 'folder-busy':
      return strip(e['folder-busy'], 'retry', 'warning');
    case 'slash-text':
      return strip(e['slash-text'], null);
    case 'keychain-unavailable':
      return strip(e['keychain-unavailable'], 'open-providers', 'warning');
    case 'attachment-refused':
      return strip(e['attachment-refused'], null);
    case 'pictures-full':
      return strip(e['pictures-full'], null);
    case 'queue-full':
      return strip([CHAT.queue.full, ''], null, 'warning');
    case 'steer-busy':
      return strip([CHAT.queue.notThisStep, ''], null, 'warning');
    default:
      break;
  }
  if (code === 'E_PAYLOAD_TOO_LARGE') return { strip: { tone: 'danger', text: CHAT.tooLong } };
  const known = reason === undefined ? undefined : BLOCKER_SENTENCES[reason as ChatFailureCode];
  if (known !== undefined) return { strip: { tone: 'danger', text: known } };
  return strip(e.other, 'retry');
}

export interface AttachFacts {
  mode: ChatMode;
  /** Chat and the model row has images = true. */
  seesImages: boolean;
  /** The model row has images = true, in either mode. */
  seesPages: boolean;
  text: string;
  files: readonly DraftAttachment[];
  /** The model's context window in tokens; null while no agent listed the model. */
  windowTokens: number | null;
  /** The model row's family; null when the endpoint names none. */
  family: string | null;
  /** The conversation's context in tokens; null when not known. */
  contextUsed: number | null;
  /** Files of this place still being read. */
  reading: number;
  /** Stored base64 of the pictures and pages the conversation already holds ({@link heldPictureChars}). */
  heldPictures: number;
  /** The images the conversation already holds (`heldImages`). */
  heldImages: number;
  /** The PDF tool is there in this mode. */
  pdfTool: boolean;
}

export function heldPictureChars(entries: readonly TranscriptEntry[]): number {
  const items: UserAttachment[] = [];
  for (const entry of entries)
    if (entry.kind === 'user' && entry.attachments !== undefined) items.push(...entry.attachments);
  return storedPictureChars(items);
}

/**
 * Which way each PDF of the files goes ({@link planPdfs}), and what the message and the readers cost; null with no
 * PDF. The tool takes them only when every PDF's pages are at the host: one drawn while this mode had no tool has none.
 */
export function attachPlan(facts: Omit<AttachFacts, 'reading'>): PdfPlan | null {
  const pdfs = facts.files.flatMap((file) => (file.kind === 'pdf' ? [file] : []));
  if (pdfs.length === 0) return null;
  const others: AttachmentInput[] = [];
  for (const file of facts.files) if (file.kind !== 'pdf') others.push(draftInput(file));
  return planPdfs({
    text: facts.text,
    attachments: others,
    pdfs,
    family: facts.family,
    window: facts.windowTokens,
    contextUsed: facts.contextUsed,
    held: facts.heldImages,
    stored: facts.heldPictures,
    tool: facts.pdfTool && pdfs.every((pdf) => pdf.stage !== null),
  });
}

/** Each file's way: a PDF's by the plan, in the order added; every other file goes in the message. */
export function fileWays(files: readonly DraftAttachment[], plan: PdfPlan | null): PdfWay[] {
  let n = 0;
  return files.map((file) => (file.kind === 'pdf' ? (plan?.ways[n++] ?? 'message') : 'message'));
}

/** The files of a message as they are sent: each PDF its pages, or its stage when the plan sends it to the tool. */
export function messageInputs(files: readonly DraftAttachment[], plan: PdfPlan | null): AttachmentInput[] {
  const ways = fileWays(files, plan);
  return files.map((file, i) => draftInput(file, ways[i]));
}

/**
 * Why a message with these files cannot go, first match wins; null when it can. The composer shows it beside Send,
 * and `send()` checks it again, so Retry and Send again cannot pass it.
 */
export function attachProblem(facts: AttachFacts): string | null {
  const a = CHAT.attach;
  if (facts.reading > 0) return a.reading;
  if (facts.files.length === 0) return null;
  const pictures = facts.files.some((file) => file.kind === 'image');
  const pdfs = facts.files.some((file) => file.kind === 'pdf');
  if (pictures && facts.mode === 'work') return a.blockedWork;
  if (pictures && !facts.seesImages) return a.blockedModel;
  if (pdfs && !facts.seesPages) return a.blockedPdfModel;
  const plan = attachPlan(facts);
  if (plan?.problem === 'one') return a.blockedPdfOne;
  if (plan?.problem === 'room') return a.blockedPdfRoom;
  const others: AttachmentInput[] = [];
  for (const file of facts.files) if (file.kind !== 'pdf') others.push(draftInput(file));
  if (pictures && facts.heldPictures + storedPictureChars(others.map(attachmentView)) > CHAT_PICTURES_MAX_BASE64) {
    return CHAT.sendErrors['pictures-full'][0];
  }
  const ways = fileWays(facts.files, plan);
  let bytes = utf8Bytes(facts.text);
  for (const file of facts.files) if (file.kind === 'file') bytes += file.numbered;
  for (const note of pdfNotes(messageInputs(facts.files, plan))) bytes += numberedBytes(note);
  if (bytes > MESSAGE_FILES_MAX_BYTES) return a.blockedLong;
  const window = facts.windowTokens;
  if (
    window !== null &&
    estimateTokens(
      facts.text,
      facts.files.map((file, i) => draftTokens(file, ways[i])),
      facts.family,
    ) >
      window * WINDOW_SHARE
  ) {
    return a.blockedWindow;
  }
  return null;
}
