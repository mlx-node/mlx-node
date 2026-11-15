/**
 * Why Send is off, first match wins, and what the page shows for it: the reason beside the button, the composer's
 * notice strip, and on the New screen the panel that takes the hero's place. Every text is fixed; a failure's own
 * message is used only for a code this table does not know.
 */

import { CHAT, fill } from '../../lib/chat-strings';
import type { ChatFailure, ChatFailureCode, ChatHostState, ChatMode } from '../../lib/chat-types';

export type BlockActionKind =
  | 'install'
  | 'restart'
  | 'open-engine'
  | 'open-models'
  | 'open-providers'
  | 'copy-details'
  | 'retry'
  | 'pick-model'
  | 'choose-folder';

export interface BlockAction {
  label: string;
  kind: BlockActionKind;
}

export type BlockIcon = 'power' | 'unplug' | 'boxes' | 'download' | 'loader' | 'alert';

export interface StripNotice {
  tone: 'neutral' | 'warning' | 'danger';
  text: string;
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

function blockedBy(failure: ChatFailure | null, mode: ChatMode): Blocker {
  const b = CHAT.blockers;
  switch (failure?.code) {
    case 'inference-failed':
      return rowBlocker(b['inference-failed'], 'danger', 'unplug', 'open-engine');
    case 'agent-incompatible':
      return rowBlocker(b['agent-incompatible'], 'danger', 'alert', 'install', {
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
  if (host.agent.binary === 'missing') {
    if (host.agent.install === 'running') {
      const text = b['agent-installing'];
      return {
        reason: text,
        notice: { tone: 'neutral', text, working: true },
        panel: { title: b['agent-missing'][0], description: text, icon: 'loader' },
      };
    }
    if (host.agent.install === 'failed') return rowBlocker(b['agent-install-failed'], 'danger', 'alert', 'install');
    return rowBlocker(b['agent-missing'], 'neutral', 'download', 'install');
  }
  if (host.agent.binary === 'too-old') return rowBlocker(b['agent-too-old'], 'neutral', 'download', 'install');
  const state = host.modes[mode];
  if (state.state === 'blocked') return blockedBy(state.failure, mode);
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

/** "agent {tag} ({agentVersion}) · needs {minTag}": versions only. */
export function copyDetails(host: ChatHostState, mode: ChatMode): string {
  return fill(CHAT.copyDetailsText, {
    tag: host.agent.tag ?? 'none',
    version: host.modes[mode].agentVersion ?? 'unknown',
    minTag: host.agent.minTag,
  });
}

const BLOCKER_SENTENCES: Partial<Record<ChatFailureCode, string>> = {
  'inference-stopped': CHAT.blockers['inference-stopped'][1],
  'inference-failed': CHAT.blockers['inference-failed'][1],
  'no-models': CHAT.blockers['no-models'][1],
  'agent-missing': CHAT.blockers['agent-missing'][1],
  'agent-too-old': CHAT.blockers['agent-too-old'][1],
  'agent-incompatible': CHAT.blockers['agent-incompatible'][1],
  'agent-failed': CHAT.blockers['agent-failed'][1],
  'skill-bundle-corrupt': CHAT.blockers['skill-bundle-corrupt'][1],
  'agent-crashed': CHAT.outcomes['agent-crashed'],
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
    case 'profile-mismatch':
      return strip(mode === 'work' ? e['profile-mismatch-work'] : e['profile-mismatch-chat'], 'install');
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
    case 'conversation-busy':
      return { toast: e['conversation-busy'][0] };
    case 'slash-text':
      return strip(e['slash-text'], null);
    case 'keychain-unavailable':
      return strip(e['keychain-unavailable'], 'open-providers', 'warning');
    default:
      break;
  }
  if (code === 'E_PAYLOAD_TOO_LARGE') return { strip: { tone: 'danger', text: CHAT.tooLong } };
  const known = reason === undefined ? undefined : BLOCKER_SENTENCES[reason as ChatFailureCode];
  if (known !== undefined) return { strip: { tone: 'danger', text: known } };
  return strip(e.other, 'retry');
}
