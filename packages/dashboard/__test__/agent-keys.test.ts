/** The key map of agent rows, the Agent view and a run's head (`ui/src/lib/agent-keys.ts`). */

import { describe, expect, it } from 'vite-plus/test';

import { agentKeyOf, type AgentKey, type KeyTarget } from '../ui/src/lib/agent-keys.js';

const el = (tagName: string, attrs: Record<string, string> = {}, parent: KeyTarget | null = null): KeyTarget => ({
  tagName,
  isContentEditable: false,
  getAttribute: (name) => attrs[name] ?? null,
  parentElement: parent,
});
const ROW = el('DIV', { role: 'row' });
const key = (k: string, target: KeyTarget | null = ROW, mods: Partial<AgentKey> = {}): AgentKey => ({
  key: k,
  target,
  ...mods,
});
const PAUSE = { pause: true };
const NO_PAUSE = { pause: false };

describe('agent keys (T40)', () => {
  // Mutation: read `key` without the target check → typing "x" in the message field stops the agent.
  it('a letter key from a field does nothing; from the row it acts', () => {
    const fields = [
      el('TEXTAREA'),
      el('INPUT'),
      el('INPUT', { type: 'search' }),
      el('SELECT'),
      { ...el('DIV'), isContentEditable: true },
      el('SPAN', {}, el('TEXTAREA')),
    ];
    for (const field of fields) {
      for (const k of ['x', 'X', 'm', 'p', 'Enter', ' ', 'ArrowDown', 'Escape', '[']) {
        expect(agentKeyOf(key(k, field), 'row', PAUSE)).toBeNull();
        expect(agentKeyOf(key(k, field), 'view', PAUSE)).toBeNull();
        expect(agentKeyOf(key(k, field), 'run-head', PAUSE)).toBeNull();
      }
    }
    expect(agentKeyOf(key('x'), 'row', PAUSE)).toBe('stop');
    expect(agentKeyOf(key('X', el('INPUT', { type: 'checkbox' })), 'row', PAUSE)).toBe('stop');
  });

  it('a row moves, opens, messages, pauses and stops; modifiers are the shell’s', () => {
    const row = (k: string) => agentKeyOf(key(k), 'row', PAUSE);
    expect(['ArrowUp', 'ArrowDown', 'Home', 'End', 'Enter', ' ', 'm', 'M', 'p', 'x'].map(row)).toEqual([
      'up',
      'down',
      'first',
      'last',
      'open',
      'open',
      'message',
      'message',
      'pause',
      'stop',
    ]);
    expect(agentKeyOf(key('k', ROW, { metaKey: true }), 'row', PAUSE)).toBeNull();
    expect(agentKeyOf(key('x', ROW, { ctrlKey: true }), 'row', PAUSE)).toBeNull();
    expect(agentKeyOf(key('x', ROW, { altKey: true }), 'row', PAUSE)).toBeNull();
    expect(agentKeyOf(key('q'), 'row', PAUSE)).toBeNull();
  });

  it('Enter and Space on a button in the row are its own press; the row opens only from itself', () => {
    expect(agentKeyOf(key('Enter', el('BUTTON')), 'row', PAUSE)).toBeNull();
    expect(agentKeyOf(key(' ', el('A')), 'row', PAUSE)).toBeNull();
    expect(agentKeyOf(key('Enter', el('SPAN', { role: 'button' })), 'row', PAUSE)).toBeNull();
    expect(agentKeyOf(key('x', el('BUTTON')), 'row', PAUSE)).toBe('stop');
  });

  // Mutation: map `P` whatever the build can do → a dead key (no Pause button is drawn).
  it('P does nothing until the agent build can pause', () => {
    expect(agentKeyOf(key('p'), 'row', NO_PAUSE)).toBeNull();
    expect(agentKeyOf(key('P'), 'run-head', NO_PAUSE)).toBeNull();
    expect(agentKeyOf(key('P'), 'run-head', PAUSE)).toBe('pause');
  });

  it('the view closes and steps through its list; a run head only pauses and stops', () => {
    expect(['Escape', '[', ']', 'x', 'ArrowDown'].map((k) => agentKeyOf(key(k), 'view', PAUSE))).toEqual([
      'close',
      'previous-agent',
      'next-agent',
      null,
      null,
    ]);
    expect(['x', 'Enter', 'm', 'ArrowDown'].map((k) => agentKeyOf(key(k), 'run-head', PAUSE))).toEqual([
      'stop',
      null,
      null,
      null,
    ]);
  });
});
