/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { ToolCall, ToolCallGroup, type ToolCallProps } from '../ui/src/components/arc-x/tool-call/tool-call.js';
import { byRole, waitFor } from './render.js';

let mounted: { root: Root; container: HTMLElement } | undefined;

function mount(element: ReactElement): HTMLElement {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  mounted = { root, container };
  return container;
}

afterEach(() => {
  if (mounted === undefined) return;
  const { root, container } = mounted;
  act(() => root.unmount());
  container.remove();
  mounted = undefined;
});

const call = (props: Partial<ToolCallProps> & { name: string }): ReactElement =>
  createElement(ToolCall, { status: 'done', key: props.name, ...props });

describe('ToolCall', () => {
  it('a failed call opens by default and shows its output; a done call keeps it out of the DOM', () => {
    const failed = mount(
      call({ name: 'search', title: 'Search the index', status: 'failed', output: 'could not reach index' }),
    );
    const button = failed.querySelector('button[aria-expanded]');
    expect(button?.getAttribute('aria-expanded')).toBe('true');
    expect(failed.textContent).toContain('could not reach index');
    expect(failed.textContent).toContain('failed');
    act(() => mounted!.root.unmount());
    mounted!.container.remove();
    mounted = undefined;

    const done = mount(call({ name: 'read_file', title: 'planner.rs', output: 'fn main() {}', durationMs: 120 }));
    expect(done.querySelector('button[aria-expanded]')?.getAttribute('aria-expanded')).toBe('false');
    expect(done.textContent).not.toContain('fn main');
    expect(done.textContent).toContain('120 ms');
    act(() => (done.querySelector('button[aria-expanded]') as HTMLButtonElement).click());
    expect(done.textContent).toContain('fn main');
  });

  it('a row without a body is not a button', () => {
    const container = mount(call({ name: 'list_dir', title: 'src' }));
    expect(container.querySelector('button')).toBeNull();
    expect(container.querySelector('[data-status="done"]')).not.toBeNull();
  });

  it('names the button with name, title, status word and duration', () => {
    const container = mount(call({ name: 'read_file', title: 'planner.rs', input: '{}', durationMs: 1240 }));
    expect(byRole(container, 'button', 'read_file, planner.rs, done, 1.24 s')).not.toBeNull();
  });

  it('a word-style row with an icon, a detail in its name, and a body that replaces input and output', () => {
    const container = mount(
      call({
        name: 'Searched',
        nameStyle: 'word',
        icon: createElement('svg', { 'data-test-icon': '' }),
        title: '"rust"',
        detail: '2 results',
        input: 'RAW INPUT',
        body: createElement('p', null, 'BODY'),
      }),
    );
    expect(container.querySelector('[data-icon]')).not.toBeNull();
    expect(container.querySelector('[data-style="word"]')).not.toBeNull();
    const button = byRole(container, 'button', 'Searched, "rust", 2 results, done')!;
    act(() => button.click());
    expect(container.textContent).toContain('BODY');
    expect(container.textContent).not.toContain('RAW INPUT');
  });

  it('a running call is live only when told so; its glyph is a loader, the word is shown when not live', () => {
    const container = mount(
      createElement(
        'div',
        null,
        call({ name: 'run', title: 'cargo test', status: 'running', live: true }),
        call({ name: 'run2', title: 'cargo build', status: 'running' }),
        call({ name: 'later', status: 'other', statusText: 'not finished' }),
      ),
    );
    const rows = container.querySelectorAll('[data-status="running"]');
    expect(rows[0]?.hasAttribute('data-live')).toBe(true);
    expect(container.querySelectorAll('[data-live]')).toHaveLength(1);
    expect(container.querySelectorAll('svg[data-status="loading"]')).toHaveLength(2);
    expect(container.textContent).toContain('not finished');
    expect(container.textContent).toContain('running');
  });
});

describe('ToolCallGroup', () => {
  function group(running: number, live = false): ReactElement {
    return createElement(ToolCallGroup, {
      count: 3,
      failed: 1,
      running,
      live,
      children: [
        call({ name: 'read_file', title: 'planner.rs', input: 'a' }),
        call({ name: 'search', status: 'failed', output: 'no index' }),
        call({ name: 'run', status: running > 0 ? 'running' : 'done', input: 'cargo test', live }),
      ],
    });
  }

  it('counts the calls and the failures in its label and name', () => {
    const container = mount(group(0));
    expect(byRole(container, 'group', '3 tool calls, 1 failed')).not.toBeNull();
    expect(container.textContent).toContain('Used 3 tools');
    expect(container.textContent).toContain('1 failed');
    act(() => mounted!.root.render(group(1, true)));
    expect(container.textContent).toContain('Using 3 tools');
    expect(container.querySelector('[data-live]')).not.toBeNull();
  });

  it('closes to its header and opens again', async () => {
    const container = mount(group(0));
    const header = byRole(container, 'button', 'Used 3 tools 1 failed') as HTMLButtonElement;
    expect(header.getAttribute('aria-expanded')).toBe('true');
    act(() => header.click());
    await waitFor(() => !container.textContent!.includes('read_file'));
    act(() => header.click());
    expect(container.textContent).toContain('read_file');
  });

  it('is one tab stop: arrows move real focus between the header and the rows', () => {
    const container = mount(group(0));
    const toggles = (): HTMLElement[] => [...container.querySelectorAll<HTMLElement>('button[aria-expanded]')];
    const stops = toggles().filter((button) => button.tabIndex === 0);
    expect(stops).toHaveLength(1);
    const header = stops[0]!;
    act(() => header.focus());
    act(() => {
      header.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
    });
    const active = document.activeElement as HTMLElement;
    expect(active.getAttribute('aria-label')).toBe('read_file, planner.rs, done');
    expect(active.getAttribute('tabindex')).toBe('0');
    expect(header.getAttribute('tabindex')).toBe('-1');
    act(() => {
      active.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true }));
    });
    expect((document.activeElement as HTMLElement).getAttribute('aria-label')).toBe('run, done');
    expect(toggles().filter((button) => button.tabIndex === 0)).toHaveLength(1);
  });
});
