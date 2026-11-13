import { act, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

let current: { root: Root; container: HTMLElement } | undefined;

/** Mount one element in a fresh container (one at a time); `unmountAll` in `afterEach` tears it down. */
export function mount(element: ReactElement): HTMLElement {
  unmountAll();
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  current = { root, container };
  return container;
}

export function rerender(element: ReactElement): void {
  if (current === undefined) throw new Error('rerender: nothing mounted');
  const { root } = current;
  act(() => root.render(element));
}

export function unmountAll(): void {
  if (current === undefined) return;
  const { root, container } = current;
  act(() => root.unmount());
  container.remove();
  current = undefined;
}

/** Flush effects and queued microtasks (a ResizeObserver stub reports on one). */
export async function flush(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
  });
}
