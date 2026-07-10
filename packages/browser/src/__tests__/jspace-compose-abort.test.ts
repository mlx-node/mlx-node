import { describe, expect, it } from 'vite-plus/test';

import { composeAbort } from '../../demo/jspace/compose-abort';

// composeAbort binds a /jspace readout's cancellation to BOTH the component
// lifetime AND the worker generation serving it, so a worker teardown (retry /
// replacement) rejects the in-flight readout promptly. The composite must abort
// when EITHER source does.
describe('composeAbort', () => {
  it('returns the base signal unchanged when there is no second signal', () => {
    const base = new AbortController().signal;
    expect(composeAbort(base)).toBe(base);
  });

  it('aborts the composite when the BASE (component) signal aborts', () => {
    const base = new AbortController();
    const extra = new AbortController();
    const composed = composeAbort(base.signal, extra.signal);
    expect(composed.aborted).toBe(false);
    base.abort();
    expect(composed.aborted).toBe(true);
  });

  it('aborts the composite when the EXTRA (worker generation) signal aborts', () => {
    const base = new AbortController();
    const extra = new AbortController();
    const composed = composeAbort(base.signal, extra.signal);
    expect(composed.aborted).toBe(false);
    extra.abort();
    expect(composed.aborted).toBe(true);
  });

  it('is already aborted when the extra signal was aborted before composing', () => {
    const base = new AbortController();
    const extra = new AbortController();
    extra.abort();
    expect(composeAbort(base.signal, extra.signal).aborted).toBe(true);
  });
});
