import type {
  ExtensionAPI,
  ExtensionContext,
  ToolCallEvent,
  ToolCallEventResult,
} from '@earendil-works/pi-coding-agent';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { createPermissionGateExtension } from '../src/extensions/permission-gate.js';

type ToolCallHandler = (
  event: ToolCallEvent,
  ctx: ExtensionContext,
) => Promise<ToolCallEventResult | undefined | void> | ToolCallEventResult | undefined | void;

/**
 * Run the real factory against a hand-rolled fake ExtensionAPI and capture
 * the registered `tool_call` handler.
 */
function loadGateHandler(): ToolCallHandler {
  const handlers = new Map<string, unknown>();
  const fakePi = {
    on(event: string, handler: unknown): void {
      handlers.set(event, handler);
    },
  } as unknown as ExtensionAPI;

  const extension = createPermissionGateExtension();
  expect(typeof extension).toBe('object');
  if (typeof extension === 'function') {
    throw new Error('expected the named InlineExtension form');
  }
  expect(extension.name).toBe('mlx-permission-gate');
  void extension.factory(fakePi);

  const handler = handlers.get('tool_call');
  expect(handler, 'factory must register a tool_call handler').toBeTypeOf('function');
  return handler as ToolCallHandler;
}

interface SelectCall {
  title: string;
  options: string[];
}

/** Fake ExtensionContext recording ui.select calls and answering with `choice`. */
function makeCtx(hasUI: boolean, choice?: string): { ctx: ExtensionContext; selectCalls: SelectCall[] } {
  const selectCalls: SelectCall[] = [];
  const ctx = {
    hasUI,
    ui: {
      select: (title: string, options: string[]): Promise<string | undefined> => {
        if (!hasUI) {
          throw new Error('ui.select must not be called when hasUI is false');
        }
        selectCalls.push({ title, options });
        return Promise.resolve(choice);
      },
    },
  } as unknown as ExtensionContext;
  return { ctx, selectCalls };
}

function toolCallEvent(toolName: string, input: unknown): ToolCallEvent {
  return { type: 'tool_call', toolCallId: 'call-1', toolName, input } as ToolCallEvent;
}

const savedAutoApprove = process.env['MLX_AGENT_AUTO_APPROVE'];

afterEach(() => {
  if (savedAutoApprove === undefined) {
    delete process.env['MLX_AGENT_AUTO_APPROVE'];
  } else {
    process.env['MLX_AGENT_AUTO_APPROVE'] = savedAutoApprove;
  }
});

describe('createPermissionGateExtension', () => {
  it('passes non-gated tools through without consulting the UI', async () => {
    const handler = loadGateHandler();
    const { ctx, selectCalls } = makeCtx(true, 'No');
    for (const toolName of ['read', 'grep', 'find', 'ls', 'my_custom_tool']) {
      const result = await handler(toolCallEvent(toolName, { path: '/tmp/x' }), ctx);
      expect(result, toolName).toBeUndefined();
    }
    expect(selectCalls).toHaveLength(0);
  });

  it('allows a gated tool when the user answers Yes', async () => {
    const handler = loadGateHandler();
    const { ctx, selectCalls } = makeCtx(true, 'Yes');
    const result = await handler(toolCallEvent('bash', { command: 'ls -la' }), ctx);
    expect(result).toBeUndefined();
    expect(selectCalls).toHaveLength(1);
    expect(selectCalls[0]!.title).toContain('Allow bash?');
    expect(selectCalls[0]!.options).toEqual(['Yes', 'Always (this session)', 'No']);
  });

  it('blocks a gated tool when the user answers No', async () => {
    const handler = loadGateHandler();
    const { ctx } = makeCtx(true, 'No');
    const result = await handler(toolCallEvent('write', { path: '/tmp/out.txt', content: 'x' }), ctx);
    expect(result).toEqual({ block: true, reason: 'Blocked by user' });
  });

  it('blocks when the select dialog is dismissed (undefined choice)', async () => {
    const handler = loadGateHandler();
    const { ctx } = makeCtx(true, undefined);
    const result = await handler(toolCallEvent('bash', { command: 'true' }), ctx);
    expect(result).toMatchObject({ block: true });
  });

  it('Always (this session) skips the prompt for the same tool but not for others', async () => {
    const handler = loadGateHandler();
    const { ctx, selectCalls } = makeCtx(true, 'Always (this session)');

    const first = await handler(toolCallEvent('bash', { command: 'echo one' }), ctx);
    expect(first).toBeUndefined();
    expect(selectCalls).toHaveLength(1);

    const second = await handler(toolCallEvent('bash', { command: 'echo two' }), ctx);
    expect(second).toBeUndefined();
    expect(selectCalls, 'allow-listed tool must not prompt again').toHaveLength(1);

    const other = await handler(toolCallEvent('write', { path: '/tmp/w.txt', content: '' }), ctx);
    expect(other).toBeUndefined();
    expect(selectCalls, 'a different gated tool must still prompt').toHaveLength(2);
    expect(selectCalls[1]!.title).toContain('Allow write?');
  });

  it('the session allow list does not leak across extension instances', async () => {
    const first = loadGateHandler();
    const always = makeCtx(true, 'Always (this session)');
    await first(toolCallEvent('bash', { command: 'echo' }), always.ctx);
    expect(always.selectCalls).toHaveLength(1);

    const second = loadGateHandler();
    const fresh = makeCtx(true, 'Yes');
    await second(toolCallEvent('bash', { command: 'echo' }), fresh.ctx);
    expect(fresh.selectCalls, 'a fresh instance must prompt again').toHaveLength(1);
  });

  it('without UI and without MLX_AGENT_AUTO_APPROVE, blocks and names the env var', async () => {
    delete process.env['MLX_AGENT_AUTO_APPROVE'];
    const handler = loadGateHandler();
    const { ctx } = makeCtx(false);
    const result = await handler(toolCallEvent('bash', { command: 'ls' }), ctx);
    expect(result).toMatchObject({ block: true });
    expect((result as ToolCallEventResult).reason).toContain('MLX_AGENT_AUTO_APPROVE');
  });

  it('without UI, MLX_AGENT_AUTO_APPROVE=1 allows gated tools', async () => {
    process.env['MLX_AGENT_AUTO_APPROVE'] = '1';
    const handler = loadGateHandler();
    const { ctx } = makeCtx(false);
    const result = await handler(toolCallEvent('edit', { path: '/tmp/a.ts', oldText: 'a', newText: 'b' }), ctx);
    expect(result).toBeUndefined();
  });

  it('without UI, other MLX_AGENT_AUTO_APPROVE values still block', async () => {
    process.env['MLX_AGENT_AUTO_APPROVE'] = 'true';
    const handler = loadGateHandler();
    const { ctx } = makeCtx(false);
    const result = await handler(toolCallEvent('bash', { command: 'ls' }), ctx);
    expect(result).toMatchObject({ block: true });
  });

  it('shows the bash command as the prompt detail', async () => {
    const handler = loadGateHandler();
    const { ctx, selectCalls } = makeCtx(true, 'Yes');
    await handler(toolCallEvent('bash', { command: 'rm -rf /tmp/scratch' }), ctx);
    expect(selectCalls[0]!.title).toContain('rm -rf /tmp/scratch');
  });

  it('shows the file path as the prompt detail for edit and write', async () => {
    const handler = loadGateHandler();
    const { ctx, selectCalls } = makeCtx(true, 'Yes');
    await handler(toolCallEvent('edit', { path: '/repo/src/main.ts', oldText: 'a', newText: 'b' }), ctx);
    expect(selectCalls[0]!.title).toContain('/repo/src/main.ts');

    await handler(toolCallEvent('write', { path: '/repo/README.md', content: 'hello' }), ctx);
    expect(selectCalls[1]!.title).toContain('/repo/README.md');
  });

  describe('prompt detail sanitization', () => {
    it('strips ANSI escape sequences (CSI, incl. the bare C1 CSI byte) from the title', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      const command = 'echo safe\u001b[2J\u001b[1;1H\u001b[31m && rm -rf /\u009b2J';
      await handler(toolCallEvent('bash', { command }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).not.toContain('\u001b');
      expect(title).not.toContain('\u009b');
      expect(title).toContain('echo safe');
      expect(title).toContain('rm -rf /');
    });

    it('strips OSC sequences (terminal title / hyperlinks) including their payload', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      const command = 'ls \u001b]0;spoofed-window-title\u0007-la';
      await handler(toolCallEvent('bash', { command }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).not.toContain('\u001b');
      expect(title).not.toContain('spoofed-window-title');
      expect(title).toContain('ls ');
      expect(title).toContain('-la');
    });

    it('replaces non-ANSI control characters with a visible placeholder', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      const command = 'echo a\u0007b\rc\u0000d';
      await handler(toolCallEvent('bash', { command }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).not.toContain('\u0007');
      expect(title).not.toContain('\r');
      expect(title).not.toContain('\u0000');
      expect(title).toContain('echo a�b�c�d');
    });

    it('keeps newlines and tabs in a short multi-line command unchanged', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      const command = 'for f in *.txt; do\n\techo "$f"\ndone';
      await handler(toolCallEvent('bash', { command }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).toContain(command);
      expect(title).not.toContain('[truncated]');
    });

    it('truncates an oversized command and marks the truncation', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      const command = `echo start ${'x'.repeat(5000)} echo end`;
      await handler(toolCallEvent('bash', { command }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).toContain('echo start');
      expect(title).toContain('… [truncated]');
      expect(title).not.toContain('echo end');
      expect(title.length).toBeLessThan(600);
    });

    it('caps a many-line command at the line limit with a marker', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      const command = Array.from({ length: 12 }, (_, i) => `step-${i}`).join('\n');
      await handler(toolCallEvent('bash', { command }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).toContain('step-0');
      expect(title).toContain('step-5');
      expect(title).not.toContain('step-6');
      expect(title).toContain('… [truncated]');
    });

    it('a command that is nothing but escape sequences shows a visible stand-in', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      await handler(toolCallEvent('bash', { command: '\u001b[2J\u001b[3J\u001b[H' }), ctx);
      expect(selectCalls[0]!.title).toContain('(unprintable content)');
    });

    it('sanitizes the write/edit path detail too', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      await handler(toolCallEvent('write', { path: '/tmp/\u001b[31mevil\u001b[0m.txt', content: '' }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).not.toContain('\u001b');
      expect(title).toContain('/tmp/evil.txt');
    });

    it('leaves a normal command and path untouched (no marker, no placeholder)', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      await handler(toolCallEvent('bash', { command: 'yarn build:native --verbose' }), ctx);
      await handler(toolCallEvent('edit', { path: '/repo/src/index.ts', oldText: 'a', newText: 'b' }), ctx);
      expect(selectCalls[0]!.title).toBe('Allow bash?\n\n  yarn build:native --verbose');
      expect(selectCalls[1]!.title).toBe('Allow edit?\n\n  /repo/src/index.ts');
    });
  });

  it('malformed event input still yields a decision without throwing', async () => {
    const handler = loadGateHandler();
    const malformedInputs: unknown[] = [undefined, null, 42, 'oops', {}, { command: 123 }, { path: { nested: true } }];
    for (const input of malformedInputs) {
      const { ctx, selectCalls } = makeCtx(true, 'No');
      const result = await handler(toolCallEvent('bash', input), ctx);
      expect(result, JSON.stringify(input)).toEqual({ block: true, reason: 'Blocked by user' });
      expect(selectCalls, 'the prompt must still be shown').toHaveLength(1);
      expect(selectCalls[0]!.title).toContain('Allow bash?');
    }
  });
});
