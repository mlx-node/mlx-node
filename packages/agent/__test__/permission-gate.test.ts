import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type {
  ExtensionAPI,
  ExtensionContext,
  SessionStartEvent,
  ToolCallEvent,
  ToolCallEventResult,
} from '@earendil-works/pi-coding-agent';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { createPermissionGateExtension } from '../src/extensions/permission-gate.js';

type ToolCallHandler = (
  event: ToolCallEvent,
  ctx: ExtensionContext,
) => Promise<ToolCallEventResult | undefined | void> | ToolCallEventResult | undefined | void;

type SessionStartHandler = (event: SessionStartEvent, ctx: ExtensionContext) => Promise<void> | void;

/**
 * Run the real factory against a hand-rolled fake ExtensionAPI and return
 * every registered handler by event name.
 */
function buildGateHandlers(): Map<string, unknown> {
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
  return handlers;
}

/** Capture just the `tool_call` handler (the bash prefix falls back to on-demand). */
function loadGateHandler(): ToolCallHandler {
  const handler = buildGateHandlers().get('tool_call');
  expect(handler, 'factory must register a tool_call handler').toBeTypeOf('function');
  return handler as ToolCallHandler;
}

/** Capture both the `tool_call` and `session_start` handlers (snapshot path). */
function loadGate(): { toolCall: ToolCallHandler; sessionStart: SessionStartHandler } {
  const handlers = buildGateHandlers();
  const toolCall = handlers.get('tool_call');
  const sessionStart = handlers.get('session_start');
  expect(toolCall, 'factory must register a tool_call handler').toBeTypeOf('function');
  expect(sessionStart, 'factory must register a session_start handler').toBeTypeOf('function');
  return { toolCall: toolCall as ToolCallHandler, sessionStart: sessionStart as SessionStartHandler };
}

/** A minimal pi `session_start` event (boot). */
function sessionStartEvent(): SessionStartEvent {
  return { type: 'session_start', reason: 'startup' };
}

interface SelectCall {
  title: string;
  options: string[];
}

/** Fake ExtensionContext recording ui.select calls and answering with `choice`. */
function makeCtx(
  hasUI: boolean,
  choice?: string,
  extra?: { cwd?: string; isProjectTrusted?: () => boolean },
): { ctx: ExtensionContext; selectCalls: SelectCall[] } {
  const selectCalls: SelectCall[] = [];
  const ctx = {
    hasUI,
    cwd: extra?.cwd,
    isProjectTrusted: extra?.isProjectTrusted,
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

/**
 * Property-style guard for the sanitizer's core invariant: encode, never
 * delete. Every printable-ASCII character of the input (anything bash
 * could interpret — letters, digits, shell metacharacters, spaces) must
 * appear in the sanitized title in the original order. Encoding may
 * insert visible `\xNN` text, but it must never remove or reorder a
 * shell-significant byte.
 */
function expectPreservesPrintables(input: string, title: string): void {
  let pos = 0;
  for (const ch of input) {
    const code = ch.codePointAt(0)!;
    if (code < 0x20 || code > 0x7e) {
      continue;
    }
    const found = title.indexOf(ch, pos);
    expect(
      found,
      `printable char ${JSON.stringify(ch)} of the input was lost after offset ${pos}`,
    ).toBeGreaterThanOrEqual(0);
    pos = found + 1;
  }
}

const savedAutoApprove = process.env['MLX_AGENT_AUTO_APPROVE'];
const savedAgentDir = process.env['PI_CODING_AGENT_DIR'];

afterEach(() => {
  if (savedAutoApprove === undefined) {
    delete process.env['MLX_AGENT_AUTO_APPROVE'];
  } else {
    process.env['MLX_AGENT_AUTO_APPROVE'] = savedAutoApprove;
  }
  if (savedAgentDir === undefined) {
    delete process.env['PI_CODING_AGENT_DIR'];
  } else {
    process.env['PI_CODING_AGENT_DIR'] = savedAgentDir;
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

  describe('prompt detail sanitization (encode, never delete)', () => {
    it('encodes CSI escape bytes visibly while keeping every printable byte', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      const command = 'echo safe\u001b[2J\u001b[1;1H\u001b[31m && rm -rf /\u009b2J';
      await handler(toolCallEvent('bash', { command }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).not.toContain('\u001b');
      expect(title).not.toContain('\u009b');
      // The escape bytes are rendered as text; the CSI parameter/final
      // bytes (which bash would still parse) stay verbatim and in place.
      expect(title).toContain('echo safe\\x1b[2J\\x1b[1;1H\\x1b[31m && rm -rf /\\x9b2J');
      expectPreservesPrintables(command, title);
    });

    it('encodes the OSC introducer and terminator; the payload text stays visible', async () => {
      // The OSC "payload" is ordinary printable text. Under the old
      // deletion regime it vanished from the prompt even though bash
      // still parses those bytes — it must be shown.
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      const command = 'ls \u001b]0;spoofed-window-title\u0007-la';
      await handler(toolCallEvent('bash', { command }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).not.toContain('\u001b');
      expect(title).not.toContain('\u0007');
      expect(title).toContain('ls \\x1b]0;spoofed-window-title\\x07-la');
      expectPreservesPrintables(command, title);
    });

    it('renders non-escape control characters (BEL, CR, NUL) as \\xNN text', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      const command = 'echo a\u0007b\rc\u0000d';
      await handler(toolCallEvent('bash', { command }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).not.toContain('\u0007');
      expect(title).not.toContain('\r');
      expect(title).not.toContain('\u0000');
      expect(title).toContain('echo a\\x07b\\x0dc\\x00d');
      expectPreservesPrintables(command, title);
    });

    it('renders DEL as \\x7f text', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      const command = 'echo x\u007fy';
      await handler(toolCallEvent('bash', { command }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).not.toContain('\u007f');
      expect(title).toContain('echo x\\x7fy');
    });

    it('regression: an OSC-shaped span cannot hide an executed command', async () => {
      // Reviewer demonstration against pi's real bash backend: bash
      // executes `printf DANGEROUS` between the ESC] and BEL, but the
      // old sanitizer deleted the whole span as an "OSC payload" and the
      // prompt showed only `printf SAFE; : `.
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      const command = 'printf SAFE; : \u001b]0; printf DANGEROUS; : \u0007';
      await handler(toolCallEvent('bash', { command }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).toContain('printf SAFE');
      expect(title).toContain('printf DANGEROUS');
      expect(title).toContain('\\x1b]0;');
      expect(title).toContain('\\x07');
      expect(title).not.toContain('\u001b');
      expect(title).not.toContain('\u0007');
      expectPreservesPrintables(command, title);
    });

    it('regression: a CSI-shaped prefix cannot hide a real pipe', async () => {
      // Reviewer demonstration: `|` is a valid CSI final byte, so the
      // old sanitizer deleted `ESC[|` wholesale and the pipe disappeared
      // from the prompt while bash still piped into `printf HIDDEN_PIPE`.
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      const command = ': \u001b[| printf HIDDEN_PIPE';
      await handler(toolCallEvent('bash', { command }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).toContain('\\x1b[| printf HIDDEN_PIPE');
      expect(title).not.toContain('\u001b');
      expectPreservesPrintables(command, title);
    });

    it('regression: ESC-backslash (two-byte ST) is encoded, not swallowed', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      const command = 'echo first\u001b\\echo second';
      await handler(toolCallEvent('bash', { command }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).toContain('echo first\\x1b\\echo second');
      expect(title).not.toContain('\u001b');
      expectPreservesPrintables(command, title);
    });

    it('regression: the C1 ST byte U+009C is encoded visibly', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      const command = 'echo a\u009cecho b';
      await handler(toolCallEvent('bash', { command }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).toContain('echo a\\x9cecho b');
      expect(title).not.toContain('\u009c');
      expectPreservesPrintables(command, title);
    });

    it('regression: an unterminated OSC prefix cannot swallow the following line', async () => {
      // The old OSC regex made the terminator optional, so an ESC] with
      // no BEL/ST consumed everything after it — including the next
      // executable line.
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      const command = 'echo first \u001b]0;no-terminator\necho second';
      await handler(toolCallEvent('bash', { command }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).toContain('echo first \\x1b]0;no-terminator\necho second');
      expect(title).not.toContain('\u001b');
      expectPreservesPrintables(command, title);
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

    it('a command that is nothing but escape bytes is shown fully encoded', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      await handler(toolCallEvent('bash', { command: '\u001b[2J\u001b[3J\u001b[H' }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).not.toContain('\u001b');
      expect(title).toContain('\\x1b[2J\\x1b[3J\\x1b[H');
      expect(title).not.toContain('(unprintable content)');
    });

    it('a whitespace-only command shows a visible stand-in', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      await handler(toolCallEvent('bash', { command: '  \n \t ' }), ctx);
      expect(selectCalls[0]!.title).toContain('(unprintable content)');
    });

    it('encodes control bytes in the write/edit path detail too', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      const path = '/tmp/\u001b[31mevil\u001b[0m.txt';
      await handler(toolCallEvent('write', { path, content: '' }), ctx);
      const title = selectCalls[0]!.title;
      expect(title).not.toContain('\u001b');
      expect(title).toContain('/tmp/\\x1b[31mevil\\x1b[0m.txt');
      expectPreservesPrintables(path, title);
    });

    it('leaves a normal command and path untouched (no marker, no encoding)', async () => {
      const handler = loadGateHandler();
      const { ctx, selectCalls } = makeCtx(true, 'Yes');
      await handler(toolCallEvent('bash', { command: 'yarn build:native --verbose' }), ctx);
      await handler(toolCallEvent('edit', { path: '/repo/src/index.ts', oldText: 'a', newText: 'b' }), ctx);
      expect(selectCalls[0]!.title).toBe('Allow bash?\n\n  yarn build:native --verbose');
      expect(selectCalls[1]!.title).toBe('Allow edit?\n\n  /repo/src/index.ts');
    });
  });

  describe('pi shellCommandPrefix is surfaced in the bash prompt', () => {
    // pi's BashTool runs `${shellCommandPrefix}\n${command}`, so a non-empty
    // prefix executes shell bytes the model never named. The gate snapshots the
    // effective prefix at `session_start` (the same instant pi bakes it into
    // BashTool) via a DIRECT, lock-free read, then shows it (displayed ==
    // executed). Resolution is fail-safe: any failure resolves to the bare
    // command and never throws.
    async function withGlobalAgentSettings(
      settings: string | undefined,
      fn: (agentDir: string) => Promise<void>,
    ): Promise<void> {
      const agentDir = await mkdtemp(join(tmpdir(), 'mlx-gate-agent-'));
      try {
        if (settings !== undefined) {
          await writeFile(join(agentDir, 'settings.json'), settings);
        }
        process.env['PI_CODING_AGENT_DIR'] = agentDir;
        await fn(agentDir);
      } finally {
        await rm(agentDir, { recursive: true, force: true });
      }
    }

    /** Fresh, unique project cwd so each test resolves against its own isolated settings. */
    function freshProject(): Promise<string> {
      return mkdtemp(join(tmpdir(), 'mlx-gate-project-'));
    }

    it('snapshots a GLOBAL settings.json shellCommandPrefix at session_start and shows it', async () => {
      await withGlobalAgentSettings(JSON.stringify({ shellCommandPrefix: 'sudo -k --' }), async () => {
        const cwd = await freshProject();
        try {
          const { toolCall, sessionStart } = loadGate();
          const { ctx, selectCalls } = makeCtx(true, 'Yes', { cwd, isProjectTrusted: () => true });
          await sessionStart(sessionStartEvent(), ctx);
          await toolCall(toolCallEvent('bash', { command: 'rm -rf /tmp/scratch' }), ctx);
          expect(selectCalls).toHaveLength(1);
          // Displayed == executed: the hidden prefix and the command both show.
          expect(selectCalls[0]!.title).toContain('sudo -k --');
          expect(selectCalls[0]!.title).toContain('rm -rf /tmp/scratch');
        } finally {
          await rm(cwd, { recursive: true, force: true });
        }
      });
    });

    it('FAITHFULNESS: keeps the session_start snapshot even after settings.json changes without a reload', async () => {
      // pi bakes the prefix into BashTool at session_start; if the file is then
      // edited WITHOUT a /reload, BashTool still runs the BAKED (snapshotted)
      // value. The gate must show that same baked value — an on-demand re-read
      // would drift to the new file contents and misrepresent what executes.
      // This is the mutation guard: reverting to a per-call / on-demand-primary
      // read makes the second assertion show PFX_B and this test fail.
      await withGlobalAgentSettings(JSON.stringify({ shellCommandPrefix: 'PFX_A' }), async (agentDir) => {
        const cwd = await freshProject();
        try {
          const { toolCall, sessionStart } = loadGate();
          const ctxTrusted = { cwd, isProjectTrusted: () => true };

          const a = makeCtx(true, 'Yes', ctxTrusted);
          await sessionStart(sessionStartEvent(), a.ctx);
          await toolCall(toolCallEvent('bash', { command: 'echo hi' }), a.ctx);
          expect(a.selectCalls[0]!.title).toContain('PFX_A');

          // Edit the file with NO further session_start.
          await writeFile(join(agentDir, 'settings.json'), JSON.stringify({ shellCommandPrefix: 'PFX_B' }));
          const b = makeCtx(true, 'Yes', ctxTrusted);
          await toolCall(toolCallEvent('bash', { command: 'echo hi' }), b.ctx);
          expect(b.selectCalls[0]!.title).toContain('PFX_A');
          expect(b.selectCalls[0]!.title).not.toContain('PFX_B');

          // A fresh session_start (simulated /reload) re-snapshots → now PFX_B.
          const c = makeCtx(true, 'Yes', ctxTrusted);
          await sessionStart(sessionStartEvent(), c.ctx);
          await toolCall(toolCallEvent('bash', { command: 'echo hi' }), c.ctx);
          expect(c.selectCalls[0]!.title).toContain('PFX_B');
          expect(c.selectCalls[0]!.title).not.toContain('PFX_A');
        } finally {
          await rm(cwd, { recursive: true, force: true });
        }
      });
    });

    it('shows a TRUSTED project .pi/settings.json prefix but HIDES an untrusted one (matches pi)', async () => {
      await withGlobalAgentSettings(undefined, async () => {
        // Trusted project → pi loads `<cwd>/.pi/settings.json` → prefix shown.
        const trustedCwd = await freshProject();
        try {
          await mkdir(join(trustedCwd, '.pi'), { recursive: true });
          await writeFile(
            join(trustedCwd, '.pi', 'settings.json'),
            JSON.stringify({ shellCommandPrefix: 'PROJECT_PREFIX' }),
          );
          const { toolCall, sessionStart } = loadGate();
          const { ctx, selectCalls } = makeCtx(true, 'Yes', { cwd: trustedCwd, isProjectTrusted: () => true });
          await sessionStart(sessionStartEvent(), ctx);
          await toolCall(toolCallEvent('bash', { command: 'make build' }), ctx);
          expect(selectCalls[0]!.title).toContain('PROJECT_PREFIX');
        } finally {
          await rm(trustedCwd, { recursive: true, force: true });
        }

        // Untrusted project → pi drops the project layer → prefix NOT shown.
        const untrustedCwd = await freshProject();
        try {
          await mkdir(join(untrustedCwd, '.pi'), { recursive: true });
          await writeFile(
            join(untrustedCwd, '.pi', 'settings.json'),
            JSON.stringify({ shellCommandPrefix: 'PROJECT_PREFIX' }),
          );
          const { toolCall, sessionStart } = loadGate();
          const { ctx, selectCalls } = makeCtx(true, 'Yes', { cwd: untrustedCwd, isProjectTrusted: () => false });
          await sessionStart(sessionStartEvent(), ctx);
          await toolCall(toolCallEvent('bash', { command: 'make build' }), ctx);
          expect(selectCalls[0]!.title).not.toContain('PROJECT_PREFIX');
          expect(selectCalls[0]!.title).toContain('make build');
        } finally {
          await rm(untrustedCwd, { recursive: true, force: true });
        }
      });
    });

    it('snapshots the bare command when no prefix is configured', async () => {
      await withGlobalAgentSettings(JSON.stringify({ theme: 'dark' }), async () => {
        const cwd = await freshProject();
        try {
          const { toolCall, sessionStart } = loadGate();
          const { ctx, selectCalls } = makeCtx(true, 'Yes', { cwd, isProjectTrusted: () => true });
          await sessionStart(sessionStartEvent(), ctx);
          await toolCall(toolCallEvent('bash', { command: 'yarn build' }), ctx);
          expect(selectCalls[0]!.title).toBe('Allow bash?\n\n  yarn build');
        } finally {
          await rm(cwd, { recursive: true, force: true });
        }
      });
    });

    it('a malformed/absent settings.json at session_start snapshots the bare command and does not crash', async () => {
      // pi's own load treats a malformed/absent settings file as `{}` and bakes
      // an empty prefix; the direct reader mirrors that, so the snapshot is `''`
      // and the prompt shows the bare command. The empty snapshot is authoritative
      // — it must NOT trigger an on-demand re-read.
      await withGlobalAgentSettings('{ this is not valid json', async () => {
        const cwd = await freshProject();
        try {
          const { toolCall, sessionStart } = loadGate();
          const { ctx, selectCalls } = makeCtx(true, 'Yes', { cwd, isProjectTrusted: () => true });
          await sessionStart(sessionStartEvent(), ctx);
          await toolCall(toolCallEvent('bash', { command: 'echo hi' }), ctx);
          expect(selectCalls).toHaveLength(1);
          expect(selectCalls[0]!.title).toBe('Allow bash?\n\n  echo hi');
        } finally {
          await rm(cwd, { recursive: true, force: true });
        }
      });
    });

    it('never crashes and shows the bare command when the ctx trust getter throws', async () => {
      // No session_start fired → the bash branch falls back to a one-shot
      // on-demand read, which must swallow a hostile `ctx.isProjectTrusted`.
      const cwd = await freshProject();
      try {
        const handler = loadGateHandler();
        const { ctx, selectCalls } = makeCtx(true, 'No', {
          cwd,
          isProjectTrusted: () => {
            throw new Error('trust check exploded');
          },
        });
        const result = await handler(toolCallEvent('bash', { command: 'echo hi' }), ctx);
        // The gate still prompted (bare command) and returned a decision.
        expect(selectCalls).toHaveLength(1);
        expect(selectCalls[0]!.title).toBe('Allow bash?\n\n  echo hi');
        expect(result).toEqual({ block: true, reason: 'Blocked by user' });
      } finally {
        await rm(cwd, { recursive: true, force: true });
      }
    });

    it('falls back to an on-demand read when a bash approval precedes session_start', async () => {
      // Structurally rare, but the snapshot may not exist yet. The bash branch
      // then reads on demand (lock-free, never throws) rather than showing empty.
      await withGlobalAgentSettings(JSON.stringify({ shellCommandPrefix: 'FALLBACK_PFX' }), async () => {
        const cwd = await freshProject();
        try {
          const { toolCall } = loadGate();
          const { ctx, selectCalls } = makeCtx(true, 'Yes', { cwd, isProjectTrusted: () => true });
          // Note: no sessionStart() call here.
          await toolCall(toolCallEvent('bash', { command: 'echo hi' }), ctx);
          expect(selectCalls[0]!.title).toContain('FALLBACK_PFX');
        } finally {
          await rm(cwd, { recursive: true, force: true });
        }
      });
    });

    it('does not consult a shell prefix for non-bash tools', async () => {
      await withGlobalAgentSettings(JSON.stringify({ shellCommandPrefix: 'SHOULD_NOT_APPEAR' }), async () => {
        const cwd = await freshProject();
        try {
          const { toolCall, sessionStart } = loadGate();
          const { ctx, selectCalls } = makeCtx(true, 'Yes', { cwd, isProjectTrusted: () => true });
          await sessionStart(sessionStartEvent(), ctx);
          await toolCall(toolCallEvent('write', { path: '/tmp/out.txt', content: 'x' }), ctx);
          expect(selectCalls[0]!.title).not.toContain('SHOULD_NOT_APPEAR');
          expect(selectCalls[0]!.title).toContain('/tmp/out.txt');
        } finally {
          await rm(cwd, { recursive: true, force: true });
        }
      });
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
