interface FixtureEntry {
  type: string;
  id: string;
  parentId?: string | null;
  timestamp: string;
  message?: { content?: unknown; stopReason?: string; [key: string]: unknown };
  [key: string]: unknown;
}

/** An explicit delegate run; prompts/reasoning/usage are deliberately much larger than its evidence. */
export function delegationFixture(id = 'delegate', evidence = 'alpha beta gamma', handoff = 'done'): FixtureEntry[] {
  const timestamp = '2026-09-14T12:00:00.000Z';
  return [
    { type: 'session', version: 3, id, timestamp, cwd: '/w' },
    {
      type: 'custom',
      id: 'd',
      parentId: null,
      timestamp,
      customType: 'mlx-delegate-session',
      data: { version: 1, sessionId: id },
    },
    {
      type: 'message',
      id: 'u',
      parentId: 'd',
      timestamp,
      message: { role: 'user', content: 'Investigate '.repeat(100) },
    },
    {
      type: 'message',
      id: 'a',
      parentId: 'u',
      timestamp,
      message: {
        role: 'assistant',
        stopReason: 'toolUse',
        model: 'local',
        content: [
          { type: 'thinking', thinking: 'reasoning '.repeat(100) },
          { type: 'toolCall', id: 'call', name: 'bash', arguments: { command: 'gh pr view 42' } },
        ],
        usage: { input: 1000, output: 1000, cacheRead: 9000 },
      },
    },
    {
      type: 'message',
      id: 'e',
      parentId: 'a',
      timestamp,
      message: {
        role: 'toolResult',
        toolCallId: 'call',
        toolName: 'bash',
        isError: false,
        content: [{ type: 'text', text: evidence }],
      },
    },
    {
      type: 'message',
      id: 'f',
      parentId: 'e',
      timestamp,
      message: {
        role: 'assistant',
        stopReason: 'stop',
        model: 'local',
        content: [{ type: 'text', text: handoff }],
        usage: { input: 1000, output: 1000, cacheRead: 9000 },
      },
    },
  ];
}
