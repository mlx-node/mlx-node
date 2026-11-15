/**
 * `mlx_web`: the stdio MCP server the Chat agent starts once per session. It imports only Node built-ins and its own
 * folder; it never spawns a process, reads or writes a file, opens a listening socket or follows a redirect; and it
 * writes only the fixed lines of `errors.ts` to stderr, because the agent saves stderr to disk.
 */

import { writeStderrLine } from './errors.js';
import { answerFrame, FrameReader } from './mcp.js';
import { readWebConfig, scrubEnv, WebTools } from './tools.js';

function fatal(): void {
  writeStderrLine('fatal');
  process.exit(1);
}

process.on('uncaughtException', fatal);
process.on('unhandledRejection', fatal);
process.removeAllListeners('warning');

const config = readWebConfig(process.env);
scrubEnv(process.env);
const tools = new WebTools(config);

function reply(message: Record<string, unknown>): void {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

const reader = new FrameReader({
  frame(text) {
    let frame: unknown;
    try {
      frame = JSON.parse(text);
    } catch {
      writeStderrLine('badFrame');
      return;
    }
    void answerFrame(frame, tools).then(
      (answer) => {
        if (answer !== null) reply(answer);
      },
      () => writeStderrLine('fatal'),
    );
  },
  tooLarge() {
    writeStderrLine('frameTooLarge');
  },
});

process.stdin.on('data', (chunk: Buffer | string) => {
  reader.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
});
// On stdin end nothing holds the loop open: the process exits once the last reply is flushed.
process.stdin.on('end', () => process.stdin.destroy());
