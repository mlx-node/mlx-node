import { delegationCommand } from '@mlx-node/agent/delegate';
import { describe, expect, it } from 'vite-plus/test';

import { DETECTION_INPUT_PREFIX, detectionMessage, detectionResult } from '../src/coding-agent-detection.js';

const command = '/Users/test/.mlx-node/bin/mlx';
const yes = { installed: true, startLine: 1, endLine: 1 };

describe('installation evidence selected by source line', () => {
  it('numbers the original text without JSON escaping quotes or treating embedded line numbers as metadata', () => {
    const text = `# Preferences\n\n99 | Run '${command}' delegate github --repo org/repo "TASK".\n`;
    expect(detectionMessage(text)).toBe(
      DETECTION_INPUT_PREFIX +
        `1 | # Preferences\n2 | \n3 | 99 | Run '${command}' delegate github --repo org/repo "TASK".\n4 | `,
    );
  });

  it.each([
    [command, `'${command}'`],
    [command, `"${command}"`],
    [command, command],
    ['/Users/Test User/bin/mlx', "'/Users/Test User/bin/mlx'"],
    ["/Users/O'Brien/bin/mlx", delegationCommand("/Users/O'Brien/bin/mlx")],
  ])('uses the original command quoting for %s', (path, spelling) => {
    expect(detectionResult(yes, `For CI, run ${spelling} delegate github.`, path)).toEqual({
      installed: true,
      needsUpdate: false,
    });
  });

  it.each(['.', ',', ';', ':', '!', '?', '', '`', ' --repo org/repo'])(
    'accepts a complete command followed by %j',
    (suffix) => {
      expect(detectionResult(yes, `Use '${command}' delegate github${suffix}`, command).installed).toBe(true);
    },
  );

  it('reads a range from the original file and ignores a different command elsewhere in the file', () => {
    const text = `Old example: '${command}' delegate github.\nUse '/old/bin/mlx'\n  delegate github for CI.`;
    expect(detectionResult({ installed: true, startLine: 2, endLine: 3 }, text, command)).toEqual({
      installed: false,
      needsUpdate: true,
    });
  });

  it.each([
    'mlx',
    '/old/bin/mlx',
    "'/old path/bin/mlx'",
    '"/old path/bin/mlx"',
    delegationCommand("/Users/O'Brien/bin/mlx"),
  ])('recognizes an active older executable %s', (spelling) => {
    expect(detectionResult(yes, `Use ${spelling} delegate github.`, command)).toEqual({
      installed: false,
      needsUpdate: true,
    });
  });

  it.each([
    `Use '${command}' agent --print.`,
    `Use '${command}' delegate github-backup.`,
    `Use '${command}' delegate github.json.`,
    `Use '${command}' delegate github/repo.`,
    `Use '/tools/notmlx' delegate github.`,
    'There is no command here.',
  ])('refuses a positive answer pointing to a different command: %s', (text) => {
    expect(() => detectionResult(yes, text, command)).toThrow('verify its answer');
  });

  it.each([
    null,
    { installed: 'true', startLine: 1, endLine: 1 },
    { installed: true, evidence: 'invented text' },
    { installed: true, startLine: '1', endLine: 1 },
    { installed: true, startLine: 0, endLine: 1 },
    { installed: true, startLine: 1, endLine: 99 },
    { installed: true, startLine: 2, endLine: 1 },
    { installed: true, startLine: 1.5, endLine: 1.5 },
    { installed: false, startLine: 1, endLine: 1 },
  ])('rejects malformed or invented source ranges: %j', (answer) => {
    expect(() => detectionResult(answer, `Use '${command}' delegate github.`, command)).toThrow('verify its answer');
  });

  it('accepts an explicit negative verdict without evidence', () => {
    expect(detectionResult({ installed: false, startLine: 0, endLine: 0 }, 'Use gh directly.', command)).toEqual({
      installed: false,
      needsUpdate: false,
    });
  });
});
