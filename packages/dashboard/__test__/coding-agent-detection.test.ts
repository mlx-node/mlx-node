import { describe, expect, it } from 'vite-plus/test';

import { DETECTION_INPUT_PREFIX, detectionMessage, detectionResult } from '../src/coding-agent-detection.js';

const command = '/Users/test/.mlx-node/bin/mlx';

describe('model installation verdicts with source references', () => {
  it('supplies the current executable separately from the numbered original file', () => {
    const text = `# Preferences\n\n99 | Run '${command}' delegate github --repo org/repo "TASK".\n`;
    expect(detectionMessage(text, command)).toBe(
      `Current app executable: ${JSON.stringify(command)}\n` +
        DETECTION_INPUT_PREFIX +
        `1 | # Preferences\n2 | \n3 | 99 | Run '${command}' delegate github --repo org/repo "TASK".\n4 | `,
    );
  });

  it('keeps spaces and quotes in the caller-provided executable', () => {
    const path = '/Users/Test "Work"/bin/mlx';
    expect(detectionMessage('Use the delegation tool.', path).split('\n')[0]).toBe(
      `Current app executable: ${JSON.stringify(path)}`,
    );
  });

  it.each([
    ['installed', { installed: true, needsUpdate: false }],
    ['needs-update', { installed: false, needsUpdate: true }],
  ] as const)('honors the model status %s without reclassifying command formatting', (status, expected) => {
    // Markdown emphasis and line breaks used to be rejected by the command regex.
    const text = `# Routing\nFor GitHub work, use **${command}**\nwith the delegate github subcommands.`;
    expect(detectionResult({ status, startLine: 2, endLine: 3 }, text)).toEqual({
      ...expected,
      source: { startLine: 2, endLine: 3 },
    });
  });

  it('accepts an explicit negative verdict without evidence', () => {
    expect(detectionResult({ status: 'not-installed', startLine: 0, endLine: 0 }, 'Use gh directly.')).toEqual({
      installed: false,
      needsUpdate: false,
    });
  });

  it.each([
    null,
    { installed: true, startLine: 1, endLine: 1 },
    { status: 'unknown', startLine: 1, endLine: 1 },
    { status: 'installed', startLine: '1', endLine: 1 },
    { status: 'installed', startLine: 0, endLine: 1 },
    { status: 'installed', startLine: 1, endLine: 99 },
    { status: 'installed', startLine: 2, endLine: 1 },
    { status: 'installed', startLine: 1.5, endLine: 1.5 },
    { status: 'needs-update', startLine: 0, endLine: 0 },
    { status: 'needs-update', startLine: 1, endLine: 99 },
    { status: 'not-installed', startLine: 1, endLine: 1 },
  ])('rejects malformed responses and invalid source ranges: %j', (answer) => {
    expect(() => detectionResult(answer, `Use '${command}' delegate github.`)).toThrow('verify its answer');
  });

  it.each(['installed', 'needs-update'])('requires nonempty source evidence for %s', (status) => {
    expect(() => detectionResult({ status, startLine: 2, endLine: 2 }, 'First line.\n   \nThird line.')).toThrow(
      'verify its answer',
    );
  });
});
