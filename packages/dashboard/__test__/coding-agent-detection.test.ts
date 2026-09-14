import { describe, expect, it } from 'vite-plus/test';

import {
  DETECTION_INPUT_PREFIX,
  detectionMessage,
  detectionResult,
  commandSelectionSource,
  commandSelectionResult,
} from '../src/coding-agent-detection.js';

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

describe('exact model-selected command spans', () => {
  it('keeps inline preferences, CRLF and Unicode outside the selected command', () => {
    const text =
      "# Preferences\r\nKeep 🌲 patches small; for PRs use '/old mlx' delegate github --repo org/repo; keep tests fast.\r\n";
    const source = { startLine: 2, endLine: 2 };
    const selected = "'/old mlx' delegate github";
    const span = commandSelectionResult({ text: selected, occurrence: 1 }, text, source);
    expect(text.slice(span.start, span.end)).toBe(selected);
    expect(text.slice(0, span.start)).toBe('# Preferences\r\nKeep 🌲 patches small; for PRs use ');
    expect(text.slice(span.end)).toBe(' --repo org/repo; keep tests fast.\r\n');
  });

  it('selects a later occurrence without touching the earlier example', () => {
    const text = 'Example: mlx delegate github. For CI, use mlx delegate github.';
    const span = commandSelectionResult({ text: 'mlx delegate github', occurrence: 2 }, text, {
      startLine: 1,
      endLine: 1,
    });
    expect(span.start).toBe(text.lastIndexOf('mlx delegate github'));
  });

  it('selects an exact multiline command using original coordinates', () => {
    const text = "Before.\nFor CI use '/old/mlx'\n delegate github. Keep tests.\nAfter.";
    const source = { startLine: 2, endLine: 3 };
    expect(commandSelectionSource(text, source)).toBe("For CI use '/old/mlx'\n delegate github. Keep tests.");
    const selected = "'/old/mlx'\n delegate github";
    const span = commandSelectionResult({ text: selected, occurrence: 1 }, text, source);
    expect(text.slice(span.start, span.end)).toBe(selected);
  });

  it.each([
    'mlx\tdelegate\tgithub',
    '"/old app/mlx" delegate github --caller-approved',
    '/old\\ app/mlx delegate github',
    "'/old'\"'\"'/mlx' delegate github",
    '"/old \\"app\\"/mlx" delegate github',
    "'/old/mlx' \\\n delegate github",
    "'/old/mlx'\r\n delegate github",
    'mlx \'delegate\' "github"',
  ])('accepts complete literal shell prefixes and preserves adjacent Markdown: %s', (selected) => {
    const text = `Keep 🌲 patches small; use \`${selected}\` for CI.`;
    const span = commandSelectionResult({ text: selected, occurrence: 1 }, text, {
      startLine: 1,
      endLine: text.split('\n').length,
    });
    expect(text.slice(span.start, span.end)).toBe(selected);
    expect(text.slice(0, span.start)).toBe('Keep 🌲 patches small; use `');
    expect(text.slice(span.end)).toBe('` for CI.');
  });

  it.each([
    'Keep patches small',
    "'Keep patches small' delegate github",
    'mlx',
    'mlx agent',
    'notmlx delegate github',
    'mlx delegate github-backup',
    'mlx delegate github --repo org/repo',
    'mlx delegate github --caller-approved=false',
    'mlx delegate github --caller-approved --allow-write',
    'mlx delegate github; keep tests fast',
    'mlx delegate github\nKeep tests fast',
    '`mlx delegate github`',
    '"mlx delegate github"',
    "'/old/mlx delegate github",
    '"$HOME/mlx" delegate github',
    '$(echo mlx) delegate github',
    '"`echo mlx`" delegate github',
    'mlx delegate github ',
    ' mlx delegate github',
    'mlx delegate github\\',
  ])('rejects exact source matches that are not a command prefix: %s', (selected) => {
    const text = `Use ${selected} for CI.`;
    expect(() =>
      commandSelectionResult({ text: selected, occurrence: 1 }, text, {
        startLine: 1,
        endLine: text.split('\n').length,
      }),
    ).toThrow('No changes were made');
  });

  it.each([
    ['notmlx delegate github', 'mlx delegate github'],
    ['/other/old/mlx delegate github', '/old/mlx delegate github'],
    ['mlx delegate github-backup', 'mlx delegate github'],
    ['mlx delegate github.com', 'mlx delegate github'],
    ['mlx delegate github --caller-approved=false', 'mlx delegate github --caller-approved'],
  ])('rejects selections that cut into another command token: %s', (invocation, selected) => {
    expect(() =>
      commandSelectionResult({ text: selected, occurrence: 1 }, `Use ${invocation} for CI.`, {
        startLine: 1,
        endLine: 1,
      }),
    ).toThrow('No changes were made');
  });

  it.each([
    null,
    {},
    { text: '', occurrence: 0 },
    { text: 'invented', occurrence: 1 },
    { text: 'mlx delegate github', occurrence: 2 },
    { text: 'mlx delegate github', occurrence: Number.MAX_SAFE_INTEGER },
    { text: 'mlx delegate github', occurrence: '1' },
    { text: 'mlx delegate github', occurrence: 0.5 },
  ])('rejects unverifiable selections without editing: %j', (answer) => {
    expect(() => commandSelectionResult(answer, 'Use mlx delegate github.', { startLine: 1, endLine: 1 })).toThrow(
      'No changes were made',
    );
  });

  it('cannot select matching text outside the model-selected lines', () => {
    expect(() =>
      commandSelectionResult({ text: 'mlx delegate github', occurrence: 1 }, 'Use mlx delegate github.\nKeep tests.', {
        startLine: 2,
        endLine: 2,
      }),
    ).toThrow('No changes were made');
  });
});
