import { describe, expect, it, vi } from 'vite-plus/test';

import { detectionCases, selectedText } from '../../../evals/agent-installed/cases.js';
import { cliInvocation, JsonLines } from '../../../evals/agent-installed/cli.js';
import { evaluateCase, grade, summarize } from '../../../evals/agent-installed/evaluate.js';
import { physicalFootprint } from '../../../evals/agent-installed/process-memory.js';
import { evalOptions } from '../../../scripts/eval-agent-installed.js';
import { DETECTION_SYSTEM, detectionMessage } from '../src/coding-agent-detection.js';

const current = detectionCases.find((c) => c.id === 'generated-prompt')!;
const connection = { url: 'http://127.0.0.1:1234', model: 'fixture-default', token: 'fixture-token' };

describe('agent-installed eval oracle and execution', () => {
  it('requires correct labels and exact annotated evidence, including alternative valid spans', () => {
    expect(grade(current, 'needs-update', [1, 1])).toMatchObject({ verdict: false });
    expect(grade(current, 'installed', [2, 2])).toMatchObject({ verdict: true, evidence: false });
    expect(grade(current, 'installed')).toMatchObject({ evidence: false });
    const multiple = detectionCases.find((c) => c.id === 'two-obsolete-routes')!;
    expect(grade(multiple, 'needs-update', [2, 2])).toEqual({ verdict: true, evidence: true });
    expect(grade(multiple, 'needs-update', [1, 2])).toEqual({ verdict: true, evidence: false });
  });

  it('keeps fixture IDs unique and source labels in the selected file, including overrides', () => {
    expect(new Set(detectionCases.map((c) => c.id)).size).toBe(detectionCases.length);
    for (const fixture of detectionCases) {
      const lines = selectedText(fixture).split('\n');
      expect(fixture.ranges.length === 0).toBe(fixture.expected === 'not-installed');
      for (const [start, end] of fixture.ranges) {
        expect(start).toBeGreaterThanOrEqual(1);
        expect(end).toBeGreaterThanOrEqual(start);
        expect(end).toBeLessThanOrEqual(lines.length);
        expect(
          lines
            .slice(start - 1, end)
            .join('\n')
            .trim(),
        ).not.toBe('');
      }
    }
  });

  it('uses the production prompt/budget and checks persisted cache reuse, invalidation and forced rechecks', async () => {
    const complete = vi
      .fn()
      .mockResolvedValueOnce('{"status":"installed","startLine":1,"endLine":1}')
      .mockResolvedValue('{"status":"not-installed","startLine":0,"endLine":0}');
    const result = await evaluateCase(current, connection, complete, undefined, true);
    expect(result.passed).toBe(true);
    expect(result.checks).toMatchObject({
      memoryCache: true,
      diskCache: true,
      contentInvalidation: true,
      forcedRecheck: true,
      preservedFiles: true,
    });
    expect(complete).toHaveBeenCalledTimes(3);
    expect(complete.mock.calls[0]).toEqual([
      connection,
      DETECTION_SYSTEM,
      [{ role: 'user', content: detectionMessage(current.text, current.command) }],
      expect.any(AbortSignal),
      16384,
    ]);
  });

  it('fails an always-installed model and keeps false positives in the denominator', async () => {
    const negative = detectionCases.find((c) => c.id === 'capability')!;
    const result = await evaluateCase(
      negative,
      connection,
      vi.fn().mockResolvedValue('{"status":"installed","startLine":1,"endLine":1}'),
    );
    expect(result.passed).toBe(false);
    expect(summarize([result])).toMatchObject({ total: 1, passed: 0, falseInstalled: 1 });
  });

  it.each(['{broken', '{"status":"installed","startLine":999,"endLine":999}'])(
    'counts invalid output as an error and a failed sample: %s',
    async (answer) => {
      const result = await evaluateCase(current, connection, vi.fn().mockResolvedValue(answer));
      expect(result.passed).toBe(false);
      expect(summarize([result])).toMatchObject({ total: 1, passed: 0, errors: 1 });
    },
  );

  it('counts inference failures instead of dropping them', async () => {
    const result = await evaluateCase(current, connection, vi.fn().mockRejectedValue(new Error('output space')));
    expect(result.passed).toBe(false);
    expect(result.calls[0].error).toContain('output space');
    expect(summarize([result])).toMatchObject({ total: 1, passed: 0, errors: 1 });
  });

  it('does not invoke a model for empty files, and selects Codex overrides through the real service', async () => {
    const complete = vi.fn().mockResolvedValue('{"status":"installed","startLine":1,"endLine":1}');
    expect(
      (
        await evaluateCase(
          detectionCases.find((c) => c.id === 'empty')!,
          connection,
          complete,
        )
      ).passed,
    ).toBe(true);
    expect(complete).not.toHaveBeenCalled();
    const override = detectionCases.find((c) => c.id === 'codex-override-enables')!;
    expect((await evaluateCase(override, connection, complete)).passed).toBe(true);
    expect(complete.mock.calls[0][2][0].content).toBe(detectionMessage(override.override!, override.command));
  });

  it('uses the real CLI without overriding runtime defaults', () => {
    expect(evalOptions(['--entrypoint', 'app']).entrypoint).toBe('app');
    expect(cliInvocation('/repo', 'agent')).toEqual(['/repo/packages/cli/dist/cli.js', 'agent', '--mode', 'rpc']);
    expect(cliInvocation('/repo', 'delegate')).toEqual(['/repo/packages/cli/dist/cli.js', 'delegate', '--mode', 'rpc']);
    expect(() => evalOptions(['--model', 'replacement'])).toThrow();
    expect(() => evalOptions(['--thinking', 'off'])).toThrow();
    expect(() => evalOptions(['--entrypoint', 'serve'])).toThrow();
  });

  it('refuses empty selections and invalid repeat counts rather than producing empty green reports', () => {
    expect(() => evalOptions(['--repeat', '0'])).toThrow();
    expect(() => evalOptions(['--repeat', 'NaN'])).toThrow();
    expect(() => evalOptions(['--max-memory-gb', 'NaN'])).toThrow();
    expect(() => evalOptions(['--cache-limit-gb', '0'])).toThrow();
    expect(() => evalOptions(['--case', 'typo'])).toThrow();
    expect(() => evalOptions(['--case', 'generated-prompt', '--split', 'holdout'])).toThrow('No cases selected');
  });

  it('reads peak physical memory without confusing it with resident memory', () => {
    expect(physicalFootprint('Physical footprint: 27.0G\nPhysical footprint (peak): 28.2G\n')).toBe(28.2 * 2 ** 30);
    expect(physicalFootprint('Physical footprint: 512M\n')).toBe(512 * 2 ** 20);
    expect(physicalFootprint('Permission denied')).toBeUndefined();
  });

  it('keeps RPC events intact across byte boundaries and Unicode line separators', () => {
    const receive = vi.fn();
    const stream = new JsonLines(receive);
    const event = {
      type: 'message_end',
      message: { role: 'assistant', content: [{ type: 'text', text: '中\u2028文\u2029é' }] },
    };
    const bytes = Buffer.from(JSON.stringify(event) + '\n' + JSON.stringify({ type: 'agent_settled' }) + '\r\n');
    for (const byte of bytes) stream.push(Buffer.from([byte]));
    expect(receive.mock.calls).toEqual([[event], [{ type: 'agent_settled' }]]);
  });
});
